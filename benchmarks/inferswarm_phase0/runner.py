"""Campaign assembly: server command construction, the run loop, and the run artifact.

Held-constant configuration (criteria section 3) is passed explicitly on every arm --
``--memory-ratio``, ``--kv-reserve-tokens``, ``--max-running-requests``,
``--cuda-graph-max-bs``, ``--max-seq-len-override``, ``--sampling-defaults`` -- so no arm can
differ in something the criteria hold fixed, and so the record does not depend on a default
that may change between FreeToken versions.

Anti-starvation, mechanically: the sweep arms always carry ``--moe-cache-auto`` (from the
arm definition), and this module offers no way to lower ``--moe-cache-size`` /
``--moe-cache-rate`` below the auto-resolved value on a performance arm.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Any, Dict, List, Sequence

from . import HARNESS_VERSION
from .artifacts import RunWriter, run_dir_name
from .baselines import Arm
from .client import (
    GenerationError,
    ServerError,
    fetch_instrumentation,
    free_port,
    measure_generation,
    prefill_seq_floor,
    start_server,
    stop_server,
)
from .manifest import Manifest, Workload, check_prompt_tokens
from .protocol import BlockTally, Protocol, iter_blocks, plan
from . import provenance as prov


@dataclass(frozen=True)
class ServeSettings:
    """Everything held constant across arms, plus where the model is.

    These are values a *campaign* fixes once; they are recorded verbatim in the artifact so
    a reader can reconstruct the command line without reading this file.
    """

    model_path: str
    model_repository: str
    model_revision: str | None
    gpu: str | None = None
    memory_ratio: float = 0.9
    kv_reserve_tokens: int | None = None
    max_running_requests: int = 1
    cuda_graph_max_bs: int = 1
    max_seq_len_override: int | None = None
    sampling_defaults: str = "none"
    server_timeout: float = 1800.0
    python_executable: str = sys.executable
    instrument_prefill: bool = True

    def env_overrides(self) -> Dict[str, str]:
        # Off by default in the runtime; the harness turns it on for every arm alike so the
        # instrumentation cannot advantage one configuration over another.
        return {"FREETOKEN_INSTRUMENT_PREFILL": "1"} if self.instrument_prefill else {}


def serve_command(arm: Arm, settings: ServeSettings, port: int) -> List[str]:
    """The full ``ft serve`` command line for one arm.

    Ordering is stable (arm flags first, held-constant flags after) so two runs of the same
    arm produce byte-identical commands, which is what makes the recorded command line a
    reproduction recipe rather than a description.
    """
    cmd = [
        settings.python_executable, "-m", "freetoken.cli", "serve",
        "--model", settings.model_path,
        "--host", "127.0.0.1", "--port", str(port),
        *arm.moe_flags(),
        "--max-running-requests", str(settings.max_running_requests),
        "--cuda-graph-max-bs", str(settings.cuda_graph_max_bs),
        "--memory-ratio", str(settings.memory_ratio),
        "--sampling-defaults", settings.sampling_defaults,
    ]
    if settings.kv_reserve_tokens is not None:
        cmd += ["--kv-reserve-tokens", str(settings.kv_reserve_tokens)]
    if settings.max_seq_len_override is not None:
        cmd += ["--max-seq-len-override", str(settings.max_seq_len_override)]
    if settings.gpu:
        cmd += ["--gpu", settings.gpu]
    return cmd


def bench_bw_command(settings: ServeSettings, dtype: str = "nvfp4") -> List[str]:
    """``ft bench bw`` for the selected GPU and expert format (criteria section 2.1, B2)."""
    cmd = [settings.python_executable, "-m", "freetoken.cli", "bench", "bw", "--dtype", dtype]
    if settings.gpu:
        cmd += ["--gpu", settings.gpu]
    return cmd


def run_bench_bw(settings: ServeSettings, dtype: str = "nvfp4") -> Dict[str, Any]:
    """Refresh the bandwidth profile B2's fetch split is resolved from, and record it."""
    cmd = bench_bw_command(settings, dtype)
    started = prov.utc_now_iso()
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        return {"command": cmd, "started_at": started, "ok": False, "error": repr(e)}
    return {
        "command": cmd,
        "started_at": started,
        "finished_at": prov.utc_now_iso(),
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


@dataclass
class Campaign:
    arms: Sequence[Arm]
    manifest: Manifest
    protocol: Protocol
    settings: ServeSettings
    out_root: Path
    short_name: str
    inferswarm_commit: str | None
    canonical: bool
    store_output_text: bool = False
    refresh_bench_bw: bool = True
    echo_server_output: bool = True
    resolved_configuration: Dict[str, Any] = field(default_factory=dict)

    # ---- planning / dry run -----------------------------------------------------------

    def class_ids(self) -> List[str]:
        return [w.class_id for w in self.manifest.workloads]

    def steps(self):
        return plan(self.protocol, [a.id for a in self.arms], self.class_ids())

    def dry_run_document(self) -> Dict[str, Any]:
        """Everything the campaign would do, without touching a GPU.

        Used by ``--dry-run`` and by the tests: the exact per-arm command lines, the
        protocol, and an unambiguous canonical/non-canonical verdict.
        """
        port = 0  # a placeholder; the real port is chosen at spawn time
        steps = self.steps()
        return {
            "canonical": self.canonical and self.protocol.canonical and self.manifest.canonical,
            "canonical_blockers": self._canonical_blockers(),
            "protocol": self.protocol.record(),
            "arms": [
                {
                    "id": arm.id,
                    "role": arm.role,
                    "description": arm.description,
                    "notes": arm.notes,
                    "moe_flags": arm.moe_flags(),
                    "requires_bench_bw": arm.requires_bench_bw,
                    "bench_bw_command": (
                        bench_bw_command(self.settings) if arm.requires_bench_bw else None
                    ),
                    "serve_command": serve_command(arm, self.settings, port),
                    "env_overrides": self.settings.env_overrides(),
                }
                for arm in self.arms
            ],
            "workload_classes": self.class_ids(),
            "workload_manifest": self.manifest.record(),
            "execution_order": [
                {
                    "execution_index": s.execution_index,
                    "arm_id": s.arm_id,
                    "class_id": s.class_id,
                    "phase": s.phase,
                    "repetition": s.repetition,
                }
                for s in steps
            ],
            "total_generations": len(steps),
            "measured_generations": sum(1 for s in steps if s.measured),
        }

    def _canonical_blockers(self) -> List[str]:
        blockers: List[str] = []
        if not self.canonical:
            blockers.append("canonical mode is off (--dev-smoke / --allow-missing-provenance)")
        blockers.extend(self.protocol.deviations)
        if not self.manifest.canonical:
            blockers.append(f"workload manifest {self.manifest.manifest_id} declares canonical=false")
        missing = self.manifest.missing_classes()
        if missing:
            blockers.append(f"workload manifest is missing classes {missing}")
        # Stable order, no duplicates: the list is read by humans and asserted on by tests.
        return list(dict.fromkeys(blockers))

    # ---- provenance --------------------------------------------------------------------

    def provenance_document(self) -> Dict[str, Any]:
        return {
            "software": prov.software_provenance(self.inferswarm_commit, HARNESS_VERSION),
            "model": prov.model_provenance(
                prov.ModelPin(
                    repository=self.settings.model_repository,
                    revision=self.settings.model_revision or "",
                    local_path=self.settings.model_path,
                )
            ),
            "host": prov.host_provenance(),
            "gpu": prov.gpu_provenance(self.settings.gpu),
        }

    # ---- execution ----------------------------------------------------------------------

    def execute(self) -> Dict[str, Any]:
        """Run the whole campaign and write the artifacts. Returns the run document."""
        provenance = self.provenance_document()
        missing = prov.missing_required(provenance)
        if missing and self.canonical:
            raise ValueError(
                "canonical run refused: required provenance is missing -> "
                + "; ".join(missing)
                + ". Supply it or run with --dev-smoke (which produces a NON-CANONICAL record)."
            )

        run_root = self.out_root / run_dir_name(
            _date.today().isoformat(), self.protocol.session_id, self.short_name
        )
        header = {
            "started_at": prov.utc_now_iso(),
            "finished_at": None,
            "canonical": self.canonical and self.protocol.canonical and self.manifest.canonical,
            "canonical_blockers": self._canonical_blockers(),
            "protocol": self.protocol.record(),
            "serve_settings": {
                "memory_ratio": self.settings.memory_ratio,
                "kv_reserve_tokens": self.settings.kv_reserve_tokens,
                "max_running_requests": self.settings.max_running_requests,
                "cuda_graph_max_bs": self.settings.cuda_graph_max_bs,
                "max_seq_len_override": self.settings.max_seq_len_override,
                "sampling_defaults": self.settings.sampling_defaults,
                "gpu": self.settings.gpu,
                "instrument_prefill": self.settings.instrument_prefill,
            },
            "workload_manifest": self.manifest.record(),
            "provenance_missing_required": missing,
            **provenance,
        }
        writer = RunWriter(run_root, header)

        steps = self.steps()
        tallies: Dict[tuple, BlockTally] = {}
        for arm_id, class_id, block in iter_blocks(steps):
            expected = sum(1 for s in block if s.measured)
            tallies[(arm_id, class_id)] = BlockTally(arm_id, class_id, expected)

        by_class = self.manifest.by_class()
        arms_by_id = {a.id: a for a in self.arms}
        # Group the plan by arm so each arm needs exactly one server process.
        for arm_id in _ordered_unique(s.arm_id for s in steps):
            arm = arms_by_id[arm_id]
            arm_steps = [s for s in steps if s.arm_id == arm_id]
            self._run_arm(arm, arm_steps, by_class, writer, tallies)

        doc = writer.finalize(
            list(tallies.values()),
            extra={
                "finished_at": prov.utc_now_iso(),
                "resolved_configuration": self.resolved_configuration,
                "run_directory": str(run_root),
                # The resolved weight format is only knowable once an engine has loaded the
                # banks, so it is backfilled here from what the arms actually reported --
                # not left saying "server not started yet" for the life of the artifact.
                "model_expert_quant_resolved": self._observed_expert_quant(),
            },
        )
        return doc

    def _observed_expert_quant(self) -> Dict[str, Any]:
        """Resolved expert weight format per arm, and whether the arms agreed.

        Criteria section 3 rule 4 holds the weight format constant across arms, so a
        disagreement here is a campaign-invalidating fact, recorded rather than smoothed.
        """
        per_arm: Dict[str, Any] = {}
        for arm_id, resolved in self.resolved_configuration.items():
            config = (resolved.get("instrumentation") or {}).get("runtime_config") or {}
            per_arm[arm_id] = (config.get("model") or {}).get("expert_quant")
        if not per_arm:
            return prov.unavailable("no arm reported a resolved configuration")
        values = sorted({v for v in per_arm.values() if v is not None})
        if not values:
            return {"per_arm": per_arm, "value": None, "consistent_across_arms": None,
                    "unavailable": "no arm's runtime report carried an expert_quant"}
        return {
            "per_arm": per_arm,
            "value": values[0] if len(values) == 1 else None,
            "consistent_across_arms": len(values) == 1,
            "unavailable": (
                None if len(values) == 1
                else f"arms reported different expert weight formats {values}; criteria "
                     "section 3 rule 4 holds the format constant, so this campaign is invalid"
            ),
        }

    def _run_arm(
        self,
        arm: Arm,
        steps: Sequence[Any],
        by_class: Dict[str, Workload],
        writer: RunWriter,
        tallies: Dict[tuple, BlockTally],
    ) -> None:
        bench_bw = None
        if arm.requires_bench_bw and self.refresh_bench_bw:
            print(f"[phase0] {arm.id}: refreshing `ft bench bw` profile", flush=True)
            bench_bw = run_bench_bw(self.settings)

        port = free_port()
        origin = f"http://127.0.0.1:{port}"
        command = serve_command(arm, self.settings, port)
        log_path = writer.server_log_path(arm.id)
        print(f"[phase0] {arm.id}: {' '.join(command)}", flush=True)
        handle = None
        try:
            handle = start_server(
                command,
                origin,
                str(log_path),
                env_overrides=self.settings.env_overrides(),
                ready_timeout=self.settings.server_timeout,
                echo=self.echo_server_output,
            )
            instrumentation = fetch_instrumentation(origin)
            model_id = _model_id(origin)
            self.resolved_configuration[arm.id] = {
                "arm": {
                    "id": arm.id,
                    "role": arm.role,
                    "moe_flags": arm.moe_flags(),
                    "notes": arm.notes,
                },
                "serve_command": command,
                "env_overrides": self.settings.env_overrides(),
                "bench_bw": bench_bw,
                "instrumentation": instrumentation,
                "served_model_id": model_id,
            }
            for step in steps:
                workload = by_class[step.class_id]
                self._run_step(arm, step, workload, origin, model_id, writer, tallies)
        except ServerError as e:
            reason = f"{arm.id}: {e}"
            print(f"[phase0] {reason}", flush=True)
            for step in steps:
                record = {
                    "arm_id": arm.id, "class_id": step.class_id,
                    "execution_index": step.execution_index, "phase": step.phase,
                    "repetition": step.repetition, "error": str(e),
                    "server_log_tail": e.log_tail,
                }
                writer.write_failure(record)
                tallies[(arm.id, step.class_id)].failures.append(
                    {"execution_index": step.execution_index, "error": str(e)}
                )
        finally:
            if handle is not None:
                stop_server(handle)

    def _run_step(
        self,
        arm: Arm,
        step: Any,
        workload: Workload,
        origin: str,
        model_id: str,
        writer: RunWriter,
        tallies: Dict[tuple, BlockTally],
    ) -> None:
        tally = tallies[(arm.id, step.class_id)]
        body = workload.request_body(model_id)
        floor = prefill_seq_floor(origin)
        started = time.time()
        try:
            metrics = measure_generation(
                origin,
                body,
                prefill_seq_floor=floor,
                store_text=self.store_output_text or arm.role == "correctness",
            )
        except (GenerationError, OSError, ValueError) as e:
            record = {
                "arm_id": arm.id, "class_id": step.class_id,
                "execution_index": step.execution_index, "phase": step.phase,
                "repetition": step.repetition, "measured": step.measured,
                "error": repr(e), "started_at_unix": started,
            }
            writer.write_failure(record)
            tally.failures.append({"execution_index": step.execution_index, "error": repr(e)})
            print(f"[phase0] {arm.id}/{step.class_id} rep {step.repetition} FAILED: {e!r}", flush=True)
            return

        deviation = check_prompt_tokens(step.class_id, metrics["prompt_tokens"])
        record = {
            "session_id": self.protocol.session_id,
            "execution_index": step.execution_index,
            "arm_id": arm.id,
            "arm_role": arm.role,
            "class_id": step.class_id,
            "phase": step.phase,
            "measured": step.measured,
            "repetition": step.repetition,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "prompt_sha256": workload.content_sha256,
            "sampling": dict(workload.sampling),
            "greedy": workload.greedy,
            "ignore_eos": workload.ignore_eos,
            "seed": None,
            "seed_unavailable": "FreeToken's SamplingParams exposes no seed parameter",
            "batch_size": 1,
            "prompt_token_deviation": deviation,
            **metrics,
        }
        writer.write_repetition(record)
        if step.measured:
            tally.observed_measured += 1
        else:
            tally.observed_warmups += 1
        print(
            f"[phase0] {arm.id}/{step.class_id} {step.phase} {step.repetition}: "
            f"decode {metrics['decode_tok_s']:.2f} tok/s, TTFT {metrics['ttft_ms']:.1f} ms",
            flush=True,
        )


def _model_id(origin: str) -> str:
    from .client import get_json

    return get_json(f"{origin}/v1/models")["data"][0]["id"]


def _ordered_unique(values) -> List[str]:
    seen: Dict[str, None] = {}
    for v in values:
        seen.setdefault(v, None)
    return list(seen)
