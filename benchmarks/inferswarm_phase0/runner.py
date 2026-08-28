"""Campaign assembly: preflight gates, the run loop, and the run artifact.

Held-constant configuration (criteria section 3) is passed explicitly on every arm --
``--memory-ratio``, ``--kv-reserve-tokens``, ``--max-running-requests``,
``--cuda-graph-max-bs``, ``--max-seq-len-override``, ``--sampling-defaults`` -- so no arm can
differ in something the criteria hold fixed, and so the record does not depend on a default
that may change between FreeToken versions. That the arms *actually* agreed is then checked
against the resolved configuration each engine reports, not assumed from the flags.

Anti-starvation, mechanically: the sweep arms always carry ``--moe-cache-auto`` (from the
arm definition), and this module offers no way to lower ``--moe-cache-size`` /
``--moe-cache-rate`` below the auto-resolved value on a performance arm.

**Three gates, in this order, and nothing measures before all three pass.**

1. *Preflight refusals* (``preflight``): a canonical run refuses to start on a dirty
   FreeToken checkout, on the wrong model repository, on a model revision the local snapshot
   contradicts, on missing required provenance, or unless the selected UUID's captured
   identity proves an RTX 3060 in the 12-GB class. These raise: there is no artifact to write,
   because no measurement is permitted.
2. *The session-level ``ft bench bw`` prerequisite* (``refresh_bench_bw``): run once, before
   the sweep traversal, in either ordering direction -- B2 *and* B3 read the profile. A
   failure aborts a canonical campaign before any server starts.
3. *Per-observation validity* (``validity.CampaignValidity``): everything only knowable while
   running -- workload shape, output length, prefill attribution, the resolved configuration,
   the physical GPU the engine actually bound, B3's resolution, cross-arm agreement. These do
   not abort; they are collected, and they make the finished campaign ``INVALID``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Any, Dict, List, Sequence

from . import CANONICAL_MODEL_REPOSITORY, HARNESS_VERSION
from . import bench_bw as bench_bw_mod
from . import gpu as gpu_mod
from . import provenance as prov
from . import validity as V
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
from .manifest import (
    CANONICAL_GREEDY_SAMPLING,
    Manifest,
    Workload,
    check_completion_tokens,
    check_prompt_tokens,
)
from .protocol import BlockTally, Protocol, iter_blocks, plan

# Resolved-configuration fields a canonical record must carry (criteria section 2.3). Every
# one is READ BACK off the live engine; none is re-derived here. A missing or null field is
# campaign-invalidating rather than an empty cell, because "auto is not a configuration
# record" and neither is a hole where the resolved value should be.
REQUIRED_RUNTIME_FIELDS: tuple[str, ...] = (
    "model.expert_quant",
    "moe.backend_requested",
    "moe.backend_resolved",
    "moe.cpu_threads",
    "moe.cpu_layers_resolved",
    "nvfp4.requested",
    "nvfp4.resolved",
    "cache.policy_requested",
    "cache.resolved_slots",
    "cache.kv_reserve_tokens",
    "runtime.attention_backend",
    "runtime.page_size",
    "runtime.memory_ratio",
    "runtime.max_running_req",
    "runtime.max_seq_len",
    "runtime.num_pages",
    "runtime.cuda_graph_max_bs",
    "runtime.cuda_graph_capture_happened",
    "runtime.max_prefill_length_resolved",
    "runtime.cache_type_resolved",
)

# Only meaningful once the engine has resolved to hybrid: the bandwidth-matched fetch split
# is what the fresh `ft bench bw` profile exists to set, so on a hybrid arm its absence means
# the profile was not consumed.
REQUIRED_RUNTIME_FIELDS_HYBRID: tuple[str, ...] = (
    "moe.hybrid_max_fetch_resolved",
    "moe.hybrid_fetch_fraction_resolved",
)

# Values criteria section 3 holds identical across arms. Deliberately NOT the resolved cache
# slots or KV page count: those legitimately differ per backend, and section 3 rule 2 asks
# for them to be reported, not equalized.
# ``model.expert_quant`` is deliberately absent: it is held constant by the same rule
# (section 3 rule 4) but has its own dedicated code (EXPERT_QUANT_MISMATCH) and its own
# backfilled per-arm record, and reporting it twice would just make the summary noisier.
HELD_CONSTANT_FIELDS: tuple[str, ...] = (
    "moe.cpu_threads",
    "runtime.attention_backend",
    "runtime.page_size",
    "runtime.memory_ratio",
    "runtime.max_running_req",
    "runtime.max_seq_len",
    "runtime.cuda_graph_max_bs",
    "runtime.max_prefill_length_resolved",
    "runtime.cache_type_resolved",
    "cache.kv_reserve_tokens",
)

# Criteria section 2.1: B3 records which MoE backend `--moe-backend auto` resolves to, and it
# "must coincide with B1 or B2".
B3_REFERENCE_ARMS = ("B1", "B2")


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


def serve_command(
    arm: Arm, settings: ServeSettings, port: int, *, gpu: str | None = None
) -> List[str]:
    """The full ``ft serve`` command line for one arm.

    Ordering is stable (arm flags first, held-constant flags after) so two runs of the same
    arm produce byte-identical commands, which is what makes the recorded command line a
    reproduction recipe rather than a description.

    ``gpu`` overrides ``settings.gpu`` with the *resolved* UUID, so the server, ``ft bench
    bw`` and the microbenchmark all name the same physical card rather than three selectors
    that merely ought to agree.
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
    selector = gpu or settings.gpu
    if selector:
        cmd += ["--gpu", selector]
    return cmd


def _lookup(doc: Any, dotted: str) -> Any:
    node = doc
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


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
    bench_bw_dtype: str = "nvfp4"
    resolved_configuration: Dict[str, Any] = field(default_factory=dict)
    validity: V.CampaignValidity = field(init=False)
    gpu_selection: gpu_mod.GpuSelection = field(init=False)
    bench_bw_record: Dict[str, Any] | None = field(default=None, init=False)
    # Checks that need every arm's resolved configuration at once (B3's resolution, the
    # held-constant comparison). Kept out of ``resolved_configuration`` so that map stays
    # exactly "one entry per arm".
    cross_arm_checks: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.validity = V.CampaignValidity(canonical_intent=self.canonical)
        self.gpu_selection = gpu_mod.resolve_gpu(self.settings.gpu)

    # ---- planning / dry run -----------------------------------------------------------

    def class_ids(self) -> List[str]:
        return [w.class_id for w in self.manifest.workloads]

    def steps(self):
        return plan(self.protocol, [a.id for a in self.arms], self.class_ids())

    def serve_gpu(self) -> str | None:
        """The selector every child process gets: the resolved UUID when one exists."""
        return self.gpu_selection.resolved_uuid or self.settings.gpu

    def bench_bw_consumers(self) -> List[str]:
        return bench_bw_mod.consuming_arms(self.arms)

    def dry_run_document(self) -> Dict[str, Any]:
        """Everything the campaign would do, without starting a server or measuring.

        Used by ``--dry-run`` and by the tests: the exact per-arm command lines, the
        protocol, and an unambiguous canonical/non-canonical verdict. It reads the already
        used nvidia-smi provenance path so a known hardware mismatch is visible here too;
        it does not initialize CUDA or run a workload.
        """
        port = 0  # a placeholder; the real port is chosen at spawn time
        steps = self.steps()
        consumers = self.bench_bw_consumers()
        gpu_provenance = prov.gpu_provenance(
            self.settings.gpu, self.gpu_selection.resolved_uuid
        )
        phase0_gpu_preflight = self._phase0_gpu_preflight(gpu_provenance)
        return {
            "canonical": self.canonical and self.protocol.canonical and self.manifest.canonical,
            "canonical_blockers": self._canonical_blockers(),
            # What would stop this plan from ever starting, checked now rather than after a
            # model load: a plan that reads "CANONICAL" while it would be refused is exactly
            # the thing a dry run exists to prevent.
            "preflight_refusals": self.preflight_refusals(
                gpu_provenance=gpu_provenance,
                phase0_gpu_preflight=phase0_gpu_preflight,
            ),
            "protocol": self.protocol.record(),
            "gpu_selection": self.gpu_selection.record(),
            "phase0_gpu_preflight": phase0_gpu_preflight,
            "bench_bw_prerequisite": {
                "runs_before_the_sweep": bool(consumers) and self.refresh_bench_bw,
                "consuming_arms": consumers,
                "dtype": self.bench_bw_dtype,
                "command": (
                    bench_bw_mod.bench_bw_command(
                        self.settings.python_executable, self.serve_gpu(), self.bench_bw_dtype
                    )
                    if consumers else None
                ),
                "note": (
                    "Session-level, not B2-local: B2 resolves its fetch split from the "
                    "profile and B3's --moe-backend auto reads the same profile, so in a "
                    "reversed session B3 would otherwise consume a stale one."
                ),
            },
            "arms": [
                {
                    "id": arm.id,
                    "role": arm.role,
                    "description": arm.description,
                    "notes": arm.notes,
                    "moe_flags": arm.moe_flags(),
                    "requires_bench_bw": arm.requires_bench_bw,
                    "consumes_bench_bw": arm.consumes_bench_bw,
                    "serve_command": serve_command(
                        arm, self.settings, port, gpu=self.serve_gpu()
                    ),
                    "env_overrides": self.settings.env_overrides(),
                    "request_sampling": (
                        dict(CANONICAL_GREEDY_SAMPLING) if arm.role == "correctness" else None
                    ),
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
        if self.canonical and not self.refresh_bench_bw and self.bench_bw_consumers():
            blockers.append(
                "the `ft bench bw` refresh was skipped, so B2's fetch split and B3's auto "
                "backend pick would read a profile this campaign did not produce"
            )
        # Stable order, no duplicates: the list is read by humans and asserted on by tests.
        return list(dict.fromkeys(blockers))

    # ---- provenance --------------------------------------------------------------------

    def model_pin(self) -> prov.ModelPin:
        return prov.ModelPin(
            repository=self.settings.model_repository,
            revision=self.settings.model_revision or "",
            local_path=self.settings.model_path,
        )

    def provenance_document(self) -> Dict[str, Any]:
        return {
            "software": prov.software_provenance(self.inferswarm_commit, HARNESS_VERSION),
            "model": prov.model_provenance(self.model_pin()),
            "host": prov.host_provenance(),
            "gpu": prov.gpu_provenance(self.settings.gpu, self.gpu_selection.resolved_uuid),
            "gpu_selection": self.gpu_selection.record(),
        }

    # ---- preflight ----------------------------------------------------------------------

    def _phase0_gpu_preflight(self, gpu_provenance: Dict[str, Any]) -> Dict[str, Any]:
        rows = gpu_provenance.get("gpus")
        if not isinstance(rows, list):
            return gpu_mod.phase0_gpu_validation_record(None)
        want = self.gpu_selection.resolved_uuid
        identity = next(
            (
                row for row in rows
                if isinstance(row, dict)
                and want
                and str(row.get("uuid", "")).upper() == want.upper()
            ),
            None,
        )
        return gpu_mod.phase0_gpu_validation_record(identity)

    def preflight_refusals(
        self,
        *,
        gpu_provenance: Dict[str, Any] | None = None,
        phase0_gpu_preflight: Dict[str, Any] | None = None,
    ) -> List[str]:
        """Everything a canonical run would refuse to start on, cheaply enough for a dry run.

        Deliberately excludes the full provenance-completeness check, which includes a
        torch-importing subprocess; ``preflight`` adds that one. The supplied GPU block is
        captured through the existing nvidia-smi provenance path. Thus ``--dry-run`` can
        show a reviewer that a plan which *looks* canonical would in fact be refused without
        initializing CUDA or running a workload.
        """
        if not self.canonical:
            return []
        reasons: List[str] = []
        if self.settings.model_repository != CANONICAL_MODEL_REPOSITORY:
            reasons.append(
                "canonical Phase 0 requires model repository "
                f"{CANONICAL_MODEL_REPOSITORY}; alternate models require --dev-smoke "
                f"(observed {self.settings.model_repository!r})"
            )
        dirty = prov.check_clean_working_tree(prov.git_commit(prov.freetoken_repo_root()))
        if dirty:
            reasons.append(dirty)
        for check, value in (
            (prov.validate_inferswarm_commit, self.inferswarm_commit),
            (prov.validate_revision, self.settings.model_revision),
        ):
            try:
                check(value, canonical=True)
            except ValueError as e:
                reasons.append(str(e))
        mismatch = prov.check_snapshot_revision(self.model_pin())
        if mismatch:
            reasons.append(mismatch)
        if not self.gpu_selection.proven:
            reasons.append(
                "--gpu must resolve to a stable GPU UUID so the record names the one "
                "physical RTX 3060 Phase 0 ran on (criteria section 2.1). "
                f"{self.gpu_selection.unavailable}"
            )
        else:
            if phase0_gpu_preflight is None:
                gpu_provenance = gpu_provenance or prov.gpu_provenance(
                    self.settings.gpu, self.gpu_selection.resolved_uuid
                )
                phase0_gpu_preflight = self._phase0_gpu_preflight(gpu_provenance)
            if phase0_gpu_preflight.get("valid") is not True:
                reasons.append(
                    f"{phase0_gpu_preflight.get('code')}: "
                    + str(phase0_gpu_preflight.get("message") or "Phase-0 GPU class is unproven")
                    + "; alternate GPUs require --dev-smoke"
                )
        if self.bench_bw_consumers() and self.bench_bw_dtype != "nvfp4":
            reasons.append(
                "canonical Phase-0 sweep requires --bench-bw-dtype nvfp4; alternate "
                "formats are developer-smoke only"
            )
        return reasons

    def preflight(self, provenance: Dict[str, Any]) -> None:
        """Refusals that must happen before *any* measurement. Raises ``ValueError``.

        A canonical run that cannot satisfy one of these has nothing to record: it is not an
        invalid campaign, it is a campaign that must not begin.
        """
        if not self.canonical:
            return
        gpu_provenance = provenance.get("gpu") if isinstance(provenance, dict) else None
        phase0_gpu_preflight = self._phase0_gpu_preflight(gpu_provenance or {})
        for reason in self.preflight_refusals(
            gpu_provenance=gpu_provenance,
            phase0_gpu_preflight=phase0_gpu_preflight,
        ):
            raise ValueError(f"canonical run refused: {reason}")

        missing = prov.missing_required(provenance)
        if missing:
            raise ValueError(
                "canonical run refused: required provenance is missing -> "
                + "; ".join(missing)
                + ". Supply it or run with --dev-smoke (which produces a NON-CANONICAL record)."
            )

    def run_bench_bw_prerequisite(self) -> Dict[str, Any]:
        """The session-level ``ft bench bw`` refresh. Runs before the sweep traversal.

        Returns the record that goes in the artifact. For a canonical campaign a skipped
        refresh, a failed command, or an unreadable / wrong-GPU profile raises **before any
        server starts**: B2's fetch split and B3's auto backend pick would otherwise be
        resolved from a profile this campaign cannot vouch for, and the numbers would look
        exactly like valid ones.
        """
        consumers = self.bench_bw_consumers()
        if not consumers:
            return {
                "skipped": True,
                "reason": "no arm in this campaign reads the `ft bench bw` profile",
                "consuming_arms": [],
            }
        if not self.refresh_bench_bw:
            reason = (
                "the `ft bench bw` refresh was skipped (--no-bench-bw), so "
                f"{consumers} would resolve from a profile this campaign did not produce"
            )
            if self.canonical:
                raise ValueError(f"canonical run refused: {reason}")
            self.validity.add(V.BENCH_BW_SKIPPED, reason)
            return {**bench_bw_mod.skipped_record(reason), "consuming_arms": consumers}

        print(
            f"[phase0] session prerequisite: refreshing `ft bench bw` before {consumers}",
            flush=True,
        )
        result = bench_bw_mod.run_bench_bw(
            python_executable=self.settings.python_executable,
            selection=self.gpu_selection,
            dtype=self.bench_bw_dtype,
        )
        record = {**result.record, "consuming_arms": consumers}
        if not result.ok:
            reason = result.failure_reason
            if self.canonical:
                raise ValueError(
                    "canonical campaign aborted before any generation: " + reason
                )
            self.validity.add(V.BENCH_BW_FAILED, reason)
            return record
        if not result.profile_usable:
            reason = result.failure_reason or "the refreshed profile could not be pinned"
            profile = record.get("profile") or {}
            if profile.get("gpu_matches") is False:
                code = V.BENCH_BW_PROFILE_GPU_MISMATCH
            elif profile.get("gpu_matches") is not True:
                code = V.BENCH_BW_PROFILE_GPU_UNVERIFIED
            elif (profile.get("nvfp4_calibration") or {}).get("usable") is not True:
                code = V.BENCH_BW_NVFP4_CALIBRATION_UNUSABLE
            else:
                code = V.BENCH_BW_PROFILE_UNREADABLE
            if self.canonical:
                raise ValueError("canonical campaign aborted before any generation: " + reason)
            self.validity.add(code, reason)
        return record

    # ---- execution ----------------------------------------------------------------------

    def execute(self) -> Dict[str, Any]:
        """Run the whole campaign and write the artifacts. Returns the run document."""
        provenance = self.provenance_document()
        missing = prov.missing_required(provenance)
        self.preflight(provenance)
        self.validity.canonical_blockers = self._canonical_blockers()
        if missing and not self.canonical:
            self.validity.add(
                V.PROVENANCE_MISSING, "required provenance is missing -> " + "; ".join(missing)
            )

        # Gate 2, before the run directory exists: nothing measures until the bandwidth
        # profile every consuming arm will read is fresh, readable, and this GPU's.
        self.bench_bw_record = self.run_bench_bw_prerequisite()

        run_root = self.out_root / run_dir_name(
            _date.today().isoformat(), self.protocol.session_id, self.short_name
        )
        header = {
            "started_at": prov.utc_now_iso(),
            "finished_at": None,
            # No bare `canonical` key here: `validity` (VALID / INVALID / NON_CANONICAL) is
            # the verdict, and a boolean beside it is exactly the field a reader would
            # mistake for one. What this run ASKED to be is validity.canonical_intent.
            "protocol": self.protocol.record(),
            "serve_settings": {
                "memory_ratio": self.settings.memory_ratio,
                "kv_reserve_tokens": self.settings.kv_reserve_tokens,
                "max_running_requests": self.settings.max_running_requests,
                "cuda_graph_max_bs": self.settings.cuda_graph_max_bs,
                "max_seq_len_override": self.settings.max_seq_len_override,
                "sampling_defaults": self.settings.sampling_defaults,
                "gpu": self.settings.gpu,
                "gpu_resolved_uuid": self.gpu_selection.resolved_uuid,
                "instrument_prefill": self.settings.instrument_prefill,
            },
            "bench_bw": self.bench_bw_record,
            "workload_manifest": self.manifest.record(),
            "provenance_missing_required": missing,
            "phase0_gpu_preflight": self._phase0_gpu_preflight(provenance.get("gpu") or {}),
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

        expert_quant = self._observed_expert_quant()
        self._check_cross_arm(expert_quant)

        doc = writer.finalize(
            list(tallies.values()),
            self.validity,
            extra={
                "finished_at": prov.utc_now_iso(),
                "resolved_configuration": self.resolved_configuration,
                "cross_arm_checks": self.cross_arm_checks,
                "run_directory": str(run_root),
                # The resolved weight format is only knowable once an engine has loaded the
                # banks, so it is backfilled here from what the arms actually reported --
                # not left saying "server not started yet" for the life of the artifact.
                "model_expert_quant_resolved": expert_quant,
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

    # ---- validity checks -----------------------------------------------------------------

    def _runtime_config(self, arm_id: str) -> Dict[str, Any] | None:
        resolved = self.resolved_configuration.get(arm_id) or {}
        instrumentation = resolved.get("instrumentation")
        if not isinstance(instrumentation, dict):
            return None
        config = instrumentation.get("runtime_config")
        return config if isinstance(config, dict) else None

    def _check_arm_runtime(self, arm: Arm, instrumentation: Dict[str, Any] | None) -> None:
        """Instrumentation availability and the required resolved fields, for one arm."""
        required = list(REQUIRED_RUNTIME_FIELDS)
        raw_config = (
            instrumentation.get("runtime_config")
            if isinstance(instrumentation, dict)
            else None
        )
        backend_resolved = (
            _lookup(raw_config, "moe.backend_resolved")
            if isinstance(raw_config, dict)
            else None
        )
        if backend_resolved == "hybrid":
            required += list(REQUIRED_RUNTIME_FIELDS_HYBRID)
        config = V.check_runtime_configuration(
            self.validity, instrumentation, required, arm_id=arm.id
        )
        if config is None:
            return
        expert_quant = _lookup(config, "model.expert_quant")
        if self.canonical and expert_quant is not None and expert_quant != "nvfp4":
            self.validity.add(
                V.MODEL_EXPERT_QUANT_UNEXPECTED,
                "canonical Phase 0 requires model.expert_quant='nvfp4'; "
                f"the live engine reported {expert_quant!r}",
                arm_id=arm.id,
            )
        if arm.id == "B2" and backend_resolved == "hybrid":
            fraction = _lookup(config, "moe.hybrid_fetch_fraction_resolved")
            if not bench_bw_mod.is_positive_finite(fraction) or float(fraction) > 1.0:
                self.validity.add(
                    V.BENCH_BW_NVFP4_CALIBRATION_UNUSABLE,
                    "B2 resolved to hybrid but reported "
                    f"hybrid_fetch_fraction_resolved={fraction!r}; canonical B2 requires a "
                    "value in (0, 1], and 0.0 is the fixed-cap fallback rather than proof "
                    "that the fresh NVFP4 bandwidth calibration was consumed",
                    arm_id=arm.id,
                )

    def _check_arm_gpu(self, arm: Arm, verification: Dict[str, Any]) -> None:
        V.check_engine_gpu(
            self.validity,
            verification,
            canonical_intent=self.canonical,
            arm_id=arm.id,
        )

    def _check_b3_resolution(self) -> Dict[str, Any] | None:
        """B3's resolved configuration must be a legitimate B1/B2 path (criteria section 2.1).

        An unexpected resolution is preserved, never silently accepted: it means the runtime
        picked something outside the predeclared sweep, and until that is understood the
        campaign is not a measurement of the sweep that was declared.
        """
        if "B3" not in {a.id for a in self.arms}:
            return None
        b3 = self._runtime_config("B3")
        if b3 is None:
            return {"checked": False, "reason": "B3 reported no resolved configuration"}
        resolved = _lookup(b3, "moe.backend_resolved")
        reference: Dict[str, Any] = {}
        for arm_id in B3_REFERENCE_ARMS:
            config = self._runtime_config(arm_id)
            if config is not None:
                reference[arm_id] = _lookup(config, "moe.backend_resolved")
        block: Dict[str, Any] = {
            "checked": True,
            "b3_backend_resolved": resolved,
            "b3_nvfp4_resolved": _lookup(b3, "nvfp4.resolved"),
            "reference_arms": reference,
        }
        if not reference:
            block["checked"] = False
            block["reason"] = (
                "neither B1 nor B2 reported a resolved backend, so there is nothing to "
                "coincide with"
            )
            return block
        expected = {v for v in reference.values() if v is not None}
        block["expected"] = sorted(expected)
        block["reference_nvfp4"] = {
            arm_id: _lookup(self._runtime_config(arm_id) or {}, "nvfp4.resolved")
            for arm_id in reference
        }
        if resolved is None:
            self.validity.add(
                V.B3_RESOLUTION_UNEXPECTED,
                "B3 did not report which MoE backend `--moe-backend auto` resolved to; the "
                "criteria require that value to be recorded (section 2.1)",
                arm_id="B3",
            )
            block["coincides"] = False
            return block
        block["coincides"] = resolved in expected
        if not block["coincides"]:
            self.validity.add(
                V.B3_RESOLUTION_UNEXPECTED,
                f"B3's `--moe-backend auto` resolved to {resolved!r}, which coincides with "
                f"neither B1 nor B2 ({sorted(expected)}). Criteria section 2.1 requires it "
                "to; the data is preserved, but the campaign is not a measurement of the "
                "declared sweep until this is understood.",
                arm_id="B3",
            )
            return block
        # Coinciding on the MoE backend is not enough: B3 also resolves --nvfp4-backend
        # auto, and if that lands somewhere the matching declared arm did not, B3 is not the
        # same executing path even though the backend name matches.
        twins = [a for a, backend in reference.items() if backend == resolved]
        twin_nvfp4 = {block["reference_nvfp4"].get(a) for a in twins}
        block["nvfp4_coincides"] = block["b3_nvfp4_resolved"] in twin_nvfp4
        if not block["nvfp4_coincides"]:
            self.validity.add(
                V.B3_RESOLUTION_UNEXPECTED,
                f"B3 resolved the same MoE backend as {twins} but a different NVFP4 expert "
                f"path: B3 reports {block['b3_nvfp4_resolved']!r}, {twins} report "
                f"{sorted(str(v) for v in twin_nvfp4)}. The data is preserved, but B3 is not "
                "an observation of the declared configuration until this is understood.",
                arm_id="B3",
            )
        return block

    def _check_held_constant(self) -> Dict[str, Any]:
        """Every criteria-fixed held-constant value, compared across the arms that ran."""
        observed: Dict[str, Dict[str, Any]] = {}
        for arm in self.arms:
            config = self._runtime_config(arm.id)
            if config is None:
                continue
            observed[arm.id] = {path: _lookup(config, path) for path in HELD_CONSTANT_FIELDS}
        report: Dict[str, Any] = {"per_arm": observed, "disagreements": []}
        if len(observed) < 2:
            report["note"] = "fewer than two arms reported a resolved configuration"
            return report
        for path in HELD_CONSTANT_FIELDS:
            values = {arm_id: fields.get(path) for arm_id, fields in observed.items()}
            distinct = {_hashable(v) for v in values.values()}
            if len(distinct) > 1:
                report["disagreements"].append({"field": path, "per_arm": values})
                self.validity.add(
                    V.HELD_CONSTANT_MISMATCH,
                    f"held-constant value {path!r} differs across arms: {values}. Criteria "
                    "section 3 requires it identical on every arm.",
                )
        return report

    def _check_cross_arm(self, expert_quant: Dict[str, Any]) -> None:
        if expert_quant.get("consistent_across_arms") is False:
            self.validity.add(
                V.EXPERT_QUANT_MISMATCH,
                str(expert_quant.get("unavailable")
                    or "arms reported different expert weight formats"),
            )
        self.cross_arm_checks = {
            "b3_resolution": self._check_b3_resolution(),
            "held_constant": self._check_held_constant(),
        }

    # ---- the run loop --------------------------------------------------------------------

    def _run_arm(
        self,
        arm: Arm,
        steps: Sequence[Any],
        by_class: Dict[str, Workload],
        writer: RunWriter,
        tallies: Dict[tuple, BlockTally],
    ) -> None:
        port = free_port()
        origin = f"http://127.0.0.1:{port}"
        command = serve_command(arm, self.settings, port, gpu=self.serve_gpu())
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
            gpu_verification = gpu_mod.verify_engine_gpu(
                self.gpu_selection, gpu_mod.engine_gpus(origin)
            )
            self.resolved_configuration[arm.id] = {
                "arm": {
                    "id": arm.id,
                    "role": arm.role,
                    "moe_flags": arm.moe_flags(),
                    "notes": arm.notes,
                },
                "serve_command": command,
                "env_overrides": self.settings.env_overrides(),
                "bench_bw": self.bench_bw_record,
                "instrumentation": instrumentation,
                "served_model_id": model_id,
                "gpu_verification": gpu_verification,
                # Confirmed from the server, not assumed from the env override the harness
                # passed: FREETOKEN_INSTRUMENT_PREFILL only takes effect in the process that
                # actually read it.
                "prefill_instrumentation_enabled": bool(
                    ((instrumentation or {}).get("prefill") or {}).get("enabled")
                ) if isinstance(instrumentation, dict) else None,
                "request_sampling": (
                    dict(CANONICAL_GREEDY_SAMPLING) if arm.role == "correctness" else None
                ),
            }
            self._check_arm_runtime(arm, instrumentation)
            self._check_arm_gpu(arm, gpu_verification)
            for step in steps:
                workload = by_class[step.class_id]
                self._run_step(arm, step, workload, origin, model_id, writer, tallies)
        except ServerError as e:
            reason = f"{arm.id}: {e}"
            print(f"[phase0] {reason}", flush=True)
            self.validity.add(V.SERVER_FAILED, reason, arm_id=arm.id)
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
        # CORRECTNESS_REFERENCE is greedy by construction (criteria section 5.3), whatever
        # the manifest's frozen performance sampling says. The performance sweep keeps the
        # manifest's sampling untouched -- that is what "frozen" means for W1-W4.
        reference = arm.role == "correctness"
        body = (
            workload.greedy_reference_body(model_id)
            if reference
            else workload.request_body(model_id)
        )
        floor = prefill_seq_floor(origin)
        started = time.time()
        try:
            metrics = measure_generation(
                origin,
                body,
                prefill_seq_floor=floor,
                store_text=self.store_output_text or reference,
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
            self.validity.add(
                V.GENERATION_FAILED, f"{arm.id}/{step.class_id} rep {step.repetition}: {e!r}",
                arm_id=arm.id, class_id=step.class_id, execution_index=step.execution_index,
            )
            print(f"[phase0] {arm.id}/{step.class_id} rep {step.repetition} FAILED: {e!r}", flush=True)
            return

        deviation = check_prompt_tokens(step.class_id, metrics["prompt_tokens"])
        length_deviation = check_completion_tokens(
            step.class_id, metrics.get("completion_tokens"), body.get("max_tokens")
        )
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
            "sampling": {k: body[k] for k in ("temperature", "top_p", "top_k") if k in body},
            "manifest_sampling": dict(workload.sampling),
            "sampling_overridden_for_correctness_reference": reference,
            "sampling_override_reason": (
                "CORRECTNESS_REFERENCE must be greedy (criteria section 5.3); the manifest's "
                "frozen performance sampling is recorded above as manifest_sampling and is "
                "unchanged for the performance sweep"
            ) if reference else None,
            "greedy": True if reference else workload.greedy,
            "ignore_eos": workload.ignore_eos,
            "seed": None,
            "seed_unavailable": "FreeToken's SamplingParams exposes no seed parameter",
            "batch_size": 1,
            "prompt_token_deviation": deviation,
            "completion_length_deviation": length_deviation,
            **metrics,
        }
        writer.write_repetition(record)
        self._check_observation(arm, step, record, deviation, length_deviation)
        if step.measured:
            tally.observed_measured += 1
        else:
            tally.observed_warmups += 1
        print(
            f"[phase0] {arm.id}/{step.class_id} {step.phase} {step.repetition}: "
            f"decode {metrics['decode_tok_s']:.2f} tok/s, TTFT {metrics['ttft_ms']:.1f} ms",
            flush=True,
        )

    def _check_observation(
        self,
        arm: Arm,
        step: Any,
        record: Dict[str, Any],
        deviation: str | None,
        length_deviation: str | None,
    ) -> None:
        """Per-generation validity. The observation is kept either way; only the verdict moves.

        Warmups are checked too: a warmup whose prompt lands outside the frozen class shape
        is the same fixture the measured repetitions use, so the violation is the block's,
        not the repetition's.
        """
        where = dict(arm_id=arm.id, class_id=step.class_id, execution_index=step.execution_index)
        if deviation:
            self.validity.add(
                V.PROMPT_SHAPE_VIOLATION,
                f"{deviation}. The observation is preserved and the prompt was NOT rewritten; "
                "the frozen class shape is part of the experimental contract, so the block is "
                "not a valid observation of this class.",
                **where,
            )
        if length_deviation:
            self.validity.add(V.COMPLETION_LENGTH_MISMATCH, length_deviation, **where)

        # Prefill is required of the observations the campaign is made of. A warmup is
        # discarded by construction (criteria section 10), so its prefill record is not part
        # of the Phase-0 data and is not held to the same bar -- unlike the fixture's shape
        # and length, which are the block's and are checked above for every generation.
        if not step.measured:
            return
        status = record.get("prefill_status") or {}
        if status.get("ok"):
            if (record.get("prefill") or {}).get("prefill_tok_s") is None:
                self.validity.add(
                    V.PREFILL_UNUSABLE,
                    "a prefill record was attributed but yielded no prefill_tok_s: "
                    + str((record.get("prefill") or {}).get("prefill_tok_s_unavailable")),
                    **where,
                )
            return
        code = str(status.get("code") or "")
        mapped = {
            "instrumentation_unavailable": V.PREFILL_UNAVAILABLE,
            "instrumentation_disabled": V.PREFILL_DISABLED,
            "no_fresh_record": V.PREFILL_MISSING,
            "ambiguous_records": V.PREFILL_AMBIGUOUS,
            "shared_batch": V.PREFILL_SHARED_BATCH,
            "unusable_timing": V.PREFILL_UNUSABLE,
        }.get(code, V.PREFILL_UNAVAILABLE)
        self.validity.add(
            mapped,
            "canonical Phase-0 records require measured prefill throughput: "
            + str(status.get("reason") or record.get("prefill_unavailable") or "no reason given"),
            **where,
        )


def _hashable(value: Any) -> Any:
    """Make a resolved value comparable in a set (a resolved layer set arrives as a list)."""
    if isinstance(value, list):
        return ("list", tuple(_hashable(v) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple(sorted((k, _hashable(v)) for k, v in value.items())))
    return value


def _model_id(origin: str) -> str:
    from .client import get_json

    return get_json(f"{origin}/v1/models")["data"][0]["id"]


def _ordered_unique(values) -> List[str]:
    seen: Dict[str, None] = {}
    for v in values:
        seen.setdefault(v, None)
    return list(seen)
