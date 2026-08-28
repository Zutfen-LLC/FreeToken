"""Issue-#3 routing/residency measurement runner.

Exact route capture and the cache-pressure curve deliberately use separate server
processes. The former is eager-only and retains bounded on-device route sequences; the
latter uses CUDA-graph-safe cache counters and runtime cache rebuilds. Neither path emits
latency evidence or prompt/generated text.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict

from . import CANONICAL_MODEL_REPOSITORY, HARNESS_VERSION
from . import gpu as gpu_mod
from . import provenance as prov
from . import validity as V
from .client import (
    fetch_instrumentation,
    free_port,
    get_json,
    start_server,
    stop_server,
    stream_generation,
)
from .manifest import (
    Manifest,
    REQUIRED_CLASSES,
    check_completion_tokens,
    check_prompt_tokens,
)

ROUTING_RUN_SCHEMA = "inferswarm.phase0.routing-run/1"
ROUTING_OBSERVATION_SCHEMA = "inferswarm.phase0.routing-observation/1"
PRESSURE_OBSERVATION_SCHEMA = "inferswarm.phase0.cache-pressure-observation/1"

ROUTING_RUNTIME_FIELDS = (
    "model.expert_quant",
    "model.is_moe",
    "model.num_moe_layers",
    "model.num_experts",
    "model.top_k",
    "moe.backend_resolved",
    "moe.decode_target",
    "moe.collect_stats",
    "moe.trace_enabled",
    "nvfp4.resolved",
    "cache.moe_cache_policy",
    "cache.resolved_slots",
    "runtime.max_running_req",
    "runtime.cuda_graph_max_bs",
)


def _lookup(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


@dataclass(frozen=True)
class RoutingSettings:
    model_path: str
    model_repository: str = CANONICAL_MODEL_REPOSITORY
    model_revision: str | None = None
    gpu: str | None = None
    memory_ratio: float = 0.9
    kv_reserve_tokens: int | None = None
    max_seq_len_override: int | None = None
    server_timeout: float = 1800.0
    python_executable: str = "python"
    trace_max_steps: int = 4096
    pressure_cuda_graph_max_bs: int = 1
    moe_backend: str = "offload"
    nvfp4_backend: str = "triton"


def cache_sweep_points(
    minimum_slots: int, auto_slots: int, num_layers: int, num_experts: int
) -> list[dict[str, Any]]:
    """Five predeclared endpoint/quartile points over [M, A], with collisions removed."""
    if minimum_slots < 1 or auto_slots < minimum_slots:
        raise ValueError(f"invalid feasible cache interval M={minimum_slots}, A={auto_slots}")
    denominator = num_layers * num_experts
    if denominator < 1:
        raise ValueError("num_layers * num_experts must be positive")
    width = auto_slots - minimum_slots
    slots = sorted({minimum_slots + (width * quartile) // 4 for quartile in range(5)})
    return [
        {"resolved_slots": value, "cache_fraction": value / denominator}
        for value in slots
    ]


def _serve_command(
    settings: RoutingSettings, *, port: int, exact: bool, gpu: str | None
) -> list[str]:
    command = [
        settings.python_executable,
        "-m",
        "freetoken.cli",
        "serve",
        "--model",
        settings.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--moe-backend",
        settings.moe_backend,
        "--nvfp4-backend",
        settings.nvfp4_backend,
        "--moe-cache-auto",
        "--moe-cache-policy",
        "lru",
        "--moe-collect-stats",
        "--max-running-requests",
        "1",
        "--cuda-graph-max-bs",
        "0" if exact else str(settings.pressure_cuda_graph_max_bs),
        "--memory-ratio",
        str(settings.memory_ratio),
        "--sampling-defaults",
        "none",
    ]
    if exact:
        command += ["--moe-trace-max-steps", str(settings.trace_max_steps)]
    else:
        # The pressure sweep intentionally crosses the 2*num_experts overlap threshold.
        # Pin prefill policy off so cache capacity is the only swept cache variable.
        command.append("--disable-moe-prefill-overlap")
    if settings.kv_reserve_tokens is not None:
        command += ["--kv-reserve-tokens", str(settings.kv_reserve_tokens)]
    if settings.max_seq_len_override is not None:
        command += ["--max-seq-len-override", str(settings.max_seq_len_override)]
    if gpu or settings.gpu:
        command += ["--gpu", str(gpu or settings.gpu)]
    return command


def build_routing_plan(
    manifest: Manifest,
    settings: RoutingSettings,
    *,
    session_id: str,
    warmups: int,
    repetitions: int,
    canonical: bool,
    reverse_order: bool = False,
) -> dict[str, Any]:
    classes = [w.class_id for w in manifest.workloads]
    if canonical and (manifest.missing_classes() or set(classes) != set(REQUIRED_CLASSES)):
        raise ValueError("a canonical routing run consumes all frozen W1-W4 classes")
    if canonical and max(w.output_tokens for w in manifest.workloads) > settings.trace_max_steps:
        raise ValueError(
            "exact trace capacity is below a frozen class's requested completion length; "
            "increase --trace-max-steps"
        )
    if warmups < 0 or repetitions < 1:
        raise ValueError("warmups must be >= 0 and repetitions must be >= 1")
    ordered = list(reversed(classes)) if reverse_order else classes
    exact_steps = []
    index = 0
    for class_id in ordered:
        for phase, count in (("discarded_warmup", warmups), ("measured", repetitions)):
            for repetition in range(count):
                exact_steps.append(
                    {
                        "execution_index": index,
                        "mode": "exact_trace",
                        "class_id": class_id,
                        "phase": phase,
                        "repetition": repetition,
                    }
                )
                index += 1
    return {
        "schema": "inferswarm.phase0.routing-plan/1",
        "canonical_intent": canonical,
        "planned_validity": (
            "PENDING_RUNTIME_VALIDATION" if canonical else V.VALIDITY_NON_CANONICAL
        ),
        "session_id": session_id,
        "workload_classes": ordered,
        "workload_manifest": manifest.record(),
        "warmups_per_block": warmups,
        "measured_repetitions_per_block": repetitions,
        "server_modes": {
            "exact_trace": {
                "command": _serve_command(settings, port=0, exact=True, gpu=settings.gpu),
                "performance_evidence": False,
                "trace_capacity_steps": settings.trace_max_steps,
            },
            "cache_pressure": {
                "command": _serve_command(settings, port=0, exact=False, gpu=settings.gpu),
                "exact_trace_enabled": False,
                "sweep_rule": "M + floor((A-M)*q/4), q=0..4; deduplicate collisions",
            },
        },
        "exact_execution_order": exact_steps,
        "pressure_execution_order": (
            "cache points are resolved before observations from authoritative M and auto A; "
            "within each point, workload order matches workload_classes"
        ),
    }


def generation_evidence(
    request_body: Dict[str, Any], result: Dict[str, Any], *, fixture_sha256: str
) -> dict[str, Any]:
    """Lossless identity/count evidence, intentionally excluding prompt/output text."""
    request_bytes = json.dumps(
        request_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    output = str(result.get("text") or "")
    usage = result.get("usage") or {}
    return {
        "fixture_sha256": fixture_sha256,
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": int(usage["completion_tokens"]),
        "requested_completion_tokens": int(request_body["max_tokens"]),
        "response_id": result.get("response_id"),
        "prompt_text_stored": False,
        "output_text_stored": False,
    }


def _post_json(origin: str, path: str, payload: dict, timeout: float = 300.0) -> dict:
    request = urllib.request.Request(
        f"{origin}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 -- retain the status even with a malformed body
            detail = {"status": "failed", "error": f"HTTP {exc.code}"}
        return detail


def _control(origin: str, operation: str) -> dict:
    result = _post_json(
        origin, "/v1/moe/instrumentation", {"operation": operation, "timeout": 30.0}, 40.0
    )
    if result.get("status") != "ok":
        raise RuntimeError(f"MoE instrumentation {operation} rejected: {result}")
    payload = result.get("payload")
    if not isinstance(payload, dict) or payload.get("schema") != "freetoken.moe-instrumentation/1":
        raise RuntimeError(f"unknown/missing MoE instrumentation payload: {result}")
    return payload


def _rebuild(origin: str, slots: int) -> dict:
    result = _post_json(
        origin,
        "/v1/cache/rebuild",
        {"moe_cache_size": slots, "mode": "if_idle", "timeout": 300.0},
        320.0,
    )
    if result.get("status") != "ok" or int(result.get("moe_cache_size") or 0) != slots:
        raise RuntimeError(f"MoE cache rebuild to {slots} failed: {result}")
    return result


def _model_id(origin: str) -> str:
    models = get_json(f"{origin}/v1/models", timeout=10).get("data") or []
    if not models or not models[0].get("id"):
        raise RuntimeError("server returned no model id")
    return str(models[0]["id"])


def _append(path: Path, record: dict) -> None:
    with path.open("a") as output:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def _ordered_workloads(manifest: Manifest, reverse: bool) -> list[Any]:
    values = list(manifest.workloads)
    return list(reversed(values)) if reverse else values


class RoutingCampaign:
    def __init__(
        self,
        *,
        manifest: Manifest,
        settings: RoutingSettings,
        out_root: Path,
        short_name: str,
        session_id: str,
        inferswarm_commit: str | None,
        warmups: int,
        repetitions: int,
        canonical: bool,
        reverse_order: bool = False,
        echo_server_output: bool = True,
    ) -> None:
        self.manifest = manifest
        self.settings = settings
        self.out_root = out_root
        self.short_name = short_name
        self.session_id = session_id
        self.inferswarm_commit = inferswarm_commit
        self.warmups = warmups
        self.repetitions = repetitions
        self.canonical = canonical
        self.reverse_order = reverse_order
        self.echo_server_output = echo_server_output
        self.gpu_selection = gpu_mod.resolve_gpu(settings.gpu)
        self.validity = V.CampaignValidity(canonical_intent=canonical)
        self.validity.canonical_blockers = self._canonical_blockers()
        self.expected_observations = 0
        self.observed_observations = 0
        self.failures: list[dict[str, Any]] = []
        self.exact_phase_completed = False
        self.pressure_plan_resolved = False
        self.pressure_phase_completed = False

    def plan(self) -> dict:
        return build_routing_plan(
            self.manifest,
            self.settings,
            session_id=self.session_id,
            warmups=self.warmups,
            repetitions=self.repetitions,
            canonical=self.canonical,
            reverse_order=self.reverse_order,
        )

    def _canonical_blockers(self) -> list[str]:
        blockers = []
        if not self.canonical:
            blockers.append("canonical mode is off (--dev-smoke / --allow-missing-provenance)")
        if not self.manifest.canonical:
            blockers.append(
                f"workload manifest {self.manifest.manifest_id} declares canonical=false"
            )
        missing = self.manifest.missing_classes()
        if missing:
            blockers.append(f"workload manifest is missing classes {missing}")
        return list(dict.fromkeys(blockers))

    def _preflight(self) -> None:
        self.plan()
        prov.validate_revision(self.settings.model_revision, canonical=self.canonical)
        prov.validate_inferswarm_commit(self.inferswarm_commit, canonical=self.canonical)
        if not self.canonical:
            return
        if self.settings.model_repository != CANONICAL_MODEL_REPOSITORY:
            raise ValueError(
                f"canonical routing requires model repository {CANONICAL_MODEL_REPOSITORY}"
            )
        mismatch = prov.check_snapshot_revision(
            prov.ModelPin(
                repository=self.settings.model_repository,
                revision=self.settings.model_revision or "",
                local_path=self.settings.model_path,
            )
        )
        if mismatch:
            raise ValueError("canonical routing run refused: " + mismatch)
        dirty = prov.check_clean_working_tree(
            prov.git_commit(prov.freetoken_repo_root())
        )
        if dirty:
            raise ValueError("canonical routing run refused: " + dirty)
        if not self.gpu_selection.proven:
            raise ValueError("canonical routing run requires --gpu resolving to a physical GPU UUID")

    def _provenance(self) -> dict:
        pin = prov.ModelPin(
            repository=self.settings.model_repository,
            revision=self.settings.model_revision or "",
            local_path=self.settings.model_path,
        )
        return {
            "software": prov.software_provenance(self.inferswarm_commit, HARNESS_VERSION),
            "model": prov.model_provenance(pin),
            "host": prov.host_provenance(),
            "gpu": prov.gpu_provenance(
                self.settings.gpu, self.gpu_selection.resolved_uuid
            ),
        }

    def _check_runtime_identity(
        self,
        *,
        mode: str,
        instrumentation: dict[str, Any] | None,
        gpu_verification: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate the live engine identity needed by this investigation mode."""
        config = V.check_runtime_configuration(
            self.validity,
            instrumentation,
            ROUTING_RUNTIME_FIELDS,
            arm_id=mode,
        )
        V.check_engine_gpu(
            self.validity,
            gpu_verification,
            canonical_intent=self.canonical,
            arm_id=mode,
        )
        expected = {
            "model.expert_quant": "nvfp4",
            "model.is_moe": True,
            "moe.backend_resolved": "offload",
            "moe.decode_target": "gpu",
            "moe.collect_stats": True,
            "moe.trace_enabled": mode == "exact_trace",
            "nvfp4.resolved": self.settings.nvfp4_backend,
            "cache.moe_cache_policy": "lru",
            "runtime.max_running_req": 1,
            "runtime.cuda_graph_max_bs": (
                0 if mode == "exact_trace" else self.settings.pressure_cuda_graph_max_bs
            ),
        }
        checks: dict[str, Any] = {}
        if config is not None:
            for path, wanted in expected.items():
                observed = _lookup(config, path)
                matches = observed == wanted
                checks[path] = {"expected": wanted, "observed": observed, "matches": matches}
                if not matches:
                    self.validity.add(
                        V.ROUTING_RUNTIME_IDENTITY_MISMATCH,
                        f"{mode} live runtime field {path!r}={observed!r}, expected {wanted!r}",
                        arm_id=mode,
                    )
            for path in ("model.num_moe_layers", "model.num_experts", "model.top_k"):
                observed = _lookup(config, path)
                positive = isinstance(observed, int) and not isinstance(observed, bool) and observed > 0
                checks[path] = {"expected": "positive integer", "observed": observed, "matches": positive}
                if not positive:
                    self.validity.add(
                        V.ROUTING_RUNTIME_IDENTITY_MISMATCH,
                        f"{mode} live runtime field {path!r} must be a positive integer, got {observed!r}",
                        arm_id=mode,
                    )
        return {
            "instrumentation": instrumentation,
            "gpu_verification": gpu_verification,
            "expected_runtime_identity": expected,
            "runtime_identity_checks": checks,
        }

    def _check_generation_contract(
        self,
        *,
        class_id: str,
        execution_index: int,
        evidence: dict[str, Any],
    ) -> None:
        prompt_deviation = check_prompt_tokens(class_id, evidence["prompt_tokens"])
        completion_deviation = check_completion_tokens(
            class_id,
            evidence.get("completion_tokens"),
            evidence.get("requested_completion_tokens"),
        )
        evidence["prompt_token_deviation"] = prompt_deviation
        evidence["completion_length_deviation"] = completion_deviation
        if prompt_deviation:
            self.validity.add(
                V.PROMPT_SHAPE_VIOLATION,
                prompt_deviation + ". The raw routing observation is preserved.",
                class_id=class_id,
                execution_index=execution_index,
            )
        if completion_deviation:
            self.validity.add(
                V.COMPLETION_LENGTH_MISMATCH,
                completion_deviation,
                class_id=class_id,
                execution_index=execution_index,
            )

    def _observe_generation(
        self,
        *,
        origin: str,
        workload: Any,
        model_id: str,
        output: Path,
        mode: str,
        phase: str,
        repetition: int,
        execution_index: int,
    ) -> dict[str, Any] | None:
        try:
            evidence = self._run_generation(origin, workload, model_id)
        except (OSError, RuntimeError, ValueError) as exc:
            failure = {
                "schema": (
                    ROUTING_OBSERVATION_SCHEMA
                    if mode == "exact_trace"
                    else PRESSURE_OBSERVATION_SCHEMA
                ),
                "record_type": "observation_failure",
                "session_id": self.session_id,
                "execution_index": execution_index,
                "mode": mode,
                "phase": phase,
                "class_id": workload.class_id,
                "repetition": repetition,
                "error": repr(exc),
                "prompt_text_stored": False,
                "output_text_stored": False,
            }
            self.failures.append(failure)
            self.validity.add(
                V.GENERATION_FAILED,
                f"{mode}/{workload.class_id} repetition {repetition}: {exc!r}",
                class_id=workload.class_id,
                execution_index=execution_index,
            )
            _append(output, failure)
            return None
        self.observed_observations += 1
        self._check_generation_contract(
            class_id=workload.class_id,
            execution_index=execution_index,
            evidence=evidence,
        )
        return evidence

    @staticmethod
    def _pressure_point_contract(snapshot: dict[str, Any], requested_slots: int) -> dict[str, Any]:
        geometry = snapshot.get("geometry") or {}
        collection = snapshot.get("collection") or {}
        checks = {
            "resolved_cache_slots": {
                "expected": requested_slots,
                "observed": geometry.get("resolved_cache_slots"),
                "matches": geometry.get("resolved_cache_slots") == requested_slots,
            },
            "cache_policy": {
                "expected": "lru",
                "observed": geometry.get("cache_policy"),
                "matches": geometry.get("cache_policy") == "lru",
            },
            "decode_target": {
                "expected": "gpu",
                "observed": geometry.get("decode_target"),
                "matches": geometry.get("decode_target") == "gpu",
            },
            "stats_enabled": {
                "expected": True,
                "observed": collection.get("stats_enabled"),
                "matches": collection.get("stats_enabled") is True,
            },
            "prefill_overlap_active": {
                "expected": False,
                "observed": geometry.get("prefill_overlap_active"),
                "matches": geometry.get("prefill_overlap_active") is False,
            },
        }
        return {"valid": all(item["matches"] for item in checks.values()), "checks": checks}

    def _run_generation(self, origin: str, workload: Any, model_id: str) -> dict:
        body = workload.request_body(model_id)
        result = stream_generation(origin, body, timeout=3600.0)
        return generation_evidence(body, result, fixture_sha256=workload.content_sha256)

    def _run_blocks(
        self,
        *,
        origin: str,
        model_id: str,
        output: Path,
        exact: bool,
        cache_point: dict[str, Any] | None = None,
        execution_index: int,
    ) -> int:
        for workload in _ordered_workloads(self.manifest, self.reverse_order):
            rebuild = None
            cache_state = "carried_from_previous_exact_block"
            cache_state_note = "actual slot-map state before this workload's warmup"
            if cache_point is not None:
                rebuild = _rebuild(origin, int(cache_point["resolved_slots"]))
                cache_state = "post_runtime_rebuild_before_workload"
                cache_state_note = (
                    "cold with respect to workload traffic; authoritative residency may include "
                    "any cache effects of graph recapture and is recorded rather than assumed empty"
                )
            elif execution_index == 0:
                cache_state = "post_server_start_before_workload"

            cold = _control(origin, "snapshot")
            point_contract = None
            if cache_point is not None:
                point_contract = self._pressure_point_contract(
                    cold, int(cache_point["resolved_slots"])
                )
                point_record = {
                    "schema": PRESSURE_OBSERVATION_SCHEMA,
                    "record_type": "point_pre_warmup",
                    "session_id": self.session_id,
                    "execution_index": execution_index,
                    "class_id": workload.class_id,
                    "cache_point": cache_point,
                    "cache_rebuild": rebuild,
                    "point_runtime_provenance": {
                        "geometry": cold.get("geometry"),
                        "collection": cold.get("collection"),
                    },
                    "point_contract": point_contract,
                    "measured": False,
                }
                _append(output, point_record)
                if not point_contract["valid"]:
                    failures = [
                        name
                        for name, check in point_contract["checks"].items()
                        if not check["matches"]
                    ]
                    self.validity.add(
                        V.ROUTING_CACHE_PRESSURE_POINT_CONTRACT,
                        f"cache-pressure point {cache_point['resolved_slots']} for "
                        f"{workload.class_id} failed live checks {failures}; the point was "
                        "refused before warmup and its boundary evidence was preserved",
                        class_id=workload.class_id,
                        execution_index=execution_index,
                    )
                    execution_index += self.warmups + self.repetitions
                    continue
            warmup_observations = []
            for repetition in range(self.warmups):
                evidence = self._observe_generation(
                    origin=origin,
                    workload=workload,
                    model_id=model_id,
                    output=output,
                    mode="exact_trace" if exact else "cache_pressure",
                    phase="discarded_warmup",
                    repetition=repetition,
                    execution_index=execution_index,
                )
                if evidence is not None:
                    warmup_observations.append(
                        {
                            "execution_index": execution_index,
                            "mode": "exact_trace" if exact else "cache_pressure",
                            "repetition": repetition,
                            **evidence,
                        }
                    )
                execution_index += 1
            before_measured = _control(origin, "reset")
            _append(
                output,
                {
                    "schema": (
                        ROUTING_OBSERVATION_SCHEMA if exact else PRESSURE_OBSERVATION_SCHEMA
                    ),
                    "record_type": "block_boundary",
                    "session_id": self.session_id,
                    "class_id": workload.class_id,
                    "cache_point": cache_point,
                    "cache_rebuild": rebuild,
                    "cold_state_kind": cache_state,
                    "cold_state_note": cache_state_note,
                    "cold_residency": cold["residency"],
                    "point_runtime_provenance": (
                        {
                            "geometry": cold.get("geometry"),
                            "collection": cold.get("collection"),
                        }
                        if cache_point is not None
                        else None
                    ),
                    "point_contract": point_contract,
                    "discarded_warmup_observations": warmup_observations,
                    "discarded_warmup_counts": before_measured["aggregate"],
                    "residency_immediately_before_measured": before_measured["residency"],
                    "reset_preserved_residency": before_measured["boundary"][
                        "residency_preserved_by_reset"
                    ],
                    "measured": False,
                },
            )
            starting_residency = before_measured["residency"]
            for repetition in range(self.repetitions):
                evidence = self._observe_generation(
                    origin=origin,
                    workload=workload,
                    model_id=model_id,
                    output=output,
                    mode="exact_trace" if exact else "cache_pressure",
                    phase="measured",
                    repetition=repetition,
                    execution_index=execution_index,
                )
                if evidence is None:
                    execution_index += 1
                    continue
                snapshot = _control(origin, "reset")
                record = {
                    "schema": (
                        ROUTING_OBSERVATION_SCHEMA if exact else PRESSURE_OBSERVATION_SCHEMA
                    ),
                    "record_type": "measured_repetition",
                    "session_id": self.session_id,
                    "execution_index": execution_index,
                    "mode": "exact_trace" if exact else "cache_pressure",
                    "class_id": workload.class_id,
                    "repetition": repetition,
                    "measured": True,
                    "cache_point": cache_point,
                    "starting_residency": starting_residency,
                    "ending_residency": snapshot["residency"],
                    "aggregate": snapshot["aggregate"],
                    "per_layer": snapshot["per_layer"],
                    "routing": snapshot["routing"],
                    "trace": snapshot["trace"] if exact else {
                        "enabled": False,
                        "reason": "cache-pressure mode uses graph-safe counters only",
                    },
                    "generation": evidence,
                    "quantity_labels": {
                        "measured": [
                            "active_selections",
                            "misses",
                            "fetches",
                            "decode_steps",
                            "routing_histogram",
                            "resident_expert_ids",
                        ] + (["exact_routes"] if exact else []),
                        "derived": ["miss_rate", "cache_fraction", "routing_concentration"],
                    },
                    "latency_is_phase0_performance_evidence": False,
                }
                if exact:
                    expected_trace_steps = max(0, evidence["completion_tokens"] - 1)
                    trace_complete = (
                        not snapshot["trace"].get("truncated")
                        and snapshot["trace"].get("steps_observed") == expected_trace_steps
                        and snapshot["trace"].get("steps_recorded") == expected_trace_steps
                    )
                    record["trace_completeness"] = {
                        "expected_decode_steps_from_completion": expected_trace_steps,
                        "observed_steps": snapshot["trace"].get("steps_observed"),
                        "recorded_steps": snapshot["trace"].get("steps_recorded"),
                        "complete": trace_complete,
                    }
                if exact and snapshot["trace"].get("truncated"):
                    message = (
                        f"exact trace truncated for {workload.class_id} repetition {repetition}; "
                        "the raw observation is preserved but invalid"
                    )
                    self.validity.add(
                        V.ROUTING_TRACE_TRUNCATED,
                        message,
                        class_id=workload.class_id,
                        execution_index=execution_index,
                    )
                    record["trace_contract_violation"] = message
                if exact and not record["trace_completeness"]["complete"]:
                    message = (
                        f"exact trace step count does not cover the completion for "
                        f"{workload.class_id} repetition {repetition}: "
                        f"{record['trace_completeness']}"
                    )
                    self.validity.add(
                        V.ROUTING_TRACE_INCOMPLETE,
                        message,
                        class_id=workload.class_id,
                        execution_index=execution_index,
                    )
                    record["trace_contract_violation"] = message
                _append(output, record)
                starting_residency = snapshot["residency"]
                execution_index += 1
        return execution_index

    def _start_mode(self, root: Path, *, exact: bool):
        port = free_port()
        origin = f"http://127.0.0.1:{port}"
        gpu = self.gpu_selection.resolved_uuid or self.settings.gpu
        command = _serve_command(self.settings, port=port, exact=exact, gpu=gpu)
        name = "exact-trace" if exact else "cache-pressure"
        handle = start_server(
            command,
            origin,
            str(root / "server-logs" / f"{name}.log"),
            ready_timeout=self.settings.server_timeout,
            echo=self.echo_server_output,
        )
        return handle, origin, command

    def _finalize_document(self, doc: dict[str, Any], root: Path) -> dict[str, Any]:
        complete = (
            self.exact_phase_completed
            and self.pressure_plan_resolved
            and self.pressure_phase_completed
            and not self.failures
            and self.observed_observations == self.expected_observations
        )
        execution_status = (
            V.EXECUTION_COMPLETE if complete else V.EXECUTION_INCOMPLETE
        )
        verdict = self.validity.verdict()
        body = {key: value for key, value in doc.items() if key != "schema"}
        final = {
            "schema": ROUTING_RUN_SCHEMA,
            "headline": V.headline(execution_status, verdict),
            "execution_status": execution_status,
            **self.validity.record(),
            "label": "MEASURED" if complete else "INCOMPLETE",
            **body,
            "observations": {
                "expected": self.expected_observations,
                "observed": self.observed_observations,
                "missing": max(0, self.expected_observations - self.observed_observations),
            },
            "phase_completion": {
                "exact_trace": self.exact_phase_completed,
                "cache_pressure_plan_resolved": self.pressure_plan_resolved,
                "cache_pressure": self.pressure_phase_completed,
            },
            "failures": list(self.failures),
            "run_directory": str(root),
        }
        (root / "run.json").write_text(json.dumps(final, indent=2) + "\n")
        return final

    def _record_mode_failure(self, mode: str, exc: Exception) -> None:
        failure = {
            "mode": mode,
            "error": repr(exc),
            "prompt_text_stored": False,
            "output_text_stored": False,
        }
        self.failures.append(failure)
        self.validity.add(V.SERVER_FAILED, f"{mode}: {exc!r}", arm_id=mode)

    def execute(self) -> dict:
        self._preflight()
        provenance = self._provenance()
        missing_provenance = prov.missing_required(provenance)
        if self.canonical and missing_provenance:
            raise ValueError(
                "canonical routing run refused: required provenance is missing -> "
                + "; ".join(missing_provenance)
            )
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in self.short_name)
        root = self.out_root / f"{date.today().isoformat()}-{self.session_id}-{safe}"
        root.mkdir(parents=True, exist_ok=False)
        (root / "server-logs").mkdir()
        exact_path = root / "exact-routing.jsonl"
        pressure_path = root / "cache-pressure.jsonl"
        exact_path.write_text("")
        pressure_path.write_text("")
        plan = self.plan()
        doc: dict[str, Any] = {
            "schema": ROUTING_RUN_SCHEMA,
            "synthetic_or_dummy_output_is_canonical_evidence": False,
            "issue": "supports/references Zutfen-LLC/inferswarm#3; does not complete it",
            "session_id": self.session_id,
            "workload_manifest": self.manifest.record(),
            "plan": plan,
            "provenance": provenance,
            "provenance_missing_required": missing_provenance,
            "artifacts": {
                "exact_routing": exact_path.name,
                "cache_pressure": pressure_path.name,
                "server_logs": "server-logs/",
            },
            "claims_generated": [],
        }
        self.expected_observations = len(self.manifest.workloads) * (
            self.warmups + self.repetitions
        )
        self._finalize_document(doc, root)

        execution_index = 0
        handle = None
        try:
            handle, origin, command = self._start_mode(root, exact=True)
            model_id = _model_id(origin)
            instrumentation = fetch_instrumentation(origin)
            reported_gpus = gpu_mod.engine_gpus(origin)
            gpu_verification = gpu_mod.verify_engine_gpu(
                self.gpu_selection, reported_gpus
            )
            doc["exact_trace_runtime"] = {
                "serve_command": command,
                **self._check_runtime_identity(
                    mode="exact_trace",
                    instrumentation=instrumentation,
                    gpu_verification=gpu_verification,
                ),
                "engine_gpus": reported_gpus,
                "cache_status_at_start": get_json(f"{origin}/v1/cache/status"),
            }
            execution_index = self._run_blocks(
                origin=origin,
                model_id=model_id,
                output=exact_path,
                exact=True,
                execution_index=execution_index,
            )
            self.exact_phase_completed = True
        except Exception as exc:  # noqa: BLE001 -- preserve an incomplete artifact
            self._record_mode_failure("exact_trace", exc)
        finally:
            if handle is not None:
                stop_server(handle)

        handle = None
        try:
            handle, origin, command = self._start_mode(root, exact=False)
            model_id = _model_id(origin)
            runtime = fetch_instrumentation(origin)
            reported_gpus = gpu_mod.engine_gpus(origin)
            gpu_verification = gpu_mod.verify_engine_gpu(
                self.gpu_selection, reported_gpus
            )
            status = get_json(f"{origin}/v1/cache/status")
            geometry = status.get("geometry") or {}
            auto_slots = int(geometry.get("moe_cache_size") or 0)
            minimum = int(((geometry.get("limits") or {}).get("moe_experts") or {}).get("min") or 0)
            if minimum < 1:
                raise RuntimeError("authoritative MoE cache minimum M is unavailable")
            points = cache_sweep_points(
                minimum,
                auto_slots,
                int(geometry.get("num_moe_layers") or 0),
                int(geometry.get("num_experts") or 0),
            )
            self.pressure_plan_resolved = True
            self.expected_observations += len(points) * len(self.manifest.workloads) * (
                self.warmups + self.repetitions
            )
            if self.canonical and len(points) < 4:
                self.validity.add(
                    V.ROUTING_CACHE_PRESSURE_POINT_CONTRACT,
                    f"canonical cache-pressure result cannot be called a curve: only {len(points)} "
                    "distinct feasible quartile points remain",
                    arm_id="cache_pressure",
                )
            doc["cache_pressure_runtime"] = {
                "serve_command": command,
                **self._check_runtime_identity(
                    mode="cache_pressure",
                    instrumentation=runtime,
                    gpu_verification=gpu_verification,
                ),
                "engine_gpus": reported_gpus,
                "authoritative_minimum_slots_M": minimum,
                "auto_resolved_slots_A": auto_slots,
                "predeclared_sweep_points": points,
                "point_selection_uses_observed_miss_rates": False,
            }
            for point in points:
                execution_index = self._run_blocks(
                    origin=origin,
                    model_id=model_id,
                    output=pressure_path,
                    exact=False,
                    cache_point=point,
                    execution_index=execution_index,
                )
            self.pressure_phase_completed = True
        except Exception as exc:  # noqa: BLE001 -- preserve an incomplete artifact
            self._record_mode_failure("cache_pressure", exc)
        finally:
            if handle is not None:
                stop_server(handle)

        doc["execution_observations"] = execution_index
        return self._finalize_document(doc, root)
