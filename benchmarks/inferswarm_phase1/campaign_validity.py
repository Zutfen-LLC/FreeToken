"""Campaign validity: states, invalidation codes, and runtime-contract validation.

Three separate answers, never conflated (the Phase-0 harness lesson, restated for
Phase 1):

``execution_status``
    Did every planned generation return? ``COMPLETE`` / ``INCOMPLETE``. Arithmetic
    over expected-vs-observed counts and the failure list.

``validity``
    Is this a valid canonical Phase-1 session? ``VALID`` / ``INVALID`` /
    ``NON_CANONICAL_DEV_SMOKE``. A session that returned every generation is still
    ``INVALID`` when a precommitted prerequisite, runtime contract, or provenance
    requirement failed.

``verdict``
    There is none. This package emits no campaign verdict of any kind; the
    performance-decision vocabulary belongs to the P6 analysis of completed
    artifacts. The states above are the complete vocabulary of runner output.

A configuration mismatch discovered at runtime is a preflight failure for that arm:
the runner does not benchmark the wrong arm and label it later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from inferswarm_phase0.validity import (
    EXECUTION_COMPLETE,
    EXECUTION_INCOMPLETE,
    VALIDITY_INVALID,
    VALIDITY_VALID,
)
from inferswarm_phase0.validity import (
    CampaignValidity as _Phase0Validity,
)

VALIDITY_NON_CANONICAL_DEV_SMOKE = "NON_CANONICAL_DEV_SMOKE"

# Re-exported for the campaign modules and tests: the shared Phase-0 execution states
# are this package's execution vocabulary too.
__all__ = [
    "EXECUTION_COMPLETE",
    "EXECUTION_INCOMPLETE",
    "VALIDITY_INVALID",
    "VALIDITY_NON_CANONICAL_DEV_SMOKE",
    "VALIDITY_STATES",
    "VALIDITY_VALID",
]

VALIDITY_STATES = (
    VALIDITY_VALID,
    VALIDITY_INVALID,
    VALIDITY_NON_CANONICAL_DEV_SMOKE,
)

# --- stable reason codes -----------------------------------------------------------------

# Provenance / preflight (fail closed before a session starts)
PROVENANCE_DIRTY_TREE = "provenance.dirty_working_tree"
PROVENANCE_MISSING = "provenance.missing_required"
PROVENANCE_MANIFEST_MISMATCH = "provenance.workload_manifest_mismatch"
PROVENANCE_PLACEMENT_MISMATCH = "provenance.placement_sha_mismatch"
PROVENANCE_PLACEMENT_UNREADABLE = "provenance.placement_unreadable"
PROVENANCE_MODEL_MISMATCH = "provenance.model_mismatch"
PROVENANCE_PREREQUISITES_MISSING = "provenance.correctness_prerequisites_missing"
PROVENANCE_RUNNER_VERSION_MISSING = "provenance.runner_version_missing"

# GPU identity
GPU0_MISMATCH = "gpu.gpu0_mismatch"
GPU1_MISMATCH = "gpu.secondary_mismatch"
GPU_UNPROVEN = "gpu.unproven"
GPU_IDLE_NOT_RESTORED = "gpu.idle_memory_not_restored"

# Runtime-resolution contracts (checked after /health ready, before warmups)
BASELINE_IDENTITY_DRIFT = "runtime.baseline_identity_drift"
CANDIDATE_CONTRACT_MISMATCH = "runtime.candidate_contract_mismatch"
BASELINE_INFERSWARM_PRESENT = "runtime.baseline_inferswarm_present"
CANDIDATE_GRAPHS_ENABLED = "runtime.candidate_cuda_graphs_enabled"
RUNTIME_CONFIG_MISSING = "runtime.config_missing"
RUNTIME_CONFIG_MISSING_FIELD = "runtime.config_missing_field"
INSTRUMENTATION_UNAVAILABLE = "runtime.instrumentation_unavailable"

# Mechanism-window contracts (checked per measured block)
REMOTE_PREFILL_NONZERO = "remote.prefill_dispatches_nonzero"
REMOTE_FORBIDDEN_FALLBACK = "remote.forbidden_fallback"
REMOTE_OWNERSHIP_ARITHMETIC = "remote.ownership_arithmetic_broken"
REMOTE_TRANSPORT_UNEXPECTED = "remote.transport_unexpected"

# Conditional supplementary arm (predeclared KV rule)
SUPPLEMENTARY_REQUIRED_BLOCK_MISSING = "supplementary.required_block_missing"

# Thermal / session boundary
SESSION_BOUNDARY_UNPROVEN = "thermal.session_boundary_unproven"

# Execution faults (incomplete AND invalid)
GENERATION_FAILED = "execution.generation_failed"
SERVER_FAILED = "execution.server_failed"

# --- the validity state ------------------------------------------------------------------


@dataclass
class SessionValidity(_Phase0Validity):
    """Phase-0's collector with the Phase-1 non-canonical state name.

    The underlying invalidation collection, canonical-blocker separation and
    ``VALID``/``INVALID`` derivation are reused unchanged; only the non-canonical
    label differs, because a Phase-1 developer smoke is labelled
    ``NON_CANONICAL_DEV_SMOKE`` everywhere it is recorded.
    """

    def verdict(self) -> str:
        if not self.canonical_intent or self.canonical_blockers:
            return VALIDITY_NON_CANONICAL_DEV_SMOKE
        return VALIDITY_INVALID if self.invalidations else VALIDITY_VALID


class RuntimeContractError(RuntimeError):
    """A runtime-contract failure discovered after ``/health`` and before warmups.

    Every such failure stops measurement for the affected arm: the runner never
    benchmarks a configuration it already knows is not the declared experiment,
    and no generation of that arm is recorded as a successful measurement. The
    arm's planned generations are preserved as not-executed evidence and the
    session is INVALID/INCOMPLETE.
    """


class BaselineIdentityError(RuntimeContractError):
    """B1 no longer resolves to the frozen Phase-0 baseline identity.

    Identity failure means the resolved identity properties drifted OR the
    InferSwarm treatment leaked into a baseline arm OR the resolved report was
    too incomplete to prove the identity — all three are the same statement:
    this is not B1.

    Session-aware by design (campaign-order amendment): in Session 1 (B1 first)
    this STOPs the session before any candidate generation — the campaign-build
    baseline identity gate must pass before the first candidate measurement
    anywhere in the campaign. In Session 2 (candidate first, by design) the
    drift is discovered when the counterbalanced B1 arm runs: the session is
    INVALID, the already-collected candidate measurements are retained as
    invalid evidence and are not eligible for the Phase-1 analysis, the Phase-0
    baseline must be refreshed, and the complete affected campaign is rerun —
    no candidate data is reused or spliced. Either way the amendment forbids
    reopening baseline selection from candidate data or silently substituting
    another arm.
    """


class CandidateContractError(RuntimeContractError):
    """The candidate arm's pre-warmup runtime contract did not hold exactly.

    Raised after ``/health`` ready and before the first warmup: wrong placement
    SHA, wrong GPU UUIDs/devices, wrong resident slots or bytes, wrong
    transport, remote mode not overlap, unexpected CUDA-graph state, wrong
    backend/CPU-layer shape, wrong GPU0 cache size, resolved KV != the pinned
    capacity, or a resolved report too incomplete to check. The arm is aborted
    with no candidate performance collected; the session is INVALID/INCOMPLETE
    with the planned generations preserved as not-executed evidence.
    """


class SupplementaryContractError(RuntimeContractError):
    """The supplementary KV-matched arm failed the B1 runtime contract.

    The arm is aborted before its first warmup exactly like a primary arm; it
    gates nothing, so the session's already-collected primary measurements
    stand, but the session is still INVALID/INCOMPLETE because a required
    supplementary block now exists only as not-executed evidence.
    """


class InstrumentationControlError(RuntimeContractError):
    """A reset/snapshot control request failed on the engine-owned boundary.

    The engine answered the instrumentation control request with a non-ok
    outcome (busy/unsupported/timeout/failed) — or the control request never
    completed — so the measured window's required mechanism/timing evidence
    (issue #5's complete timing population) cannot be collected. The session
    stops immediately: every generation already collected is preserved
    unchanged, every remaining planned generation of this arm AND every later
    arm is preserved as not-executed evidence, the session is INVALID/
    INCOMPLETE, ``run-session`` returns nonzero, and no retry, no splicing,
    and no performance ratio exists. This is control-plane behavior only: no
    timing record is truncated and no performance threshold moves.
    """


def _lookup(doc: Any, dotted: str) -> Any:
    node = doc
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def check_required_fields(
    state: SessionValidity,
    runtime_config: Mapping[str, Any] | None,
    required: Sequence[str],
    *,
    arm_id: str,
) -> bool:
    """Every required resolved field must be present and non-null (Phase-0 contract)."""
    if not isinstance(runtime_config, Mapping) or runtime_config.get("error"):
        state.add(
            INSTRUMENTATION_UNAVAILABLE,
            "the resolved configuration could not be read: "
            + str((runtime_config or {}).get("error") if isinstance(runtime_config, Mapping) else "no document"),
            arm_id=arm_id,
        )
        return False
    missing = [p for p in required if _lookup(runtime_config, p) is None]
    for path in missing:
        state.add(
            RUNTIME_CONFIG_MISSING_FIELD,
            f"required resolved field {path!r} is missing or null",
            arm_id=arm_id,
        )
    return not missing


# Fields the baseline and candidate runtime contracts share.
COMMON_REQUIRED_FIELDS: tuple[str, ...] = (
    "model.expert_quant",
    "moe.backend_requested",
    "moe.backend_resolved",
    "moe.cpu_layers_resolved",
    "moe.auto_cpu_layers_fired",
    "moe.decode_target",
    "nvfp4.requested",
    "nvfp4.resolved",
    "cache.policy_requested",
    "cache.resolved_slots",
    "cache.kv_reserve_tokens",
    "runtime.page_size",
    "runtime.num_pages",
    "runtime.memory_ratio",
    "runtime.max_running_req",
    "runtime.cuda_graph_max_bs",
    "runtime.cuda_graph_capture_happened",
)


def validate_baseline_runtime(
    state: SessionValidity,
    runtime_config: Mapping[str, Any] | None,
    *,
    arm_id: str,
    strict: bool = True,
) -> dict[str, Any]:
    """The B1 identity contract, read off the live engine before warmups.

    ``strict`` (the primary baseline) raises ``BaselineIdentityError`` on any
    identity failure; the supplementary KV-matched arm records its findings and
    the runner aborts the arm before its first warmup without stopping the
    session (it gates nothing). There is deliberately NO numeric validity band
    on the resolved auto expert-cache slot count: the methodology requires
    ``--moe-cache-auto`` and the other resolved identity properties, records
    the exact resolved slot count as provenance, and lets the predeclared
    supplementary-KV rule own capacity differences.
    """
    ok = check_required_fields(
        state, runtime_config, COMMON_REQUIRED_FIELDS, arm_id=arm_id
    )
    if not ok or not isinstance(runtime_config, Mapping):
        if strict:
            raise BaselineIdentityError(
                "the baseline runtime report is incomplete; the frozen B1 identity "
                "cannot be proven on this build"
            )
        return {"arm_id": arm_id, "checked": False, "strict": strict}

    findings: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            findings.append(message)

    # one GPU, no InferSwarm treatment: leakage IS a B1 identity failure
    secondary = _lookup(runtime_config, "inferswarm_secondary_device") or {}
    resident = _lookup(runtime_config, "inferswarm_resident_bank") or {}
    remote = _lookup(runtime_config, "inferswarm_remote_decode") or {}
    require(
        secondary.get("configured") is False,
        f"secondary device block says configured={secondary.get('configured')!r}",
    )
    require(
        resident.get("placement_configured") is False,
        "a resident placement bank is configured on a baseline arm",
    )
    require(
        remote.get("enabled") is not True,
        "remote decode is enabled on a baseline arm",
    )
    if findings:
        state.add(BASELINE_INFERSWARM_PRESENT, "; ".join(findings), arm_id=arm_id)

    identity: list[str] = []
    # offload, GPU decode
    if _lookup(runtime_config, "moe.backend_resolved") != "offload":
        identity.append(
            f"backend_resolved={_lookup(runtime_config, 'moe.backend_resolved')!r}, expected 'offload'"
        )
    if _lookup(runtime_config, "moe.decode_target") != "gpu":
        identity.append(
            f"decode_target={_lookup(runtime_config, 'moe.decode_target')!r}, expected 'gpu'"
        )
    # zero CPU MoE layers, no auto-locking
    cpu_layers = _lookup(runtime_config, "moe.cpu_layers_resolved")
    auto_fired = _lookup(runtime_config, "moe.auto_cpu_layers_fired")
    if cpu_layers != [] or auto_fired is not False:
        identity.append(
            f"cpu_layers_resolved={cpu_layers!r}, auto_cpu_layers_fired={auto_fired!r}; "
            "expected no CPU MoE layers"
        )
    # B1 auto NVFP4 resolution recorded and matching the Phase-0 record
    nvfp4 = _lookup(runtime_config, "nvfp4.resolved")
    if _lookup(runtime_config, "nvfp4.requested") != "auto":
        identity.append(f"nvfp4.requested={_lookup(runtime_config, 'nvfp4.requested')!r}, expected 'auto'")
    if nvfp4 != "triton":
        identity.append(
            f"nvfp4 auto resolved to {nvfp4!r}; the Phase-0 record resolved triton. This "
            "is a material resolution change on the campaign build"
        )
    # expert cache must be the AUTO policy; the resolved slot count is provenance,
    # never a numeric validity gate (no hidden band; the supplementary-KV rule
    # owns KV-capacity differences)
    slots = _lookup(runtime_config, "cache.resolved_slots")
    if _lookup(runtime_config, "cache.policy_requested") != "auto":
        identity.append(
            f"cache.policy_requested={_lookup(runtime_config, 'cache.policy_requested')!r}, expected 'auto'"
        )
    from .campaign_arms import PHASE0_RECORDED_BASELINE_CACHE_SLOTS

    # graphs stay enabled on the baseline
    if _lookup(runtime_config, "runtime.cuda_graph_max_bs") != 1 or _lookup(
        runtime_config, "runtime.cuda_graph_capture_happened"
    ) is not True:
        identity.append(
            "CUDA graph capture did not happen with cuda_graph_max_bs=1; the frozen B1 "
            "identity runs graph-enabled (criteria section 12 forbids disabling graphs "
            "on the baseline to make the comparison look cleaner)"
        )

    record: dict[str, Any] = {
        "arm_id": arm_id,
        "checked": True,
        "strict": strict,
        "resolved": {
            "backend": _lookup(runtime_config, "moe.backend_resolved"),
            "decode_target": _lookup(runtime_config, "moe.decode_target"),
            "cpu_layers_resolved": cpu_layers,
            "auto_cpu_layers_fired": auto_fired,
            "nvfp4_requested": _lookup(runtime_config, "nvfp4.requested"),
            "nvfp4_resolved": nvfp4,
            "cache_policy": _lookup(runtime_config, "cache.policy_requested"),
            "cache_resolved_slots": slots,
            "cuda_graph_max_bs": _lookup(runtime_config, "runtime.cuda_graph_max_bs"),
            "cuda_graph_capture_happened": _lookup(
                runtime_config, "runtime.cuda_graph_capture_happened"
            ),
        },
        "expected": {
            "backend": "offload",
            "decode_target": "gpu",
            "cpu_layers_resolved": [],
            "auto_cpu_layers_fired": False,
            "nvfp4_resolved": "triton",
            "cache_policy": "auto",
            "cuda_graph_max_bs": 1,
            "cuda_graph_capture_happened": True,
        },
        "cache_slots_rule": (
            "the resolved auto expert-cache slot count is recorded exactly as "
            f"provenance (Phase-0 recorded {PHASE0_RECORDED_BASELINE_CACHE_SLOTS}); it is "
            "not a validity threshold, and KV-capacity differences are handled by the "
            "predeclared supplementary-KV rule, never by a numeric slot band"
        ),
        "identity_findings": identity,
        "inferswarm_leakage_findings": findings,
        "historical_note": (
            "B1 is the frozen Phase-0 CANONICAL_PERFORMANCE_BASELINE identity "
            "remeasured on the current campaign build; this check is a drift guard, "
            "not a new baseline selection"
        ),
    }
    if identity:
        state.add(
            BASELINE_IDENTITY_DRIFT,
            "; ".join(identity),
            arm_id=arm_id,
        )
        record["stop_reason"] = (
            "the Phase-0 baseline must be refreshed before candidate performance; this "
            "campaign does not substitute another arm"
        )
    return record

def validate_candidate_runtime(
    state: SessionValidity,
    runtime_config: Mapping[str, Any] | None,
    *,
    arm_id: str,
    gpu0_uuid: str,
    gpu1_uuid: str,
    placement_sha256: str,
    expected_gpu1_slots: int,
    expected_gpu1_expert_bytes: int,
) -> dict[str, Any]:
    """The landed-candidate runtime contract, read off the live engine before warmups."""
    required = COMMON_REQUIRED_FIELDS + (
        "inferswarm_secondary_device.secondary.uuid",
        "inferswarm_remote_decode.enabled",
        "inferswarm_remote_decode.execution_mode",
        "inferswarm_remote_decode.transport",
        "inferswarm_remote_decode.placement_sha256",
        "inferswarm_remote_decode.primary.uuid",
        "inferswarm_remote_decode.secondary.uuid",
        "inferswarm_resident_bank.resident_slots",
    )
    ok = check_required_fields(state, runtime_config, required, arm_id=arm_id)
    if not ok or not isinstance(runtime_config, Mapping):
        return {"arm_id": arm_id, "checked": False, "stop_reason": "runtime report incomplete"}

    findings: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            findings.append(message)

    primary_uuid = _lookup(runtime_config, "inferswarm_remote_decode.primary.uuid")
    secondary_uuid = _lookup(
        runtime_config, "inferswarm_remote_decode.secondary.uuid"
    )
    require(
        str(primary_uuid or "").upper() == gpu0_uuid.upper(),
        f"remote-decode primary UUID is {primary_uuid!r}, expected {gpu0_uuid!r}",
    )
    require(
        str(secondary_uuid or "").upper() == gpu1_uuid.upper(),
        f"remote-decode secondary UUID is {secondary_uuid!r}, expected {gpu1_uuid!r}",
    )
    dev_secondary = _lookup(
        runtime_config, "inferswarm_secondary_device.secondary.uuid"
    )
    require(
        str(dev_secondary or "").upper() == gpu1_uuid.upper(),
        f"secondary-device UUID is {dev_secondary!r}, expected {gpu1_uuid!r}",
    )
    resolved_sha = _lookup(runtime_config, "inferswarm_remote_decode.placement_sha256")
    require(
        resolved_sha == placement_sha256,
        f"runtime placement SHA-256 is {resolved_sha!r}, expected {placement_sha256!r}",
    )
    slots = _lookup(runtime_config, "inferswarm_resident_bank.resident_slots")
    require(
        slots == expected_gpu1_slots,
        f"GPU1 resident slots = {slots!r}, expected {expected_gpu1_slots}",
    )
    banks = _lookup(runtime_config, "inferswarm_resident_bank.banks") or []
    gpu1_bytes = (
        sum(int(b.get("total_resident_bytes") or 0) for b in banks if isinstance(b, dict))
        if isinstance(banks, list)
        else None
    )
    require(
        gpu1_bytes == expected_gpu1_expert_bytes,
        f"GPU1 expert bank tensor bytes = {gpu1_bytes!r}, expected {expected_gpu1_expert_bytes}",
    )
    transport = _lookup(runtime_config, "inferswarm_remote_decode.transport")
    require(
        transport == "host_staged",
        f"transport mode is {transport!r}, expected 'host_staged'",
    )
    mode = _lookup(runtime_config, "inferswarm_remote_decode.execution_mode")
    require(
        _lookup(runtime_config, "inferswarm_remote_decode.enabled") is True
        and mode == "overlap",
        f"remote decode enabled={_lookup(runtime_config, 'inferswarm_remote_decode.enabled')!r} "
        f"mode={mode!r}; the canonical candidate runs overlap",
    )
    require(
        _lookup(runtime_config, "runtime.cuda_graph_max_bs") == 0
        and _lookup(runtime_config, "runtime.cuda_graph_capture_happened") is False,
        "candidate CUDA graph capture state is not disabled",
    )
    require(
        _lookup(runtime_config, "moe.backend_resolved") == "offload"
        and _lookup(runtime_config, "moe.decode_target") == "gpu"
        and _lookup(runtime_config, "moe.cpu_layers_resolved") == [],
        "candidate MoE backend/decode-target/CPU-layer contract is not the landed shape",
    )
    require(
        _lookup(runtime_config, "cache.resolved_slots") == 3774,
        f"candidate GPU0 cache slots = {_lookup(runtime_config, 'cache.resolved_slots')!r}, expected 3774",
    )
    resolved_kv = kv_capacity_tokens(runtime_config)
    require(
        resolved_kv == 17075,
        f"candidate resolved KV capacity = {resolved_kv!r} tokens, expected 17075 "
        "(the canonical candidate pins --num-tokens 17075; this pin is what makes "
        "the predeclared supplementary arm's capacity fully known)",
    )

    if findings:
        state.add(CANDIDATE_CONTRACT_MISMATCH, "; ".join(findings), arm_id=arm_id)
    return {
        "arm_id": arm_id,
        "checked": True,
        "resolved": {
            "primary_uuid": primary_uuid,
            "secondary_uuid": secondary_uuid,
            "placement_sha256": resolved_sha,
            "gpu1_resident_slots": slots,
            "gpu1_expert_bank_tensor_bytes": gpu1_bytes,
            "transport": transport,
            "remote_execution_mode": mode,
            "remote_decode_enabled": _lookup(
                runtime_config, "inferswarm_remote_decode.enabled"
            ),
            "cuda_graph_max_bs": _lookup(runtime_config, "runtime.cuda_graph_max_bs"),
            "cuda_graph_capture_happened": _lookup(
                runtime_config, "runtime.cuda_graph_capture_happened"
            ),
            "gpu0_cache_slots": _lookup(runtime_config, "cache.resolved_slots"),
            "peer_access": _lookup(
                runtime_config, "inferswarm_secondary_device.peer_access"
            ),
        },
        "contract_findings": findings,
    }


def validate_candidate_block_snapshot(
    state: SessionValidity,
    snapshot: Mapping[str, Any],
    *,
    arm_id: str,
    class_id: str,
    block_id: str,
) -> dict[str, Any]:
    """Post-block mechanism-window contracts: remote prefill zero, no fallback.

    The F-gate evaluation itself belongs to P6's analysis of the retained snapshot;
    what is validated here is only the hard runtime invariants that would make the
    arm not be the declared experiment at all.
    """
    findings: list[str] = []
    aggregate = snapshot.get("aggregate") or {}
    prefill_dispatches = aggregate.get("prefill_remote_dispatches")
    if prefill_dispatches != 0:
        findings.append(
            f"remote prefill dispatches = {prefill_dispatches!r}; the candidate "
            "executes prefill on GPU0 only"
        )
    fallback = aggregate.get("fallback_elsewhere")
    if fallback != 0:
        findings.append(
            f"fallback_elsewhere = {fallback!r}; silent fallback is forbidden, a "
            "GPU1-assigned route that cannot execute is an explicit failure"
        )
    ownership = snapshot.get("ownership") or {}
    if ownership.get("successful_selection_arithmetic_exact") is not True:
        findings.append("route-ownership arithmetic did not account every selection exactly")
    where = {"arm_id": arm_id, "class_id": class_id}
    if findings:
        state.add(REMOTE_FORBIDDEN_FALLBACK, "; ".join(findings), **where)
    return {
        "block_id": block_id,
        "prefill_remote_dispatches": prefill_dispatches,
        "fallback_elsewhere": fallback,
        "ownership": ownership,
        "findings": findings,
    }


def kv_capacity_tokens(runtime_config: Mapping[str, Any]) -> int | None:
    pages = _lookup(runtime_config, "runtime.num_pages")
    page_size = _lookup(runtime_config, "runtime.page_size")
    if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
        return pages * page_size
    return None
