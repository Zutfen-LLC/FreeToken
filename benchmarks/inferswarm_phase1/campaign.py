"""The P5 campaign runner: plan, validate, and (for P6) execute one session at a time.

Built on the Phase-0 harness modules — manifest, provenance, client, statistics,
validity — not a second benchmark framework. What this module adds is the Phase-1
campaign shape:

* the whole session plan exists before the first server starts;
* provenance preflight fails closed (dirty tree, wrong model/manifest/placement SHA,
  wrong GPUs, missing correctness prerequisites, missing runner version);
* one fresh server per arm per session, started/stopped by the runner, with startup
  timestamps and an idle-memory check between arms;
* runtime-resolution validation after ``/health`` ready and before warmups — the
  baseline identity drift check STOPs the session before candidate performance;
* per-class instrumentation windows (reset after warmups, snapshot after the measured
  repetitions) retaining the mechanism counters and issue-#5 layer timing;
* failures preserved, blocks marked incomplete, no repetition deleted or invisibly
  retried, no dynamic shortening;
* per-arm descriptive statistics after a block completes; never a cross-arm ratio,
  never a campaign verdict.

This package does not run the canonical campaign; P6 does, after the campaign-order
amendment merges.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferswarm_phase0 import CANONICAL_MODEL_REPOSITORY
from inferswarm_phase0 import gpu as gpu_mod
from inferswarm_phase0 import provenance as prov
from inferswarm_phase0.client import (
    GenerationError,
    ServerError,
    fetch_instrumentation,
    free_port,
    get_json,
    measure_generation,
    prefill_seq_floor,
    start_server,
    stop_server,
)
from inferswarm_phase0.manifest import (
    Manifest,
    check_completion_tokens,
    check_prompt_tokens,
    load_manifest,
)

from . import CAMPAIGN_RUNNER_VERSION
from .campaign_arms import (
    BASELINE_ARM_ID,
    CANDIDATE_ARM_ID,
    CANONICAL_PLACEMENT_SHA256,
    EXPECTED_GPU1_EXPERT_BYTES,
    EXPECTED_GPU1_SLOTS,
    GPU0_UUID,
    GPU1_UUID,
    PHASE0_BASELINE_COMMIT,
    ArmDefinitionError,
    CampaignArm,
    compare_primary_arms,
    load_placement_reference,
    supplementary_requirement,
    validate_arm_definitions,
)
from .campaign_artifacts import (
    CAMPAIGN_PLAN_SCHEMA,
    BlockTally,
    SessionWriter,
    baseline_noise_floor,
    execution_status,
    summarize_block,
)
from .campaign_protocol import (
    CAMPAIGN_PRIMARY_GENERATIONS,
    CANONICAL_CLASSES,
    PER_SESSION_PRIMARY_GENERATIONS,
    CampaignProtocol,
    PlannedGeneration,
    build_session_plan,
    iter_blocks,
    session_arm_order,
)
from .campaign_validity import (
    GENERATION_FAILED,
    GPU_IDLE_NOT_RESTORED,
    SERVER_FAILED,
    BaselineIdentityError,
    SessionValidity,
    validate_baseline_runtime,
    validate_candidate_block_snapshot,
    validate_candidate_runtime,
)
from .campaign_validity import (
    kv_capacity_tokens as kv_tokens_of,
)

# The frozen workload identity: canonical manifest and its SHA-256. A canonical run
# with an altered manifest is refused.
CANONICAL_MANIFEST_ID = "phase0-v1-2026-08-27"
CANONICAL_MANIFEST_SHA256 = (
    "10f81e5418a71a68f387632de422c3337cc7ba0518111a8746ad856d0210b24a"
)

# Between arms: the previous arm's server must be gone and GPU memory back in its
# idle range before the next arm starts. Frozen here, before any campaign.
GPU_IDLE_MEMORY_MAX_BYTES = 1 << 30

# Driver settle time between stopping a server and reading idle memory, so the
# reading reflects released rather than in-flight teardown.
BETWEEN_ARM_SETTLE_SECONDS = 3.0

# One measured block per class = repetitions x output tokens decode steps; W1/W2 are
# 512, so 10 x 512 = 5120 covers every class's window after the post-warmup reset.
DEFAULT_TIMING_MAX_STEPS = 10 * 512

_PREREQUISITE_KEYS = (
    "correctness_reference_v2_artifact_sha256",
    "candidate_c3_artifact_sha256",
    "p2_p3_p4_requalification_artifact_sha256",
    "freetoken_runtime_commit",
)


class CampaignRefused(ValueError):
    """A canonical campaign that must not start. Nothing has been measured."""


@dataclass(frozen=True)
class CampaignSettings:
    """Everything the campaign fixes once, outside the per-arm flag lists."""

    model_path: str
    manifest_path: str
    model_repository: str = CANONICAL_MODEL_REPOSITORY
    model_revision: str | None = None
    placement_path: str | None = None
    inferswarm_commit: str | None = None
    out_root: Path = Path("phase1-campaign")
    server_timeout: float = 3600.0
    python_executable: str = sys.executable
    timing_max_steps: int = DEFAULT_TIMING_MAX_STEPS
    store_output_text: bool = False
    echo_server_output: bool = True
    # The correctness/mechanism prerequisite manifest (P6 supplies real artifacts).
    prerequisites_path: str | None = None

    def env_overrides(self) -> dict[str, str]:
        # Same env on every arm, exactly as Phase 0 did, so instrumentation cannot
        # advantage one arm over the other. Prefill CUDA-event instrumentation is the
        # Phase-0 contract's performance-compatible surface.
        return {"FREETOKEN_INSTRUMENT_PREFILL": "1"}


@dataclass
class CampaignDefinition:
    """Arms + protocol + settings: everything the plan and session derive from."""

    arms: Sequence[CampaignArm]
    protocol: CampaignProtocol
    settings: CampaignSettings
    canonical: bool = True

    def arms_by_id(self) -> dict[str, CampaignArm]:
        return {a.id: a for a in self.arms}

    def primary_arms(self) -> list[CampaignArm]:
        return [a for a in self.arms if a.role == "primary"]


# ----------------------------------------------------------------------------------------
# serve command
# ----------------------------------------------------------------------------------------


def serve_command(
    arm: CampaignArm, settings: CampaignSettings, port: int
) -> list[str]:
    """The exact ``ft serve`` command for one arm (byte-stable for a fixed port)."""
    return [
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
        *arm.flags(placement_path=settings.placement_path),
        "--moe-layer-timing-max-steps",
        str(settings.timing_max_steps),
    ]


# ----------------------------------------------------------------------------------------
# provenance preflight
# ----------------------------------------------------------------------------------------


def _triton_version() -> Any:
    try:
        from importlib.metadata import version

        return version("triton")
    except Exception:  # noqa: BLE001 -- provenance must never raise
        return prov.unavailable("triton distribution metadata unavailable")


def phase1_software_provenance(settings: CampaignSettings) -> dict[str, Any]:
    base = prov.software_provenance(settings.inferswarm_commit, CAMPAIGN_RUNNER_VERSION)
    base["phase1_campaign_runner_version"] = CAMPAIGN_RUNNER_VERSION
    base["triton"] = _triton_version()
    return base


def load_prerequisites(path: str | None) -> dict[str, Any]:
    """The correctness/mechanism prerequisite manifest.

    Records the exact passing correctness-reference-v2 artifact, candidate C3
    artifact, P2/P3/P4 requalification artifact, and the FreeToken runtime commit the
    performance campaign build was qualified with. A canonical session refuses to
    start without it; correctness is re-established on the exact campaign build
    before performance is accepted (P6). The performance runner itself never enables
    the C3 full-logit recorder.
    """
    if path is None:
        return {
            "supplied": False,
            "required_keys": list(_PREREQUISITE_KEYS),
            "note": (
                "a canonical session requires the prerequisite manifest naming the "
                "exact passing correctness artifacts for this build"
            ),
        }
    p = Path(path)
    if not p.is_file():
        raise CampaignRefused(f"correctness prerequisites manifest not found: {path}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise CampaignRefused(
            f"correctness prerequisites manifest must be a JSON object: {path}"
        )
    for key in _PREREQUISITE_KEYS:
        if not doc.get(key):
            raise CampaignRefused(
                f"correctness prerequisites manifest is missing {key!r}: performance "
                "from an unqualified build is not evidence"
            )
    return {"supplied": True, **doc}


def _hostname() -> Any:
    name = platform.node()
    return name if name else prov.unavailable("platform.node() returned empty")


def _gpu_row(gpu_block: dict[str, Any], uuid: str | None) -> dict[str, Any] | None:
    rows = gpu_block.get("gpus")
    if not isinstance(rows, list) or uuid is None:
        return None
    for row in rows:
        if isinstance(row, dict) and str(row.get("uuid", "")).upper() == uuid.upper():
            return row
    return None


def provenance_document(
    definition: CampaignDefinition,
    manifest: Manifest,
    placement: dict[str, Any] | None,
) -> dict[str, Any]:
    """The full §8 provenance record: repo, model, workloads, GPUs, software, host,
    placement, prerequisites."""
    settings = definition.settings
    gpu0 = gpu_mod.resolve_gpu(GPU0_UUID)
    gpu1 = gpu_mod.resolve_gpu(GPU1_UUID)
    gpu_block = prov.gpu_provenance(GPU0_UUID, gpu0.resolved_uuid)
    return {
        "software": phase1_software_provenance(settings),
        "model": prov.model_provenance(
            prov.ModelPin(
                repository=settings.model_repository,
                revision=settings.model_revision or "",
                local_path=settings.model_path,
            )
        ),
        "workloads": {
            "manifest_path": manifest.source_path,
            "manifest_id": manifest.manifest_id,
            "manifest_sha256": manifest.manifest_sha256,
            "canonical_manifest_id": CANONICAL_MANIFEST_ID,
            "canonical_manifest_sha256": CANONICAL_MANIFEST_SHA256,
            "fixture_sha256": {w.class_id: w.content_sha256 for w in manifest.workloads},
            "output_tokens": {w.class_id: w.output_tokens for w in manifest.workloads},
        },
        "gpus": {
            "gpu0": {
                "declared_uuid": GPU0_UUID,
                "resolved": gpu0.record(),
                "row": _gpu_row(gpu_block, gpu0.resolved_uuid),
            },
            "gpu1": {
                "declared_uuid": GPU1_UUID,
                "resolved": gpu1.record(),
                "row": _gpu_row(gpu_block, gpu1.resolved_uuid),
            },
            "topology": gpu_block.get("topology"),
            "topology_p2p": gpu_block.get("topology_p2p"),
            "note": (
                "peer-access result and transport mode are additionally recorded from "
                "the running candidate engine (inferswarm_secondary_device) in the "
                "candidate arm's runtime.json"
            ),
        },
        "host": {
            **prov.host_provenance(),
            "hostname": _hostname(),
            "hostname_note": "host-local session identity; not published evidence",
        },
        "placement": (
            {
                "policy": placement["policy_id"],
                "canonical_remote_placement": placement["canonical_remote_placement"],
                "artifact_sha256": placement["sha256"],
                "frozen_sha256": CANONICAL_PLACEMENT_SHA256,
                "remote_slots": placement["budget"]["remote_slots"],
                "remote_resident_bytes": placement["budget"]["remote_resident_bytes"],
                "model_revision": placement["model_revision"],
                "source_artifact_shas": placement["source_artifact_shas"],
            }
            if placement
            else prov.unavailable("no candidate arm requires a placement")
        ),
        "prerequisites": load_prerequisites(settings.prerequisites_path),
        "historical_phase0_baseline_commit": PHASE0_BASELINE_COMMIT,
        "historical_phase0_baseline_note": (
            "the Phase-0 baseline was measured on that commit; this campaign "
            "remeasures the same frozen B1 identity on the current campaign build "
            "instead of dividing by numbers from another commit/day"
        ),
    }


def _model_pin(settings: CampaignSettings) -> prov.ModelPin:
    return prov.ModelPin(
        repository=settings.model_repository,
        revision=settings.model_revision or "",
        local_path=settings.model_path,
    )


def _ordered_primary_pair(
    definition: CampaignDefinition,
) -> tuple[CampaignArm, CampaignArm]:
    arms = definition.arms_by_id()
    return arms[BASELINE_ARM_ID], arms[CANDIDATE_ARM_ID]


def preflight_refusals(
    definition: CampaignDefinition,
    manifest: Manifest,
    placement: dict[str, Any] | None,
) -> list[str]:
    """Everything a canonical campaign would refuse to start on, cheaply.

    Runs in ``validate`` and again before a session executes; nothing measures until
    this list is empty (a dev-smoke campaign returns no refusals and is stamped
    non-canonical everywhere).
    """
    if not definition.canonical:
        return []
    settings = definition.settings
    reasons: list[str] = []

    if not CAMPAIGN_RUNNER_VERSION:
        reasons.append("the campaign runner version is missing from provenance")
    if settings.model_repository != CANONICAL_MODEL_REPOSITORY:
        reasons.append(
            f"canonical Phase 1 requires model repository {CANONICAL_MODEL_REPOSITORY}; "
            f"observed {settings.model_repository!r}"
        )
    for check, value in (
        (prov.validate_revision, settings.model_revision),
        (prov.validate_inferswarm_commit, settings.inferswarm_commit),
    ):
        try:
            check(value, canonical=True)
        except ValueError as e:
            reasons.append(str(e))
    dirty = prov.check_clean_working_tree(prov.git_commit(prov.freetoken_repo_root()))
    if dirty:
        reasons.append(dirty)
    mismatch = prov.check_snapshot_revision(_model_pin(settings))
    if mismatch:
        reasons.append(mismatch)

    if manifest.manifest_id != CANONICAL_MANIFEST_ID or (
        manifest.manifest_sha256 != CANONICAL_MANIFEST_SHA256
    ):
        reasons.append(
            f"the canonical workload manifest is {CANONICAL_MANIFEST_ID} at SHA-256 "
            f"{CANONICAL_MANIFEST_SHA256}; observed {manifest.manifest_id!r} at "
            f"{manifest.manifest_sha256!r}. A canonical run with an altered manifest "
            "is refused; developer smoke manifests must be explicitly non-canonical."
        )
    if [w.class_id for w in manifest.workloads] != list(CANONICAL_CLASSES):
        reasons.append(
            f"the canonical class order is {list(CANONICAL_CLASSES)}; observed "
            f"{[w.class_id for w in manifest.workloads]}"
        )

    if placement is None:
        if any(a.id == CANDIDATE_ARM_ID for a in definition.arms):
            reasons.append(
                "the candidate arm requires the frozen placement artifact "
                f"(--placement-path) pinned to SHA-256 {CANONICAL_PLACEMENT_SHA256}"
            )
    else:
        if placement["sha256"] != CANONICAL_PLACEMENT_SHA256:
            reasons.append("placement SHA-256 does not match the frozen artifact")
        if (
            settings.model_revision
            and placement["model_revision"]
            and placement["model_revision"].lower() != settings.model_revision.lower()
        ):
            reasons.append(
                f"the placement artifact pins model revision {placement['model_revision']}, "
                f"which disagrees with --model-revision {settings.model_revision}"
            )
        if placement["workload_manifest_sha256"] != CANONICAL_MANIFEST_SHA256:
            reasons.append(
                "the placement artifact pins a different workload manifest SHA-256"
            )

    reasons.extend(validate_arm_definitions(definition.arms))
    comparison = compare_primary_arms(*_ordered_primary_pair(definition))
    for diff in comparison["undeclared_differences"]:
        reasons.append(
            f"undeclared arm difference {diff['flag']}: baseline={diff['baseline']!r} "
            f"candidate={diff['candidate']!r}"
        )
    for name, entry in comparison["held_equal"].items():
        if not entry["equal"]:
            detail = entry.get("reason") or (
                f"baseline={entry.get('baseline')!r} candidate={entry.get('candidate')!r}"
            )
            reasons.append(
                f"held-constant flag {name} differs between the primary arms: {detail}"
            )

    try:
        prerequisites = load_prerequisites(settings.prerequisites_path)
        if not prerequisites.get("supplied"):
            reasons.append(
                "canonical performance requires the correctness prerequisite manifest "
                "(--correctness-prerequisites): the exact passing "
                "correctness-reference-v2, candidate C3, and P2/P3/P4 requalification "
                "artifacts plus the FreeToken runtime commit. Performance from an "
                "unqualified build is not evidence."
            )
    except CampaignRefused as e:
        reasons.append(str(e))

    for name, selection in (
        ("GPU0", gpu_mod.resolve_gpu(GPU0_UUID)),
        ("GPU1", gpu_mod.resolve_gpu(GPU1_UUID)),
    ):
        if not selection.proven:
            reasons.append(
                f"{name} must resolve to its frozen physical UUID "
                f"({selection.requested}); {selection.unavailable}"
            )
    return reasons


# ----------------------------------------------------------------------------------------
# thermal / session boundary
# ----------------------------------------------------------------------------------------

_SMI_FIELDS = (
    "uuid",
    "temperature_gpu_c",
    "clocks_sm",
    "power_draw",
    "pstate",
    "memory_used_mib",
)


def thermal_observation() -> dict[str, Any]:
    """GPU temperatures/clocks/power/state, host load, background GPU processes.

    Recorded before each arm. Reuses the Phase-0 thermal/session-reset rules; no new
    temperature threshold is invented here — the observation is evidence, the reset
    requirement itself is the Phase-0 amendment's and is operator-attested.
    """
    query = prov._run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,temperature.gpu,clocks.sm,power.draw,pstate,memory.used",
            "--format=csv,noheader",
        ]
    )
    gpus = (
        [
            dict(
                zip(
                    _SMI_FIELDS,
                    (c.strip() for c in line.split(",")),
                )
            )
            for line in query.splitlines()
            if line.strip()
        ]
        if query
        else prov.unavailable("nvidia-smi thermal query failed")
    )
    apps = prov._run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    )
    try:
        load = Path("/proc/loadavg").read_text().strip()
    except OSError:
        load = prov.unavailable("could not read /proc/loadavg")
    return {
        "observed_at": prov.utc_now_iso(),
        "gpus": gpus,
        "host_load": load,
        "background_compute_apps": (
            [line for line in apps.splitlines() if line.strip()]
            if apps is not None
            else prov.unavailable("nvidia-smi compute-apps query failed")
        ),
        "note": (
            "Phase-0 thermal/session-reset rules apply unchanged; a session boundary "
            "is an operator-attested independently cooled thermal reset, never an "
            "elapsed-wall-time assumption"
        ),
    }


def gpu_memory_used_bytes(uuid: str) -> int | None:
    out = prov._run(
        ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader"]
    )
    if not out:
        return None
    for line in out.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) == 2 and cells[0].upper() == uuid.upper():
            value = cells[1].split()[0] if cells[1] else ""
            try:
                return int(float(value)) * (1 << 20)
            except ValueError:
                return None
    return None


def session_boundary_check(
    session_number: int, thermal_reset_attested: str | None
) -> dict[str, Any]:
    """Session 2 (and any later session) requires an attested independent thermal reset."""
    attested = bool(thermal_reset_attested and thermal_reset_attested.strip())
    required = session_number >= 2
    record: dict[str, Any] = {
        "session_number": session_number,
        "attestation_required": required,
        "attestation": thermal_reset_attested or None,
        "observed": thermal_observation(),
        "passed": True if not required else attested,
    }
    if required and not attested:
        record["reason"] = (
            "an independently cooled thermal reset (Phase-0 Session-2 amendment) must "
            "be attested before this session starts; a session boundary is never "
            "assumed from elapsed wall time"
        )
    return record


# ----------------------------------------------------------------------------------------
# manifest / placement loading
# ----------------------------------------------------------------------------------------


def load_campaign_manifest(path: str | Path, *, canonical: bool) -> Manifest:
    """Load the frozen W1-W4 manifest; a canonical run refuses an altered identity."""
    manifest = load_manifest(path, canonical=canonical)
    if canonical and (
        manifest.manifest_id != CANONICAL_MANIFEST_ID
        or manifest.manifest_sha256 != CANONICAL_MANIFEST_SHA256
    ):
        raise CampaignRefused(
            f"the canonical workload manifest is {CANONICAL_MANIFEST_ID} at SHA-256 "
            f"{CANONICAL_MANIFEST_SHA256}; observed {manifest.manifest_id!r} at "
            f"{manifest.manifest_sha256!r}. Developer smoke manifests must be "
            "explicitly non-canonical."
        )
    return manifest


def build_placement_reference(definition: CampaignDefinition) -> dict[str, Any] | None:
    """Read and validate the frozen placement artifact before any model starts."""
    if not any(a.id == CANDIDATE_ARM_ID for a in definition.arms):
        return None
    try:
        return load_placement_reference(definition.settings.placement_path or "")
    except ArmDefinitionError as e:
        raise CampaignRefused(f"placement preflight failed: {e}") from e


def session_full_arm_order(definition: CampaignDefinition, session_number: int) -> list[str]:
    """The executed arm order: the counterbalanced primaries, then any supplementary.

    Supplementary arms run after both primaries resolve (their requirement is
    mechanical from the primaries' recorded KV capacities), are clearly labelled, and
    are counted separately — they never replace a primary arm.
    """
    order = list(session_arm_order(session_number))
    order.extend(
        a.id for a in definition.arms if a.role == "supplementary"
    )
    return order


# ----------------------------------------------------------------------------------------
# planning / dry run
# ----------------------------------------------------------------------------------------


def plan_document(
    definition: CampaignDefinition, manifest: Manifest | None
) -> dict[str, Any]:
    """The whole two-session campaign as an explicit document, before any server.

    Every expected generation of both sessions appears with session/arm/class,
    warmup/measured phase, repetition and execution index. There is no dynamic
    shortening: the executed session must match this plan exactly.
    """
    arms_by_id = definition.arms_by_id()
    sessions: list[dict[str, Any]] = []
    for number in (1, 2):
        order = session_full_arm_order(definition, number)
        steps = build_session_plan(
            session_number=number,
            arm_order=order,
            arms_by_id=arms_by_id,
            protocol=definition.protocol,
        )
        sessions.append(
            {
                "session_id": f"session-{number}",
                "session_number": number,
                "arm_order": list(order),
                "primary_arm_order": list(session_arm_order(number)),
                "supplementary_arm_ids": [
                    a.id for a in definition.arms if a.role == "supplementary"
                ],
                "class_order": list(definition.protocol.classes),
                "class_order_reversed": False,
                "fresh_server_per_arm": True,
                "radix_cache_cleared_between_classes": False,
                "thermal_reset_required": True,
                "arm_order_note": (
                    "session 2 reverses the ARM order only; workload order is "
                    "identical in both sessions (campaign-order amendment)"
                ),
                "generations": [s.record() for s in steps],
                "generation_count": len(steps),
                "primary_generation_count": sum(
                    1 for s in steps if arms_by_id[s.arm_id].role == "primary"
                ),
            }
        )
    primary_total = sum(s["primary_generation_count"] for s in sessions)
    return {
        "schema": CAMPAIGN_PLAN_SCHEMA,
        "campaign_runner_version": CAMPAIGN_RUNNER_VERSION,
        "canonical": (
            definition.canonical
            and definition.protocol.canonical
            and (manifest is None or manifest.canonical)
        ),
        "protocol": definition.protocol.record(),
        "arms": [a.record() for a in definition.arms],
        "serve_commands": {
            a.id: serve_command(a, definition.settings, port=0) for a in definition.arms
        },
        "env_overrides": definition.settings.env_overrides(),
        "workload_manifest": manifest.record() if manifest is not None else None,
        "sessions": sessions,
        "primary_generation_counts": {
            "per_arm_per_session": len(definition.protocol.classes)
            * (definition.protocol.warmups + definition.protocol.repetitions),
            "per_session": primary_total // 2,
            "campaign": primary_total,
            "canonical_per_session": PER_SESSION_PRIMARY_GENERATIONS,
            "canonical_campaign": CAMPAIGN_PRIMARY_GENERATIONS,
        },
        "counterbalanced_order": {
            "session-1": list(session_arm_order(1)),
            "session-2": list(session_arm_order(2)),
        },
        "supplementary_arm_support": (
            "the KV-matched supplementary baseline is resolved mechanically after "
            "both primary arms resolve; it never replaces a primary arm"
        ),
    }


def validation_document(definition: CampaignDefinition) -> dict[str, Any]:
    """The ``validate`` dry-run: proof the campaign is internally comparable.

    No model is started. Proves both arms fully specified, held constants equal,
    intended differences exactly enumerated, manifest/placement hashes canonical,
    session ordering canonical, expected counts exact, no protocol deviation, and no
    forbidden flag leakage — plus the preflight refusals a real session would hit.
    """
    canonical = definition.canonical and definition.protocol.canonical
    manifest = load_campaign_manifest(
        definition.settings.manifest_path, canonical=canonical
    )
    placement: dict[str, Any] | None = None
    placement_error: str | None = None
    if any(a.id == CANDIDATE_ARM_ID for a in definition.arms):
        try:
            placement = load_placement_reference(
                definition.settings.placement_path or ""
            )
        except ArmDefinitionError as e:
            placement_error = f"placement preflight failed: {e}"
    refusals = preflight_refusals(definition, manifest, placement)
    if placement_error is not None:
        refusals.append(placement_error)
    plan = plan_document(definition, manifest)
    baseline, candidate = _ordered_primary_pair(definition)
    comparison = compare_primary_arms(baseline, candidate)

    count_ok = (
        plan["primary_generation_counts"]["per_session"]
        == PER_SESSION_PRIMARY_GENERATIONS
        and plan["primary_generation_counts"]["campaign"]
        == CAMPAIGN_PRIMARY_GENERATIONS
    )
    session_orders = [s["primary_arm_order"] for s in plan["sessions"]]
    ordering_ok = session_orders == [
        list(session_arm_order(1)),
        list(session_arm_order(2)),
    ]
    class_orders_ok = all(
        s["class_order"] == list(CANONICAL_CLASSES) and not s["class_order_reversed"]
        for s in plan["sessions"]
    )

    return {
        "schema": "inferswarm.phase1.campaign-validation/1",
        "campaign_runner_version": CAMPAIGN_RUNNER_VERSION,
        "canonical": bool(
            canonical
            and manifest.canonical
            and not refusals
            and count_ok
            and ordering_ok
            and class_orders_ok
            and comparison["held_equal_all"]
            and not comparison["undeclared_differences"]
            and not definition.protocol.deviations
        ),
        "canonical_blockers": [
            *([] if definition.canonical else ["canonical mode is off (--dev-smoke)"]),
            *definition.protocol.deviations,
            *([] if manifest.canonical else ["workload manifest declares canonical=false"]),
        ],
        "preflight_refusals": refusals,
        "held_equal": comparison["held_equal"],
        "held_equal_all": comparison["held_equal_all"],
        "intended_differences": comparison["intended_differences"],
        "undeclared_differences": comparison["undeclared_differences"],
        "counts": {
            **plan["primary_generation_counts"],
            "counts_ok": count_ok,
        },
        "session_orders": session_orders,
        "session_ordering_ok": ordering_ok,
        "class_orders_ok": class_orders_ok,
        "manifest": manifest.record(),
        "placement": (
            {
                "policy": placement["policy_id"],
                "artifact_sha256": placement["sha256"],
                "frozen_sha256": CANONICAL_PLACEMENT_SHA256,
                "remote_slots": placement["budget"]["remote_slots"],
                "remote_resident_bytes": placement["budget"]["remote_resident_bytes"],
                "model_revision": placement["model_revision"],
            }
            if placement
            else None
        ),
        "placement_error": placement_error,
        "workload_identity": {
            "manifest_id": manifest.manifest_id,
            "manifest_sha256": manifest.manifest_sha256,
            "canonical": (
                manifest.manifest_id == CANONICAL_MANIFEST_ID
                and manifest.manifest_sha256 == CANONICAL_MANIFEST_SHA256
            ),
        },
        "verdict_firewall": (
            "this runner computes no cross-arm ratio and emits no campaign verdict; "
            "execution/validity states and per-arm descriptive statistics are its "
            "complete output vocabulary"
        ),
    }

# ----------------------------------------------------------------------------------------
# session execution (P6 surface; exercised by mocked tests in P5)
# ----------------------------------------------------------------------------------------


def _post_json(url: str, body: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
    import urllib.request

    request = urllib.request.Request(
        url,
        data=json.dumps(dict(body)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def moe_instrumentation(
    origin: str, operation: str, *, timeout: float = 60.0
) -> dict[str, Any]:
    """POST /v1/moe/instrumentation — the engine-owned idle-boundary reset/snapshot."""
    response = _post_json(
        f"{origin}/v1/moe/instrumentation",
        {"operation": operation, "timeout": timeout},
        timeout=timeout + 5,
    )
    if response.get("status") != "ok" or "payload" not in response:
        raise ServerError(f"MoE instrumentation {operation} failed: {response}")
    return response["payload"]


@dataclass
class SessionExecution:
    """Executes one session of the campaign. P6's entry point; P5 proves it by mock."""

    definition: CampaignDefinition
    session_number: int
    thermal_reset_attested: str | None = None
    validity: SessionValidity = field(init=False)

    def __post_init__(self) -> None:
        self.canonical = self.definition.canonical
        self.validity = SessionValidity(canonical_intent=self.canonical)
        settings = self.definition.settings
        self.manifest = load_campaign_manifest(
            settings.manifest_path, canonical=self.canonical
        )
        self.placement = build_placement_reference(self.definition)
        self.arms_by_id = self.definition.arms_by_id()
        self.root = Path(settings.out_root) / f"session-{self.session_number}"
        self.writer = SessionWriter(self.root)
        self.steps = build_session_plan(
            session_number=self.session_number,
            arm_order=session_full_arm_order(self.definition, self.session_number),
            arms_by_id=self.arms_by_id,
            protocol=self.definition.protocol,
        )
        self.tallies: dict[tuple, BlockTally] = {}
        for arm_id, class_id, block in iter_blocks(self.steps):
            self.tallies[(arm_id, class_id)] = BlockTally(
                arm_id=arm_id,
                class_id=class_id,
                block_id=block[0].block_id,
                expected_warmups=sum(1 for s in block if not s.measured),
                expected_measured=sum(1 for s in block if s.measured),
            )
        self.runtime_records: dict[str, dict[str, Any]] = {}
        self.startup_records: dict[str, dict[str, Any]] = {}
        self.thermal_records: list[dict[str, Any]] = []
        self.idle_records: list[dict[str, Any]] = []
        self.kv_capacities: dict[str, int | None] = {}
        self.stopped_early_reason: str | None = None
        self._recorded_indices: set[int] = set()
        self.boundary_record: dict[str, Any] | None = None

    # ---- gates -----------------------------------------------------------------------

    def preflight(self) -> None:
        """Fail-closed checks before the plan is written. Raises CampaignRefused."""
        if not self.canonical:
            self.validity.canonical_blockers.extend(self.definition.protocol.deviations)
            self.validity.canonical_blockers.append(
                "--dev-smoke: this session is NON_CANONICAL_DEV_SMOKE by construction"
            )
            self.boundary_record = session_boundary_check(self.session_number, None)
            return
        self.boundary_record = session_boundary_check(
            self.session_number, self.thermal_reset_attested
        )
        if not self.boundary_record["passed"]:
            raise CampaignRefused(str(self.boundary_record.get("reason")))
        refusals = preflight_refusals(self.definition, self.manifest, self.placement)
        if refusals:
            raise CampaignRefused("canonical session refused: " + "; ".join(refusals))

    # ---- execution -------------------------------------------------------------------

    def execute(self) -> dict[str, Any]:
        self.preflight()
        plan = plan_document(self.definition, self.manifest)
        session_plan = next(
            s for s in plan["sessions"] if s["session_number"] == self.session_number
        )
        self.writer.write_plan(
            {
                **session_plan,
                "campaign_runner_version": CAMPAIGN_RUNNER_VERSION,
                "protocol": self.definition.protocol.record(),
                "no_dynamic_shortening": True,
            }
        )
        provenance = provenance_document(self.definition, self.manifest, self.placement)
        provenance["thermal"] = {"session_boundary": self.boundary_record}
        self.writer.write_provenance(provenance)

        by_class = self.manifest.by_class()
        for arm_id in dict.fromkeys(s.arm_id for s in self.steps):
            if self.stopped_early_reason is not None:
                self._record_arm_not_executed(arm_id, self.stopped_early_reason)
                continue
            idle = self._observe_before_arm(arm_id)
            if idle.get("refuse_start"):
                self._record_arm_not_executed(arm_id, idle["reason"])
                continue
            try:
                self._run_arm(self.arms_by_id[arm_id], by_class)
            except BaselineIdentityError as e:
                # The amendment's STOP: no candidate performance after a drifted
                # baseline. The baseline arm's own records are already written; the
                # remaining arms are recorded as not executed.
                self.stopped_early_reason = str(e)
                continue
        return self._finalize()

    def _record_arm_not_executed(self, arm_id: str, reason: str) -> None:
        """Preserve the absence of an arm's generations; never silently skip."""
        for step in self.steps:
            if step.arm_id != arm_id or step.execution_index in self._recorded_indices:
                continue
            self._write_generation(
                arm_id,
                {
                    "session_id": step.session_id,
                    "session_number": step.session_number,
                    "execution_index": step.execution_index,
                    "arm_id": arm_id,
                    "arm_role": step.arm_role,
                    "class_id": step.class_id,
                    "phase": step.phase,
                    "measured": step.measured,
                    "repetition": step.repetition,
                    "block_id": step.block_id,
                    "failed": True,
                    "error": f"not executed ({reason})",
                    "started_at_unix": None,
                },
            )
            self.tallies[(arm_id, step.class_id)].failures.append(
                {"execution_index": step.execution_index, "error": f"not executed ({reason})"}
            )

    # ---- one arm ---------------------------------------------------------------------

    def _run_arm(self, arm: CampaignArm, by_class: Mapping[str, Any]) -> None:
        port = free_port()
        origin = f"http://127.0.0.1:{port}"
        command = serve_command(arm, self.definition.settings, port)
        log_path = self.writer.server_log_path(arm.id)
        print(
            f"[phase1] session-{self.session_number} {arm.id}: {' '.join(command)}",
            flush=True,
        )
        launched = time.time()
        handle = None
        try:
            handle = start_server(
                command,
                origin,
                str(log_path),
                env_overrides=self.definition.settings.env_overrides(),
                ready_timeout=self.definition.settings.server_timeout,
                echo=self.definition.settings.echo_server_output,
            )
            ready = time.time()
            self.startup_records[arm.id] = {
                "arm_id": arm.id,
                "launched_at_unix": launched,
                "ready_at_unix": ready,
                "m_start_duration_s": ready - launched,
                "serve_command": command,
                "env_overrides": self.definition.settings.env_overrides(),
                "note": "startup is reported, never amortized into tokens/s (criteria section 11)",
            }
            self.writer.write_startup(arm.id, self.startup_records[arm.id])

            instrumentation = fetch_instrumentation(origin)
            runtime_config = (
                instrumentation.get("runtime_config")
                if isinstance(instrumentation, dict)
                else None
            )
            gpu_verification = gpu_mod.verify_engine_gpu(
                gpu_mod.resolve_gpu(GPU0_UUID), gpu_mod.engine_gpus(origin)
            )
            self._validate_arm_runtime(arm, runtime_config, gpu_verification)
            self.kv_capacities[arm.id] = kv_tokens_of(runtime_config or {})
            model_id = _model_id(origin)

            # Arm-major: every class of this arm runs inside this one server process,
            # in the fixed protocol order; no restart and no radix-cache clearing
            # between classes.
            for class_id in self.definition.protocol.classes:
                block_steps = [
                    s for s in self.steps if s.arm_id == arm.id and s.class_id == class_id
                ]
                if block_steps:
                    self._run_block(arm, class_id, block_steps, by_class, origin, model_id)
        except ServerError as e:
            reason = f"{arm.id}: {e}"
            print(f"[phase1] {reason}", flush=True)
            self.validity.add(SERVER_FAILED, reason, arm_id=arm.id)
            self._record_arm_not_executed(arm.id, f"server failed: {e}")
        finally:
            if handle is not None:
                stop_server(handle)
            self.idle_records.append(self._observe_after_arm(arm.id))
            self._write_arm_summary(arm.id)

    def _validate_arm_runtime(
        self,
        arm: CampaignArm,
        runtime_config: Mapping[str, Any] | None,
        gpu_verification: dict[str, Any],
    ) -> None:
        from inferswarm_phase0.validity import check_engine_gpu

        check_engine_gpu(
            self.validity,
            gpu_verification,
            canonical_intent=self.canonical,
            arm_id=arm.id,
        )
        if arm.id == CANDIDATE_ARM_ID:
            record = validate_candidate_runtime(
                self.validity,
                runtime_config,
                arm_id=arm.id,
                gpu0_uuid=GPU0_UUID,
                gpu1_uuid=GPU1_UUID,
                placement_sha256=CANONICAL_PLACEMENT_SHA256,
                expected_gpu1_slots=EXPECTED_GPU1_SLOTS,
                expected_gpu1_expert_bytes=EXPECTED_GPU1_EXPERT_BYTES,
            )
        else:
            # Both baseline arms use the B1 contract. The primary baseline drifts ->
            # STOP; the supplementary arm's findings invalidate it without stopping
            # the session (it gates nothing), and its pinned --num-tokens legitimately
            # moves the auto cache plan, so the slot band is not applied to it.
            strict = arm.id == BASELINE_ARM_ID
            record = validate_baseline_runtime(
                self.validity, runtime_config, arm_id=arm.id, strict=strict
            )
            if not strict:
                record["supplementary"] = True
                record["note"] = (
                    "supplementary KV-matched arm: the B1 identity contract applies "
                    "except the auto-slot band, which the pinned --num-tokens "
                    "legitimately moves"
                )
        self.runtime_records[arm.id] = {
            "arm_id": arm.id,
            "runtime_config": runtime_config,
            "gpu_verification": gpu_verification,
            "validation": record,
        }
        self.writer.write_runtime(arm.id, self.runtime_records[arm.id])
        if (
            arm.id == BASELINE_ARM_ID
            and record.get("identity_findings")
            and record.get("strict", True)
        ):
            raise BaselineIdentityError(
                "baseline B1 identity drift: "
                + "; ".join(record["identity_findings"])
                + ". The Phase-0 baseline must be refreshed before candidate "
                "performance; this campaign does not substitute another arm."
            )

    # ---- one block -------------------------------------------------------------------

    def _run_block(
        self,
        arm: CampaignArm,
        class_id: str,
        block_steps: Sequence[PlannedGeneration],
        by_class: Mapping[str, Any],
        origin: str,
        model_id: str,
    ) -> None:
        tally = self.tallies[(arm.id, class_id)]
        workload = by_class[class_id]
        body = workload.request_body(model_id)
        warmups = [s for s in block_steps if not s.measured]
        measured = [s for s in block_steps if s.measured]
        for step in warmups:
            self._generate(arm, step, workload, origin, body)
        if not measured:
            return
        # The measurement window is the measured repetitions: counters and timing are
        # reset at an engine-owned idle boundary after the discarded warmups (warmup
        # residency is preserved), and snapshotted after the last measured repetition.
        reset_snapshot = moe_instrumentation(origin, "reset")
        for step in measured:
            self._generate(arm, step, workload, origin, body)
        window = moe_instrumentation(origin, "snapshot")
        self.writer.write_block_mechanism(
            arm.id,
            class_id,
            {
                "arm_id": arm.id,
                "class_id": class_id,
                "block_id": tally.block_id,
                "window_note": (
                    "reset after the discarded warmups, snapshot after the last "
                    "measured repetition; counters cover the measured window only"
                ),
                "reset_boundary": reset_snapshot.get("boundary"),
                "remote_decode": window.get("inferswarm_remote_decode"),
                "resident_bank": window.get("inferswarm_resident_bank"),
                "moe_layer_timing": window.get("moe_layer_timing"),
            },
        )
        if arm.id == CANDIDATE_ARM_ID:
            validate_candidate_block_snapshot(
                self.validity,
                window.get("inferswarm_remote_decode") or {},
                arm_id=arm.id,
                class_id=class_id,
                block_id=tally.block_id,
            )

    def _generate(
        self,
        arm: CampaignArm,
        step: PlannedGeneration,
        workload: Any,
        origin: str,
        body: Mapping[str, Any],
    ) -> None:
        tally = self.tallies[(arm.id, step.class_id)]
        floor = prefill_seq_floor(origin)
        started = time.time()
        record = {
            "session_id": step.session_id,
            "session_number": step.session_number,
            "execution_index": step.execution_index,
            "arm_id": arm.id,
            "arm_role": arm.role,
            "class_id": step.class_id,
            "phase": step.phase,
            "measured": step.measured,
            "repetition": step.repetition,
            "block_id": step.block_id,
            "started_at_unix": started,
        }
        try:
            metrics = measure_generation(
                origin,
                dict(body),
                prefill_seq_floor=floor,
                store_text=self.definition.settings.store_output_text,
            )
        except (GenerationError, OSError, ValueError) as e:
            record.update({"failed": True, "error": repr(e)})
            self._write_generation(arm.id, record)
            tally.failures.append(
                {"execution_index": step.execution_index, "error": repr(e)}
            )
            self.validity.add(
                GENERATION_FAILED,
                f"{arm.id}/{step.class_id} {step.phase} {step.repetition}: {e!r}",
                arm_id=arm.id,
                class_id=step.class_id,
                execution_index=step.execution_index,
            )
            print(
                f"[phase1] {arm.id}/{step.class_id} {step.phase} {step.repetition} FAILED: {e!r}",
                flush=True,
            )
            return

        deviation = check_prompt_tokens(step.class_id, metrics["prompt_tokens"])
        length_deviation = check_completion_tokens(
            step.class_id, metrics.get("completion_tokens"), body.get("max_tokens")
        )
        record.update(
            {
                "failed": False,
                "finished_at_unix": time.time(),
                "prompt_sha256": workload.content_sha256,
                "sampling": {
                    k: body[k] for k in ("temperature", "top_p", "top_k") if k in body
                },
                "ignore_eos": workload.ignore_eos,
                "batch_size": 1,
                "prompt_token_deviation": deviation,
                "completion_length_deviation": length_deviation,
                **metrics,
            }
        )
        self._write_generation(arm.id, record)
        if step.measured:
            tally.observed_measured += 1
        else:
            tally.observed_warmups += 1
        print(
            f"[phase1] {arm.id}/{step.class_id} {step.phase} {step.repetition}: "
            f"decode {metrics['decode_tok_s']:.2f} tok/s, TTFT {metrics['ttft_ms']:.1f} ms",
            flush=True,
        )

    def _write_generation(self, arm_id: str, record: Mapping[str, Any]) -> None:
        self.writer.write_generation(arm_id, record)
        self._recorded_indices.add(int(record["execution_index"]))

    # ---- idle / between arms ----------------------------------------------------------

    def _observe_before_arm(self, arm_id: str) -> dict[str, Any]:
        thermal = thermal_observation()
        thermal.update({"arm_id": arm_id, "position": "before_arm"})
        self.thermal_records.append(thermal)
        observed = {
            uuid: gpu_memory_used_bytes(uuid) for uuid in (GPU0_UUID, GPU1_UUID)
        }
        record: dict[str, Any] = {
            "position": "before_arm",
            "arm_id": arm_id,
            "observed_used_bytes": observed,
            "bound_bytes": GPU_IDLE_MEMORY_MAX_BYTES,
        }
        breached = {
            uuid: used
            for uuid, used in observed.items()
            if used is not None and used > GPU_IDLE_MEMORY_MAX_BYTES
        }
        record["breached"] = breached
        record["refuse_start"] = False
        if breached:
            reason = (
                f"GPU memory did not return to the idle range before arm {arm_id}: "
                f"{breached}. A stale server from the previous arm would make this "
                "arm incomparable; the arm is not started."
            )
            record["reason"] = reason
            self.validity.add(
                GPU_IDLE_NOT_RESTORED, reason, arm_id=arm_id
            )
            if self.canonical:
                record["refuse_start"] = True
        self.idle_records.append(record)
        return record

    def _observe_after_arm(self, arm_id: str) -> dict[str, Any]:
        time.sleep(BETWEEN_ARM_SETTLE_SECONDS)
        observed = {
            uuid: gpu_memory_used_bytes(uuid) for uuid in (GPU0_UUID, GPU1_UUID)
        }
        return {
            "position": "after_arm",
            "arm_id": arm_id,
            "observed_used_bytes": observed,
            "bound_bytes": GPU_IDLE_MEMORY_MAX_BYTES,
        }

    # ---- summaries --------------------------------------------------------------------

    def _write_arm_summary(self, arm_id: str) -> None:
        arm_reps = [
            r for r in self.writer.generations() if r.get("arm_id") == arm_id
        ]
        blocks = [t for t in self.tallies.values() if t.arm_id == arm_id]
        doc = {
            "arm_id": arm_id,
            "arm_role": self.arms_by_id[arm_id].role,
            "runner_version": CAMPAIGN_RUNNER_VERSION,
            "blocks": [t.record() for t in blocks],
            "statistics": {
                t.class_id: summarize_block(
                    [r for r in arm_reps if r.get("class_id") == t.class_id]
                )
                for t in blocks
            },
        }
        if arm_id == BASELINE_ARM_ID:
            doc["noise_floor_status"] = baseline_noise_floor(blocks, arm_reps)
        self.writer.write_arm_summary(arm_id, doc)

    def _finalize(self) -> dict[str, Any]:
        reps = self.writer.generations()
        tallies = list(self.tallies.values())
        failure_count = sum(1 for r in reps if r.get("failed"))
        status = execution_status(tallies, failure_count)

        held: dict[str, Any] = {}
        for field_path in (
            "runtime.memory_ratio",
            "runtime.page_size",
            "runtime.max_running_req",
        ):
            values = {
                arm_id: _dig(rec.get("runtime_config") or {}, field_path)
                for arm_id, rec in self.runtime_records.items()
            }
            held[field_path] = {
                "per_arm": values,
                "equal": len(set(map(str, values.values()))) <= 1,
            }

        baseline_kv = self.kv_capacities.get(BASELINE_ARM_ID)
        candidate_kv = self.kv_capacities.get(CANDIDATE_ARM_ID)
        doc = {
            "session_id": f"session-{self.session_number}",
            "session_number": self.session_number,
            "runner_version": CAMPAIGN_RUNNER_VERSION,
            "execution_status": status,
            **self.validity.record(),
            "execution_order": {
                "arm_order": list(dict.fromkeys(s.arm_id for s in self.steps)),
                "primary_arm_order": list(session_arm_order(self.session_number)),
                "class_order": list(self.definition.protocol.classes),
                "all_block_identities": [t.block_id for t in tallies],
                "execution_indices_recorded": sorted(
                    r.get("execution_index") for r in reps
                ),
            },
            "completion": {
                "expected_primary_generations": (
                    PER_SESSION_PRIMARY_GENERATIONS
                    if self.definition.protocol.canonical
                    else None
                ),
                "observed_generations": len(reps),
                "failed_generations": failure_count,
                "incomplete_blocks": [t.record() for t in tallies if not t.complete],
            },
            "provenance_consistency": {
                "manifest_sha256": self.manifest.manifest_sha256,
                "manifest_canonical": (
                    self.manifest.manifest_sha256 == CANONICAL_MANIFEST_SHA256
                ),
                "placement_sha256": self.placement["sha256"] if self.placement else None,
                "placement_canonical": (
                    self.placement is None
                    or self.placement["sha256"] == CANONICAL_PLACEMENT_SHA256
                ),
                "runner_version": CAMPAIGN_RUNNER_VERSION,
            },
            "held_constant_validation": held,
            "baseline_noise_floor_status": baseline_noise_floor(
                [t for t in tallies if t.arm_id == BASELINE_ARM_ID], reps
            ),
            "supplementary_arm_requirement": supplementary_requirement(
                baseline_kv, candidate_kv
            ),
            "kv_capacities_tokens": self.kv_capacities,
            "thermal_records": self.thermal_records,
            "idle_records": self.idle_records,
            "stopped_early_reason": self.stopped_early_reason,
            "startup_records": self.startup_records,
            "runtime_validation": {
                arm_id: rec.get("validation")
                for arm_id, rec in self.runtime_records.items()
            },
            "no_verdict_note": (
                "this session summary contains per-arm descriptive statistics only; "
                "no cross-arm ratio and no campaign verdict is computed by this runner"
            ),
        }
        self.writer.write_session_summary(doc)
        doc["artifact_sha256"] = self.writer.artifact_sha256_index()
        doc["run_directory"] = str(self.root)
        return doc


def _dig(doc: Mapping[str, Any], dotted: str) -> Any:
    node: Any = doc
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _model_id(origin: str) -> str:
    return get_json(f"{origin}/v1/models")["data"][0]["id"]
