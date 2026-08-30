"""The P5 campaign runner: plan, validate, and (for P6) execute one session at a time.

Built on the Phase-0 harness modules — manifest, provenance, client, statistics,
validity — not a second benchmark framework. What this module adds is the Phase-1
campaign shape:

* the whole session plan exists before the first server starts;
* provenance preflight fails closed (dirty tree, wrong model/manifest/placement SHA,
  wrong GPUs, correctness prerequisites not bound to the exact current clean
  FreeToken HEAD, missing runner version);
* one fresh server per arm per session, started/stopped by the runner, with startup
  timestamps and an idle-memory check between arms;
* runtime-resolution validation after ``/health`` ready and before warmups — and
  EVERY failure found there stops measurement for that arm before its first warmup:
  no planned generation is measured on a configuration already known not to be the
  declared experiment; the planned generations are preserved as not-executed
  evidence. Session 1's baseline resolution is the campaign-build baseline identity
  gate (a drift STOPs the session before any candidate generation); Session 2 runs the
  candidate first by design and revalidates B1 when its counterbalanced B1 arm
  runs — a drift there invalidates the whole session, its candidate measurements
  are retained as invalid evidence and are not eligible for the Phase-1 analysis,
  and the complete affected campaign must be rerun;
* the whole campaign carries a deterministic campaign-identity fingerprint (one
  SHA-256 over a stable canonical-JSON component set: FreeToken HEAD, InferSwarm
  methodology commit, runner version, model repository + exact revision, workload
  manifest SHA, placement SHA, canonical protocol identity, primary arm
  definitions). Session 1 records it in provenance and its session summary; Session 2
  refuses to start — before any server — unless the session-1 gate record is
  COMPLETE/VALID/passed, comes from the expected artifact set, AND its recorded
  campaign identity equals the current one exactly, with every differing component
  reported by name;
* the conditional supplementary KV-matched baseline is predeclared in every
  canonical plan with its trigger and pinned capacity fixed before execution;
  after both primary runtime reports exist it executes only when the two primary
  arms resolved different KV capacities — no performance number controls the
  branch;
* per-class instrumentation windows (reset after warmups, snapshot after the measured
  repetitions) retaining the mechanism counters and issue-#5 layer timing;
* failures preserved, blocks marked incomplete, no repetition deleted or invisibly
  retried, no dynamic shortening;
* per-arm descriptive statistics after a block completes; never a cross-arm
  comparison, never a campaign verdict.

This package does not run the canonical campaign; P6 does, after the campaign-order
amendment merges.
"""

from __future__ import annotations

import json
import hashlib
import platform
import re
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
    CANDIDATE_PINNED_KV_TOKENS,
    CANONICAL_PLACEMENT_SHA256,
    EXPECTED_GPU1_EXPERT_BYTES,
    EXPECTED_GPU1_SLOTS,
    GPU0_UUID,
    GPU1_UUID,
    KV_MATCHED_ARM_ID,
    KV_RULE_CONDITION,
    KV_RULE_UNRESOLVED,
    NOT_REQUIRED_BY_KV_RULE,
    PHASE0_BASELINE_COMMIT,
    REQUIRED_BY_KV_RULE,
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
    SUPPLEMENTARY_REQUIRED_BLOCK_MISSING,
    BaselineIdentityError,
    CandidateContractError,
    InstrumentationControlError,
    RuntimeContractError,
    SessionValidity,
    SupplementaryContractError,
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

# The frozen Phase-1 campaign instrumentation-control timeout: the server-side
# operation budget sent with every POST /v1/moe/instrumentation reset/snapshot
# request, in seconds. MEASURED on the canonical Session-1 build: the B1/W1
# post-block snapshot of ~204,800 retained complete-layer timing records
# (5,120 decode steps x 40 MoE layers) takes 136.93 s, so the previous frozen
# 60 s budget returned HTTP 504 after a fully completed block. 300 s covers the
# measured snapshot with margin while staying an explicit, auditable constant.
# This is control-plane behavior only: it changes no performance threshold and
# no measured inference interval.
MOE_INSTRUMENTATION_TIMEOUT_SECONDS = 300.0

# The HTTP client waits slightly longer than the server-side operation budget so
# a server-produced timeout response (HTTP 504 with the engine's structured
# status/request id) always wins over a local socket timeout.
MOE_INSTRUMENTATION_HTTP_GRACE_SECONDS = 5.0


def instrumentation_control_record() -> dict[str, Any]:
    """The frozen control-plane budget for the engine-owned instrumentation windows.

    Recorded in the campaign plan, every session plan, and provenance, so the
    reset/snapshot control behavior is auditable. Deliberately distinct from
    ``--server-timeout``: model startup and instrumentation snapshot collection
    are different operations with different recorded budgets, and neither is
    derived from the other.
    """
    return {
        "endpoint": "/v1/moe/instrumentation",
        "operations": ["reset", "snapshot"],
        "operation_timeout_seconds": MOE_INSTRUMENTATION_TIMEOUT_SECONDS,
        "http_client_timeout_seconds": (
            MOE_INSTRUMENTATION_TIMEOUT_SECONDS + MOE_INSTRUMENTATION_HTTP_GRACE_SECONDS
        ),
        "frozen_before_execution": True,
        "cli_override": None,
        "note": (
            "control-plane only: the operation budget sent with every "
            "reset/snapshot request and the HTTP client budget waiting slightly "
            "longer so a server-produced timeout response wins over a local "
            "socket timeout; distinct from --server-timeout (model startup) and "
            "from every measured inference interval; changing it moves no "
            "performance threshold and no timing record"
        ),
    }

_PREREQUISITE_KEYS = (
    "correctness_reference_v2_artifact_sha256",
    "candidate_c3_artifact_sha256",
    "p2_p3_p4_requalification_artifact_sha256",
    "freetoken_runtime_commit",
)
# Declared evidence digests that must be lowercase, normalized 64-hex SHA-256
# values. A nonempty string alone proves nothing.
_PREREQUISITE_SHA_KEYS: tuple[str, ...] = (
    "correctness_reference_v2_artifact_sha256",
    "candidate_c3_artifact_sha256",
    "p2_p3_p4_requalification_artifact_sha256",
)
# Optional companion keys: when a manifest names the artifact path, the runner
# rehashes the bytes and compares; when it does not, the record distinguishes
# "identity syntactically verified" from "bytes independently rehashed".
_PREREQUISITE_PATH_KEY = {k: k[: -len("sha256")] + "path" for k in _PREREQUISITE_SHA_KEYS}

_HEX40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def load_prerequisites(
    path: str | None, *, repo_head: str | None = None
) -> dict[str, Any]:
    """The correctness/mechanism prerequisite manifest, bound to this checkout.

    Records the exact passing correctness-reference-v2 artifact, candidate C3
    artifact, P2/P3/P4 requalification artifact, and the FreeToken runtime
    commit the performance campaign build was qualified with. A canonical
    session refuses to start without it, and refuses on any of these (fail
    closed, before a server starts):

    * every required key present and nonempty;
    * ``freetoken_runtime_commit`` is a valid 40-hex commit SHA, and —
      mandatory — EQUALS the current clean FreeToken HEAD the campaign runs
      from (``repo_head``); correctness qualified on a different build is not
      correctness for this campaign;
    * every declared evidence digest is a lowercase, normalized 64-hex
      SHA-256 value;
    * when the manifest supplies an ``*_artifact_path`` for a declared digest,
      the file is rehashed and must agree; when it does not, the verification
      record distinguishes "identity syntactically verified" from "bytes
      independently rehashed / not available".

    The performance runner itself never enables the C3 full-logit recorder.
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

    commit = str(doc["freetoken_runtime_commit"]).strip()
    if not _HEX40_RE.match(commit):
        raise CampaignRefused(
            f"prerequisites freetoken_runtime_commit {commit!r} is not a valid 40-hex "
            "commit SHA; the campaign build must be named exactly"
        )
    normalized = commit.lower()
    commit_record: dict[str, Any] = {
        "declared": commit,
        "normalized": normalized,
        "valid_40_hex": True,
        "current_head": repo_head.lower() if repo_head else None,
        "equals_current_head": (
            normalized == repo_head.lower() if repo_head else None
        ),
        "rule": (
            "mandatory: the declared runtime commit must equal the current clean "
            "FreeToken HEAD this campaign runs from; correctness qualified on "
            "another commit is not correctness for this campaign"
        ),
    }
    if repo_head and normalized != repo_head.lower():
        raise CampaignRefused(
            f"prerequisites freetoken_runtime_commit {normalized} does not equal the "
            f"current FreeToken HEAD {repo_head.lower()}: the correctness artifacts "
            "were qualified on a different build. Requalify correctness on the exact "
            "campaign checkout (current-commit equality is mandatory)"
        )

    evidence: dict[str, Any] = {}
    for sha_key in _PREREQUISITE_SHA_KEYS:
        declared = str(doc[sha_key]).strip()
        if not _SHA256_RE.match(declared):
            raise CampaignRefused(
                f"prerequisites {sha_key}={declared!r} is not a lowercase normalized "
                "64-hex SHA-256 value; uppercase or malformed digests are refused"
            )
        path_key = _PREREQUISITE_PATH_KEY[sha_key]
        artifact_path = doc.get(path_key)
        entry: dict[str, Any] = {
            "declared": declared,
            "valid_lowercase_sha256": True,
            "artifact_path": artifact_path,
            "bytes_independently_rehashed": False,
            "rehashed_sha256": None,
            "matches_declared": None,
        }
        if artifact_path:
            artifact = Path(artifact_path)
            if not artifact.is_file():
                raise CampaignRefused(
                    f"prerequisites {path_key}={artifact_path!r} does not exist; a "
                    "declared artifact path must be readable so its bytes can be "
                    "rehashed"
                )
            rehashed = hashlib.sha256(artifact.read_bytes()).hexdigest()
            entry.update(
                bytes_independently_rehashed=True,
                rehashed_sha256=rehashed,
                matches_declared=rehashed == declared,
            )
            if rehashed != declared:
                raise CampaignRefused(
                    f"prerequisites {sha_key} disagrees with the bytes at "
                    f"{path_key}={artifact_path!r}: declared {declared}, rehashed "
                    f"{rehashed}. The declared evidence identity is wrong"
                )
            entry["status"] = "identity syntactically verified; bytes independently rehashed and agreeing"
        else:
            entry["status"] = (
                "identity syntactically verified; bytes not independently rehashed "
                "(no artifact path supplied; artifact is external to this manifest)"
            )
        evidence[sha_key] = entry

    return {
        "supplied": True,
        **doc,
        "verification": {
            "freetoken_runtime_commit": commit_record,
            "evidence_sha256": evidence,
        },
    }


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
    head_block = prov.git_commit(prov.freetoken_repo_root())
    head = (
        head_block.get("value")
        if isinstance(head_block, dict) and isinstance(head_block.get("value"), str)
        else None
    )
    return {
        "software": phase1_software_provenance(settings),
        "campaign_identity": campaign_identity(
            definition, manifest, placement, repo_head=head
        ),
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
        "prerequisites": load_prerequisites(
            settings.prerequisites_path, repo_head=head
        ),
        "instrumentation_control": instrumentation_control_record(),
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


# ----------------------------------------------------------------------------------------
# campaign identity
# ----------------------------------------------------------------------------------------

CAMPAIGN_IDENTITY_SCHEMA = "inferswarm.phase1.campaign-identity/1"

# The components whose union IS this campaign, for the session-1 -> session-2 gate.
# Order is irrelevant (the fingerprint hashes canonical JSON); the names are stable
# and appear verbatim in refusal messages when a component differs.
IDENTITY_COMPONENT_KEYS: tuple[str, ...] = (
    "freetoken_head",
    "inferswarm_methodology_commit",
    "campaign_runner_version",
    "model_repository",
    "model_revision",
    "workload_manifest_sha256",
    "placement_sha256",
    "protocol",
    "primary_arms",
)


def _canonical_json(doc: Mapping[str, Any]) -> str:
    """The stable representation hashed into the fingerprint: sorted keys, no
    incidental whitespace, ASCII — byte-identical for equal component values."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _current_repo_head() -> str | None:
    head_block = prov.git_commit(prov.freetoken_repo_root())
    return (
        head_block.get("value")
        if isinstance(head_block, dict) and isinstance(head_block.get("value"), str)
        else None
    )


def campaign_identity(
    definition: CampaignDefinition,
    manifest: Manifest,
    placement: dict[str, Any] | None,
    *,
    repo_head: str | None = None,
    runner_version: str | None = None,
) -> dict[str, Any]:
    """The deterministic campaign identity/fingerprint binding session 1 to session 2.

    One SHA-256 over a stable canonical-JSON representation of every component that
    defines the experiment: the exact FreeToken build, the InferSwarm methodology
    commit, the campaign runner version, the model repository + exact revision, the
    frozen workload-manifest SHA, the frozen placement SHA, the canonical protocol
    identity, and the primary arm definitions. The human-readable component values
    are retained alongside the digest, so a refusal can name exactly what moved.
    """
    settings = definition.settings
    if repo_head is None:
        repo_head = _current_repo_head()
    components: dict[str, Any] = {
        "freetoken_head": (repo_head or "").strip().lower() or None,
        "inferswarm_methodology_commit": (settings.inferswarm_commit or "")
        .strip()
        .lower()
        or None,
        "campaign_runner_version": runner_version or CAMPAIGN_RUNNER_VERSION,
        "model_repository": settings.model_repository,
        "model_revision": (settings.model_revision or "").strip().lower() or None,
        "workload_manifest_sha256": manifest.manifest_sha256,
        "placement_sha256": placement["sha256"] if placement else None,
        "protocol": {
            "classes": list(definition.protocol.classes),
            "warmups": definition.protocol.warmups,
            "repetitions": definition.protocol.repetitions,
            "deviations": list(definition.protocol.deviations),
            "session_arm_orders": [
                list(session_arm_order(1)),
                list(session_arm_order(2)),
            ],
        },
        "primary_arms": {
            arm.id: list(arm.config_flags)
            for arm in sorted(definition.primary_arms(), key=lambda a: a.id)
        },
    }
    return {
        "schema": CAMPAIGN_IDENTITY_SCHEMA,
        "sha256": hashlib.sha256(
            _canonical_json(components).encode("utf-8")
        ).hexdigest(),
        "components": components,
        "rule": (
            "session 1 records this fingerprint in its provenance and session "
            "summary; session 2 refuses to start unless the session-1 gate record "
            "carries exactly this identity — any differing component is reported "
            "by name before any server starts"
        ),
    }


def campaign_identity_differences(
    recorded: Mapping[str, Any] | None, current: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Which identity components differ between the session-1 record and now.

    Compares the human-readable component values with the same canonical-JSON
    normalization the fingerprint hashes, so 'equal' here is exactly 'hashes equal'.
    """
    recorded_components = (
        recorded.get("components")
        if isinstance(recorded, Mapping) and isinstance(recorded.get("components"), Mapping)
        else None
    )
    differences: list[dict[str, Any]] = []
    for key in IDENTITY_COMPONENT_KEYS:
        expected = _canonical_json({key: current.get(key)})
        observed = (
            _canonical_json({key: recorded_components.get(key)})
            if recorded_components is not None and key in recorded_components
            else None
        )
        if observed != expected:
            differences.append(
                {
                    "component": key,
                    "session_1": (
                        recorded_components.get(key)
                        if recorded_components is not None and key in recorded_components
                        else "<absent>"
                    ),
                    "current": current.get(key),
                }
            )
    return differences


# The session-1 artifacts the session-2 gate requires beyond the summary booleans:
# the gate must come from a real session-1 artifact set, not a bare JSON claim.
SESSION_ONE_GATE_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "plan.json",
    "provenance.json",
    f"{BASELINE_ARM_ID}/runtime.json",
)


def session_one_gate_record(
    out_root: Path,
    *,
    current_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read the sibling session-1 record that gates any session-2 start.

    Session 2 runs the candidate first by design, so the only thing that can
    stand between the campaign-build baseline identity gate and the first
    candidate measurement anywhere in the campaign is this precondition:
    session-1 (B1 first) must already exist under the same ``--out-root``,
    its B1 resolution must have passed, its gate must come from the expected
    artifact set (not bare COMPLETE/VALID/passed booleans), and its recorded
    campaign identity must equal the current campaign identity exactly. Fail
    closed on anything unreadable or unequal; name the differing component(s).
    """
    summary_path = Path(out_root) / "session-1" / "session-summary.json"
    record: dict[str, Any] = {
        "required": True,
        "path": str(summary_path),
        "present": False,
        "ok": False,
        "reason": None,
    }
    if not summary_path.is_file():
        record["reason"] = (
            "session-1 session-summary.json not found: Session 1 runs B1 first and "
            "its campaign-build baseline identity gate must pass before any candidate "
            "measurement anywhere in the campaign — including before Session 2, "
            "whose first arm is the candidate"
        )
        return record
    record["present"] = True
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        record["reason"] = f"session-1 session-summary.json is unreadable: {e}"
        return record
    gate = summary.get("baseline_identity_gate") or {}
    record["execution_status"] = summary.get("execution_status")
    record["validity"] = summary.get("validity")
    record["baseline_identity_gate_passed"] = gate.get("passed")
    session_dir = summary_path.parent

    # The gate must come from the expected session-1 artifact set: the files must
    # exist, be listed in the summary's own SHA-256 index, and hash to the recorded
    # values. Booleans alone prove nothing about where this record came from.
    artifact_index = summary.get("artifact_sha256")
    artifact_index = artifact_index if isinstance(artifact_index, dict) else {}
    artifact_findings: list[str] = []
    for relative in SESSION_ONE_GATE_REQUIRED_ARTIFACTS:
        artifact = session_dir / relative
        if not artifact.is_file():
            artifact_findings.append(f"{relative} is missing")
            continue
        recorded_sha = artifact_index.get(relative)
        if not isinstance(recorded_sha, str) or not _SHA256_RE.match(recorded_sha):
            artifact_findings.append(
                f"{relative} is not listed in session-1's artifact SHA-256 index"
            )
            continue
        actual_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual_sha != recorded_sha:
            artifact_findings.append(
                f"{relative} hashes to {actual_sha}, not the recorded {recorded_sha}"
            )
    record["artifact_set"] = {
        "required": list(SESSION_ONE_GATE_REQUIRED_ARTIFACTS),
        "verified": not artifact_findings,
        "findings": artifact_findings,
    }

    # The campaign identity: session 1's fingerprint must equal the current one
    # exactly. A COMPLETE/VALID/passed record measured under any other campaign
    # (older FreeToken HEAD, different model revision, manifest, placement,
    # runner version, protocol, or arm definitions) is not this campaign's gate.
    recorded_identity = summary.get("campaign_identity")
    identity_block: dict[str, Any] = {"required": current_identity is not None}
    if current_identity is not None:
        current_components = current_identity.get("components") or {}
        if not isinstance(recorded_identity, Mapping) or not (
            isinstance(recorded_identity.get("sha256"), str)
            and _SHA256_RE.match(recorded_identity["sha256"])
        ):
            identity_block.update(
                {
                    "equal": False,
                    "recorded_sha256": None,
                    "expected_sha256": current_identity.get("sha256"),
                    "differences": [
                        {
                            "component": "campaign_identity",
                            "session_1": "<absent>",
                            "current": current_identity.get("sha256"),
                        }
                    ],
                }
            )
        else:
            differences = campaign_identity_differences(
                recorded_identity, current_components
            )
            equal = recorded_identity["sha256"] == current_identity.get("sha256")
            identity_block.update(
                {
                    "equal": equal and not differences,
                    "recorded_sha256": recorded_identity["sha256"],
                    "expected_sha256": current_identity.get("sha256"),
                    "differences": differences,
                }
            )
        record["identity"] = identity_block

    if summary.get("execution_status") != "COMPLETE":
        record["reason"] = (
            "session-1 is not COMPLETE; a complete session-1 whose B1 campaign-build "
            "identity gate passed must exist before session 2 starts"
        )
    elif summary.get("validity") != "VALID":
        record["reason"] = (
            "session-1 is not VALID; session 2 may not collect candidate "
            "measurements on top of an invalid session-1"
        )
    elif gate.get("passed") is not True:
        record["reason"] = (
            "session-1's campaign-build baseline identity gate did not pass; the "
            "Phase-0 baseline must be refreshed and the campaign rerun before any "
            "candidate measurement"
        )
    elif artifact_findings:
        record["reason"] = (
            "session-1's gate does not come from the expected artifact set: "
            + "; ".join(artifact_findings)
        )
    elif current_identity is not None and not identity_block.get("equal"):
        differences = identity_block.get("differences") or []
        names = ", ".join(
            str(d.get("component")) for d in differences
        ) or "the campaign identity digest"
        record["reason"] = (
            f"session-1 was measured under a different campaign identity ({names}); "
            "session 2 must run on exactly the campaign session 1 measured. Rerun "
            "session 1 on the current campaign before any candidate-first "
            "measurement starts"
        )
    else:
        record["ok"] = True
    return record


def preflight_refusals(
    definition: CampaignDefinition,
    manifest: Manifest,
    placement: dict[str, Any] | None,
    *,
    session_number: int | None = None,
) -> list[str]:
    """Everything a canonical campaign would refuse to start on, cheaply.

    Runs in ``validate`` and again before a session executes; nothing measures until
    this list is empty (a dev-smoke campaign returns no refusals and is stamped
    non-canonical everywhere). ``session_number`` adds the session-scoped gates:
    session 2 cannot start unless session-1's campaign-build baseline identity
    gate already passed.
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
    commit_block = prov.git_commit(prov.freetoken_repo_root())
    dirty = prov.check_clean_working_tree(commit_block)
    if dirty:
        reasons.append(dirty)
    head = (
        commit_block.get("value")
        if isinstance(commit_block, dict) and isinstance(commit_block.get("value"), str)
        else None
    )
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
        prerequisites = load_prerequisites(
            settings.prerequisites_path, repo_head=head
        )
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
    if settings.prerequisites_path and head is None:
        reasons.append(
            "the current FreeToken HEAD could not be resolved, so the prerequisite "
            "freetoken_runtime_commit cannot be bound to this checkout; "
            "current-commit equality is mandatory"
        )

    # Every canonical campaign predeclares the conditional supplementary arm with
    # its trigger and pinned capacity fixed before execution; the plan may not
    # claim to be fully specified while a supplementary arm could surprise it.
    predeclared = [
        a
        for a in definition.arms
        if a.id == KV_MATCHED_ARM_ID and a.execution_condition == KV_RULE_CONDITION
    ]
    if not predeclared:
        reasons.append(
            f"a canonical campaign predeclares the conditional supplementary arm "
            f"{KV_MATCHED_ARM_ID} (trigger {KV_RULE_CONDITION}; pinned capacity "
            f"{CANDIDATE_PINNED_KV_TOKENS} tokens) in the plan before execution; "
            "add it with predeclared_kv_matched_arm()"
        )

    if session_number is not None and session_number >= 2:
        gate = session_one_gate_record(
            settings.out_root,
            current_identity=campaign_identity(
                definition, manifest, placement, repo_head=head
            ),
        )
        if not gate["ok"]:
            reasons.append(
                f"session {session_number} cannot start: {gate['reason']} "
                f"(looked for {gate['path']})"
            )

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

    The predeclared supplementary arm is planned in this position — after both
    primaries — and executes only when its fixed condition (the KV rule,
    evaluated from the two primary arms' recorded resolved KV capacities once
    both runtime reports exist) resolves true. It is clearly labelled, counted
    separately, and never replaces a primary arm; no performance number
    controls the branch.
    """
    order = list(session_arm_order(session_number))
    order.extend(
        a.id for a in definition.arms if a.role == "supplementary"
    )
    return order


# ----------------------------------------------------------------------------------------
# planning / dry run
# ----------------------------------------------------------------------------------------


def conditional_supplementary_declaration(
    definition: CampaignDefinition, protocol: CampaignProtocol
) -> list[dict[str, Any]]:
    """The pre-execution declaration of every conditional supplementary arm.

    Everything about the arm is fixed here — definition, exact flags, trigger,
    pinned capacity, possible generation count, position, and its non-gating
    status — so the plan is fully specified before any performance exists and
    no runtime observation can reveal a mandatory-but-unplanned arm.
    """
    declarations: list[dict[str, Any]] = []
    possible = len(protocol.classes) * (protocol.warmups + protocol.repetitions)
    for arm in definition.arms:
        if arm.role != "supplementary" or arm.execution_condition is None:
            continue
        declarations.append(
            {
                "arm_id": arm.id,
                "definition": arm.record(),
                "exact_flags": arm.flags(),
                "condition": arm.execution_condition,
                "trigger_fixed_before_execution": True,
                "evaluated_from": (
                    "the two primary arms' resolved runtime reports, after both exist; "
                    "no performance number controls this branch"
                ),
                "pinned_kv_capacity_tokens": CANDIDATE_PINNED_KV_TOKENS,
                "possible_generations_per_session": possible,
                "position": "after both primary arms",
                "status_vocabulary": [
                    REQUIRED_BY_KV_RULE,
                    NOT_REQUIRED_BY_KV_RULE,
                    KV_RULE_UNRESOLVED,
                ],
                "supplementary_status": (
                    "non-gating; never a primary comparator; never enters the "
                    "primary cross-arm comparison"
                ),
            }
        )
    return declarations


def plan_document(
    definition: CampaignDefinition, manifest: Manifest | None
) -> dict[str, Any]:
    """The whole two-session campaign as an explicit document, before any server.

    Every expected generation of both sessions appears with session/arm/class,
    warmup/measured phase, repetition and execution index. Conditional
    supplementary generations appear too, tagged ``conditional``: the arm is
    fully specified before execution and its generations execute only when the
    fixed condition resolves true. There is no dynamic shortening: the executed
    session must match this plan exactly.
    """
    arms_by_id = definition.arms_by_id()
    supplementary_ids = [
        a.id for a in definition.arms if a.role == "supplementary"
    ]
    conditional = conditional_supplementary_declaration(
        definition, definition.protocol
    )
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
                "supplementary_arm_ids": list(supplementary_ids),
                "conditional_supplementary_arms": [dict(c) for c in conditional],
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
                "conditional_generation_count": sum(
                    1 for s in steps if s.conditional
                ),
            }
        )
    primary_total = sum(s["primary_generation_count"] for s in sessions)
    conditional_total = sum(s["conditional_generation_count"] for s in sessions)
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
        "instrumentation_control": instrumentation_control_record(),
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
        "conditional_supplementary_generation_counts": {
            "per_session": conditional_total // 2 if conditional else 0,
            "possible_only": True,
            "note": (
                "possible generations of the predeclared conditional supplementary "
                "arm; executed only when the fixed KV rule resolves true"
            ),
        },
        "counterbalanced_order": {
            "session-1": list(session_arm_order(1)),
            "session-2": list(session_arm_order(2)),
        },
        "session_order_gates": {
            "session_1_baseline_identity_gate": (
                "session 1 runs B1 first; its runtime resolution is the "
                "campaign-build baseline identity gate and must pass before the "
                "first candidate measurement anywhere in the campaign"
            ),
            "session_2_precondition": (
                "session 2 (candidate first, by design) refuses to start unless a "
                "complete, valid session-1 record with a passed baseline identity "
                "gate exists under --out-root, comes from the expected session-1 "
                "artifact set, and records EXACTLY the current campaign identity "
                "(one SHA-256 over FreeToken HEAD, InferSwarm methodology commit, "
                "runner version, model repository+revision, workload manifest SHA, "
                "placement SHA, canonical protocol identity, and primary arm "
                "definitions); any differing component is reported by name"
            ),
        },
        "supplementary_arm_support": (
            "the conditional supplementary KV-matched baseline is predeclared in "
            "every canonical plan with its trigger and pinned capacity fixed before "
            "execution; it executes only when the primary arms resolve different KV "
            "capacities and never replaces a primary arm"
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
    declaration = conditional_supplementary_declaration(
        definition, definition.protocol
    )
    predeclared_ok = bool(declaration) and all(
        d["pinned_kv_capacity_tokens"] == CANDIDATE_PINNED_KV_TOKENS
        and d["possible_generations_per_session"]
        == len(definition.protocol.classes)
        * (definition.protocol.warmups + definition.protocol.repetitions)
        and "--num-tokens" in d["exact_flags"]
        and d["exact_flags"][d["exact_flags"].index("--num-tokens") + 1]
        == str(CANDIDATE_PINNED_KV_TOKENS)
        for d in declaration
    )

    return {
        "schema": "inferswarm.phase1.campaign-validation/1",
        "campaign_runner_version": CAMPAIGN_RUNNER_VERSION,
        "campaign_identity": campaign_identity(
            definition, manifest, placement
        ),
        "canonical": bool(
            canonical
            and manifest.canonical
            and not refusals
            and count_ok
            and ordering_ok
            and class_orders_ok
            and predeclared_ok
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
        "supplementary_predeclaration": {
            "predeclared": predeclared_ok,
            "conditional_arms": declaration,
            "session_one_gate": (
                "session 1 B1 runtime resolution is the campaign-build baseline "
                "identity gate and must pass before the first candidate measurement "
                "anywhere in the campaign; session 2 (candidate first) refuses to "
                "start without a passing session-1 record under --out-root whose "
                "recorded campaign identity equals the current one exactly"
            ),
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


def _bounded_body_snippet(raw: bytes, limit: int = 200) -> str:
    """A bounded, printable diagnostic of an unreadable response body."""
    text = raw[:limit].decode("utf-8", errors="replace")
    suffix = "..." if len(raw) > limit else ""
    return text + suffix


def _post_json(url: str, body: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
    """POST one control request and decode its response, failing closed.

    The instrumentation endpoint uses HTTP status codes for outcomes (200 ok,
    409 busy, 422 unsupported, 504 timeout, 503 failed). ``urlopen`` raises
    ``HTTPError`` for every non-2xx response, so the expected error statuses are
    decoded here and returned as structured documents (tagged with
    ``http_status``) for the caller to fail closed on; a body that is not a
    JSON object, and any transport-level failure (URL/socket/timeout), become a
    ``ServerError`` rather than a raw urllib exception terminating
    SessionExecution.
    """
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        url,
        data=json.dumps(dict(body)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as e:
        try:
            raw = e.read()
        except OSError:
            raw = b""
        try:
            decoded = json.loads(raw) if raw else None
        except ValueError:
            decoded = None
        if isinstance(decoded, dict):
            return {**decoded, "http_status": e.code}
        raise ServerError(
            f"HTTP {e.code} from POST {url} with a non-JSON response body: "
            f"{_bounded_body_snippet(raw)!r}"
        ) from e
    except (urllib.error.URLError, OSError) as e:
        raise ServerError(
            f"POST {url} failed before a response arrived ({e!r})"
        ) from e
    try:
        decoded = json.loads(raw)
    except ValueError as e:
        raise ServerError(
            f"POST {url} returned a non-JSON body: {_bounded_body_snippet(raw)!r}"
        ) from e
    if not isinstance(decoded, dict):
        raise ServerError(
            f"POST {url} returned a non-object JSON body: {decoded!r}"
        )
    return decoded


def moe_instrumentation(
    origin: str,
    operation: str,
    *,
    timeout: float = MOE_INSTRUMENTATION_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """POST /v1/moe/instrumentation — the engine-owned idle-boundary reset/snapshot.

    The server-side operation budget is the frozen campaign constant (never
    ``--server-timeout``) and the HTTP client waits ``timeout`` plus a small
    grace so a server-produced timeout response wins over a local socket
    timeout. Every non-ok outcome — including the expected HTTP error statuses
    the server answers with — raises ``ServerError`` carrying the operation,
    the HTTP status, the engine status, and the engine error/request id where
    present. No retry, no hiding.
    """
    response = _post_json(
        f"{origin}/v1/moe/instrumentation",
        {"operation": operation, "timeout": timeout},
        timeout=timeout + MOE_INSTRUMENTATION_HTTP_GRACE_SECONDS,
    )
    if response.get("status") != "ok" or "payload" not in response:
        details = "; ".join(
            f"{key}={response[key]!r}"
            for key in ("http_status", "status", "error", "request_id")
            if response.get(key) is not None
        )
        raise ServerError(
            f"MoE instrumentation {operation} failed: {details or response!r}"
        )
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
        # Primary (and any unconditionally-forced supplementary) blocks are
        # expected unconditionally. A conditional supplementary arm's tallies are
        # created only when its condition resolves true, so a NOT_REQUIRED arm can
        # never pollute completeness with blocks that were correctly never executed.
        self.tallies: dict[tuple, BlockTally] = {}
        for arm_id, class_id, block in iter_blocks(self.steps):
            arm = self.arms_by_id[arm_id]
            if arm.role == "supplementary" and arm.execution_condition:
                continue
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
        self.drift_disposition: dict[str, Any] | None = None
        self.supplementary_decision: dict[str, Any] | None = None
        self.baseline_gate: dict[str, Any] = {"checked": False, "passed": False}
        self.campaign_identity: dict[str, Any] | None = None
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
        refusals = preflight_refusals(
            self.definition,
            self.manifest,
            self.placement,
            session_number=self.session_number,
        )
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
                "instrumentation_control": instrumentation_control_record(),
                "no_dynamic_shortening": True,
            }
        )
        provenance = provenance_document(self.definition, self.manifest, self.placement)
        provenance["thermal"] = {"session_boundary": self.boundary_record}
        self.campaign_identity = provenance["campaign_identity"]
        self.writer.write_provenance(provenance)

        by_class = self.manifest.by_class()
        for arm_id in dict.fromkeys(s.arm_id for s in self.steps):
            arm = self.arms_by_id[arm_id]
            # A predeclared conditional supplementary arm executes only when its
            # fixed condition — evaluated from the two primary arms' recorded
            # resolved KV capacities, never from a performance number — is true.
            if arm.role == "supplementary" and arm.execution_condition:
                self.supplementary_decision = decision = self._kv_rule_decision()
                if decision["required"] is not True:
                    # NOT_REQUIRED_BY_KV_RULE or UNRESOLVED: no generations, no
                    # failure records — the disposition lives in the session
                    # summary. An unresolved condition always co-occurs with a
                    # primary-arm invalidation.
                    continue
                self._ensure_conditional_tallies(arm_id)
            if self.stopped_early_reason is not None:
                self._record_arm_not_executed(arm_id, self.stopped_early_reason)
                continue
            idle = self._observe_before_arm(arm_id)
            if idle.get("refuse_start"):
                self._record_arm_not_executed(arm_id, idle["reason"])
                continue
            try:
                self._run_arm(arm, by_class)
            except BaselineIdentityError as e:
                # Session-aware STOP (campaign-order amendment). Session 1: B1 ran
                # first, so the campaign-build baseline identity gate failed before
                # any candidate generation — the remaining arms are recorded as not
                # executed. Session 2: the candidate ran first by design; the
                # revalidation drift makes the ENTIRE session invalid, its candidate
                # measurements are retained as invalid evidence and are not
                # eligible for the Phase-1 analysis, and the complete affected
                # campaign must be rerun.
                self.stopped_early_reason = str(e)
                if self.session_number >= 2:
                    self.drift_disposition = {
                        "session": self.session_number,
                        "finding": str(e),
                        "session_validity": "INVALID",
                        "candidate_measurements": (
                            "retained as invalid evidence; not eligible for any "
                            "Phase-1 analysis, cross-arm comparison, or campaign "
                            "verdict"
                        ),
                        "required_remediation": (
                            "refresh the Phase-0 baseline and rerun the complete "
                            "affected campaign"
                        ),
                        "reuse_policy": (
                            "no candidate data from this session is reused or "
                            "spliced into the rerun"
                        ),
                    }
                continue
            except RuntimeContractError as e:
                # Any other pre-warmup runtime-contract failure (candidate contract
                # mismatch, supplementary-arm contract failure, engine-GPU
                # mismatch): the arm is aborted before its first warmup — no
                # generation of the aborted arm is measured — the session stops,
                # and every remaining planned generation (the aborted arm's
                # included) is preserved as not-executed evidence.
                self.stopped_early_reason = str(e)
                self._record_arm_not_executed(arm_id, str(e))
                continue
        return self._finalize()

    def _kv_rule_decision(self) -> dict[str, Any]:
        """Evaluate the predeclared KV rule from the primary arms' recorded reports."""
        baseline_kv = self.kv_capacities.get(BASELINE_ARM_ID)
        candidate_kv = self.kv_capacities.get(CANDIDATE_ARM_ID)
        requirement = supplementary_requirement(baseline_kv, candidate_kv)
        if requirement["decidable"]:
            required = requirement["required"]
            status = REQUIRED_BY_KV_RULE if required else NOT_REQUIRED_BY_KV_RULE
            reason = (
                "the primary arms resolved different KV capacities; the predeclared "
                "supplementary arm is required"
                if required
                else "the primary arms resolved equal KV capacities; the predeclared "
                "supplementary arm is not required and its generations are not executed"
            )
        else:
            required = None
            status = KV_RULE_UNRESOLVED
            reason = requirement["unavailable_reason"] or (
                "the condition could not be evaluated from the primary arms' "
                "runtime reports"
            )
        return {
            **requirement,
            "required": required,
            "status": status,
            "reason": reason,
            "branch_inputs": {
                "baseline_kv_tokens": baseline_kv,
                "candidate_kv_tokens": candidate_kv,
            },
        }

    def _ensure_conditional_tallies(self, arm_id: str) -> None:
        for aid, class_id, block in iter_blocks(self.steps):
            if aid != arm_id or (aid, class_id) in self.tallies:
                continue
            self.tallies[(aid, class_id)] = BlockTally(
                arm_id=aid,
                class_id=class_id,
                block_id=block[0].block_id,
                expected_warmups=sum(1 for s in block if not s.measured),
                expected_measured=sum(1 for s in block if s.measured),
            )

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
        # The engine must provably run on the declared physical GPU (and, on a
        # canonical run, on Phase-0-proven hardware). A wrong or unproven engine
        # GPU stops this arm before its first warmup: benchmarking the wrong
        # device and labelling it later is exactly what this gate forbids.
        engine_gpu_failed = gpu_verification.get("matches") is not True or (
            self.canonical
            and gpu_verification.get("matches") is True
            and (gpu_verification.get("phase0_hardware") or {}).get("valid") is not True
        )
        engine_gpu_reason = str(
            gpu_verification.get("mismatch")
            or gpu_verification.get("unavailable")
            or (gpu_verification.get("phase0_hardware") or {}).get("message")
            or "the engine GPU identity did not verify"
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
            self.runtime_records[arm.id] = {
                "arm_id": arm.id,
                "runtime_config": runtime_config,
                "gpu_verification": gpu_verification,
                "validation": record,
            }
            self.writer.write_runtime(arm.id, self.runtime_records[arm.id])
            # A resolved-arm mismatch discovered after /health and before warmups
            # must not generate performance observations: the arm is aborted with
            # no candidate generation recorded as a successful measurement.
            findings = list(record.get("contract_findings") or [])
            if engine_gpu_failed:
                findings.append(
                    "the engine did not run on the declared physical GPU: "
                    + engine_gpu_reason
                )
            if record.get("checked") is not True:
                raise CandidateContractError(
                    f"{arm.id}: the candidate runtime report is incomplete; the "
                    "pre-warmup runtime contract could not be validated "
                    f"({record.get('stop_reason')}). The arm is aborted before the "
                    "first warmup; no candidate performance is collected"
                )
            if findings:
                raise CandidateContractError(
                    f"{arm.id} pre-warmup runtime-contract mismatch: "
                    + "; ".join(findings)
                    + ". The arm is aborted before the first warmup; no candidate "
                    "performance is collected and the planned generations are "
                    "preserved as not-executed evidence"
                )
            return

        # Both baseline arms use the B1 contract. The primary baseline failing the
        # contract (identity drift, InferSwarm leakage, or an unprovable report) is
        # the session-aware STOP below; the supplementary arm's failure aborts the
        # arm without stopping the session (it gates nothing).
        strict = arm.id == BASELINE_ARM_ID
        record = validate_baseline_runtime(
            self.validity, runtime_config, arm_id=arm.id, strict=strict
        )
        if not strict:
            record["supplementary"] = True
            record["note"] = (
                "supplementary KV-matched arm: the same B1 identity contract; the "
                "resolved auto cache plan legitimately moves with the pinned "
                "--num-tokens, and no numeric slot band exists to be moved out of"
            )
        self.runtime_records[arm.id] = {
            "arm_id": arm.id,
            "runtime_config": runtime_config,
            "gpu_verification": gpu_verification,
            "validation": record,
        }
        self.writer.write_runtime(arm.id, self.runtime_records[arm.id])
        identity_findings = list(record.get("identity_findings") or [])
        leakage_findings = list(record.get("inferswarm_leakage_findings") or [])
        gpu_findings = (
            ["the engine did not run on the declared physical GPU: " + engine_gpu_reason]
            if engine_gpu_failed
            else []
        )
        if arm.id == BASELINE_ARM_ID:
            checked = bool(record.get("checked"))
            passed = (
                checked
                and not identity_findings
                and not leakage_findings
                and not gpu_findings
            )
            self.baseline_gate = {"checked": checked, "passed": passed}
        if not strict:
            if not record.get("checked") or (
                identity_findings or leakage_findings or gpu_findings
            ):
                raise SupplementaryContractError(
                    f"{arm.id} failed the B1 runtime contract before its first "
                    "warmup: "
                    + "; ".join([*leakage_findings, *identity_findings, *gpu_findings])
                    + ". The supplementary arm is aborted with its planned "
                    "generations preserved as not-executed evidence"
                )
            return
        if identity_findings or leakage_findings or gpu_findings:
            findings = "; ".join(
                [
                    *(f"InferSwarm leakage: {f}" for f in leakage_findings),
                    *identity_findings,
                    *gpu_findings,
                ]
            )
            if self.session_number == 1:
                raise BaselineIdentityError(
                    "baseline B1 identity failure (session-1 campaign-build "
                    "baseline identity gate): "
                    + findings
                    + ". The session stops before any candidate generation; the "
                    "Phase-0 baseline must be refreshed and the campaign rerun; this "
                    "campaign does not substitute another arm."
                )
            raise BaselineIdentityError(
                "baseline B1 identity failure (session-2 revalidation): "
                + findings
                + ". Session 2 is INVALID; the candidate measurements already "
                "collected are retained as invalid evidence and are not eligible "
                "for the Phase-1 analysis; the Phase-0 baseline must be refreshed "
                "and the complete affected campaign rerun; no candidate data is "
                "reused or spliced."
            )

    # ---- one block -------------------------------------------------------------------

    def _instrumentation_boundary(
        self, arm: CampaignArm, class_id: str, origin: str, operation: str
    ) -> dict[str, Any]:
        """One reset/snapshot control request at an engine-owned idle boundary.

        A failed control operation cannot be retried or worked around: the
        measured window's required evidence (mechanism counters and issue #5's
        complete timing population) would not exist. The failure invalidates
        the session and stops it here — every generation already collected is
        preserved unchanged and every remaining planned generation, of this arm
        and of every later arm, is preserved as not-executed evidence.
        """
        try:
            return moe_instrumentation(
                origin, operation, timeout=MOE_INSTRUMENTATION_TIMEOUT_SECONDS
            )
        except ServerError as e:
            reason = (
                f"{arm.id}/{class_id}: the instrumentation {operation} boundary "
                f"failed: {e}"
            )
            self.validity.add(
                SERVER_FAILED, reason, arm_id=arm.id, class_id=class_id
            )
            raise InstrumentationControlError(
                reason
                + ". The session stops here: the remaining planned generations "
                "are preserved as not-executed evidence; no retry, no splicing, "
                "and no performance ratio is computed"
            ) from e

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
        reset_snapshot = self._instrumentation_boundary(arm, class_id, origin, "reset")
        for step in measured:
            self._generate(arm, step, workload, origin, body)
        window = self._instrumentation_boundary(arm, class_id, origin, "snapshot")
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

        decision = self.supplementary_decision
        required = decision.get("required") if decision else None
        possible_supplementary = len(self.definition.protocol.classes) * (
            self.definition.protocol.warmups + self.definition.protocol.repetitions
        )
        required_supplementary_tallies = (
            [t for t in tallies if t.arm_id == KV_MATCHED_ARM_ID]
            if required is True
            else []
        )
        required_block_completed = (
            bool(required_supplementary_tallies)
            and all(t.complete for t in required_supplementary_tallies)
            if required is True
            else None
        )
        # A canonical session can never look COMPLETE/VALID when the predeclared
        # condition resolved true and the required supplementary block is missing.
        if required is True and required_block_completed is not True:
            self.validity.add(
                SUPPLEMENTARY_REQUIRED_BLOCK_MISSING,
                f"the KV rule resolved {REQUIRED_BY_KV_RULE} but the "
                f"{KV_MATCHED_ARM_ID} block is missing or incomplete",
                arm_id=KV_MATCHED_ARM_ID,
            )
        status = execution_status(tallies, failure_count)
        if required is True and required_block_completed is not True:
            status = "INCOMPLETE"

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
        supplementary_record = (
            decision
            if decision is not None
            else supplementary_requirement(baseline_kv, candidate_kv)
        )
        gate_role = (
            "campaign-build baseline identity gate (session 1 runs B1 first)"
            if self.session_number == 1
            else "session-2 revalidation of the campaign-build baseline identity gate"
        )
        gate_consequence = (
            "a drift here stops the session before any candidate generation, "
            "anywhere in the campaign"
            if self.session_number == 1
            else "a drift here makes the entire session INVALID: candidate "
            "measurements already collected are retained as invalid evidence, "
            "excluded from every Phase-1 analysis, and the complete affected "
            "campaign must be rerun with no candidate data reused or spliced"
        )
        doc = {
            "session_id": f"session-{self.session_number}",
            "session_number": self.session_number,
            "runner_version": CAMPAIGN_RUNNER_VERSION,
            "campaign_identity": self.campaign_identity,
            "execution_status": status,
            **self.validity.record(),
            "execution_order": {
                "arm_order": list(dict.fromkeys(s.arm_id for s in self.steps)),
                "executed_arm_order": list(
                    dict.fromkeys(
                        r.get("arm_id") for r in reps if r.get("arm_id")
                    )
                ),
                "primary_arm_order": list(session_arm_order(self.session_number)),
                "class_order": list(self.definition.protocol.classes),
                "all_block_identities": [t.block_id for t in tallies],
                "execution_indices_recorded": sorted(
                    r.get("execution_index") for r in reps
                ),
            },
            "baseline_identity_gate": {
                "role": gate_role,
                "arm_id": BASELINE_ARM_ID,
                **self.baseline_gate,
                "consequence_on_drift": gate_consequence,
            },
            "completion": {
                "expected_primary_generations": (
                    PER_SESSION_PRIMARY_GENERATIONS
                    if self.definition.protocol.canonical
                    else None
                ),
                "conditional_supplementary_generations": (
                    possible_supplementary if required is True else None
                ),
                "supplementary_condition": {
                    "arm_id": KV_MATCHED_ARM_ID,
                    "condition": KV_RULE_CONDITION,
                    "pinned_kv_capacity_tokens": CANDIDATE_PINNED_KV_TOKENS,
                    "resolved": decision.get("decidable") if decision else None,
                    "required": required,
                    "status": decision.get("status") if decision else None,
                    "reason": decision.get("reason") if decision else None,
                    "required_supplementary_block_completed": (
                        required_block_completed
                    ),
                },
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
            "supplementary_arm_requirement": supplementary_record,
            "kv_capacities_tokens": self.kv_capacities,
            "thermal_records": self.thermal_records,
            "idle_records": self.idle_records,
            "stopped_early_reason": self.stopped_early_reason,
            "baseline_drift_disposition": self.drift_disposition,
            "startup_records": self.startup_records,
            "runtime_validation": {
                arm_id: rec.get("validation")
                for arm_id, rec in self.runtime_records.items()
            },
            "no_verdict_note": (
                "this session summary contains per-arm descriptive statistics only; "
                "no cross-arm comparison and no campaign verdict is computed by "
                "this runner"
            ),
        }
        # The SHA-256 index must be INSIDE the on-disk summary: session 2 reads
        # session-summary.json — not this returned document — and verifies the
        # session-1 artifact set against the embedded index. It is computed after
        # every session artifact required by the gate is written but before the
        # summary itself, so it deliberately excludes session-summary.json: a file
        # cannot contain its own hash, and nothing here attempts one.
        doc["artifact_sha256"] = self.writer.artifact_sha256_index()
        self.writer.write_session_summary(doc)
        doc["run_directory"] = str(self.root)
        # Returned only, never embedded: the full-directory index (the summary
        # included), reported under its own name so the embedded gate index keeps
        # its meaning.
        doc["full_directory_sha256"] = self.writer.artifact_sha256_index()
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
