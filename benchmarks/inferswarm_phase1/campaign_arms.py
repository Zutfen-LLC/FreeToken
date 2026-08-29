"""Canonical Phase-1 campaign arms: exact, machine-readable server definitions.

Two primary arms and one optional supplementary arm, each a frozen flag set from the
landed Phase-1 candidate configuration (InferSwarm issues #4/#5 and the campaign-order
amendment). Every arm is data: the runner builds the exact ``ft serve`` command from it,
compares the arms mechanically for held constants and intended differences, and refuses
to start a canonical session when any undeclared difference survives.

Arm A ``baseline_b1`` is the already-frozen Phase-0 ``CANONICAL_PERFORMANCE_BASELINE``
identity (B1) observed on the current campaign build. It is not a new baseline
selection. Session 1's B1 runtime resolution is the campaign-build baseline
identity gate: it must pass before the first candidate measurement anywhere in
the campaign. Session 2 revalidates the same identity when its counterbalanced
B1 arm runs; if it then drifts, session 2 is INVALID, its candidate
measurements are retained as invalid evidence and are not eligible for the
Phase-1 analysis, the baseline must be refreshed, and the complete affected
campaign is rerun (see ``campaign_validity``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --- frozen identities ------------------------------------------------------------------

GPU0_UUID = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
GPU1_UUID = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"

CANONICAL_PLACEMENT_SHA256 = (
    "2f62bb84df40d4cc5649e940a39cb53d2975eadecbc320fb97d2b037d4e005f4"
)
CANONICAL_PLACEMENT_POLICY = "phase1-qwen36-placement-v2"
CANONICAL_PLACEMENT_NAME = "coverage_constrained_complement_5442"

EXPECTED_GPU1_SLOTS = 5442
EXPECTED_GPU1_EXPERT_BYTES = 9_662_902_272

# The candidate explicitly pins --num-tokens, and its runtime contract requires
# the RESOLVED KV capacity to equal this value. That makes the conditional
# supplementary arm fully specified before any performance exists: it is B1 plus
# exactly this --num-tokens.
CANDIDATE_PINNED_KV_TOKENS = 17075

# The fixed, predeclared trigger for the conditional supplementary arm. It is
# evaluated once both primary runtime reports exist; no performance number
# controls the branch.
KV_RULE_CONDITION = "candidate_resolved_kv_capacity != baseline_resolved_kv_capacity"
REQUIRED_BY_KV_RULE = "REQUIRED_BY_KV_RULE"
NOT_REQUIRED_BY_KV_RULE = "NOT_REQUIRED_BY_KV_RULE"
KV_RULE_UNRESOLVED = "UNRESOLVED"

# Phase-0's recorded B1 auto resolution. The resolved state is authoritative; these are
# the frozen expectations a material deviation from is a preflight failure, never
# something the runner fixes.
EXPECTED_BASELINE_NVFP4 = "triton"
# The exact slot count Phase-0 recorded for B1's auto expert cache. PROVENANCE ONLY:
# the canonical methodology (campaign-order amendment) requires --moe-cache-auto and
# records the exact resolved slot count, but fixes NO numeric validity band on it --
# a hidden threshold living only in runner code is exactly what the amendment forbids.
# KV-capacity consequences are owned by the predeclared supplementary-KV rule.
PHASE0_RECORDED_BASELINE_CACHE_SLOTS = 3774

# The Phase-0 baseline was measured on this FreeToken commit (criteria section 2.2
# record). P5/P6 remeasures the same B1 identity on the campaign build instead of
# dividing by numbers from another commit/day; the historical commit is recorded in
# provenance for that statement.
PHASE0_BASELINE_COMMIT = "2c3da952e47391bf392e0ece8ae4c67acbc91762"

# Flags a canonical performance arm may never carry. The C3 full-logit recorder does
# device-to-host copies and is explicitly performance-incompatible; the serialized
# remote mode is a diagnostic control, not the canonical candidate.
FORBIDDEN_EVERYWHERE: tuple[str, ...] = (
    "--inferswarm-correctness-diagnostics",
    "--inferswarm-remote-mode",
)
FORBIDDEN_BASELINE_ONLY: tuple[str, ...] = (
    "--inferswarm-secondary-gpu",
    "--inferswarm-placement",
    "--inferswarm-remote-decode",
)

BASELINE_ARM_ID = "baseline_b1"
CANDIDATE_ARM_ID = "candidate_v2"
KV_MATCHED_ARM_ID = "baseline_b1_kv_matched"


class ArmDefinitionError(ValueError):
    """An arm definition that cannot be a canonical campaign arm."""


@dataclass(frozen=True)
class CampaignArm:
    """One campaign arm as an exact, ordered ``ft serve`` flag set.

    ``config_flags`` contains everything after ``--model/--host/--port``. The order is
    stable so the same arm always produces a byte-identical command line: the recorded
    command is a reproduction recipe, not a description.
    """

    id: str
    role: str  # "primary" | "supplementary"
    description: str
    gpu0: str
    config_flags: tuple[str, ...]
    notes: str = ""
    # The supplementary KV-matched arm exists to separate additional KV capacity from
    # remote-expert effects; it never replaces a primary arm anywhere.
    supplementary_reason: str | None = None
    # When set, the arm is a CONDITIONAL arm: it executes only when the named
    # condition (evaluated from the primary arms' resolved runtime reports, before
    # any of its own measurements) is true. Primary arms are never conditional.
    execution_condition: str | None = None

    def flags(self, placement_path: str | None = None) -> list[str]:
        """The arm's flags; ``<placement-path>`` stays as its placeholder until the
        validated path is supplied (dry-run output keeps the placeholder visible)."""
        out: list[str] = []
        for flag in self.config_flags:
            if flag == "<placement-path>":
                out.append(placement_path if placement_path else "<placement-path>")
            else:
                out.append(flag)
        return out

    def record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "description": self.description,
            "gpu0": self.gpu0,
            "config_flags": list(self.config_flags),
            "notes": self.notes,
            "supplementary_reason": self.supplementary_reason,
            "execution_condition": self.execution_condition,
        }


# --- the two primary arms ----------------------------------------------------------------

def baseline_b1_arm() -> CampaignArm:
    """Arm A: the frozen B1 identity, single GPU0, no InferSwarm treatment.

    ``--nvfp4-backend auto`` is deliberate: B1 is the hardware-selected backend, and
    what ``auto`` resolves to is recorded from the running engine. ``--cuda-graph-max-bs
    1`` is passed explicitly even though it is the default, so the arm is reproducible
    from the command line alone and the graph state is declared rather than inherited.
    """
    return CampaignArm(
        id=BASELINE_ARM_ID,
        role="primary",
        description="frozen B1 identity on the campaign build: offload + auto NVFP4 + auto expert cache, single GPU0",
        gpu0=GPU0_UUID,
        config_flags=(
            "--gpu", GPU0_UUID,
            "--moe-backend", "offload",
            "--moe-cache-auto",
            "--nvfp4-backend", "auto",
            "--kv-reserve-tokens", "17075",
            "--memory-ratio", "0.85",
            "--max-running-requests", "1",
            "--cuda-graph-max-bs", "1",
            "--sampling-defaults", "none",
            "--moe-layer-timing-role", "baseline",
        ),
        notes=(
            "Phase-0 CANONICAL_PERFORMANCE_BASELINE = B1, remeasured on the exact "
            "campaign build per the campaign-order amendment. No secondary GPU, no "
            "placement, no remote decode, no InferSwarm remote mode; CUDA graphs stay "
            "enabled. The resolved auto values (NVFP4 backend, cache slots, CPU layers, "
            "decode target) are read back off the live engine before warmups."
        ),
    )


def candidate_v2_arm() -> CampaignArm:
    """Arm B: the exact landed Phase-1 candidate.

    The placement path is the one host-local value in the arm; it is supplied by the
    campaign (validated against the frozen SHA-256 before anything starts) and appears
    as ``<placement-path>`` in the recorded template.
    """
    return CampaignArm(
        id=CANDIDATE_ARM_ID,
        role="primary",
        description="landed Phase-1 candidate: fixed 3,774-slot GPU0 cache + 5,442-slot resident GPU1 bank, host-staged remote decode, overlap",
        gpu0=GPU0_UUID,
        config_flags=(
            "--gpu", GPU0_UUID,
            "--moe-backend", "offload",
            "--moe-cpu-layers", "0",
            "--nvfp4-backend", "triton",
            "--moe-cache-size", "3774",
            "--kv-reserve-tokens", "17075",
            "--num-tokens", "17075",
            "--memory-ratio", "0.85",
            "--cuda-graph-max-bs", "0",
            "--max-running-requests", "1",
            "--sampling-defaults", "none",
            "--inferswarm-secondary-gpu", GPU1_UUID,
            "--inferswarm-placement", "<placement-path>",
            "--inferswarm-remote-decode",
            "--moe-layer-timing-role", "candidate",
        ),
        notes=(
            "Canonical placement phase1-qwen36-placement-v2 / "
            "coverage_constrained_complement_5442, SHA-256 pinned and verified before "
            "startup. Remote mode is overlap; the serialized mode is a diagnostic "
            "control and is not part of the canonical arm. CUDA graph capture is "
            "disabled for the cross-device path (criteria section 12) and that cost "
            "stays inside the candidate's end-to-end result."
        ),
    )


def kv_matched_arm(
    kv_tokens: int, *, conditional: bool = True
) -> CampaignArm:
    """The supplementary KV-matched arm: B1 at a pinned ``--num-tokens``.

    Same as B1 except ``--num-tokens``. It exists to separate additional KV
    capacity from remote-expert effects when the two primary arms resolve
    different KV capacities, and can never replace B1 in the primary comparison.

    ``conditional=True`` (canonical) predeclares the arm under the fixed KV rule
    with the capacity the canonical candidate pins (17,075 tokens); it executes
    only when the rule fires. ``conditional=False`` is the dev-smoke/testing
    override that forces the arm unconditionally.
    """
    if int(kv_tokens) <= 0:
        raise ArmDefinitionError(
            f"the KV-matched arm needs the candidate's resolved KV capacity in tokens, "
            f"got {kv_tokens!r}; resolve both primary arms first and derive it from "
            "their recorded KV capacities"
        )
    return CampaignArm(
        id=KV_MATCHED_ARM_ID,
        role="supplementary",
        description=(
            "conditional supplementary B1 predeclared at the candidate's pinned "
            "--num-tokens 17075"
            if conditional
            else "supplementary B1 at a pinned --num-tokens (dev-smoke override)"
        ),
        gpu0=GPU0_UUID,
        config_flags=(
            "--gpu", GPU0_UUID,
            "--moe-backend", "offload",
            "--moe-cache-auto",
            "--nvfp4-backend", "auto",
            "--kv-reserve-tokens", "17075",
            "--num-tokens", str(int(kv_tokens)),
            "--memory-ratio", "0.85",
            "--max-running-requests", "1",
            "--cuda-graph-max-bs", "1",
            "--sampling-defaults", "none",
            "--moe-layer-timing-role", "baseline",
        ),
        notes=(
            "identical to baseline_b1 except --num-tokens; never a primary arm"
        ),
        supplementary_reason=(
            f"anti-starvation contract (criteria section 3 rule 2): required exactly "
            f"when {KV_RULE_CONDITION}; predeclared before execution with the "
            f"capacity pinned to the candidate's --num-tokens {int(kv_tokens)}"
            if conditional
            else "dev-smoke override: forced unconditionally, recorded as a deviation"
        ),
        execution_condition=KV_RULE_CONDITION if conditional else None,
    )


def predeclared_kv_matched_arm() -> CampaignArm:
    """The conditional supplementary arm every canonical campaign predeclares.

    The trigger and the pinned capacity are fixed before execution: the
    canonical candidate requests ``--num-tokens 17075`` and its runtime contract
    requires the resolved KV capacity to equal 17,075 tokens, so this arm is
    fully specified in the plan — the operator never guesses or passes
    ``--num-tokens`` manually.
    """
    return kv_matched_arm(CANDIDATE_PINNED_KV_TOKENS, conditional=True)


def primary_arms() -> tuple[CampaignArm, CampaignArm]:
    return (baseline_b1_arm(), candidate_v2_arm())


# --- flag comparison: held constants and intended differences -----------------------------

# Flag-level differences the frozen methodology declares between the primary arms.
# Every differing flag must map to exactly one named bucket; an unmapped difference is
# an undeclared configuration change and fails validation.
INTENDED_DIFFERENCE_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "inferswarm_remote_execution",
        (
            "--inferswarm-secondary-gpu", "--inferswarm-placement",
            "--inferswarm-remote-decode",
        ),
    ),
    (
        "fixed_candidate_expert_placement_cache",
        (
            "--moe-cache-auto", "--moe-cache-size", "--nvfp4-backend",
            "--moe-cpu-layers",
        ),
    ),
    ("cuda_graph_state", ("--cuda-graph-max-bs",)),
    (
        "candidate_kv_capacity_pin",
        ("--num-tokens",),
    ),
    ("moe_layer_timing_role_label", ("--moe-layer-timing-role",)),
)

# Flags whose VALUES must be identical across the primary arms (held constants stated
# at the command-line level; the manifest supplies the rest, and the resolved report
# supplies the runtime half).
HELD_FLAG_NAMES: tuple[str, ...] = (
    "--gpu", "--moe-backend", "--kv-reserve-tokens", "--memory-ratio",
    "--max-running-requests", "--sampling-defaults",
)


def flags_to_dict(flags: Sequence[str]) -> dict[str, str | bool]:
    """Parse a flag list into ``{flag: value | True}`` (boolean flags are ``True``)."""
    out: dict[str, str | bool] = {}
    i = 0
    while i < len(flags):
        token = flags[i]
        if token.startswith("--"):
            if i + 1 < len(flags) and not flags[i + 1].startswith("--"):
                out[token] = flags[i + 1]
                i += 2
                continue
            out[token] = True
            i += 1
            continue
        raise ArmDefinitionError(
            f"unexpected bare value {token!r} in flag list; every value must follow a flag"
        )
    return out


def intended_difference_map() -> dict[str, tuple[str, ...]]:
    return dict(INTENDED_DIFFERENCE_BUCKETS)


def compare_primary_arms(
    baseline: CampaignArm, candidate: CampaignArm
) -> dict[str, Any]:
    """The machine-readable arm comparison: held_equal / intended_differences.

    Any flag-level difference that is not covered by a declared intended-difference
    bucket is an ``undeclared_differences`` entry, and canonical validation fails on a
    non-empty list. This is the check that makes "the arms differ in exactly the
    declared ways" auditable rather than asserted.
    """
    base = flags_to_dict(baseline.flags())
    cand = flags_to_dict(candidate.flags())
    flag_names = sorted(set(base) | set(cand))

    held_equal: dict[str, Any] = {}
    for name in HELD_FLAG_NAMES:
        in_base, in_cand = name in base, name in cand
        if in_base != in_cand:
            held_equal[name] = {
                "equal": False,
                "reason": f"present on {'candidate' if in_cand else 'baseline'} only",
                "baseline": base.get(name), "candidate": cand.get(name),
            }
        else:
            held_equal[name] = {
                "equal": base[name] == cand[name],
                "baseline": base[name], "candidate": cand[name],
            }

    bucket_of: dict[str, str] = {}
    for bucket, flags in INTENDED_DIFFERENCE_BUCKETS:
        for flag in flags:
            if flag in bucket_of:
                raise ArmDefinitionError(
                    f"flag {flag} is declared in two intended-difference buckets "
                    f"({bucket_of[flag]} and {bucket}); each difference has exactly one name"
                )
            bucket_of[flag] = bucket

    intended: dict[str, Any] = {bucket: [] for bucket, _ in INTENDED_DIFFERENCE_BUCKETS}
    undeclared: list[dict[str, Any]] = []
    for name in flag_names:
        if name in HELD_FLAG_NAMES:
            continue
        differs = base.get(name) != cand.get(name)
        if not differs:
            continue
        bucket = bucket_of.get(name)
        entry = {"flag": name, "baseline": base.get(name), "candidate": cand.get(name)}
        if bucket is None:
            undeclared.append({**entry, "reason": "no intended-difference bucket declares this flag"})
        else:
            intended[bucket].append(entry)

    intended = {k: v for k, v in intended.items() if v}
    return {
        "held_equal": held_equal,
        "held_constant_flag_names": list(HELD_FLAG_NAMES),
        "held_equal_all": all(v["equal"] for v in held_equal.values()),
        "intended_differences": intended,
        "intended_difference_bucket_names": [b for b, _ in INTENDED_DIFFERENCE_BUCKETS],
        "undeclared_differences": undeclared,
        "note": (
            "workload manifest, output lengths, sampling, model revision, GPU0 identity "
            "and prefill instrumentation are held by the campaign outside the per-arm "
            "flag list and are validated separately (manifest SHA, request bodies, "
            "resolved runtime report)"
        ),
    }


def check_forbidden_flags(arm: CampaignArm) -> list[str]:
    """Flags this arm may never carry. Returns the violation reasons."""
    reasons: list[str] = []
    flags = arm.flags()
    present = {f for f in flags if f.startswith("--")}
    for flag in FORBIDDEN_EVERYWHERE:
        if flag in present:
            reasons.append(
                f"{arm.id}: {flag} is performance-incompatible or diagnostic-only and is "
                "forbidden in every canonical campaign arm"
            )
    if arm.id in (BASELINE_ARM_ID, KV_MATCHED_ARM_ID):
        for flag in FORBIDDEN_BASELINE_ONLY:
            if flag in present:
                reasons.append(
                    f"{arm.id}: {flag} leaks the InferSwarm treatment into a baseline arm"
                )
    return reasons


def validate_arm_definitions(arms: Sequence[CampaignArm]) -> list[str]:
    """Structural refusals: wrong GPU, missing placement, role violations."""
    reasons: list[str] = []
    primaries = [a for a in arms if a.role == "primary"]
    ids = [a.id for a in arms]
    if sorted(a.id for a in primaries) != sorted(
        [BASELINE_ARM_ID, CANDIDATE_ARM_ID]
    ):
        reasons.append(
            f"a canonical campaign has exactly the two primary arms {BASELINE_ARM_ID} "
            f"and {CANDIDATE_ARM_ID}; observed {[a.id for a in primaries]}"
        )
    if KV_MATCHED_ARM_ID in ids:
        kv = next(a for a in arms if a.id == KV_MATCHED_ARM_ID)
        if kv.role != "supplementary":
            reasons.append(
                f"{KV_MATCHED_ARM_ID} is a supplementary arm and can never be primary"
            )
        if kv.execution_condition == KV_RULE_CONDITION:
            tokens = kv.flags()
            if (
                "--num-tokens" not in tokens
                or tokens[tokens.index("--num-tokens") + 1] != str(CANDIDATE_PINNED_KV_TOKENS)
            ):
                reasons.append(
                    f"{KV_MATCHED_ARM_ID}: the predeclared conditional arm pins "
                    f"--num-tokens {CANDIDATE_PINNED_KV_TOKENS} (the candidate's "
                    "pinned KV capacity); got a different value"
                )
    for arm in arms:
        if arm.gpu0 != GPU0_UUID:
            reasons.append(
                f"{arm.id}: GPU0 must be the frozen physical UUID {GPU0_UUID}; got {arm.gpu0!r}"
            )
        parsed = flags_to_dict(arm.flags())
        if parsed.get("--gpu") != GPU0_UUID:
            reasons.append(
                f"{arm.id}: --gpu must be the frozen physical UUID {GPU0_UUID}; got "
                f"{parsed.get('--gpu')!r}"
            )
        reasons.extend(check_forbidden_flags(arm))
    candidate = next((a for a in arms if a.id == CANDIDATE_ARM_ID), None)
    if candidate is not None:
        flags = candidate.flags()
        if "--inferswarm-placement" not in flags:
            reasons.append(f"{CANDIDATE_ARM_ID}: --inferswarm-placement is required")
        idx = flags.index("--inferswarm-secondary-gpu") + 1 if "--inferswarm-secondary-gpu" in flags else None
        if idx is None or flags[idx] != GPU1_UUID:
            reasons.append(
                f"{CANDIDATE_ARM_ID}: --inferswarm-secondary-gpu must be the frozen "
                f"physical UUID {GPU1_UUID}"
            )
        if "--inferswarm-remote-decode" not in flags:
            reasons.append(f"{CANDIDATE_ARM_ID}: --inferswarm-remote-decode is required")
    return reasons


# --- frozen placement artifact ------------------------------------------------------------

def load_placement_reference(path: str | Path) -> dict[str, Any]:
    """Read and validate the frozen placement artifact before any model starts.

    Fails closed: an unreadable artifact, a SHA-256 disagreement with the frozen
    canonical value, or a budget/geometry disagreement with the frozen contract is a
    preflight refusal, raised as ``ArmDefinitionError``.
    """
    import hashlib

    path = Path(path)
    if not path.is_file():
        raise ArmDefinitionError(f"placement artifact not found: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != CANONICAL_PLACEMENT_SHA256:
        raise ArmDefinitionError(
            f"placement artifact SHA-256 disagreement: expected "
            f"{CANONICAL_PLACEMENT_SHA256}, got {digest}. The canonical placement is "
            "frozen; do not regenerate or edit it."
        )
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ArmDefinitionError(f"placement artifact is not valid UTF-8 JSON: {e}") from e

    def expect(cond: bool, message: str) -> None:
        if not cond:
            raise ArmDefinitionError(f"placement artifact: {message}")

    expect(doc.get("schema") == "inferswarm.phase1.placement/1", "unexpected schema")
    expect(doc.get("policy_id") == CANONICAL_PLACEMENT_POLICY, "unexpected policy_id")
    expect(
        doc.get("canonical_remote_placement") == CANONICAL_PLACEMENT_NAME,
        "unexpected canonical_remote_placement",
    )
    budget = doc.get("budget") or {}
    expect(budget.get("remote_slots") == EXPECTED_GPU1_SLOTS, "budget.remote_slots != 5442")
    expect(
        budget.get("remote_resident_bytes") == EXPECTED_GPU1_EXPERT_BYTES,
        "budget.remote_resident_bytes != 9662902272",
    )
    source = doc.get("source") or {}
    return {
        "sha256": digest,
        "policy_id": doc.get("policy_id"),
        "canonical_remote_placement": doc.get("canonical_remote_placement"),
        "status": doc.get("status"),
        "model_repository": source.get("model_repository"),
        "model_revision": source.get("model_revision"),
        "workload_manifest_sha256": source.get("workload_manifest_sha256"),
        "geometry": doc.get("geometry"),
        "budget": {
            "bytes_per_slot": budget.get("bytes_per_slot"),
            "remote_budget_bytes": budget.get("remote_budget_bytes"),
            "remote_slots": budget.get("remote_slots"),
            "remote_resident_bytes": budget.get("remote_resident_bytes"),
            "gpu0_primary_proxy_slots": budget.get("gpu0_primary_proxy_slots"),
        },
        "source_artifact_shas": {
            "run_json": source.get("run_json_sha256"),
            "exact_routing": source.get("exact_routing_sha256"),
            "cache_pressure": source.get("cache_pressure_sha256"),
        },
    }


def placement_flags_present(flags: Sequence[str]) -> bool:
    return "--inferswarm-placement" in flags


def kv_capacity_tokens(runtime_config: Mapping[str, Any]) -> int | None:
    """Resolved KV capacity in tokens, read off the resolved runtime report."""
    runtime = runtime_config.get("runtime") if isinstance(runtime_config, dict) else None
    if not isinstance(runtime, dict):
        return None
    pages = runtime.get("num_pages")
    page_size = runtime.get("page_size")
    if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
        return pages * page_size
    return None


def supplementary_requirement(
    baseline_kv_tokens: int | None, candidate_kv_tokens: int | None
) -> dict[str, Any]:
    """The mechanical anti-starvation determination from recorded KV capacities.

    The condition itself is fixed before execution (the predeclared KV rule);
    this function only evaluates it against the two primary arms' recorded
    resolved capacities. No performance number enters the determination.
    """
    undecidable = baseline_kv_tokens is None or candidate_kv_tokens is None
    required = (
        None if undecidable else baseline_kv_tokens != candidate_kv_tokens
    )
    return {
        "required": required,
        "decidable": not undecidable,
        "baseline_kv_tokens": baseline_kv_tokens,
        "candidate_kv_tokens": candidate_kv_tokens,
        "arm_id": KV_MATCHED_ARM_ID,
        "condition": KV_RULE_CONDITION,
        "pinned_kv_capacity_tokens": CANDIDATE_PINNED_KV_TOKENS,
        "rule": (
            "the conditional supplementary KV-matched baseline is predeclared in "
            "every canonical plan; it is required exactly when "
            f"{KV_RULE_CONDITION}, evaluated from the primary arms' resolved "
            "runtime reports after both exist; it never replaces "
            f"{BASELINE_ARM_ID} in the primary comparison and never enters the "
            "primary cross-arm comparison"
        ),
        "unavailable_reason": (
            "both primary arms must resolve first; the condition is evaluated "
            "from their recorded KV capacities" if undecidable else None
        ),
    }
