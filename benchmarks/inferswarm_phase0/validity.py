"""The one authoritative campaign-validity gate.

Three questions a Phase-0 artifact must answer separately, because they are not the same
question and conflating them is how an invalid campaign comes to look publishable:

``execution_status``
    Did every planned generation actually return? ``COMPLETE`` / ``INCOMPLETE``. This is
    arithmetic over expected-vs-observed repetition counts and the failure list.

``validity``
    Is this a *valid canonical Phase-0 baseline campaign*? ``VALID`` / ``INVALID`` /
    ``NON_CANONICAL``. A campaign that ran every repetition it planned is still ``INVALID``
    if any precommitted prerequisite, held-constant rule, workload-shape rule,
    instrumentation requirement or provenance requirement failed.

``label``
    The InferSwarm evidence label (``BENCHMARKING.md``). Individual observations that
    genuinely occurred are ``MEASURED`` whatever the campaign verdict is -- a real number
    was really observed. That never promotes the *campaign* to a valid baseline.

An invalidation is a structured record, not a warning string: a stable ``code``, a
human-readable ``message``, and whatever locates it (``arm_id`` / ``class_id`` /
``execution_index``). Codes are stable so a later analysis step can filter on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

EXECUTION_COMPLETE = "COMPLETE"
EXECUTION_INCOMPLETE = "INCOMPLETE"

VALIDITY_VALID = "VALID"
VALIDITY_INVALID = "INVALID"
VALIDITY_NON_CANONICAL = "NON_CANONICAL"

# --- stable reason codes ---------------------------------------------------------------
#
# Grouped by the criteria section that makes each one invalidating. Adding a code is fine;
# changing what an existing one means is not -- a consumer filters on these.

# criteria section 2.1 (B2's fresh `ft bench bw` profile) and 2.3 (the profile used)
BENCH_BW_SKIPPED = "bench_bw.skipped"
BENCH_BW_FAILED = "bench_bw.failed"
BENCH_BW_PROFILE_UNREADABLE = "bench_bw.profile_unreadable"
BENCH_BW_PROFILE_GPU_MISMATCH = "bench_bw.profile_gpu_mismatch"

# criteria section 2.3 (resolved configuration) -- read off the live engine
INSTRUMENTATION_UNAVAILABLE = "runtime.instrumentation_unavailable"
RUNTIME_CONFIG_MISSING = "runtime.config_missing"
RUNTIME_CONFIG_MISSING_FIELD = "runtime.config_missing_field"

# criteria section 2.1 (B3 must coincide with B1 or B2)
B3_RESOLUTION_UNEXPECTED = "sweep.b3_resolution_unexpected"

# criteria section 3 (anti-starvation / held constant across arms)
EXPERT_QUANT_MISMATCH = "held_constant.expert_quant_mismatch"
HELD_CONSTANT_MISMATCH = "held_constant.mismatch"

# criteria section 9 (frozen workload shape) and section 3 rule 5 (exact output length)
PROMPT_SHAPE_VIOLATION = "workload.prompt_shape_violation"
COMPLETION_LENGTH_MISMATCH = "workload.completion_length_mismatch"

# criteria section 6 / the benchmark contract's required prefill throughput
PREFILL_UNAVAILABLE = "prefill.unavailable"
PREFILL_DISABLED = "prefill.instrumentation_disabled"
PREFILL_MISSING = "prefill.no_fresh_record"
PREFILL_AMBIGUOUS = "prefill.ambiguous_records"
PREFILL_SHARED_BATCH = "prefill.shared_batch"
PREFILL_UNUSABLE = "prefill.unusable_timing"

# criteria section 2.1 (one specific physical RTX 3060) and the benchmark contract
GPU_UNPROVEN = "gpu.unproven"
GPU_MISMATCH = "gpu.mismatch"

# criteria section 1.1 (pinned upstream revision) + the benchmark contract's provenance
MODEL_REVISION_MISMATCH = "provenance.model_revision_mismatch"
MODEL_REPOSITORY_MISMATCH = "provenance.model_repository_mismatch"
DIRTY_WORKING_TREE = "provenance.dirty_working_tree"
PROVENANCE_MISSING = "provenance.missing_required"

# execution faults -- they make a campaign incomplete AND invalid
GENERATION_FAILED = "execution.generation_failed"
SERVER_FAILED = "execution.server_failed"


@dataclass(frozen=True)
class Invalidation:
    """One reason this campaign is not a valid canonical Phase-0 baseline."""

    code: str
    message: str
    arm_id: str | None = None
    class_id: str | None = None
    execution_index: int | None = None

    def record(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "arm_id": self.arm_id,
            "class_id": self.class_id,
            "execution_index": self.execution_index,
        }


@dataclass
class CampaignValidity:
    """Collects invalidating conditions and derives the campaign's single verdict.

    ``canonical_intent`` is whether this run *asked* to be canonical. A developer smoke run
    never becomes ``INVALID``: it was never claiming to be a baseline, so its verdict is
    ``NON_CANONICAL`` and its invalidations are recorded as observations. A canonical run
    with even one invalidation is ``INVALID``, whatever its repetition counts say.
    """

    canonical_intent: bool
    invalidations: List[Invalidation] = field(default_factory=list)
    # Reasons the run was non-canonical by construction (--dev-smoke, protocol overrides,
    # a non-canonical manifest, --allow-missing-provenance). Distinct from invalidations:
    # these say "this was never a baseline attempt", not "this attempt failed".
    canonical_blockers: List[str] = field(default_factory=list)

    def add(
        self,
        code: str,
        message: str,
        *,
        arm_id: str | None = None,
        class_id: str | None = None,
        execution_index: int | None = None,
    ) -> None:
        self.invalidations.append(
            Invalidation(code, message, arm_id, class_id, execution_index)
        )

    def extend(self, others: Sequence[Invalidation]) -> None:
        self.invalidations.extend(others)

    @property
    def codes(self) -> List[str]:
        # Stable order, deduplicated: this is read by humans and asserted on by tests.
        return list(dict.fromkeys(i.code for i in self.invalidations))

    def verdict(self) -> str:
        if not self.canonical_intent or self.canonical_blockers:
            return VALIDITY_NON_CANONICAL
        return VALIDITY_INVALID if self.invalidations else VALIDITY_VALID

    def record(self) -> Dict[str, Any]:
        return {
            "validity": self.verdict(),
            "canonical_intent": self.canonical_intent,
            "canonical_blockers": list(self.canonical_blockers),
            "campaign_invalidations": [i.record() for i in self.invalidations],
            "campaign_invalidation_codes": self.codes,
            "validity_note": (
                "execution_status answers 'did every planned generation return?'. validity "
                "answers 'is this a valid canonical Phase-0 baseline campaign?'. They are "
                "independent: a campaign that ran every repetition is still INVALID if any "
                "precommitted prerequisite, held-constant rule, workload-shape rule, "
                "instrumentation requirement or provenance requirement failed."
            ),
        }


def headline(execution_status: str, validity: str) -> str:
    """The one line that goes at the very top of run.json and SUMMARY.md.

    Incompleteness is stated first because it subsumes everything else: a campaign that
    lost repetitions has no verdict to give.
    """
    if execution_status != EXECUTION_COMPLETE:
        return "INCOMPLETE RUN"
    if validity == VALIDITY_VALID:
        return "VALID CANONICAL CAMPAIGN"
    if validity == VALIDITY_INVALID:
        return "INVALID CANONICAL ATTEMPT"
    return "NON-CANONICAL DEVELOPER RUN"
