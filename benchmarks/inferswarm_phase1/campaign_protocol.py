"""The Phase-1 campaign protocol: two counterbalanced arm-major sessions.

Frozen by the InferSwarm campaign-order amendment (before any candidate performance
existed), replacing the physically impossible repetition-level A/B/A/B interleaving:

* Session 1: fresh thermal-reset state, ``baseline_b1`` then ``candidate_v2``;
* Session 2: independent thermal-reset state, arm order reversed;
* within every arm/server process the class order is ``W1 -> W2 -> W3 -> W4`` in BOTH
  sessions (workload order is never reversed);
* per class: 2 discarded warmups + 10 measured generations;
* one fresh server process per arm per session; no restart between classes; no radix
  cache clearing between classes.

Per arm: 4 x (2 + 10) = 48 generations. Per session: 96 primary generations. Two
sessions: 192 primary generations. Supplementary arms are counted separately.

The plan is built whole before the first server starts, contains every expected
generation, and cannot be shortened dynamically.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .campaign_arms import BASELINE_ARM_ID, CANDIDATE_ARM_ID

CANONICAL_WARMUPS = 2
CANONICAL_REPETITIONS = 10
CANONICAL_CLASSES: tuple[str, ...] = ("W1", "W2", "W3", "W4")

SESSION_1_ID = "session-1"
SESSION_2_ID = "session-2"
SESSION_1_ARM_ORDER: tuple[str, str] = (BASELINE_ARM_ID, CANDIDATE_ARM_ID)
SESSION_2_ARM_ORDER: tuple[str, str] = (CANDIDATE_ARM_ID, BASELINE_ARM_ID)

PER_ARM_PRIMARY_GENERATIONS = len(CANONICAL_CLASSES) * (CANONICAL_WARMUPS + CANONICAL_REPETITIONS)
PER_SESSION_PRIMARY_GENERATIONS = 2 * PER_ARM_PRIMARY_GENERATIONS
CAMPAIGN_PRIMARY_GENERATIONS = 2 * PER_SESSION_PRIMARY_GENERATIONS


class ProtocolError(ValueError):
    """A protocol that cannot be a canonical Phase-1 campaign."""


@dataclass(frozen=True)
class CampaignProtocol:
    """The precommitted repetition/warmup/class contract for the whole campaign."""

    warmups: int = CANONICAL_WARMUPS
    repetitions: int = CANONICAL_REPETITIONS
    classes: tuple[str, ...] = CANONICAL_CLASSES
    # Every way this campaign departs from the canonical protocol, human-readable.
    # Empty means canonical. Built when overrides are parsed, never inferred later.
    deviations: tuple[str, ...] = ()

    @property
    def canonical(self) -> bool:
        return not self.deviations

    def record(self) -> dict[str, Any]:
        return {
            "canonical": self.canonical,
            "warmups_per_class": self.warmups,
            "measured_repetitions_per_class": self.repetitions,
            "classes": list(self.classes),
            "canonical_warmups": CANONICAL_WARMUPS,
            "canonical_repetitions": CANONICAL_REPETITIONS,
            "canonical_classes": list(CANONICAL_CLASSES),
            "deviations": list(self.deviations),
            "counts": {
                "per_arm_generations": len(self.classes) * (self.warmups + self.repetitions),
                "per_session_primary_generations": PER_SESSION_PRIMARY_GENERATIONS,
                "campaign_primary_generations": CAMPAIGN_PRIMARY_GENERATIONS,
            },
            "rules": [
                "two counterbalanced arm-major sessions (campaign-order amendment)",
                "session 2 reverses the ARM order only; W1->W2->W3->W4 in both sessions",
                "one fresh server process per arm per session; no restart between classes",
                "no radix cache clearing between classes",
                "every measured repetition is preserved; no repetition is discarded",
                "no cross-arm ratio and no campaign verdict is computed by this runner",
            ],
        }


def build_protocol(
    *,
    warmups: int | None,
    repetitions: int | None,
    classes: Sequence[str] | None,
    dev_smoke: bool,
) -> CampaignProtocol:
    """Resolve overrides into a protocol, recording each deviation.

    Overrides are refused outside ``--dev-smoke``: silently accepting
    ``--repetitions 2`` on a canonical run is exactly how a campaign stops being the
    frozen experiment.
    """
    deviations: list[str] = []
    resolved_warmups = CANONICAL_WARMUPS if warmups is None else int(warmups)
    resolved_reps = CANONICAL_REPETITIONS if repetitions is None else int(repetitions)
    resolved_classes = tuple(CANONICAL_CLASSES if classes is None else classes)
    overridden = (
        resolved_warmups != CANONICAL_WARMUPS
        or resolved_reps != CANONICAL_REPETITIONS
        or resolved_classes != CANONICAL_CLASSES
    )
    if overridden and not dev_smoke:
        raise ProtocolError(
            "--warmups/--repetitions/--classes change the precommitted protocol "
            "(campaign-order amendment; 2 warmups + 10 measured over W1-W4). Pass "
            "--dev-smoke to run a clearly-marked NON_CANONICAL_DEV_SMOKE campaign."
        )
    if dev_smoke:
        deviations.append(
            "--dev-smoke: developer smoke test; every artifact is non-canonical"
        )
    if resolved_warmups != CANONICAL_WARMUPS:
        deviations.append(f"warmups={resolved_warmups} (canonical {CANONICAL_WARMUPS})")
    if resolved_reps != CANONICAL_REPETITIONS:
        deviations.append(
            f"measured repetitions={resolved_reps} (canonical {CANONICAL_REPETITIONS})"
        )
    if resolved_classes != CANONICAL_CLASSES:
        deviations.append(
            f"classes={list(resolved_classes)} (canonical {list(CANONICAL_CLASSES)})"
        )
    if resolved_warmups < 0 or resolved_reps < 1:
        raise ProtocolError("warmups must be >= 0 and repetitions >= 1")
    if not resolved_classes:
        raise ProtocolError("at least one workload class is required")
    return CampaignProtocol(
        warmups=resolved_warmups,
        repetitions=resolved_reps,
        classes=resolved_classes,
        deviations=tuple(deviations),
    )


@dataclass(frozen=True)
class PlannedGeneration:
    """One generation the campaign will perform, in exact campaign execution order."""

    session_id: str
    session_number: int
    execution_index: int  # monotonic within the session; the recorded order
    arm_id: str
    arm_role: str
    class_id: str
    phase: str  # "warmup" | "measured"
    repetition: int  # 0-based within its phase and block
    measured: bool
    block_id: str

    def record(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_number": self.session_number,
            "execution_index": self.execution_index,
            "arm_id": self.arm_id,
            "arm_role": self.arm_role,
            "class_id": self.class_id,
            "phase": self.phase,
            "repetition": self.repetition,
            "measured": self.measured,
            "block_id": self.block_id,
        }


def session_arm_order(session_number: int) -> tuple[str, str]:
    """The counterbalanced arm order for a session. Session 2 reverses session 1."""
    if session_number == 1:
        return SESSION_1_ARM_ORDER
    if session_number == 2:
        return SESSION_2_ARM_ORDER
    raise ProtocolError(
        f"the canonical campaign has exactly two sessions; got session {session_number!r}"
    )


def build_session_plan(
    *,
    session_number: int,
    arm_order: Sequence[str],
    arms_by_id: dict[str, Any],
    protocol: CampaignProtocol,
) -> list[PlannedGeneration]:
    """The full session as an explicit ordered list, built before anything runs.

    Arm-major (one server process per arm, all of that arm's classes inside it),
    classes always in protocol order — the class order is never reversed, not even in
    session 2: reversal applies only to the arm order.
    """
    steps: list[PlannedGeneration] = []
    index = 0
    for arm_id in arm_order:
        arm = arms_by_id[arm_id]
        for class_id in protocol.classes:
            block_id = f"session-{session_number}/{arm_id}/{class_id}/block-1"
            for i in range(protocol.warmups):
                steps.append(
                    PlannedGeneration(
                        session_id=f"session-{session_number}",
                        session_number=session_number,
                        execution_index=index,
                        arm_id=arm_id,
                        arm_role=arm.role,
                        class_id=class_id,
                        phase="warmup",
                        repetition=i,
                        measured=False,
                        block_id=block_id,
                    )
                )
                index += 1
            for i in range(protocol.repetitions):
                steps.append(
                    PlannedGeneration(
                        session_id=f"session-{session_number}",
                        session_number=session_number,
                        execution_index=index,
                        arm_id=arm_id,
                        arm_role=arm.role,
                        class_id=class_id,
                        phase="measured",
                        repetition=i,
                        measured=True,
                        block_id=block_id,
                    )
                )
                index += 1
    return steps


def iter_blocks(steps: Sequence[PlannedGeneration]):
    """Group a session plan into (arm_id, class_id, block) preserving order."""
    current: tuple[str, str] | None = None
    bucket: list[PlannedGeneration] = []
    for step in steps:
        key = (step.arm_id, step.class_id)
        if key != current:
            if current is not None:
                yield current[0], current[1], bucket
            current, bucket = key, []
        bucket.append(step)
    if current is not None:
        yield current[0], current[1], bucket


def rerun_block_id(existing_block_ids: Sequence[str], arm_id: str, class_id: str, session_id: str) -> str:
    """A NEW block identity for a rerun; the original block is never overwritten."""
    n = sum(
        1
        for bid in existing_block_ids
        if bid.startswith(f"{session_id}/{arm_id}/{class_id}/block-")
    )
    return f"{session_id}/{arm_id}/{class_id}/block-{n + 1}"
