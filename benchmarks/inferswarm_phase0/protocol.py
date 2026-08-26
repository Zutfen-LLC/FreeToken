"""The precommitted repetition / warmup protocol (criteria section 10).

The contract, restated so the code and the document cannot drift apart:

* server fully ready (``/health`` maintenance == "serving") before any measurement;
* **2 discarded warmup generations** per (configuration, workload class);
* **10 measured generations** per (configuration, workload class, session);
* every measured repetition is preserved -- no selective outlier deletion, ever;
* execution order is recorded;
* sessions are distinct identifiers so a campaign can be repeated on another day and
  thermal state;
* no early stopping: the campaign completes before any ratio is computed. This harness
  computes no ratios at all.

Deviations exist only for developer smoke tests, are opt-in, and are stamped into the run
artifact and the summary so a non-canonical run can never be mistaken for a canonical one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Sequence, Tuple

CANONICAL_WARMUPS = 2
CANONICAL_REPETITIONS = 10


@dataclass(frozen=True)
class Protocol:
    warmups: int = CANONICAL_WARMUPS
    repetitions: int = CANONICAL_REPETITIONS
    session_id: str = "session-1"
    reverse_order: bool = False
    # Every way this run departs from the criteria's protocol, human-readable. Empty means
    # canonical. Never inferred at report time -- built when the overrides are parsed.
    deviations: Tuple[str, ...] = ()

    @property
    def canonical(self) -> bool:
        return not self.deviations

    def record(self) -> Dict[str, Any]:
        return {
            "canonical": self.canonical,
            "warmups_per_block": self.warmups,
            "measured_repetitions_per_block": self.repetitions,
            "session_id": self.session_id,
            "order_reversed": self.reverse_order,
            "deviations": list(self.deviations),
            "canonical_warmups": CANONICAL_WARMUPS,
            "canonical_repetitions": CANONICAL_REPETITIONS,
            "rules": [
                "every measured repetition is preserved; no rep is discarded",
                "no ratio is computed by this harness (criteria section 10, no early stopping)",
                "CANONICAL_PERFORMANCE_BASELINE is selected by a human from the completed "
                "campaign (criteria section 2.2), never by this runner",
            ],
        }


def build_protocol(
    *,
    warmups: int | None,
    repetitions: int | None,
    session_id: str,
    reverse_order: bool,
    dev_smoke: bool,
) -> Protocol:
    """Resolve CLI overrides into a Protocol, recording each deviation.

    Overrides are refused outside ``--dev-smoke``: silently accepting ``--repetitions 2``
    on a canonical run is precisely how a campaign stops being reproducible.
    """
    deviations: List[str] = []
    resolved_warmups = CANONICAL_WARMUPS if warmups is None else int(warmups)
    resolved_reps = CANONICAL_REPETITIONS if repetitions is None else int(repetitions)
    overridden = (
        resolved_warmups != CANONICAL_WARMUPS or resolved_reps != CANONICAL_REPETITIONS
    )
    if overridden and not dev_smoke:
        raise ValueError(
            "--warmups/--repetitions change the precommitted protocol (criteria section 10). "
            "Pass --dev-smoke to run a clearly-marked NON-CANONICAL developer smoke test."
        )
    if dev_smoke:
        deviations.append("--dev-smoke: this run is a developer smoke test, not a baseline")
    if resolved_warmups != CANONICAL_WARMUPS:
        deviations.append(
            f"warmups={resolved_warmups} (canonical {CANONICAL_WARMUPS})"
        )
    if resolved_reps != CANONICAL_REPETITIONS:
        deviations.append(
            f"measured repetitions={resolved_reps} (canonical {CANONICAL_REPETITIONS})"
        )
    if resolved_warmups < 0 or resolved_reps < 1:
        raise ValueError("warmups must be >= 0 and repetitions >= 1")
    return Protocol(
        warmups=resolved_warmups,
        repetitions=resolved_reps,
        session_id=session_id,
        reverse_order=reverse_order,
        deviations=tuple(deviations),
    )


@dataclass(frozen=True)
class Step:
    """One generation the runner will perform, in campaign execution order."""

    execution_index: int   # monotonic across the whole campaign; the recorded order
    arm_id: str
    class_id: str
    phase: str             # "warmup" | "measured"
    repetition: int        # 0-based within its phase and block
    measured: bool


def plan(
    protocol: Protocol, arm_ids: Sequence[str], class_ids: Sequence[str]
) -> List[Step]:
    """The full campaign as an explicit ordered list, built before anything runs.

    Config-major: one ``ft serve`` process per arm, all of that arm's classes inside it.
    (Criteria section 10's A/B interleaving governs the Phase-1 candidate-vs-baseline
    comparison, where both arms are alive at once; the Phase-0 sweep's five arms are five
    different server configurations and cannot be interleaved without a model reload per
    generation. The realized order is recorded either way, and ``--reverse-order`` gives
    session 2 the reversed traversal the section asks for.)
    """
    arms = list(reversed(arm_ids)) if protocol.reverse_order else list(arm_ids)
    classes = list(reversed(class_ids)) if protocol.reverse_order else list(class_ids)
    steps: List[Step] = []
    index = 0
    for arm_id in arms:
        for class_id in classes:
            for i in range(protocol.warmups):
                steps.append(Step(index, arm_id, class_id, "warmup", i, False))
                index += 1
            for i in range(protocol.repetitions):
                steps.append(Step(index, arm_id, class_id, "measured", i, True))
                index += 1
    return steps


def iter_blocks(steps: Sequence[Step]) -> Iterator[Tuple[str, str, List[Step]]]:
    """Group a plan into (arm_id, class_id) blocks, preserving order."""
    current: Tuple[str, str] | None = None
    bucket: List[Step] = []
    for step in steps:
        key = (step.arm_id, step.class_id)
        if key != current:
            if current is not None:
                yield current[0], current[1], bucket
            current, bucket = key, []
        bucket.append(step)
    if current is not None:
        yield current[0], current[1], bucket


@dataclass
class BlockTally:
    """Expected vs observed repetitions for one (arm, class) block.

    The point of this type is that a summary cannot look healthy while hiding a shortfall:
    ``complete`` is computed from the counts, not asserted by the writer.
    """

    arm_id: str
    class_id: str
    expected_measured: int
    observed_measured: int = 0
    observed_warmups: int = 0
    failures: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.observed_measured == self.expected_measured and not self.failures

    def record(self) -> Dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "class_id": self.class_id,
            "expected_measured": self.expected_measured,
            "observed_measured": self.observed_measured,
            "observed_warmups": self.observed_warmups,
            "failures": list(self.failures),
            "complete": self.complete,
        }
