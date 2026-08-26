"""Warmup / repetition sequencing, and the canonical-vs-smoke distinction.

Expectations are criteria section 10's precommitted protocol: 2 discarded warmups and 10
measured generations per (configuration, class, session), recorded execution order, no
early stopping, no discarded repetition.
"""

from __future__ import annotations

import pytest

from inferswarm_phase0.protocol import (
    CANONICAL_REPETITIONS,
    CANONICAL_WARMUPS,
    BlockTally,
    build_protocol,
    iter_blocks,
    plan,
)


def _protocol(**kw):
    base = dict(
        warmups=None, repetitions=None, session_id="session-1",
        reverse_order=False, dev_smoke=False,
    )
    base.update(kw)
    return build_protocol(**base)


def test_the_canonical_protocol_is_two_warmups_and_ten_measured():
    p = _protocol()
    assert (p.warmups, p.repetitions) == (2, 10)
    assert (CANONICAL_WARMUPS, CANONICAL_REPETITIONS) == (2, 10)
    assert p.canonical and p.deviations == ()


def test_protocol_overrides_are_refused_outside_dev_smoke():
    with pytest.raises(ValueError, match="--dev-smoke"):
        _protocol(repetitions=3)
    with pytest.raises(ValueError, match="--dev-smoke"):
        _protocol(warmups=0)


def test_dev_smoke_records_every_deviation():
    p = _protocol(dev_smoke=True, warmups=1, repetitions=2)
    assert not p.canonical
    joined = " ".join(p.deviations)
    assert "developer smoke test" in joined
    assert "warmups=1" in joined and "repetitions=2" in joined
    assert p.record()["canonical"] is False


def test_dev_smoke_alone_is_still_non_canonical():
    """Even at canonical counts, an explicitly-declared smoke test must not masquerade as
    a baseline."""
    p = _protocol(dev_smoke=True)
    assert (p.warmups, p.repetitions) == (2, 10)
    assert not p.canonical


def test_plan_sequences_warmups_before_measured_in_every_block():
    steps = plan(_protocol(), ["B1", "B2"], ["W1", "W2"])
    assert len(steps) == 2 * 2 * (2 + 10)
    for _arm, _cls, block in iter_blocks(steps):
        phases = [s.phase for s in block]
        assert phases == ["warmup"] * 2 + ["measured"] * 10
        assert [s.repetition for s in block if s.measured] == list(range(10))


def test_execution_order_is_recorded_and_gapless():
    steps = plan(_protocol(), ["B1", "B2", "B3"], ["W1", "W2"])
    assert [s.execution_index for s in steps] == list(range(len(steps)))


def test_reverse_order_gives_session_two_the_opposite_traversal():
    forward = plan(_protocol(), ["B1", "B2", "B3"], ["W1", "W2"])
    reverse = plan(_protocol(reverse_order=True), ["B1", "B2", "B3"], ["W1", "W2"])
    assert [s.arm_id for s in forward][:12] == ["B1"] * 12
    assert [s.arm_id for s in reverse][:12] == ["B3"] * 12
    assert forward[0].class_id == "W1" and reverse[0].class_id == "W2"
    # same total work, different order -- nothing is dropped by reversing
    assert len(forward) == len(reverse)


def test_sessions_are_distinguishable():
    a, b = _protocol(session_id="s1"), _protocol(session_id="thermal-day-2")
    assert a.record()["session_id"] != b.record()["session_id"]


def test_protocol_record_states_the_no_early_stopping_rule():
    rules = " ".join(_protocol().record()["rules"])
    assert "no ratio is computed" in rules
    assert "CANONICAL_PERFORMANCE_BASELINE is selected by a human" in rules


def test_a_block_is_complete_only_when_every_repetition_landed():
    tally = BlockTally("B1", "W2", expected_measured=10)
    for _ in range(9):
        tally.observed_measured += 1
    assert not tally.complete
    tally.observed_measured += 1
    assert tally.complete
    tally.failures.append({"execution_index": 3, "error": "boom"})
    assert not tally.complete, "a failure must keep the block incomplete even at full count"


def test_iter_blocks_preserves_order_and_grouping():
    steps = plan(_protocol(warmups=0, repetitions=1, dev_smoke=True), ["B1", "B2"], ["W1", "W2"])
    keys = [(arm, cls) for arm, cls, _ in iter_blocks(steps)]
    assert keys == [("B1", "W1"), ("B1", "W2"), ("B2", "W1"), ("B2", "W2")]
