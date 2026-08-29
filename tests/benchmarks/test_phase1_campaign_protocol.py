"""Protocol and plan tests: counts, counterbalanced ordering, and non-overridability."""

from __future__ import annotations

import pytest
from inferswarm_phase1.campaign_arms import (
    BASELINE_ARM_ID,
    CANDIDATE_ARM_ID,
    baseline_b1_arm,
    candidate_v2_arm,
)
from inferswarm_phase1.campaign_protocol import (
    CAMPAIGN_PRIMARY_GENERATIONS,
    CANONICAL_CLASSES,
    CANONICAL_REPETITIONS,
    CANONICAL_WARMUPS,
    PER_ARM_PRIMARY_GENERATIONS,
    PER_SESSION_PRIMARY_GENERATIONS,
    ProtocolError,
    build_protocol,
    build_session_plan,
    iter_blocks,
    rerun_block_id,
    session_arm_order,
)


def _arms():
    b, c = baseline_b1_arm(), candidate_v2_arm()
    return {b.id: b, c.id: c}


def _canonical_protocol():
    return build_protocol(warmups=None, repetitions=None, classes=None, dev_smoke=False)


# --- the frozen constants ----------------------------------------------------------------


def test_canonical_protocol_is_two_warmups_ten_measured_over_w1_w4():
    protocol = _canonical_protocol()
    assert protocol.warmups == CANONICAL_WARMUPS == 2
    assert protocol.repetitions == CANONICAL_REPETITIONS == 10
    assert protocol.classes == CANONICAL_CLASSES == ("W1", "W2", "W3", "W4")
    assert protocol.canonical
    assert protocol.deviations == ()


def test_counts_are_48_per_arm_96_per_session_192_campaign():
    assert PER_ARM_PRIMARY_GENERATIONS == 48
    assert PER_SESSION_PRIMARY_GENERATIONS == 96
    assert CAMPAIGN_PRIMARY_GENERATIONS == 192
    arms = _arms()
    for number in (1, 2):
        steps = build_session_plan(
            session_number=number,
            arm_order=session_arm_order(number),
            arms_by_id=arms,
            protocol=_canonical_protocol(),
        )
        assert len(steps) == 96
        per_arm = {arm: sum(1 for s in steps if s.arm_id == arm) for arm in arms}
        assert per_arm == {BASELINE_ARM_ID: 48, CANDIDATE_ARM_ID: 48}


# --- counterbalanced ordering --------------------------------------------------------------


def test_session_1_runs_baseline_then_candidate():
    assert session_arm_order(1) == ("baseline_b1", CANDIDATE_ARM_ID)


def test_session_2_reverses_the_arm_order_only():
    assert session_arm_order(2) == (CANDIDATE_ARM_ID, "baseline_b1")
    assert session_arm_order(2) == tuple(reversed(session_arm_order(1)))


def test_workload_order_is_w1_w4_in_both_sessions_and_never_reversed():
    arms = _arms()
    protocol = _canonical_protocol()
    for number in (1, 2):
        steps = build_session_plan(
            session_number=number,
            arm_order=session_arm_order(number),
            arms_by_id=arms,
            protocol=protocol,
        )
        for arm_id in arms:
            class_order = list(
                dict.fromkeys(s.class_id for s in steps if s.arm_id == arm_id)
            )
            assert class_order == ["W1", "W2", "W3", "W4"]


def test_within_block_warmups_precede_measured_with_contiguous_indices():
    arms = _arms()
    steps = build_session_plan(
        session_number=1,
        arm_order=session_arm_order(1),
        arms_by_id=arms,
        protocol=_canonical_protocol(),
    )
    assert [s.execution_index for s in steps] == list(range(96))
    for arm_id, class_id, block in iter_blocks(steps):
        phases = [s.phase for s in block]
        assert phases == ["warmup"] * 2 + ["measured"] * 10
        # repetition is 0-based within its phase: warmups 0-1, then measured 0-9
        assert [s.repetition for s in block] == [0, 1] + list(range(10))
        assert all(s.block_id == f"session-1/{arm_id}/{class_id}/block-1" for s in block)
        assert all(s.session_number == 1 and s.session_id == "session-1" for s in block)


def test_session_three_is_refused():
    with pytest.raises(ProtocolError):
        session_arm_order(3)


# --- dev overrides ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warmups": 1, "repetitions": None, "classes": None},
        {"warmups": None, "repetitions": 3, "classes": None},
        {"warmups": None, "repetitions": None, "classes": ["W1", "W2"]},
    ],
)
def test_protocol_overrides_are_rejected_without_dev_smoke(kwargs):
    with pytest.raises(ProtocolError, match="dev-smoke"):
        build_protocol(dev_smoke=False, **kwargs)


def test_dev_smoke_overrides_are_recorded_as_deviations_and_force_noncanonical():
    protocol = build_protocol(
        warmups=1, repetitions=2, classes=["W1"], dev_smoke=True
    )
    assert not protocol.canonical
    assert len(protocol.deviations) == 4  # dev-smoke + warmups + reps + classes
    record = protocol.record()
    assert record["canonical"] is False
    assert "warmups=1 (canonical 2)" in record["deviations"]
    assert "measured repetitions=2 (canonical 10)" in record["deviations"]
    assert "classes=['W1'] (canonical ['W1', 'W2', 'W3', 'W4'])" in record["deviations"]


# --- no early stopping by construction -----------------------------------------------------


def test_the_plan_cannot_be_shortened_dynamically():
    """The plan is a pure function of the protocol: nothing that happened during a
    campaign can change how many generations remain."""
    protocol = _canonical_protocol()
    steps_a = build_session_plan(
        session_number=1,
        arm_order=session_arm_order(1),
        arms_by_id=_arms(),
        protocol=protocol,
    )
    # Re-planning mid-campaign (same inputs) yields the identical full plan.
    steps_b = build_session_plan(
        session_number=1,
        arm_order=session_arm_order(1),
        arms_by_id=_arms(),
        protocol=protocol,
    )
    assert [s.record() for s in steps_a] == [s.record() for s in steps_b]
    assert len(steps_a) == 96


def test_reruns_create_new_block_identities_never_reusing_one():
    existing = [
        "session-1/baseline_b1/W1/block-1",
        "session-1/candidate_v2/W3/block-1",
    ]
    assert (
        rerun_block_id(existing, "baseline_b1", "W1", "session-1")
        == "session-1/baseline_b1/W1/block-2"
    )
    assert (
        rerun_block_id(existing + ["session-1/baseline_b1/W1/block-2"], "baseline_b1", "W1", "session-1")
        == "session-1/baseline_b1/W1/block-3"
    )
