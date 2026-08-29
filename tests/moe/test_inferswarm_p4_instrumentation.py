from __future__ import annotations

from copy import deepcopy

from freetoken.moe.inferswarm_remote_decode import (
    LayerCounters,
    TransferByteCounters,
    evaluate_mechanism_gates,
)


def _counters(**changes):
    values = LayerCounters().__dict__.copy()
    values.update(changes)
    values.update(failure_events=0, prefill_remote_dispatches=0)
    return values


def _evaluate(counters, transfer=None, *, gpu0=300, gpu1=100):
    return evaluate_mechanism_gates(
        gpu0_expert_cache_bytes=gpu0,
        gpu1_expert_cache_bytes=gpu1,
        counters=counters,
        transfer_bytes=transfer or TransferByteCounters().as_dict(),
    )


def test_f1_uses_actual_bank_bytes_once_and_threshold_edge_is_inclusive():
    gates = _evaluate(_counters(), gpu0=300, gpu1=100)
    assert gates["F1"]["combined_expert_cache_bytes"] == 400
    assert gates["F1"]["gpu1_expert_byte_fraction"] == 0.25
    assert gates["F1"]["passed"] is True
    # A prefill view alias is intentionally not an input to the arithmetic.
    assert _evaluate(_counters(), gpu0=301, gpu1=99)["F1"]["passed"] is False


def test_f2_is_exact_reset_window_ratio_and_threshold_edge_is_inclusive():
    at_edge = _evaluate(_counters(total_router_selections=10, executed_on_gpu1=2))["F2"]
    assert at_edge["gpu1_execution_fraction"] == 0.2
    assert at_edge["passed"] is True
    below = _evaluate(_counters(total_router_selections=11, executed_on_gpu1=2))["F2"]
    assert below["passed"] is False
    assert "reset-delimited" in below["scope"]


def test_f3_per_layer_mismatch_cannot_be_hidden_by_aggregate_cancellation():
    exact = _evaluate(_counters(expected_remote_dispatches=2, remote_dispatches=2))[
        "F3"
    ]
    assert exact["passed"] is True
    cancelled_aggregate = _evaluate(
        _counters(
            expected_remote_dispatches=2,
            remote_dispatches=2,
            dispatch_mismatch_layer_calls=2,
        )
    )["F3"]
    assert cancelled_aggregate["expected_dispatches"] == 2
    assert cancelled_aggregate["actual_dispatches"] == 2
    assert cancelled_aggregate["passed"] is False


def test_f5_uses_complete_h2g_payload_unique_identity_denominator_and_strict_edge():
    transfer = TransferByteCounters().as_dict()
    transfer["host_to_gpu1"].update(
        activation=60, routing_weights=20, routing_ids=20, expert_weights=0
    )
    gates = _evaluate(
        _counters(hypothetical_streamed_remote_weight_bytes=20_000), transfer
    )["F5"]
    assert gates["steady_state_host_to_gpu1_bytes"] == 100
    assert gates["steady_state_expert_weight_bytes_host_to_gpu1"] == 0
    assert gates["ratio"] == 0.005
    assert gates["passed"] is True

    # Repeated route selections do not enter this input: the producer accumulates only
    # unique remote identities per layer/step. At exactly 1%, the strict gate fails.
    strict = _evaluate(
        _counters(hypothetical_streamed_remote_weight_bytes=10_000), transfer
    )["F5"]
    assert strict["ratio"] == 0.01
    assert strict["passed"] is False

    startup = deepcopy(transfer)
    startup["host_to_gpu1"]["expert_weights"] = 1
    assert (
        _evaluate(_counters(hypothetical_streamed_remote_weight_bytes=20_000), startup)[
            "F5"
        ]["steady_state_expert_weight_bytes_host_to_gpu1"]
        == 1
    )


def test_f6_keeps_mismatch_failure_and_fallback_distinct():
    valid = _evaluate(_counters(selected_for_gpu1=4, executed_on_gpu1=4))["F6"]
    assert valid["passed"] is True
    for changes in (
        {"selected_for_gpu1": 4, "executed_on_gpu1": 3},
        {"selected_for_gpu1": 4, "executed_on_gpu1": 4, "explicit_failure": 1},
        {"selected_for_gpu1": 4, "executed_on_gpu1": 4, "fallback_elsewhere": 1},
    ):
        assert _evaluate(_counters(**changes))["F6"]["passed"] is False


def test_f4_is_only_a_correctness_reference_not_a_fake_mechanism_pass():
    assert _evaluate(_counters())["F4"] == {
        "status": "evaluated_by_correctness",
        "passed": None,
    }
