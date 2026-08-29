from __future__ import annotations

from copy import deepcopy

import pytest
from inferswarm_phase1.p4_overlap import _timing_value
from inferswarm_phase1.p4_workload_smoke import _score_c3, evaluate_candidate_snapshot


def _snapshot():
    gates = {
        name: {"passed": True, "numerator": 1, "denominator": 2}
        for name in ("F1", "F2", "F3", "F5", "F6")
    }
    return {
        "inferswarm_remote_decode": {
            "gates": gates,
            "aggregate": {"prefill_remote_dispatches": 0},
            "steady_state_transfer_bytes": {"host_to_gpu1": {"expert_weights": 0}},
        }
    }


def test_workload_smoke_preserves_gate_arithmetic_and_zero_weight_boundary():
    summary = evaluate_candidate_snapshot(_snapshot())
    assert set(summary["gates"]) == {"F1", "F2", "F3", "F5", "F6"}
    assert summary["remote_prefill_dispatches"] == 0
    assert summary["steady_state_expert_weight_bytes_host_to_gpu1"] == 0


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value["gates"]["F3"].update(passed=False), "F3"),
        (
            lambda value: value["aggregate"].update(prefill_remote_dispatches=1),
            "remote_prefill_zero",
        ),
        (
            lambda value: value["steady_state_transfer_bytes"]["host_to_gpu1"].update(
                expert_weights=10
            ),
            "steady_state_remote_expert_weights_zero",
        ),
    ],
)
def test_workload_smoke_records_mechanism_failures(mutate, match):
    snapshot = _snapshot()
    remote = snapshot["inferswarm_remote_decode"]
    mutate(remote)
    summary = evaluate_candidate_snapshot(snapshot)
    assert summary["passed"] is False
    assert match in summary["failures"]


def test_overlap_fixture_reads_only_valid_independent_complete_wall():
    record = {
        "durations": {
            "complete_layer": {
                "status": "valid",
                "value_ms": 4.0,
                "source": "cuda_globaltimer_marker_gpu0",
            },
            "gpu0_branch": {
                "complete_local_branch": {
                    "status": "valid",
                    "value_ms": 3.0,
                    "source": "cuda_globaltimer_marker_gpu0",
                }
            },
        }
    }
    assert _timing_value(record, "complete_layer") == 4.0
    assert _timing_value(record, "gpu0_branch", "complete_local_branch") == 3.0
    invalid = deepcopy(record)
    invalid["durations"]["complete_layer"]["status"] = "unavailable"
    with pytest.raises(RuntimeError, match="complete_layer"):
        _timing_value(invalid, "complete_layer")


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]


def test_serving_text_roundtrip_never_claims_exact_c3():
    score = _score_c3(_Tokenizer(), "a" * 64, "a" * 64)
    assert score["text_reencoding_diagnostic"]["first64_reencoded_equal"] is True
    assert score["evaluated"] is False
    assert score["passed"] is None
    assert score["first64_equal"] is None
