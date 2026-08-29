from __future__ import annotations

import pytest
from inferswarm_phase1.c3_correctness import (
    V2_ARTIFACT_SHA256,
    _generation_state,
    _validate_role_runtime,
    score_c3,
)


def _record(*, tokens=None, logits=None):
    token_ids = list(range(80)) if tokens is None else tokens
    values = [0.0, 1.0, 5.0, 4.0, 3.0, 2.0] if logits is None else logits
    top5 = sorted(range(len(values)), key=lambda index: values[index], reverse=True)[:5]
    return {
        "uid": 7,
        "generated_token_ids": token_ids,
        "generated_token_count": len(token_ids),
        "step0": {
            "available": True,
            "source_dtype": "torch.float32",
            "serialized_dtype": "float32",
            "vocab_size": len(values),
            "argmax": max(range(len(values)), key=lambda index: values[index]),
            "top5_order": top5,
            "full_logits": values,
        },
    }


def test_generation_state_requires_one_exact_reset_delimited_record():
    record = _record(tokens=[2] + list(range(1, 80)))
    snapshot = {
        "inferswarm_correctness_diagnostics": {
            "enabled": True,
            "truncated": False,
            "overflow_requests": 0,
            "records": [record],
        }
    }
    assert _generation_state(snapshot, 80) is record

    snapshot["inferswarm_correctness_diagnostics"]["records"] = []
    with pytest.raises(RuntimeError, match="exactly one request record"):
        _generation_state(snapshot, 80)


def test_c3_exact_first64_and_frozen_step0_tolerances_pass():
    reference = _record(tokens=[2] + list(range(1, 80)))
    candidate = _record(
        tokens=[2] + list(range(1, 80)),
        logits=[0.001, 1.001, 5.001, 4.001, 3.001, 2.001],
    )
    result = score_c3(candidate, reference)
    assert result["passed"] is True
    assert result["token_gate"]["first64_equal"] is True
    assert result["step0"]["argmax_equal"] is True
    assert result["step0"]["top5_order_equal"] is True
    assert result["step0"]["full_logits_within_tolerance"] is True


def test_c3_rejects_token_top5_or_full_logit_failure_without_new_tolerance():
    reference = _record(tokens=[2] + list(range(1, 80)))

    failed_tokens = list(reference["generated_token_ids"])
    failed_tokens[30] = 999
    token_failure = _record(tokens=failed_tokens)
    assert score_c3(token_failure, reference)["passed"] is False

    top5_failure = _record(
        tokens=[2] + list(range(1, 80)), logits=[0.0, 1.0, 4.0, 5.0, 3.0, 2.0]
    )
    assert score_c3(top5_failure, reference)["passed"] is False

    logit_failure = _record(
        tokens=[2] + list(range(1, 80)), logits=[0.0, 1.0, 5.0, 4.0, 3.0, 2.1]
    )
    result = score_c3(logit_failure, reference)
    assert result["passed"] is False
    assert result["step0"]["rtol"] == 2e-3
    assert result["step0"]["atol"] == 2e-3


def test_divergence_after_token64_is_diagnostic_only():
    reference = _record(tokens=[2] + list(range(1, 80)))
    candidate_tokens = list(reference["generated_token_ids"])
    candidate_tokens[70] = 999
    candidate = _record(tokens=candidate_tokens)
    result = score_c3(candidate, reference)
    assert result["passed"] is True
    assert result["token_gate"]["first_divergence_token_index"] == 70
    assert result["token_gate"]["beyond_token_64_divergence_is_diagnostic_only"] is True


def test_candidate_runtime_must_report_exact_v2_and_overlap():
    runtime = {
        "resident_bank": {
            "artifact": {
                "sha256": V2_ARTIFACT_SHA256,
                "policy": "phase1-qwen36-placement-v2",
            }
        },
        "remote_decode": {
            "enabled": True,
            "overlap_active": True,
            "placement_sha256": V2_ARTIFACT_SHA256,
        },
    }
    _validate_role_runtime(runtime, "candidate")
    runtime["remote_decode"]["placement_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="placement SHA disagrees"):
        _validate_role_runtime(runtime, "candidate")
