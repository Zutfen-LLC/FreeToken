from __future__ import annotations

import base64
import hashlib

import pytest
import torch
from freetoken.engine.correctness_diagnostics import (
    CorrectnessDiagnostics,
    absent_correctness_diagnostics_report,
)


def test_exact_step0_logits_and_accepted_tokens_are_captured_before_decoding():
    diagnostics = CorrectnessDiagnostics(max_requests=2)
    logits = torch.tensor([0.25, -1.0, 4.0, 3.0, 2.0, 1.0], dtype=torch.float32)
    diagnostics.capture_step0_logits(17, logits)
    diagnostics.record_accepted_token(17, 2)
    diagnostics.record_accepted_token(17, 91)

    # A later decode row for the same request must not replace step 0.
    diagnostics.capture_step0_logits(17, torch.arange(6, dtype=torch.float32))
    record = diagnostics.snapshot()["records"][0]
    assert record["uid"] == 17
    assert record["generated_token_ids"] == [2, 91]
    assert record["step0"]["source_dtype"] == "torch.float32"
    assert record["step0"]["serialized_dtype"] == "float32"
    assert record["step0"]["full_logits"] == logits.tolist()
    assert record["step0"]["argmax"] == 2
    assert record["step0"]["top5_order"] == [2, 3, 4, 5, 0]


def test_capture_clones_replay_buffer_and_reset_clears_only_diagnostics():
    diagnostics = CorrectnessDiagnostics()
    logits = torch.arange(6, dtype=torch.float32)
    diagnostics.capture_step0_logits(1, logits)
    logits.fill_(-100)
    assert diagnostics.snapshot()["records"][0]["step0"]["full_logits"] == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]
    diagnostics.reset()
    snapshot = diagnostics.snapshot()
    assert snapshot["records"] == []
    assert snapshot["truncated"] is False


def test_request_bound_is_loud_and_disabled_report_is_empty():
    diagnostics = CorrectnessDiagnostics(max_requests=1)
    diagnostics.record_accepted_token(1, 10)
    diagnostics.record_accepted_token(2, 20)
    snapshot = diagnostics.snapshot()
    assert snapshot["records_retained"] == 1
    assert snapshot["overflow_requests"] == 1
    assert snapshot["truncated"] is True

    absent = absent_correctness_diagnostics_report()
    assert absent["enabled"] is False
    assert absent["records"] == []
    assert absent["ordinary_sampling_unchanged"] is True
    assert absent["ordinary_sse_unchanged"] is True
    assert absent["moe_root_cause"]["enabled"] is False
    assert absent["moe_root_cause"]["performance_fields_collected"] is False


def test_root_cause_capture_is_step_bounded_lossless_and_stable():
    diagnostics = CorrectnessDiagnostics(
        root_cause_mode="trace",
        root_cause_decode_step=1,
        root_cause_expected_layers=2,
    )
    hidden = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    ids = torch.tensor([[3, 7]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)

    diagnostics.begin_decode_step(0)
    diagnostics.capture_moe_input(0, hidden, ids, weights)
    assert diagnostics.snapshot()["moe_root_cause"]["layers_retained"] == 0

    diagnostics.begin_decode_step(1)
    for layer_id in range(2):
        diagnostics.capture_moe_input(layer_id, hidden + layer_id, ids, weights)
        diagnostics.capture_moe_output(layer_id, hidden + layer_id)
    snapshot = diagnostics.snapshot()["moe_root_cause"]
    assert snapshot["exactly_expected_layers"] is True
    assert snapshot["layer_ids"] == [0, 1]
    assert snapshot["truncated"] is False
    assert snapshot["performance_fields_collected"] is False

    record = snapshot["layers"][0]["tensors"]["hidden_input"]
    raw = base64.b64decode(record["raw_bytes_base64"])
    assert hashlib.sha256(raw).hexdigest() == record["raw_byte_sha256"]
    assert record["dtype"] == "bfloat16"
    assert record["shape"] == [1, 2]
    assert (
        diagnostics.snapshot()["moe_root_cause"]["layers"][0]["tensors"][
            "hidden_input"
        ]["raw_byte_sha256"]
        == record["raw_byte_sha256"]
    )


def test_root_cause_capture_fails_loudly_at_storage_bound():
    diagnostics = CorrectnessDiagnostics(
        root_cause_mode="trace",
        root_cause_expected_layers=1,
        root_cause_max_tensor_bytes=3,
    )
    diagnostics.begin_decode_step(0)
    with pytest.raises(RuntimeError, match="explicit byte bound"):
        diagnostics.capture_moe_input(
            0,
            torch.ones(2, dtype=torch.float32),
            torch.ones(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.float32),
        )
    snapshot = diagnostics.snapshot()["moe_root_cause"]
    assert snapshot["truncated"] is True
    assert snapshot["overflow_records"] == 1


def test_root_cause_default_contract_retains_exactly_40_moe_layers():
    diagnostics = CorrectnessDiagnostics(root_cause_mode="trace")
    diagnostics.begin_decode_step(0)
    for layer_id in range(40):
        hidden = torch.tensor([[layer_id]], dtype=torch.bfloat16)
        diagnostics.capture_moe_input(
            layer_id,
            hidden,
            torch.tensor([[layer_id % 8]], dtype=torch.int32),
            torch.ones((1, 1), dtype=torch.float32),
        )
        diagnostics.capture_moe_output(layer_id, hidden)
    root = diagnostics.snapshot()["moe_root_cause"]
    assert root["expected_moe_layers"] == 40
    assert root["layer_ids"] == list(range(40))
    assert root["layers_retained"] == 40
    assert root["exactly_expected_layers"] is True
    assert root["tensor_bytes_retained"] <= root["max_tensor_bytes"]
