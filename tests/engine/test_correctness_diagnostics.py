from __future__ import annotations

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
