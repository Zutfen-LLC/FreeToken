"""Per-generation metrics, and the rules that stop a number from being invented.

Specifically: prefill throughput is never derived from TTFT, a prefill record that could
belong to an earlier generation is refused rather than attributed, and a shared prefill
batch yields no per-request rate.
"""

from __future__ import annotations

import pytest

from inferswarm_phase0 import client as client_mod


def _stream_result(stamps, completion=4, prompt=900, text="hello"):
    return {
        "t0": stamps[0] - 0.5,          # 500 ms before the first token
        "t_end": stamps[-1] + 0.01,
        "stamps": list(stamps),
        "text": text,
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


@pytest.fixture
def patched(monkeypatch):
    state = {"instrumentation": {"prefill": {"enabled": False, "observed": 0, "records": []}}}

    monkeypatch.setattr(
        client_mod, "stream_generation",
        lambda origin, body, timeout=0: _stream_result([10.0, 10.05, 10.10, 10.15]),
    )
    monkeypatch.setattr(client_mod, "get_json", lambda url, timeout=10: {"vram_bytes": 11 << 30})
    monkeypatch.setattr(
        client_mod, "fetch_instrumentation",
        lambda origin, limit=8: state["instrumentation"],
    )
    return state


def _measure(**kw):
    body = {"max_tokens": 4, "messages": [{"role": "user", "content": "x"}]}
    return client_mod.measure_generation(
        "http://x", body, prefill_seq_floor=kw.pop("floor", 0), store_text=kw.pop("store", False)
    )


def test_decode_window_is_anchored_on_the_first_and_last_token(patched):
    rec = _measure()
    # 3 steps across 0.15 s of arrivals
    assert rec["decode_steps"] == 3
    assert rec["decode_tok_s"] == pytest.approx(3 / 0.15, rel=1e-6)
    assert rec["ttft_ms"] == pytest.approx(500.0, rel=1e-6)


def test_raw_inter_token_gaps_are_preserved(patched):
    rec = _measure()
    assert len(rec["inter_token_ms"]) == 3
    assert rec["inter_token_ms_p50"] is not None
    assert rec["inter_token_ms_max"] == max(rec["inter_token_ms"])


def test_output_text_is_stored_only_when_asked(patched):
    assert _measure(store=False)["output_text"] is None
    assert _measure(store=True)["output_text"] == "hello"
    assert _measure()["output_sha256"]


def test_prefill_is_absent_with_a_reason_when_instrumentation_is_off(patched):
    rec = _measure()
    assert rec["prefill"] is None
    assert "FREETOKEN_INSTRUMENT_PREFILL" in rec["prefill_unavailable"]
    # and the trap is named rather than quietly available
    assert "not prefill throughput" in rec["prefill_tps_from_ttft_deliberately_absent"]


def test_a_stale_prefill_record_is_refused(patched):
    patched["instrumentation"] = {
        "prefill": {
            "enabled": True, "observed": 5,
            "records": [{"seq": 5, "gpu_ms": 40.0, "new_tokens": 900, "shared_batch": False}],
        }
    }
    rec = _measure(floor=5)  # the record predates this generation
    assert rec["prefill"] is None
    assert "stale" in rec["prefill_unavailable"]


def test_a_fresh_prefill_record_yields_a_rate_with_a_defined_denominator(patched):
    patched["instrumentation"] = {
        "prefill": {
            "enabled": True, "observed": 6, "measurement": "CUDA-event ...",
            "records": [{"seq": 6, "gpu_ms": 40.0, "new_tokens": 900, "shared_batch": False}],
        }
    }
    rec = _measure(floor=5)
    assert rec["prefill"]["prefill_tok_s"] == pytest.approx(900 / 0.04)
    # ... and it is NOT prompt_tokens / TTFT (which would be 900 / 0.5 = 1800)
    assert rec["prefill"]["prefill_tok_s"] != pytest.approx(900 / 0.5)


def test_a_shared_prefill_batch_yields_no_per_request_rate(patched):
    patched["instrumentation"] = {
        "prefill": {
            "enabled": True, "observed": 2,
            "records": [{"seq": 2, "gpu_ms": 40.0, "new_tokens": 0, "shared_batch": True}],
        }
    }
    rec = _measure(floor=1)
    assert rec["prefill"]["prefill_tok_s"] is None
    assert "more than one request" in rec["prefill"]["prefill_tok_s_unavailable"]


def test_a_truncated_stream_is_an_error_not_a_zero(monkeypatch, patched):
    monkeypatch.setattr(
        client_mod, "stream_generation",
        lambda origin, body, timeout=0: _stream_result([10.0], completion=1),
    )
    with pytest.raises(client_mod.GenerationError, match=">=2 token events"):
        _measure()


def test_completion_length_mismatch_is_recorded(monkeypatch, patched):
    monkeypatch.setattr(
        client_mod, "stream_generation",
        lambda origin, body, timeout=0: _stream_result([10.0, 10.1, 10.2], completion=3),
    )
    rec = _measure()
    assert rec["completion_tokens"] == 3
    assert rec["requested_max_tokens"] == 4
    assert rec["completion_matches_request"] is False


def test_missing_stats_is_an_explicit_null(monkeypatch, patched):
    def boom(url, timeout=10):
        raise OSError("connection reset")

    monkeypatch.setattr(client_mod, "get_json", boom)
    rec = _measure()
    assert rec["vram_bytes"] is None
    assert "/v1/stats" in rec["vram_unavailable"]


# --- attributing a prefill record to THIS request -----------------------------------------------

def _records(*specs):
    return {"enabled": True, "observed": max(s["seq"] for s in specs) if specs else 0,
            "records": list(specs)}


def test_a_record_is_matched_by_request_uid_when_the_response_id_carries_one():
    """/v1/chat/completions ids are chatcmpl-<uid> and each prefill record is stamped with
    the same uid, so the match is request identity rather than 'newest above a floor'."""
    block = _records(
        {"seq": 7, "uid": 41, "gpu_ms": 10.0, "new_tokens": 100},
        {"seq": 8, "uid": 42, "gpu_ms": 40.0, "new_tokens": 900},
        {"seq": 9, "uid": 43, "gpu_ms": 20.0, "new_tokens": 300},
    )
    rec, status = client_mod.select_prefill_record(block, prefill_seq_floor=0, request_uid=42)
    assert rec["uid"] == 42
    assert status["ok"] is True and status["attribution"] == "uid"


def test_a_uid_with_no_record_is_refused_rather_than_falling_back_to_the_newest():
    block = _records({"seq": 9, "uid": 43, "gpu_ms": 20.0, "new_tokens": 300})
    rec, status = client_mod.select_prefill_record(block, prefill_seq_floor=0, request_uid=42)
    assert rec is None
    assert status["code"] == client_mod.PREFILL_MISSING
    assert "uid 42" in status["reason"]


def test_multiple_fresh_records_without_a_uid_are_ambiguous_not_the_newest():
    """Selecting fresh[-1] would attribute another request's measured interval to this one."""
    block = _records(
        {"seq": 8, "gpu_ms": 40.0, "new_tokens": 900},
        {"seq": 9, "gpu_ms": 20.0, "new_tokens": 300},
    )
    rec, status = client_mod.select_prefill_record(block, prefill_seq_floor=7, request_uid=None)
    assert rec is None
    assert status["code"] == client_mod.PREFILL_AMBIGUOUS
    assert status["candidates"] == 2


def test_exactly_one_fresh_record_without_a_uid_is_accepted():
    block = _records({"seq": 8, "gpu_ms": 40.0, "new_tokens": 900})
    rec, status = client_mod.select_prefill_record(block, prefill_seq_floor=7, request_uid=None)
    assert rec["seq"] == 8
    assert status["ok"] is True and status["attribution"] == "sequence"


def test_the_request_uid_is_parsed_from_the_openai_response_id():
    assert client_mod.request_uid("chatcmpl-1234") == 1234
    assert client_mod.request_uid("cmpl-7") == 7
    assert client_mod.request_uid("chatcmpl-abcdef") is None
    assert client_mod.request_uid(None) is None


def test_every_prefill_refusal_carries_a_stable_code(patched):
    rec = _measure()
    assert rec["prefill_status"]["code"] == client_mod.PREFILL_DISABLED
    assert rec["prefill_status"]["ok"] is False


def test_a_shared_batch_is_flagged_by_code_not_only_by_prose(patched):
    patched["instrumentation"] = {
        "prefill": {
            "enabled": True, "observed": 2,
            "records": [{"seq": 2, "gpu_ms": 40.0, "new_tokens": 0, "shared_batch": True}],
        }
    }
    rec = _measure(floor=1)
    assert rec["prefill_status"]["code"] == client_mod.PREFILL_SHARED_BATCH


def test_zero_gpu_ms_is_flagged_as_unusable(patched):
    patched["instrumentation"] = {
        "prefill": {
            "enabled": True, "observed": 2,
            "records": [{"seq": 2, "gpu_ms": 0.0, "new_tokens": 900, "shared_batch": False}],
        }
    }
    rec = _measure(floor=1)
    assert rec["prefill_status"]["code"] == client_mod.PREFILL_UNUSABLE
    assert rec["prefill"]["prefill_tok_s"] is None


def test_missing_instrumentation_is_flagged_as_unavailable(monkeypatch, patched):
    monkeypatch.setattr(
        client_mod, "fetch_instrumentation",
        lambda origin, limit=8: {"unavailable": "HTTP 404 from /v1/instrumentation"},
    )
    rec = _measure()
    assert rec["prefill_status"]["code"] == client_mod.PREFILL_UNAVAILABLE
