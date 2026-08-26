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
