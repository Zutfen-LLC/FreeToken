"""``/v1/instrumentation``: the benchmark-facing provenance document.

Two things must hold for it to be usable as benchmark evidence:

* a consumer can tell "no new prefill record" from "a record I missed" -- hence the
  monotonic sequence number and the observed total;
* every absence is explicit. "prefill not measured" must never be readable as "prefill was
  free", and a missing readiness ack must not silently look like a resolved configuration.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.server.stats import StatsTracker, build_instrumentation


def _reply(**kw):
    base = dict(uid=1, prefill=None, gpu_mem_bytes=0, completion_tokens_delta=0,
                prompt_tokens_delta=0, finished=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _state(tracker, runtime_config=None):
    return SimpleNamespace(stats=tracker, runtime_config=runtime_config, instance_id="i-1")


def _prefill(gpu_ms=40.0, new_tokens=900):
    return {"gpu_ms": gpu_ms, "new_tokens": new_tokens, "cached_tokens": 0,
            "chunks": 1, "shared_batch": False}


def test_prefill_records_are_sequenced_and_counted():
    tr = StatsTracker()
    for i in range(3):
        tr.observe(_reply(prefill=_prefill(gpu_ms=10.0 + i)))
    assert tr.prefill_observed == 3
    seqs = [r["seq"] for r in tr.prefill_records()]
    assert seqs == [1, 2, 3]
    assert [r["gpu_ms"] for r in tr.prefill_records()] == [10.0, 11.0, 12.0]


def test_replies_without_a_measurement_do_not_advance_the_counter():
    tr = StatsTracker()
    tr.observe(_reply(prefill=None, completion_tokens_delta=1))
    assert tr.prefill_observed == 0
    assert tr.prefill_records() == []


def test_the_record_ring_is_bounded_but_the_counter_is_not():
    tr = StatsTracker()
    for _ in range(300):
        tr.observe(_reply(prefill=_prefill()))
    assert tr.prefill_observed == 300
    records = tr.prefill_records(limit=256)
    assert len(records) <= 256
    assert records[-1]["seq"] == 300  # a consumer can still detect what it missed


def test_disabled_instrumentation_is_stated_not_implied(monkeypatch):
    from freetoken.server import stats as stats_mod

    monkeypatch.setattr(stats_mod.ENV, "INSTRUMENT_PREFILL", False)
    doc = build_instrumentation(_state(StatsTracker()))
    assert doc["prefill"]["enabled"] is False
    assert doc["prefill"]["records"] == []
    assert doc["prefill"]["observed"] == 0


def test_enabled_instrumentation_documents_its_measurement_boundary(monkeypatch):
    from freetoken.server import stats as stats_mod

    monkeypatch.setattr(stats_mod.ENV, "INSTRUMENT_PREFILL", True)
    doc = build_instrumentation(_state(StatsTracker()))
    assert doc["prefill"]["enabled"] is True
    measurement = doc["prefill"]["measurement"]
    assert "CUDA-event" in measurement
    # the whole point: TTFT is a different quantity
    assert "TTFT" in measurement


def test_runtime_config_absence_is_explicit():
    doc = build_instrumentation(_state(StatsTracker(), runtime_config=None))
    assert doc["runtime_config"] is None
    assert doc["runtime_config_unavailable"]


def test_runtime_config_is_passed_through_when_the_ack_delivered_it():
    config = {"moe": {"backend_resolved": "offload"}}
    doc = build_instrumentation(_state(StatsTracker(), runtime_config=config))
    assert doc["runtime_config"] == config
    assert doc["runtime_config_unavailable"] is None


def test_stats_tracker_keeps_working_for_its_existing_consumers():
    """The prefill ring is additive; /v1/stats must be unaffected."""
    tr = StatsTracker()
    tr.observe(_reply(completion_tokens_delta=4, gpu_mem_bytes=123), now=100.0)
    assert tr.completion_tokens_total == 4
    assert tr.vram_bytes == 123
    assert tr.decode_tps(now=101.0) == pytest.approx(4.0)
