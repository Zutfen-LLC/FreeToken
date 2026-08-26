"""Per-request prefill attribution (``Scheduler._accumulate_prefill``).

CPU-only: the CUDA events are stood in for by a stub whose ``elapsed_time`` returns a known
number, so what is under test is the *attribution*, not the timer.

The property that matters: a prefill interval is a property of the batch. Attributing a
shared batch's interval to one request would manufacture a number, so a multi-request batch
is marked and its token counts are left at zero for the consumer to reject.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.message import PrefillMeasurement
from freetoken.scheduler.scheduler import Scheduler


class _Event:
    def __init__(self, ms: float) -> None:
        self._ms = ms

    def elapsed_time(self, _other) -> float:
        return self._ms


def _timing(ms: float):
    return (_Event(ms), object())


def _batch(uids, *, new_tokens=900, cached_tokens=0, prefill=True):
    return SimpleNamespace(
        reqs=[SimpleNamespace(uid=u) for u in uids],
        is_prefill=prefill,
        log_new_tokens=new_tokens,
        log_cached_tokens=cached_tokens,
    )


@pytest.fixture
def scheduler():
    """A bare object with just the accumulator state; Scheduler.__init__ needs a GPU."""
    obj = Scheduler.__new__(Scheduler)
    obj._prefill_probe = {}
    return obj


def test_a_single_request_batch_is_attributed_exactly(scheduler):
    scheduler._accumulate_prefill(_batch([7], new_tokens=900, cached_tokens=100), _timing(42.0))
    rec = scheduler._prefill_probe[7]
    assert isinstance(rec, PrefillMeasurement)
    assert rec.gpu_ms == 42.0
    assert rec.new_tokens == 900 and rec.cached_tokens == 100
    assert rec.chunks == 1 and rec.shared_batch is False


def test_chunked_prefill_sums_across_chunks(scheduler):
    for chunk_ms, chunk_tokens in ((10.0, 512), (12.0, 512), (5.0, 200)):
        scheduler._accumulate_prefill(_batch([1], new_tokens=chunk_tokens), _timing(chunk_ms))
    rec = scheduler._prefill_probe[1]
    assert rec.gpu_ms == pytest.approx(27.0)
    assert rec.new_tokens == 1224
    assert rec.chunks == 3


def test_a_shared_batch_is_marked_and_its_token_counts_withheld(scheduler):
    scheduler._accumulate_prefill(_batch([1, 2], new_tokens=1800), _timing(50.0))
    for uid in (1, 2):
        rec = scheduler._prefill_probe[uid]
        assert rec.shared_batch is True
        assert rec.new_tokens == 0 and rec.cached_tokens == 0
        assert rec.gpu_ms == 50.0  # the batch's interval, honestly labelled as shared


def test_a_shared_chunk_taints_the_whole_request(scheduler):
    scheduler._accumulate_prefill(_batch([1], new_tokens=512), _timing(10.0))
    scheduler._accumulate_prefill(_batch([1, 2], new_tokens=512), _timing(20.0))
    assert scheduler._prefill_probe[1].shared_batch is True


def test_no_timing_records_nothing(scheduler):
    """Instrumentation is off by default; the accumulator must be inert then."""
    scheduler._accumulate_prefill(_batch([1]), None)
    assert scheduler._prefill_probe == {}


def test_decode_batches_are_ignored(scheduler):
    scheduler._accumulate_prefill(_batch([1], prefill=False), _timing(99.0))
    assert scheduler._prefill_probe == {}


def test_the_measurement_message_defaults_to_not_measured():
    """None, never 0: "not measured" must not be readable as "0 ms"."""
    from freetoken.message import DetokenizeMsg

    msg = DetokenizeMsg(uid=1, next_token=5, finished=False)
    assert msg.prefill is None


def test_the_measurement_survives_the_message_wire():
    """The scheduler and the frontend are different processes; the record must serialize."""
    from freetoken.message import DetokenizeMsg

    original = DetokenizeMsg(
        uid=3, next_token=9, finished=False,
        prefill=PrefillMeasurement(gpu_ms=12.5, new_tokens=900, cached_tokens=0, chunks=2),
    )
    restored = DetokenizeMsg.decoder(DetokenizeMsg.encoder(original))
    assert isinstance(restored.prefill, PrefillMeasurement)
    assert restored.prefill.gpu_ms == 12.5
    assert restored.prefill.new_tokens == 900
    assert restored.prefill.chunks == 2
