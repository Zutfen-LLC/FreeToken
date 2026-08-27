"""Regression for terminal length accounting under overlap scheduling.

The overlap loop can launch the next decode forward before the previous sampled token is
host-drained.  ``Engine.forward_batch`` advances ``Req.device_len`` for that future forward,
so terminal accounting must use host-appended output tokens rather than the future-looking
``Req.can_decode`` state.  Otherwise an N-token request terminates after N-1 delivered tokens.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.scheduler.scheduler import Scheduler


class _CopyDone:
    def synchronize(self) -> None:
        pass


class _ForwardOutput:
    prefill_timing = None

    def __init__(self, token: int) -> None:
        self._parts = (None, torch.tensor([token], dtype=torch.int32), _CopyDone())

    def __getitem__(self, index: int):
        return self._parts[index]


def _scheduler_for_drain(captured: list) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.cache_manager = SimpleNamespace(lazy_free_region=lambda: nullcontext())
    scheduler.decode_manager = SimpleNamespace(
        remove_req=lambda req: None,
        running_reqs=set(),
    )
    scheduler.finished_reqs = set()
    scheduler._prefill_probe = {}
    scheduler.toolcall_anchor_id = None
    scheduler.eos_token_ids = set()
    scheduler.config = SimpleNamespace(page_size=1)
    scheduler.status_reporter = SimpleNamespace(report_batch=lambda *args, **kwargs: None)
    scheduler.send_result = lambda replies: captured.extend(replies)
    scheduler._accumulate_prefill = lambda *args, **kwargs: None
    scheduler._free_req_resources = lambda req: setattr(req, "table_idx", -1)
    scheduler._kv_usage_pages = lambda: (0, 1)
    scheduler._mamba_slot_usage = lambda: None
    scheduler._swa_token_usage = lambda: None
    scheduler._gpu_mem_bytes = lambda: 0
    return scheduler


def _last_data(batch: Batch, token: int):
    return (SimpleNamespace(batch=batch), _ForwardOutput(token))


def test_overlap_does_not_finish_penultimate_token_when_future_forward_reaches_limit():
    captured = []
    scheduler = _scheduler_for_drain(captured)

    req = Req(
        input_ids=torch.tensor([10, 11], dtype=torch.int64),
        table_idx=0,
        cached_len=1,
        output_len=3,
        uid=7,
        sampling_params=SamplingParams(ignore_eos=True, max_tokens=3),
        cache_handle=object(),
    )
    batch = Batch(reqs=[req], phase="decode")

    # Token 1 was sampled and drained normally.
    req.complete_one()
    req.append_host(torch.tensor([101], dtype=torch.int64))

    # Token 2 is the last_data waiting to be drained.  Before that happens, overlap launches
    # token 3 and Engine.forward_batch advances device_len to max_device_len.  This is the
    # exact state that made ``not req.can_decode`` falsely terminate token 2.
    req.complete_one()  # token 2 sampled
    req.complete_one()  # future token 3 sampled; device_len now at the output limit
    assert req.device_len == req.max_device_len
    assert req.input_ids.numel() == req.max_device_len - 2

    scheduler._process_last_data(_last_data(batch, 102))

    assert len(captured) == 1
    assert captured[0].next_token == 102
    assert captured[0].finished is False
    assert captured[0].finish_reason is None
    assert req.input_ids.numel() == req.max_device_len - 1
    assert req not in scheduler.finished_reqs

    # The actual Nth token, already sampled by the overlapped future forward, is the one that
    # must carry finish_reason=length.
    scheduler._process_last_data(_last_data(batch, 103))

    assert len(captured) == 2
    assert captured[1].next_token == 103
    assert captured[1].finished is True
    assert captured[1].finish_reason == "length"
    assert req.input_ids.numel() == req.max_device_len
