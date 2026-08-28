from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.message import MoeInstrumentationBackendMsg
from freetoken.scheduler.scheduler import Scheduler


def _scheduler(*, prefill: bool = False, decode: bool = False, inflight=False):
    sched = Scheduler.__new__(Scheduler)
    sched.prefill_manager = SimpleNamespace(runnable=prefill)
    sched.decode_manager = SimpleNamespace(runnable=decode)
    sched._last_data = object() if inflight else None
    sched.config = SimpleNamespace(tp_info=SimpleNamespace(size=1))
    sched.device = torch.device("cpu")
    calls = []
    sched.engine = SimpleNamespace(
        moe_offload_cache=object(),
        moe_instrumentation=lambda operation: calls.append(operation) or {"schema": "x"},
    )
    replies = []
    sched._reply_moe_instrumentation = (
        lambda request_id, status, payload=None, error=None: replies.append(
            (request_id, status, payload, error)
        )
    )
    return sched, calls, replies


def test_snapshot_is_rejected_while_generation_is_active():
    for state in ({"prefill": True}, {"decode": True}, {"inflight": True}):
        sched, calls, replies = _scheduler(**state)
        Scheduler._process_one_msg(
            sched, MoeInstrumentationBackendMsg(request_id="i1", operation="snapshot")
        )
        assert calls == []
        assert replies[0][1] == "busy"


def test_reset_runs_only_at_synchronized_idle_boundary(monkeypatch):
    sched, calls, replies = _scheduler()
    syncs = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: syncs.append(device))

    Scheduler._process_one_msg(
        sched, MoeInstrumentationBackendMsg(request_id="i2", operation="reset")
    )

    assert syncs == [torch.device("cpu")]
    assert calls == ["reset"]
    assert replies == [("i2", "ok", {"schema": "x"}, None)]
