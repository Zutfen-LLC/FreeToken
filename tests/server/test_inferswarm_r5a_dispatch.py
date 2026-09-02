from __future__ import annotations

import asyncio
from types import SimpleNamespace

from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.server.api_server import FrontendManager


class _Queue:
    def __init__(self):
        self.items = []

    async def put(self, value):
        self.items.append(value)

    def stop(self):
        pass


class _Dispatcher:
    def __init__(self):
        self.items = []
        self.closed = False

    async def submit(self, msg, state):
        self.items.append((msg, state))

    def close(self):
        self.closed = True


def test_existing_tokenize_waist_routes_to_r5a_without_zmq_submission():
    send = _Queue()
    state = FrontendManager(
        config=SimpleNamespace(),
        send_tokenizer=send,
        recv_tokenizer=_Queue(),
    )
    dispatcher = _Dispatcher()
    state.inferswarm_r5a_dispatcher = dispatcher
    msg = TokenizeMsg(uid=7, text="ordinary request", sampling_params=SamplingParams(max_tokens=2))
    asyncio.run(state.send_one(msg))
    assert dispatcher.items == [(msg, state)]
    assert send.items == []
    state.shutdown()
    assert dispatcher.closed


def test_default_serving_path_is_unchanged_without_r5a_dispatcher():
    send = _Queue()
    state = FrontendManager(
        config=SimpleNamespace(),
        send_tokenizer=send,
        recv_tokenizer=_Queue(),
    )
    # Avoid starting the real listener; the assertion here is only that the
    # unchanged queue path remains the default.
    state._create_listener_once = lambda: None
    msg = TokenizeMsg(uid=8, text="ordinary request", sampling_params=SamplingParams(max_tokens=2))
    asyncio.run(state.send_one(msg))
    assert send.items == [msg]


def test_existing_tokenize_waist_routes_to_r5b_without_zmq_submission():
    send = _Queue()
    state = FrontendManager(
        config=SimpleNamespace(),
        send_tokenizer=send,
        recv_tokenizer=_Queue(),
    )
    dispatcher = _Dispatcher()
    state.inferswarm_r5b_dispatcher = dispatcher
    msg = TokenizeMsg(
        uid=9, text="ordinary epoch request", sampling_params=SamplingParams(max_tokens=2)
    )
    asyncio.run(state.send_one(msg))
    assert dispatcher.items == [(msg, state)]
    assert send.items == []
    state.shutdown()
    assert dispatcher.closed
