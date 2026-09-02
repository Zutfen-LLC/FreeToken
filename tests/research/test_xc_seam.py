"""Tests for the coordinator-side remote realizer and the Node-local
execution agent seam (inferswarm #67).

Uses an in-memory fake runtime so these run on the CPU-only Coordinator host
with no torch, no model, and no GPU.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from freetoken.research.xc_coordinator import (
    RemoteEpochRuntime,
    RemoteNodeAgentConnection,
    RemoteRealizationError,
    make_remote_realizer,
)
from freetoken.research.xc_wire import (
    PROTOCOL_ID,
    XCWireError,
    body_checksum,
    encode_frame,
    recv_frame,
    send_exact,
)

import benchmarks.inferswarm_xc.node_agent as node_agent_module


class FakeRuntime:
    """Deterministic stand-in for the accepted R5B-isolated R4 runtime.

    Stateful like the real substrate: each session's first call records the
    prompt length; replay inputs (prompt + committed output) yield the next
    reference tokens from that position.
    """

    def __init__(self, tokens=(9764, 393, 45)):
        self.tokens = list(tokens)
        self.generate_calls = 0
        self.report_calls = 0
        self.closed = False
        self._base: int | None = None
        self.observation = {
            "plan_digest": None,  # filled by the agent harness below
            "participants": ["node.inferswarm01", "node.inferswarm03"],
            "compute_units": ["gpu.node-a.0", "gpu.node-b.0"],
        }

    def generate(self, *, session_id, prompt_token_ids, max_new_tokens, on_token=None):
        self.generate_calls += 1
        # The real substrate derives decode state from the replayed token
        # stream (prompt + committed output), not from the wire session id.
        if self._base is None:
            self._base = len(prompt_token_ids)
        offset = len(prompt_token_ids) - self._base
        out = self.tokens[offset : offset + max_new_tokens]
        assert out, "fake runtime exhausted its reference tokens"
        result = {
            "session_id": session_id,
            "generated_token_ids": list(out),
            "plan_digest": self.observation["plan_digest"],
            "boundaries": [],
        }
        if on_token is not None:
            for step, token in enumerate(out):
                on_token(step, token, {})
        return result

    def report(self):
        self.report_calls += 1
        return {"kind": "fake-report", "generate_calls": self.generate_calls}

    def close(self):
        self.closed = True


PLAN_DIGEST = "sha256:" + "b" * 64
SCOPE = "scope/test-xc"


def _authorization(plan, generation=2, attempt=1, realization_id=None):
    """A Coordinator-shaped realization authorization for the plan."""
    tag = plan["digest"].split(":")[-1][:12]
    return {
        "schema": "inferswarm.r5b.realization-authorization/1",
        "epoch_id": f"research-generation-{generation}:{tag}",
        "generation": generation,
        "plan_digest": plan["digest"],
        "realization_id": realization_id or f"realization-{attempt}-{attempt:012x}",
        "attempt": attempt,
        "allocated_at_ns": 1,
    }


def _frozen_plan():
    """A minimal, genuinely frozen execution plan (digest verifiable)."""
    from freetoken.research.r3_planner import freeze

    return freeze(
        {
            "schema": "inferswarm.r5a.execution-plan/1",
            "candidate_id": "resident-two-node-two-slot[test]",
            "strategy_identity": {"id": "test.strategy/1"},
            "model_identity": {"repository": "test/model", "revision": "testrev"},
            "participants": ["node.inferswarm01", "node.inferswarm03"],
            "compute_units": ["gpu.node-a.0", "gpu.node-b.0"],
            "representations": [],
            "backend_choices": [],
            "state_placement": [],
            "state_authority": [],
            "semantic_boundaries": [],
            "expected_resource_accounting": {},
        }
    )


def _matching_observation(plan):
    return {
        "plan_digest": plan["digest"],
        "participants": plan["participants"],
        "compute_units": plan["compute_units"],
        "representations": plan["representations"],
        "backend_choices": plan["backend_choices"],
        "state_placement": plan["state_placement"],
        "state_authority": plan["state_authority"],
        "semantic_boundaries": plan["semantic_boundaries"],
    }


class _AgentHarness:
    """Run the real NodeAgent serve loop in-thread against a fake runtime."""

    def __init__(self, runtime: FakeRuntime):
        self.runtime = runtime

        import argparse

        args = argparse.Namespace(
            listen_host="127.0.0.1",
            listen_port=0,
            scope_id=SCOPE,
            participant_plan="",
            model="",
            peer_host="",
            peer_port=18485,
            diagnostic=False,
            ready_file=None,
        )
        self.agent = node_agent_module.NodeAgent(args)
        self.agent.build_runtime = self._build
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(4)
        self.port = self._listener.getsockname()[1]
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _build(self, plan):
        self.runtime.observation["plan_digest"] = plan["digest"]
        holder = _FakeRemote(self.runtime)
        holder.observation = _matching_observation(plan)
        return holder

    def _accept_loop(self):
        while True:
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return
            session = node_agent_module._Session(self.agent, conn)
            threading.Thread(target=session.run, daemon=True).start()


class _FakeRemote:
    def __init__(self, runtime):
        self._runtime = runtime
        self.observation = {}
        self.reclamation_report = {"kind": "fake-reclamation"}

    def generate(self, **kwargs):
        return self._runtime.generate(**kwargs)

    def report(self):
        return self._runtime.report()

    def close(self):
        self._runtime.close()


@pytest.fixture()
def agent():
    harness = _AgentHarness(FakeRuntime())
    yield harness
    harness._listener.close()


def _connect(port):
    return RemoteNodeAgentConnection(
        host="127.0.0.1", port=port, scope_id=SCOPE
    )


class TestRemoteRealization:
    def test_realize_generate_close_round_trip(self, agent):
        plan = _frozen_plan()
        runtime = _connect(agent.port).realize(plan, _authorization(plan))
        assert runtime.observation["plan_digest"] == plan["digest"]
        result = runtime.generate(
            session_id=1, prompt_token_ids=[9764, 393], max_new_tokens=2
        )
        assert result["generated_token_ids"] == [9764, 393]
        report = runtime.report()
        assert report["generate_calls"] == 1
        runtime.close()
        assert agent.runtime.closed
        assert runtime.reclamation_report["kind"] == "remote-node-agent-epoch-close"

    def test_generate_before_realize_rejected(self, agent):
        sock = socket.create_connection(("127.0.0.1", agent.port), timeout=10)
        body = {
            "kind": "request",
            "protocol": PROTOCOL_ID,
            "scope_id": SCOPE,
            "epoch_id": "research-generation-1:bbbbbbbbbbbb",
            "generation": 1,
            "realization_id": "realization-9-fff",
            "plan_digest": PLAN_DIGEST,
            "operation": "GENERATE",
            "session_id": 1,
            "position": 1,
            "prompt_token_ids": [1],
            "max_new_tokens": 1,
        }
        send_exact(sock, encode_frame(body))
        response = recv_frame(sock)
        assert response["ok"] is False
        assert "no authorized realization" in response["error"]
        sock.close()

    def test_scope_mismatch_rejected(self, agent):
        connection = RemoteNodeAgentConnection(
            host="127.0.0.1", port=agent.port, scope_id="scope/other"
        )
        with pytest.raises(RemoteRealizationError):
            connection.realize(_frozen_plan(), _authorization(_frozen_plan()))

    def test_stale_position_rejected(self, agent):
        runtime = _connect(agent.port).realize(
            _frozen_plan(), _authorization(_frozen_plan())
        )
        sock = runtime._sock
        # sequence 1 (fresh) then a replay of position 1 -> stale
        runtime.generate(session_id=1, prompt_token_ids=[1], max_new_tokens=1)
        body = {
            "kind": "request",
            "protocol": PROTOCOL_ID,
            "scope_id": SCOPE,
            "epoch_id": runtime._epoch_id,
            "generation": runtime._generation,
            "realization_id": runtime._realization_id,
            "plan_digest": runtime._plan_digest,
            "operation": "GENERATE",
            "session_id": 1,
            "position": 1,
            "prompt_token_ids": [1],
            "max_new_tokens": 1,
        }
        send_exact(sock, encode_frame(body))
        response = recv_frame(sock)
        assert response["ok"] is False
        assert "stale/reordered execution position" in response["error"]
        # After a rejection the agent tears the connection down: the runtime
        # must fail closed rather than guess.
        with pytest.raises(RemoteRealizationError):
            runtime.report()

    def test_plan_digest_mismatch_rejected(self, agent):
        runtime = _connect(agent.port).realize(
            _frozen_plan(), _authorization(_frozen_plan())
        )
        sock = runtime._sock
        body = {
            "kind": "request",
            "protocol": PROTOCOL_ID,
            "scope_id": SCOPE,
            "epoch_id": runtime._epoch_id,
            "generation": runtime._generation,
            "realization_id": runtime._realization_id,
            "plan_digest": "sha256:" + "c" * 64,
            "operation": "REPORT",
        }
        send_exact(sock, encode_frame(body))
        response = recv_frame(sock)
        assert response["ok"] is False
        assert "identity" in response["error"]
        sock.close()

    def test_result_checksum_tamper_detected_by_client(self, agent):
        runtime = _connect(agent.port).realize(
            _frozen_plan(), _authorization(_frozen_plan())
        )
        sock = runtime._sock
        request = {
            "kind": "request",
            "protocol": PROTOCOL_ID,
            "scope_id": SCOPE,
            "epoch_id": runtime._epoch_id,
            "generation": runtime._generation,
            "realization_id": runtime._realization_id,
            "plan_digest": runtime._plan_digest,
            "operation": "GENERATE",
            "session_id": 7,
            "position": 1,
            "prompt_token_ids": [9764],
            "max_new_tokens": 1,
        }
        send_exact(sock, encode_frame(request))
        response = recv_frame(sock)
        assert response["ok"] is True
        # Tamper: wrong checksum -> the coordinator facade must fail closed.
        response["result_checksum"] = "sha256:" + "0" * 64
        from freetoken.research.xc_wire import validate_response

        with pytest.raises(XCWireError):
            validate_response(response)
        runtime.close()

    def test_disconnect_during_exchange_fails_closed(self, agent):
        runtime = _connect(agent.port).realize(
            _frozen_plan(), _authorization(_frozen_plan())
        )
        sock = runtime._sock
        # half-send a request then kill the socket
        frame = encode_frame(
            {
                "kind": "request",
                "protocol": PROTOCOL_ID,
                "scope_id": SCOPE,
                "epoch_id": runtime._epoch_id,
            "generation": runtime._generation,
            "realization_id": runtime._realization_id,
                "plan_digest": runtime._plan_digest,
                "operation": "REPORT",
            }
        )
        sock.sendall(frame[:5])
        sock.close()
        with pytest.raises(RemoteRealizationError):
            runtime.report()


class TestMakeRemoteRealizer:
    def test_realizer_returns_reconciled_plan(self, agent):
        realizer = make_remote_realizer(
            host="127.0.0.1", port=agent.port, scope_id=SCOPE
        )
        realized = realizer(_frozen_plan(), _authorization(_frozen_plan()))
        assert realized.observation["plan_digest"] == realized.observation["plan_digest"]
        result = realized.runtime.generate(
            session_id=1, prompt_token_ids=[9764], max_new_tokens=1
        )
        assert result["generated_token_ids"] == [9764]
        realized.runtime.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
