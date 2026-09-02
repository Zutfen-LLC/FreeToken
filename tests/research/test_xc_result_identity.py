"""Coordinator-side result-identity validation tests (inferswarm #67 final
review blocker).

Proves the Coordinator rejects a syntactically valid remote response whose
result identity does not match the exact request just sent: wrong scope,
wrong session, wrong position, and a stale previous position — plus, through
the controller path, a result whose epoch/generation/realization/plan are all
correct but whose session or position does not.  Every mismatch must fail
closed before any token callback/emission or ledger mutation, and the
connection/scope must fail closed afterwards.  No TCP-ordering assumption is
relied upon: the response identity is compared against the request just
sent, never against socket arrival order.

Runs on the CPU-only Coordinator host: no torch, no model, no GPU.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
from typing import Any

import pytest

from freetoken.research.xc_coordinator import (
    RemoteNodeAgentConnection,
    RemoteRealizationError,
    make_remote_realizer,
)
from freetoken.research.xc_wire import (
    PROTOCOL_ID,
    HEADER_STRUCT,
    body_checksum,
    canonical_body,
    encode_frame,
    send_exact,
)

from tests.research.test_xc_coordinator_epoch import (
    _fake_environment,
    _snapshot_from,
    _test_evidence_catalog,
    _test_objective,
    _test_problem,
    _StepStrategy,
)
from tests.research.test_xc_seam import (
    SCOPE,
    FakeRuntime,
    _AgentHarness,
    _authorization,
    _frozen_plan,
)


class _SessionSpoofRuntime(FakeRuntime):
    """A misbehaving substrate whose result claims a different session."""

    def generate(self, *, session_id, prompt_token_ids, max_new_tokens, on_token=None):
        result = dict(
            super().generate(
                session_id=session_id,
                prompt_token_ids=prompt_token_ids,
                max_new_tokens=max_new_tokens,
                on_token=None,
            )
        )
        result["session_id"] = session_id + 100  # attributed to another session
        return result


def _recv_frame_raw(sock: socket.socket) -> bytes:
    header = b""
    while len(header) < HEADER_STRUCT.size:
        chunk = sock.recv(HEADER_STRUCT.size - len(header))
        assert chunk, "disconnect while reading a frame header"
        header += chunk
    body_len = HEADER_STRUCT.unpack(header)[2]
    body = b""
    while len(body) < body_len:
        chunk = sock.recv(body_len - len(body))
        assert chunk, "disconnect while reading a frame body"
        body += chunk
    return header + body


def _edit_frame(raw: bytes, field: str, value: Any) -> bytes:
    """Rewrite one field of a captured frame, keeping it syntactically valid.

    For GENERATE responses the result checksum is recomputed over the
    (untouched) result so only the echoed identity field is wrong.
    """
    magic, version, _ = HEADER_STRUCT.unpack(raw[: HEADER_STRUCT.size])
    body = json.loads(raw[HEADER_STRUCT.size :])
    body[field] = value
    if body.get("operation") == "GENERATE" and "result" in body:
        body["result_checksum"] = body_checksum(body["result"])
    data = canonical_body(body)
    return HEADER_STRUCT.pack(magic, version, len(data)) + data


def _capture_honest_frames(generate_count: int = 1) -> list[bytes]:
    """Capture honest REALIZE + N GENERATE response frames from the real
    Node agent (fake runtime), then tear the session down."""
    harness = _AgentHarness(FakeRuntime())
    plan = _frozen_plan()
    auth = _authorization(plan)
    sock = socket.create_connection(("127.0.0.1", harness.port), timeout=10)
    try:
        realize_request = {
            "kind": "request",
            "protocol": PROTOCOL_ID,
            "scope_id": SCOPE,
            "epoch_id": auth["epoch_id"],
            "generation": auth["generation"],
            "realization_id": auth["realization_id"],
            "plan_digest": plan["digest"],
            "operation": "REALIZE",
            "execution_plan": plan,
        }
        send_exact(sock, encode_frame(realize_request))
        frames = [_recv_frame_raw(sock)]
        for position in range(1, generate_count + 1):
            generate_request = {
                "kind": "request",
                "protocol": PROTOCOL_ID,
                "scope_id": SCOPE,
                "epoch_id": auth["epoch_id"],
                "generation": auth["generation"],
                "realization_id": auth["realization_id"],
                "plan_digest": plan["digest"],
                "operation": "GENERATE",
                "session_id": 1,
                "position": position,
                "prompt_token_ids": [9764, 393],
                "max_new_tokens": 1,
            }
            send_exact(sock, encode_frame(generate_request))
            frames.append(_recv_frame_raw(sock))
        return frames
    finally:
        sock.close()
        harness._listener.close()


class _ReplayAgent:
    """Stand-in Node agent that replays pre-captured response frames.

    Reads one request per response it serves; the Coordinator-side facade
    cannot tell it from a misbehaving real agent: the frames are
    syntactically valid, checksummed, and identity-bearing.
    """

    def __init__(self, frames: list[bytes]):
        self._listener = socket.socket()
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self.port = self._listener.getsockname()[1]
        self._frames = list(frames)
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        try:
            conn, _ = self._listener.accept()
        except OSError:
            return
        try:
            for frame in self._frames:
                # consume the request (one full frame) then answer
                _recv_frame_raw(conn)
                send_exact(conn, frame)
            # frames exhausted: fail closed by disconnecting
        except (OSError, AssertionError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        self._listener.close()


def _realize_against(port: int):
    plan = _frozen_plan()
    return RemoteNodeAgentConnection(
        host="127.0.0.1", port=port, scope_id=SCOPE
    ).realize(plan, _authorization(plan))


def _generate_with_tokens(runtime, session_id: int = 1):
    tokens: list[int] = []
    runtime.generate(
        session_id=session_id,
        prompt_token_ids=[9764, 393],
        max_new_tokens=1,
        on_token=lambda step, token, boundary: tokens.append(token),
    )
    return tokens


class TestResponseIdentityValidation:
    """A syntactically valid response with wrong identity must fail closed
    before any token callback/emission, and the scope must fail closed."""

    def _rejected(self, frames: list[bytes], expected_fragment: str):
        replay = _ReplayAgent(frames)
        try:
            runtime = _realize_against(replay.port)
            with pytest.raises(RemoteRealizationError) as excinfo:
                _generate_with_tokens(runtime)
            assert expected_fragment in str(excinfo.value)
            # The connection/scope fails closed: the runtime is unusable
            # after a result-identity rejection.
            with pytest.raises(RemoteRealizationError):
                runtime.report()
            return excinfo.value
        finally:
            replay.close()

    def test_wrong_scope_id_rejected(self):
        honest = _capture_honest_frames()
        tampered = _edit_frame(honest[1], "scope_id", "scope/other")
        self._rejected([honest[0], tampered], "scope")

    def test_wrong_session_id_rejected(self):
        honest = _capture_honest_frames()
        tampered = _edit_frame(honest[1], "session_id", 42)
        self._rejected([honest[0], tampered], "session")

    def test_wrong_position_rejected(self):
        honest = _capture_honest_frames()
        tampered = _edit_frame(honest[1], "position", 7)
        self._rejected([honest[0], tampered], "position")

    def test_stale_previous_position_rejected(self):
        honest = _capture_honest_frames(generate_count=2)
        assert json.loads(honest[2][HEADER_STRUCT.size :])["position"] == 2
        # The second GENERATE's response echoes the stale position 1.
        stale = _edit_frame(honest[2], "position", 1)
        replay = _ReplayAgent([honest[0], honest[1], stale])
        try:
            runtime = _realize_against(replay.port)
            first = _generate_with_tokens(runtime)  # position 1: accepted
            assert first == [9764]
            tokens: list[int] = []
            with pytest.raises(RemoteRealizationError) as excinfo:
                runtime.generate(
                    session_id=1,
                    prompt_token_ids=[9764, 393, 9764],
                    max_new_tokens=1,
                    on_token=lambda step, token, boundary: tokens.append(token),
                )
            assert "position" in str(excinfo.value)
            assert tokens == []  # the stale-position result emitted nothing
        finally:
            replay.close()

    def test_result_session_identity_mismatch_rejected(self):
        # The runtime-level result claims a session the Coordinator did not
        # request: the facade must fail closed before on_token even though
        # every wire-level identity field is honest.
        harness = _AgentHarness(_SessionSpoofRuntime())
        try:
            plan = _frozen_plan()
            runtime = RemoteNodeAgentConnection(
                host="127.0.0.1", port=harness.port, scope_id=SCOPE
            ).realize(plan, _authorization(plan))
            tokens: list[int] = []
            with pytest.raises(RemoteRealizationError) as excinfo:
                runtime.generate(
                    session_id=1,
                    prompt_token_ids=[9764, 393],
                    max_new_tokens=1,
                    on_token=lambda step, token, boundary: tokens.append(token),
                )
            assert "session" in str(excinfo.value)
            assert tokens == []
        finally:
            harness._listener.close()


class TestControllerPathFencing:
    """Correct epoch/generation/realization/plan but wrong session/position,
    exercised through the controller: no emission, no ledger mutation."""

    def _controller_with(self, harness):
        from freetoken.research.r3_planner import freeze
        from freetoken.research.r5b_epochs import EpochServingController

        realizer = make_remote_realizer(
            host="127.0.0.1", port=harness.port, scope_id=SCOPE
        )
        return EpochServingController(
            problem=_test_problem(),
            initial_snapshot=_snapshot_from(
                _fake_environment("test" * 10), gpu_a1_available=False
            ),
            policy=freeze({"schema": "t/2"}),
            objective=_test_objective(),
            evidence_catalog=_test_evidence_catalog(),
            compiler=lambda evaluation: _frozen_plan(),
            realizer=realizer,
            transition_strategy=_StepStrategy(),
            transition_policy=lambda old, new, event: {"authorize": False},
        )

    def test_wrong_session_result_rejected_no_ledger_mutation(self):
        harness = _AgentHarness(_SessionSpoofRuntime())
        try:
            controller = self._controller_with(harness)
            emitted: list[int] = []
            with pytest.raises(RemoteRealizationError) as excinfo:
                controller.serve_tokens(
                    session_id=1,
                    prompt_token_ids=[5, 6],
                    max_new_tokens=2,
                    sampling_inputs={"temperature": 0.0},
                    on_token=lambda step, token, commit: emitted.append(token),
                )
            assert "session" in str(excinfo.value)
            assert emitted == []
            report = controller.report()
            sessions = report["sessions"]
            assert len(sessions) == 1
            # No committed ledger mutation: the failed session committed
            # nothing and emitted nothing.
            assert sessions[0]["generated_token_ids"] == []
            assert sessions[0]["committed_epoch_ids"] == []
            assert sessions[0]["latest_committed_position"] == 0
            assert sessions[0].get("failed") is True
            controller.close()
        finally:
            harness._listener.close()

    def test_wrong_position_result_rejected_no_ledger_mutation(self):
        harness = _AgentHarness(FakeRuntime())
        try:
            controller = self._controller_with(harness)

            # Forge a wrong position echo at the wire-response level: the
            # frame the Coordinator receives is syntactically valid, honest
            # in epoch/generation/realization/plan/session, and echoes a
            # position that does not match the request just sent.  The
            # controller path (not just the unit path) must fail closed
            # before commit.
            import freetoken.research.xc_coordinator as xc_coordinator_module

            original_recv = xc_coordinator_module.recv_frame

            def forged_recv(sock):
                body = original_recv(sock)
                if (
                    body.get("kind") == "response"
                    and body.get("operation") == "GENERATE"
                    and body.get("ok")
                ):
                    body = dict(body)
                    body["position"] = 99
                return body

            xc_coordinator_module.recv_frame = forged_recv
            emitted: list[int] = []
            try:
                with pytest.raises(RemoteRealizationError) as excinfo:
                    controller.serve_tokens(
                        session_id=1,
                        prompt_token_ids=[5, 6],
                        max_new_tokens=2,
                        sampling_inputs={"temperature": 0.0},
                        on_token=lambda step, token, commit: emitted.append(token),
                    )
            finally:
                xc_coordinator_module.recv_frame = original_recv
            assert "position" in str(excinfo.value)
            assert emitted == []
            report = controller.report()
            sessions = report["sessions"]
            assert len(sessions) == 1
            assert sessions[0]["generated_token_ids"] == []
            assert sessions[0]["committed_epoch_ids"] == []
            assert sessions[0]["latest_committed_position"] == 0
            assert sessions[0].get("failed") is True
            controller.close()
        finally:
            harness._listener.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
