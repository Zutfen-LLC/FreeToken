from __future__ import annotations

import itertools
import time

from freetoken.research.n1_local_boundary import (
    BF16_DTYPE_CODE,
    FLAG_PREFILL_FINAL,
    Frame,
    HIDDEN_SIZE,
    MessageType,
    SessionStateMachine,
    UnixSocketTransport,
    decode_token_result,
)

from .service import expected_handshake


class N1BlockClient:
    def __init__(self, transport: UnixSocketTransport):
        self.transport = transport
        self.state = SessionStateMachine()
        self.step_ids = itertools.count()
        self.timings: list[dict] = []

    @classmethod
    def connect(cls, socket_path: str) -> "N1BlockClient":
        client = cls(UnixSocketTransport.connect(socket_path))
        hello = Frame(MessageType.HELLO, payload=expected_handshake().encode())
        client.transport.send(hello)
        response = client.transport.receive()
        if response.message_type is not MessageType.HELLO_ACK:
            raise RuntimeError(f"expected HELLO_ACK, got {response.message_type.name}")
        from freetoken.research.n1_local_boundary import Handshake
        Handshake.decode(response.payload).require_exact(expected_handshake())
        return client

    def _round_trip(self, frame: Frame) -> Frame:
        started = time.perf_counter_ns()
        self.transport.send(frame)
        sent = time.perf_counter_ns()
        response = self.transport.receive()
        received = time.perf_counter_ns()
        if response.message_type is MessageType.ERROR:
            raise RuntimeError(response.payload.decode("utf-8", errors="replace"))
        self.timings.append({
            "session_id": frame.session_id,
            "step_id": frame.step_id,
            "message_type": frame.message_type.name,
            "frame_bytes": 60 + len(frame.payload),
            "send_ns": sent - started,
            "round_trip_ns": received - started,
        })
        return response

    def open(self, session_id: int) -> None:
        self.state.open(session_id)
        self.step_ids = itertools.count()
        response = self._round_trip(Frame(MessageType.OPEN_SESSION, session_id=session_id))
        if response.message_type is not MessageType.ACK:
            raise RuntimeError("OPEN_SESSION was not acknowledged")

    def hidden(
        self, *, session_id: int, operation: MessageType, position: int,
        token_count: int, payload: bytes, final_prefill: bool = False,
    ) -> int | None:
        flags = FLAG_PREFILL_FINAL if final_prefill else 0
        frame = Frame(
            operation, session_id, next(self.step_ids), position, token_count,
            1, HIDDEN_SIZE, BF16_DTYPE_CODE, flags, payload,
        )
        self.state.accept_hidden(frame)
        response = self._round_trip(frame)
        if final_prefill or operation is MessageType.DECODE:
            if response.message_type is not MessageType.TOKEN_RESULT:
                raise RuntimeError("expected TOKEN_RESULT")
            return decode_token_result(response.payload)
        if response.message_type is not MessageType.ACK:
            raise RuntimeError("prefill chunk was not acknowledged")
        return None

    def reset(self, session_id: int) -> None:
        response = self._round_trip(Frame(MessageType.RESET_SESSION, session_id=session_id))
        if response.message_type is not MessageType.ACK:
            raise RuntimeError("RESET_SESSION was not acknowledged")
        self.state.reset(session_id)
        self.step_ids = itertools.count()

    def close(self, session_id: int) -> None:
        response = self._round_trip(Frame(MessageType.CLOSE_SESSION, session_id=session_id))
        if response.message_type is not MessageType.ACK:
            raise RuntimeError("CLOSE_SESSION was not acknowledged")
        self.state.close(session_id)

    def close_transport(self) -> None:
        self.transport.close()


__all__ = ["N1BlockClient"]
