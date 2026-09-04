"""#76 wire client: R6 RemoteLastStageClient + case-scoped ops."""

from __future__ import annotations

from typing import Any

from freetoken.research.r4_wire import (
    WIRE_PROTOCOL_ID,
    encode_frame,
    recv_frame,
    send_exact,
)


class I76LastStageClient:
    role = "last"

    def __init__(
        self,
        *,
        host: str,
        port: int,
        experiment_id: str,
        session_id: int = 1,
        connect_timeout: float = 30.0,
    ) -> None:
        import socket

        self._host = host
        self._port = int(port)
        self._identity = {
            "protocol": WIRE_PROTOCOL_ID,
            "experiment_id": experiment_id,
        }
        self._session_id = int(session_id)
        self._sock = socket.create_connection(
            (host, int(port)), timeout=connect_timeout
        )
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send_exact(
            self._sock,
            encode_frame({
                "kind": "hello",
                "protocol": WIRE_PROTOCOL_ID,
                "experiment_id": experiment_id,
                "session_id": self._session_id,
            }),
        )
        header, _payload = recv_frame(self._sock, self._identity)
        if header.get("op") != "HELLO_ACK":
            raise RuntimeError(f"last-stage hello rejected: {header}")

    def send(self, message):
        pass  # StageClient API compatibility

    def recv(self):
        raise RuntimeError("I76LastStageClient is request/response only")

    def _control(self, op: str, **fields: Any) -> dict:
        header = {
            "kind": "request",
            "protocol": WIRE_PROTOCOL_ID,
            "experiment_id": self._identity["experiment_id"],
            "session_id": self._session_id,
            "op": op,
            **fields,
        }
        send_exact(self._sock, encode_frame(header))
        response, _ = recv_frame(self._sock, self._identity)
        return response

    def _boundary(
        self, *, operation: str, position: int, hidden, capture_step: int | None = None
    ) -> dict:
        import torch

        from freetoken.research.r4_wire import payload_checksum

        token_count = int(hidden.shape[0])
        payload = (
            hidden.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        )
        header = {
            "kind": "request",
            "protocol": WIRE_PROTOCOL_ID,
            "experiment_id": self._identity["experiment_id"],
            "session_id": self._session_id,
            "op": "BOUNDARY",
            "operation": operation,
            "position": int(position),
            "token_count": token_count,
            "dtype": "bfloat16",
            "layout": "plane-major-contiguous",
            "payload_len": len(payload),
            "payload_sha256": payload_checksum(payload),
        }
        if capture_step is not None:
            header["capture_step"] = int(capture_step)
        send_exact(self._sock, encode_frame(header, payload))
        response, _ = recv_frame(self._sock, self._identity)
        if response.get("op") != "TOKEN_RESULT":
            raise RuntimeError(f"last-stage rejected boundary: {response}")
        return response

    def request(self, message):
        op = message["op"]
        if op == "PREFILL":
            return self._boundary(
                operation="prefill",
                position=int(message["position"]),
                hidden=message["hidden"],
                capture_step=message.get("capture_step"),
            )
        if op == "DECODE":
            return self._boundary(
                operation="decode",
                position=int(message["position"]),
                hidden=message["hidden"],
            )
        if op == "CASE_BEGIN":
            response = self._control("CASE_BEGIN",
                                     case_id=message["case_id"])
            if response.get("op") != "CASE_ACK":
                raise RuntimeError(f"CASE_BEGIN rejected: {response}")
            return response
        if op == "CASE_SAVE":
            response = self._control("CASE_SAVE", tag=message["tag"])
            if response.get("op") != "SAVE_ACK":
                raise RuntimeError(f"CASE_SAVE rejected: {response}")
            return response
        if op == "RESET":
            send_exact(
                self._sock,
                encode_frame({
                    "kind": "hello",
                    "protocol": WIRE_PROTOCOL_ID,
                    "experiment_id": self._identity["experiment_id"],
                    "session_id": self._session_id,
                }),
            )
            header, _ = recv_frame(self._sock, self._identity)
            if header.get("op") != "HELLO_ACK":
                raise RuntimeError(f"last-stage re-hello rejected: {header}")
            return {"op": "ACK"}
        if op == "REPORT":
            return {"op": "REPORT", "report": {"remote": True}}
        if op == "SHUTDOWN":
            try:
                self._sock.close()
            except OSError:
                pass
            return {"op": "ACK"}
        raise RuntimeError(f"unsupported wire op {op!r}")

    def shutdown(self):
        try:
            self.request({"op": "SHUTDOWN"})
        except Exception:  # noqa: BLE001
            try:
                self._sock.close()
            except OSError:
                pass
