"""R6 chain-side client for the remote last stage (R4 wire).

Speaks the accepted r4_wire framing to last_stage_service.py on the remote
compute node; presents the same request/response surface as a local
StageClient so GemmaStageChainRuntime can compose it as the chain tail.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Any

from freetoken.research.r4_wire import (
    encode_frame,
    payload_checksum,
    recv_frame,
    send_exact,
)

WIRE_PROTOCOL_ID = "inferswarm.r4.boundary-wire/1"
ROW_WIDTH = 3840


class RemoteLastStageClient:
    """Wire client for the dense chain's remote last stage."""

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
            encode_frame(
                {
                    "kind": "hello",
                    "protocol": WIRE_PROTOCOL_ID,
                    "experiment_id": experiment_id,
                    "session_id": self._session_id,
                }
            ),
        )
        header, _payload = recv_frame(self._sock, self._identity)
        if header.get("op") != "HELLO_ACK":
            raise RuntimeError(f"last-stage hello rejected: {header}")

    def send(self, message):
        # StageClient API compatibility: no-op (wire is synchronous).
        pass

    def recv(self):
        raise RuntimeError("RemoteLastStageClient is request/response only")

    def _boundary(
        self, *, operation: str, position: int, hidden
    ) -> dict[str, Any]:
        import torch

        token_count = int(hidden.shape[0])
        payload = (
            hidden.detach()
            .contiguous()
            .view(torch.uint8)
            .cpu()
            .numpy()
            .tobytes()
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
        send_exact(self._sock, encode_frame(header, payload))
        response, _ = recv_frame(self._sock, self._identity)
        if response.get("op") != "TOKEN_RESULT":
            raise RuntimeError(f"last-stage rejected boundary: {response}")
        return response

    def request(self, message):
        op = message["op"]
        if op == "PREFILL":
            response = self._boundary(
                operation="prefill",
                position=int(message["position"]),
                hidden=message["hidden"],
            )
            return {"op": "TOKEN_RESULT", "token_id": response["token_id"]}
        if op == "DECODE":
            response = self._boundary(
                operation="decode",
                position=int(message["position"]),
                hidden=message["hidden"],
            )
            return {"op": "TOKEN_RESULT", "token_id": response["token_id"]}
        if op == "RESET":
            # Session reset re-hellos over the same connection.
            send_exact(
                self._sock,
                encode_frame(
                    {
                        "kind": "hello",
                        "protocol": WIRE_PROTOCOL_ID,
                        "experiment_id": self._identity["experiment_id"],
                        "session_id": self._session_id,
                    }
                ),
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


__all__ = ["RemoteLastStageClient"]
