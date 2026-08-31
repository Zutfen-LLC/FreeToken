from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import torch

from freetoken.research.n0_model_block import write_json_with_sha
from freetoken.research.n1_local_boundary import (
    BF16_DTYPE_CODE,
    BoundaryProtocolError,
    FLAG_PREFILL_FINAL,
    Frame,
    Handshake,
    HIDDEN_SIZE,
    MessageType,
    SessionStateMachine,
    UnixSocketTransport,
    encode_token_result,
)

from .runtime import N1BlockRuntime


MODEL_REPOSITORY = "nvidia/Qwen3.6-35B-A3B-NVFP4"
MODEL_REVISION = "491c2f1ea524c639598bf8fa787a93fed5a6fbce"
N0_PLAN_SHA256 = "a00c87e85cf60178e30e167117cc621de5cc8d9463e06f1abda25511b13a57ae"
PROCESS_A_UUID = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
PROCESS_B_UUID = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"
LOGIT_CHECKPOINTS = frozenset((0, 1, 15, 31))


def expected_handshake() -> Handshake:
    return Handshake(
        MODEL_REPOSITORY, MODEL_REVISION, N0_PLAN_SHA256,
        0, 19, 19, 40, HIDDEN_SIZE, BF16_DTYPE_CODE,
        PROCESS_A_UUID, PROCESS_B_UUID,
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


class FinalBlockService:
    def __init__(self, runtime: N1BlockRuntime, *, report_out: str):
        self.runtime = runtime
        self.report_out = report_out
        self.state = SessionStateMachine()
        self.handshake_record: dict | None = None
        self.boundaries: list[dict] = []
        self.logit_checkpoints: list[dict] = []
        self.timings: list[dict] = []
        self.sessions_closed = 0
        self.generated_step = 0

    def _ack(self, frame: Frame) -> Frame:
        return Frame(MessageType.ACK, frame.session_id, frame.step_id)

    def _token(self, request: Frame, token_id: int) -> Frame:
        return Frame(
            MessageType.TOKEN_RESULT,
            request.session_id,
            request.step_id,
            request.absolute_start_position,
            payload=encode_token_result(token_id),
        )

    def _record_logits(self, logits: torch.Tensor, session_id: int) -> None:
        if self.generated_step in LOGIT_CHECKPOINTS:
            values = logits.detach().float()
            self.logit_checkpoints.append({
                "generated_step": self.generated_step,
                "session_id": session_id,
                "shape": list(values.shape),
                "float32_sha256": _tensor_sha256(values),
                "argmax": int(values.argmax(dim=-1).item()),
                "nan_count": int(torch.isnan(values).sum().item()),
                "inf_count": int(torch.isinf(values).sum().item()),
                "full_logits": values.tolist(),
            })
        self.generated_step += 1

    def _hidden(self, request: Frame) -> Frame:
        self.state.accept_hidden(request)
        started = time.perf_counter_ns()
        sha_before = hashlib.sha256(request.payload).hexdigest()
        hidden, residual = self.runtime.receive_boundary(
            request.payload, request.token_count
        )
        reconstructed = self.runtime.boundary_payload(hidden, residual)
        sha_after = hashlib.sha256(reconstructed).hexdigest()
        if sha_after != sha_before:
            raise BoundaryProtocolError("reconstructed boundary SHA-256 mismatch")
        received_ns = time.perf_counter_ns()
        if request.message_type is MessageType.PREFILL:
            token, logits = self.runtime.prefill_b(
                hidden, residual, start=request.absolute_start_position
            )
            if request.flags & FLAG_PREFILL_FINAL:
                self.runtime.populate_all_experts()
                self._record_logits(logits, request.session_id)
                response = self._token(request, token)
            else:
                response = self._ack(request)
        else:
            token, logits = self.runtime.decode_b(
                hidden, residual, position=request.absolute_start_position
            )
            self._record_logits(logits, request.session_id)
            response = self._token(request, token)
        computed_ns = time.perf_counter_ns()
        self.boundaries.append({
            "session_id": request.session_id,
            "step_id": request.step_id,
            "operation": request.message_type.name,
            "absolute_start_position": request.absolute_start_position,
            "token_count": request.token_count,
            "dtype": "torch.bfloat16",
            "shape": [2, request.token_count, HIDDEN_SIZE],
            "payload_bytes": len(request.payload),
            "sha256_before": sha_before,
            "sha256_after": sha_after,
        })
        self.timings.append({
            "session_id": request.session_id,
            "step_id": request.step_id,
            "receive_and_h2d_ns": received_ns - started,
            "b_compute_and_argmax_ns": computed_ns - received_ns,
        })
        return response

    def _write_report(self) -> None:
        write_json_with_sha(self.report_out, {
            "schema": "inferswarm.n1.final-block-service/1",
            "handshake": self.handshake_record,
            "runtime": self.runtime.report(),
            "boundaries": self.boundaries,
            "local_logit_checkpoints": self.logit_checkpoints,
            "timings": self.timings,
            "sessions_closed": self.sessions_closed,
            "state_bytes_transferred": 0,
            "full_logits_transferred": 0,
        })

    def serve_connection(self, transport: UnixSocketTransport) -> None:
        hello = transport.receive()
        if hello.message_type is not MessageType.HELLO:
            raise BoundaryProtocolError("first frame must be HELLO")
        peer = Handshake.decode(hello.payload)
        peer.require_exact(expected_handshake())
        self.handshake_record = {
            "protocol_version": 1,
            "peer": peer.__dict__,
            "local": expected_handshake().__dict__,
            "passed": True,
            "process_b_pid": os.getpid(),
        }
        transport.send(Frame(MessageType.HELLO_ACK, payload=expected_handshake().encode()))
        while True:
            request = transport.receive()
            if request.message_type is MessageType.OPEN_SESSION:
                self.state.open(request.session_id)
                self.runtime.reset_session_state()
                self.generated_step = 0
                transport.send(self._ack(request))
            elif request.message_type in (MessageType.PREFILL, MessageType.DECODE):
                transport.send(self._hidden(request))
            elif request.message_type is MessageType.RESET_SESSION:
                self.state.reset(request.session_id)
                self.runtime.reset_session_state()
                self.generated_step = 0
                transport.send(self._ack(request))
            elif request.message_type is MessageType.CLOSE_SESSION:
                self.state.close(request.session_id)
                self.runtime.reset_session_state()
                self.sessions_closed += 1
                transport.send(self._ack(request))
                self._write_report()
            else:
                raise BoundaryProtocolError(
                    f"unsupported service message {request.message_type.name}"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--ready-file", required=True)
    args = parser.parse_args(argv)
    socket_path = Path(args.socket)
    if socket_path.exists():
        socket_path.unlink()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(1)
    runtime = N1BlockRuntime(
        role="b", model_path=args.model, plan_path=args.plan,
        device_index=args.device_index,
    )
    Path(args.ready_file).write_text(json.dumps({"pid": os.getpid()}) + "\n")
    connection, _ = listener.accept()
    transport = UnixSocketTransport(connection)
    service = FinalBlockService(runtime, report_out=args.report_out)
    try:
        service.serve_connection(transport)
    except (BoundaryProtocolError, EOFError) as exc:
        try:
            transport.send(Frame(MessageType.ERROR, payload=str(exc).encode("utf-8")))
        except Exception:
            pass
        service._write_report()
        if "truncated frame" not in str(exc):
            raise
    finally:
        transport.close()
        listener.close()
        if socket_path.exists():
            socket_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
