"""Node A coordinator for the R4 two-node boundary.

Runs on inferswarm01.  Hosts exactly one resident captured Block A runtime
(accepted R2 substrate, unchanged) and one persistent full-duplex TCP
connection to the Node B service.  For each boundary: Block A compute, D2H
into a bounded registered host buffer, framed send of the exact semantic
activation bytes, then receive of the small framed result.

Research-internal; fail-closed on any wire violation.  All timing uses
coordinator-local clocks only (no cross-host one-way arithmetic).
"""

from __future__ import annotations

import json
import socket
import statistics
import time
from pathlib import Path

from freetoken.research.r4_wire import (
    WireError,
    encode_frame,
    payload_checksum,
    recv_frame,
    send_exact,
    validate_request,
)

WIRE_PROTOCOL_ID = "inferswarm.r4.boundary-wire/1"
BOUNDARY_CONTRACT = {
    "dtype": "bfloat16",
    "layout": "plane-major-contiguous",
    "planes": 2,
    "row_width": 2048,
    "element_bytes": 2,
    "max_token_count": 64,
}


class NodeACoordinator:
    def __init__(
        self,
        *,
        plan: dict,
        model_path: str,
        peer_host: str,
        peer_port: int,
        diagnostic: bool,
        runtime,
        host_buffer,
    ) -> None:
        import torch

        self.torch = torch
        self.plan = plan
        self.diagnostic = diagnostic
        self.runtime = runtime
        self.host_u8 = host_buffer
        self.experiment_id = plan["digest"]
        self.session_id = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        deadline = time.time() + 60
        self.sock.settimeout(60)
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                self.sock.connect((peer_host, peer_port))
                break
            except OSError as exc:
                last_error = exc
                time.sleep(1.0)
        else:
            raise WireError(f"cannot connect to Node B service: {last_error}")
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.tcp_nodelay = bool(
            self.sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
        )
        self.accounting = {
            "frames_tx": 0,
            "frames_rx": 0,
            "wire_bytes_tx": 0,
            "wire_bytes_rx": 0,
            "semantic_payload_bytes_tx": 0,
            "framing_control_bytes_tx": 0,
            "result_control_bytes_rx": 0,
        }

    # -- session -------------------------------------------------------
    def hello(self, session_id: int) -> None:
        header = {
            "kind": "hello",
            "protocol": WIRE_PROTOCOL_ID,
            "experiment_id": self.experiment_id,
            "session_id": session_id,
            "payload_len": 0,
        }
        frame = encode_frame(header)
        send_exact(self.sock, frame)
        self._account_tx(frame, 0)
        reply, _ = recv_frame(
            self.sock,
            {"protocol": WIRE_PROTOCOL_ID, "experiment_id": self.experiment_id},
        )
        self._account_rx(encode_frame(reply))
        if reply.get("op") != "HELLO_ACK":
            raise WireError(f"unexpected hello reply {reply.get('op')!r}")
        if not reply.get("runtime_ready"):
            raise WireError("Node B reported runtime not ready")
        self.session_id = session_id

    def open_session(self) -> None:
        header = {
            "kind": "request",
            "protocol": WIRE_PROTOCOL_ID,
            "experiment_id": self.experiment_id,
            "session_id": self.session_id,
            "op": "OPEN_SESSION",
            "payload_len": 0,
        }
        frame = encode_frame(header)
        send_exact(self.sock, frame)
        self._account_tx(frame, 0)
        reply, _ = recv_frame(
            self.sock,
            {
                "protocol": WIRE_PROTOCOL_ID,
                "experiment_id": self.experiment_id,
                "session_id": self.session_id,
            },
        )
        self._account_rx(encode_frame(reply))
        if reply.get("op") != "SESSION_ACK":
            raise WireError(f"unexpected session reply {reply.get('op')!r}")

    # -- boundary ------------------------------------------------------
    def boundary_step(
        self,
        *,
        operation: str,
        token_ids: list[int],
        position: int,
        capture_logits: bool = False,
        capture_state: bool = False,
    ) -> tuple[int, dict]:
        from benchmarks.inferswarm_r2.qwen_split_adapter import (
            BOUNDARY_PLANES,
            HIDDEN_SIZE,
        )
        from freetoken.research.r2_local_split import validate_boundary_payload

        torch = self.torch
        token_count = len(token_ids)
        step_started = time.perf_counter_ns()
        compute_started = time.perf_counter_ns()
        if operation == "prefill":
            hidden, residual = self.runtime.prefill_a(token_ids, position)
        else:
            if token_count != 1:
                raise WireError("decode requires exactly one token")
            hidden, residual = self.runtime.decode_a(token_ids[0], position)
        torch.cuda.synchronize(0)
        compute_ended = time.perf_counter_ns()
        payload_bytes = token_count * BOUNDARY_PLANES * HIDDEN_SIZE * 2
        validate_boundary_payload(
            self.plan,
            producer_execution_id="exec.block-a",
            consumer_execution_id="exec.block-b",
            dtype="bfloat16",
            layout="plane-major-contiguous",
            token_count=token_count,
            payload_bytes=payload_bytes,
        )
        host = (
            self.host_u8[:payload_bytes]
            .view(torch.bfloat16)
            .reshape(BOUNDARY_PLANES, token_count, HIDDEN_SIZE)
        )
        d2h_started = time.perf_counter_ns()
        host[0].copy_(hidden, non_blocking=True)
        host[1].copy_(residual, non_blocking=True)
        torch.cuda.synchronize(0)
        d2h_ended = time.perf_counter_ns()
        payload = bytes(self.host_u8[:payload_bytes])
        checksum = payload_checksum(payload) if self.diagnostic else None
        validate_request(
            {
                "kind": "request",
                "operation": operation,
                "position": position,
                "token_count": token_count,
                "dtype": "bfloat16",
                "layout": "plane-major-contiguous",
                "payload_len": payload_bytes,
            },
            contract=BOUNDARY_CONTRACT,
            payload=payload,
        )
        header = {
            "kind": "request",
            "protocol": WIRE_PROTOCOL_ID,
            "experiment_id": self.experiment_id,
            "session_id": self.session_id,
            "op": "BOUNDARY",
            "operation": operation,
            "position": position,
            "token_count": token_count,
            "dtype": "bfloat16",
            "layout": "plane-major-contiguous",
            "payload_len": payload_bytes,
            "capture_logits": bool(capture_logits and self.diagnostic),
            "capture_state": bool(capture_state and self.diagnostic),
        }
        if checksum is not None:
            header["payload_sha256"] = checksum
        frame = encode_frame(header, payload)
        send_started = time.perf_counter_ns()
        send_exact(self.sock, frame)
        send_ended = time.perf_counter_ns()
        self._account_tx(frame, payload_bytes)
        reply, _ = recv_frame(
            self.sock,
            {
                "protocol": WIRE_PROTOCOL_ID,
                "experiment_id": self.experiment_id,
                "session_id": self.session_id,
            },
        )
        reply_ended = time.perf_counter_ns()
        self._account_rx(encode_frame(reply))
        if reply.get("op") != "TOKEN_RESULT":
            raise WireError(f"unexpected boundary reply {reply.get('op')!r}")
        if self.diagnostic and reply.get("consumer_sha256") != checksum:
            raise WireError("producer checksum != receiver checksum")
        record = {
            "operation": operation,
            "position": position,
            "token_count": token_count,
            "payload_bytes": payload_bytes,
            "producer_sha256": checksum,
            "consumer_sha256": reply.get("consumer_sha256"),
            "block_a_compute_ns": compute_ended - compute_started,
            "d2h_ns": d2h_ended - d2h_started,
            "sender_send_ns": send_ended - send_started,
            "h2d_ns": reply.get("h2d_ns"),
            "block_b_compute_ns": reply.get("compute_ns"),
            "receiver_service_ns": reply.get("service_ns"),
            "step_wall_ns": reply_ended - step_started,
            "tcp_nodelay": self.tcp_nodelay,
        }
        if "logits" in reply:
            record["logits"] = reply["logits"]
        return int(reply["token_id"]), record

    # -- accounting ----------------------------------------------------
    def _account_tx(self, frame: bytes, semantic: int) -> None:
        self.accounting["frames_tx"] += 1
        self.accounting["wire_bytes_tx"] += len(frame)
        self.accounting["semantic_payload_bytes_tx"] += semantic
        self.accounting["framing_control_bytes_tx"] += len(frame) - semantic

    def _account_rx(self, frame: bytes) -> None:
        self.accounting["frames_rx"] += 1
        self.accounting["wire_bytes_rx"] += len(frame)
        self.accounting["result_control_bytes_rx"] += len(frame)

    def report(self) -> dict:
        return {
            "tcp_nodelay": self.tcp_nodelay,
            "experiment_id": self.experiment_id,
            **self.accounting,
        }

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def run_session(
    coordinator: NodeACoordinator,
    *,
    session_id: int,
    prompt_ids: list[int],
    max_new_tokens: int,
    prefill_chunk: int,
    capture_steps: set[int] | None = None,
    capture_prefill_state: bool = False,
) -> dict:
    """One generation session over the persistent connection (coordinator clock)."""

    capture_steps = capture_steps or set()
    coordinator.hello(session_id)
    coordinator.open_session()
    request_started = time.perf_counter_ns()
    boundaries: list[dict] = []
    generated: list[int] = []
    first_token_ns = None
    prefill_started = time.perf_counter_ns()
    for start in range(0, len(prompt_ids), prefill_chunk):
        chunk = prompt_ids[start : start + prefill_chunk]
        final = start + len(chunk) == len(prompt_ids)
        token, boundary = coordinator.boundary_step(
            operation="prefill",
            token_ids=chunk,
            position=start,
            capture_logits=final and 0 in capture_steps,
            capture_state=capture_prefill_state,
        )
        boundary["generated_step"] = len(generated)
        boundaries.append(boundary)
        if final:
            generated.append(token)
            first_token_ns = time.perf_counter_ns()
    prefill_ended = first_token_ns or time.perf_counter_ns()
    decode_started = time.perf_counter_ns()
    while len(generated) < max_new_tokens:
        step = len(generated)
        position = len(prompt_ids) + step - 1
        token, boundary = coordinator.boundary_step(
            operation="decode",
            token_ids=[generated[-1]],
            position=position,
            capture_logits=step in capture_steps,
        )
        generated.append(token)
        boundary["generated_step"] = step
        boundaries.append(boundary)
    decode_ended = time.perf_counter_ns()
    decode_latencies = [
        item["step_wall_ns"] for item in boundaries if item["operation"] == "decode"
    ]
    return {
        "session_id": session_id,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated,
        "boundaries": boundaries,
        "prompt_token_count": len(prompt_ids),
        "generated_token_count": len(generated),
        "prefill_wall_ns": prefill_ended - prefill_started,
        "ttft_ns": (first_token_ns or decode_started) - request_started,
        "decode_wall_ns": decode_ended - decode_started,
        "total_request_wall_ns": decode_ended - request_started,
        "decode_tokens_per_second": (max_new_tokens - 1)
        / ((decode_ended - decode_started) / 1e9),
        "inter_token_latency_ns": decode_latencies,
        "inter_token_p50_ns": statistics.median(decode_latencies),
        "boundary_semantic_bytes": sum(
            item["payload_bytes"] for item in boundaries
        ),
        "block_a_compute_ns": sum(
            item["block_a_compute_ns"] for item in boundaries
        ),
        "block_b_compute_ns": sum(
            item["block_b_compute_ns"] for item in boundaries
        ),
    }


__all__ = ["BOUNDARY_CONTRACT", "NodeACoordinator", "WIRE_PROTOCOL_ID", "run_session"]
