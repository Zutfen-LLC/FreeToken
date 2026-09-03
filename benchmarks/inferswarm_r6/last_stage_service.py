"""R6 remote last-stage service (inferswarm03) over the accepted R4 wire.

Serves the LAST stage of a dense Gemma chain: receives framed boundary
activations (single-plane bf16 hidden state, plane-major-contiguous, row
width 3840), executes the stage's owned decoder layers + final norm +
tied lm_head, and returns the framed token result.  Reuses the accepted
r4_wire framing/checksum/identity machinery unchanged; only the boundary
contract row width and the resident runtime differ (dense Gemma vs. MoE
Qwen).

Research-internal: not a public daemon API.  Fail-closed; no retries.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
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
MAX_TOKEN_COUNT = 64  # matches strategy PREFILL_CHUNK (single-chunk replays)
ROW_WIDTH = 3840
BOUNDARY_CONTRACT = {
    "dtype": "bfloat16",
    "layout": "plane-major-contiguous",
    "planes": 1,
    "row_width": ROW_WIDTH,
    "element_bytes": 2,
    "max_token_count": MAX_TOKEN_COUNT,
}


def serve(
    *,
    listen_host: str,
    listen_port: int,
    participant_plan: str,
    model_path: str,
    gpu_uuid: str,
    diagnostic: bool,
    ready_file: str | None = None,
    capture_logits: bool = False,
    allow_producer: str | None = None,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    import torch

    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.rotary import set_rope_device

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    set_rope_device(torch.device("cuda:0"))

    repo_root = Path(__file__).resolve().parents[2]
    running_sha = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo_root}", "-C", str(repo_root),
         "rev-parse", "HEAD"],
        text=True,
    ).strip()
    plan = json.loads(Path(participant_plan).read_text())
    plan_producer = plan.get("provenance", {}).get("r6", {}).get("producer_sha")
    if plan_producer and running_sha != plan_producer:
        if allow_producer and allow_producer == running_sha:
            # Explicit evidence-arm requalification: a NEW producer runs the
            # same frozen plan for a focused capture the frozen producer
            # cannot produce (e.g. secondary-comparator logits).  Never
            # silent: the mode is recorded in every report this process
            # emits, and the canonical run never uses this path.
            producer_check = {
                "mode": "EXPLICIT_OVERRIDE_EVIDENCE_ARM",
                "plan_frozen_producer": plan_producer,
                "running_producer": running_sha,
                "reason": "focused evidence capture not implemented by the "
                          "frozen producer; distinct requalification "
                          "producer recorded honestly",
            }
        else:
            raise RuntimeError(
                f"last-stage running producer {running_sha!r} != plan's frozen "
                f"producer {plan_producer!r}; canonical execution refuses to "
                f"proceed (an evidence arm may pass --allow-producer "
                f"{running_sha!r} to record an explicit override)"
            )
    else:
        producer_check = {
            "mode": "PLAN_FROZEN",
            "plan_frozen_producer": plan_producer,
            "running_producer": running_sha,
        }
    from benchmarks.inferswarm_r6.stage_runtime import GemmaDenseStage, LogitCapture

    block = plan["blocks"][-1]
    runtime = GemmaDenseStage(
        role="last",
        model_path=model_path,
        adapter_data={
            **block,
            "runtime_capacity_tokens": int(
                plan.get("runtime_capacity_tokens", 256)
            ),
            "declared_shared_state": plan.get("declared_shared_state"),
        },
    )
    if capture_logits:
        runtime._logit_capture = LogitCapture()

    buffer_bytes = MAX_TOKEN_COUNT * ROW_WIDTH * 2
    host_u8 = torch.empty(buffer_bytes, dtype=torch.uint8)
    stats = {"boundaries_served": 0, "activation_bytes_rx": 0, "result_bytes_tx": 0}

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((listen_host, listen_port))
    server.listen(1)
    if ready_file:
        Path(ready_file).write_text(
            json.dumps(
                {
                    "plan_digest": plan.get("digest"),
                    "stage": "last",
                    "gpu_uuid": gpu_uuid,
                    "listen": [listen_host, listen_port],
                    "diagnostic": diagnostic,
                    "pid": os.getpid(),
                    "producer_freetoken_sha": running_sha,
                    "producer_check": producer_check,
                    "runtime": runtime.report("P4_ready_for_resident_execution"),
                }
            )
        )
    experiment_id = plan.get("digest") or "r6-dense-chain"
    try:
        conn, _addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        session_id = None
        while True:
            identity = {"protocol": WIRE_PROTOCOL_ID, "experiment_id": experiment_id}
            header, payload = recv_frame(conn, identity)
            kind = header.get("kind")
            if kind == "hello":
                session_id = header["session_id"]
                runtime.reset_session_state()
                response = {
                    "kind": "response",
                    "protocol": WIRE_PROTOCOL_ID,
                    "experiment_id": experiment_id,
                    "session_id": session_id,
                    "op": "HELLO_ACK",
                    "runtime_ready": True,
                }
                send_exact(conn, encode_frame(response))
                continue
            if kind != "request" or session_id is None:
                raise WireError(f"unexpected frame kind {kind!r} before session")
            if header.get("session_id") != session_id:
                raise WireError("session identity mismatch")
            op = header["op"]
            if op == "OPEN_SESSION":
                runtime.reset_session_state()
                response = {
                    "kind": "response",
                    "protocol": WIRE_PROTOCOL_ID,
                    "experiment_id": experiment_id,
                    "session_id": session_id,
                    "op": "SESSION_ACK",
                }
                send_exact(conn, encode_frame(response))
                continue
            if op != "BOUNDARY":
                raise WireError(f"unsupported request op {op!r}")
            token_count = int(header["token_count"])
            validate_request(
                header,
                contract=BOUNDARY_CONTRACT,
                checksum=payload_checksum(payload) if diagnostic else None,
                payload=payload,
            )
            hidden = (
                torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                .view(torch.bfloat16)
                .reshape(token_count, ROW_WIDTH)
                .to(device="cuda:0", non_blocking=False)
            )
            if header["operation"] == "prefill":
                token, _logits = runtime.prefill(
                    None, hidden, int(header["position"])
                )
            else:
                if token_count != 1:
                    raise WireError("decode boundary requires exactly one token")
                token, _logits = runtime.decode(hidden, int(header["position"]))
            stats["boundaries_served"] += 1
            stats["activation_bytes_rx"] += len(payload)
            response = {
                "kind": "response",
                "protocol": WIRE_PROTOCOL_ID,
                "experiment_id": experiment_id,
                "session_id": session_id,
                "op": "TOKEN_RESULT",
                "token_id": int(token),
                "consumer_sha256": payload_checksum(payload),
                "compute_ns": 0,
            }
            frame = encode_frame(response)
            send_exact(conn, frame)
            stats["result_bytes_tx"] += len(frame)
            del hidden
    finally:
        try:
            capture_payload = None
            if runtime._logit_capture is not None:
                capture_payload = {
                    "steps_captured": sorted(
                        runtime._logit_capture.steps.keys()
                    ),
                    "nan_inf_count": runtime._logit_capture.nan_inf_count,
                    "logits_by_step": {
                        str(step): values
                        for step, values in runtime._logit_capture.steps.items()
                    },
                }
            Path(
                os.environ.get("R6_LAST_STAGE_FINAL_REPORT", "/tmp/r6-last-stage.json")
            ).write_text(
                json.dumps(
                    {
                        "schema": "inferswarm.r6.last-stage-final-report/1",
                        "plan_digest": plan.get("digest"),
                        "producer_freetoken_sha": running_sha,
                        "producer_check": producer_check,
                        "stats": stats,
                        "logit_capture": capture_payload,
                        "runtime": runtime.report("P5_post_run"),
                    }
                )
            )
        except Exception:  # noqa: BLE001
            pass
        server.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=18485)
    parser.add_argument("--participant-plan", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--ready-file")
    parser.add_argument("--capture-logits", action="store_true",
                        help="retain full-vocab last-stage logits per served "
                        "step in the final report (comparator arm only)")
    parser.add_argument("--allow-producer", default=None,
                        help="explicit evidence-arm override: record this "
                        "exact running producer instead of the plan's frozen "
                        "producer (never valid for a canonical run)")
    args = parser.parse_args(argv)
    serve(
        listen_host=args.listen_host,
        listen_port=int(args.listen_port),
        participant_plan=args.participant_plan,
        model_path=args.model,
        gpu_uuid=args.gpu_uuid,
        diagnostic=bool(args.diagnostic),
        ready_file=args.ready_file,
        capture_logits=bool(args.capture_logits),
        allow_producer=args.allow_producer,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
