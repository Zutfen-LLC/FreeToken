"""#88 v3 remote last-stage service (inferswarm03) over the accepted R4 wire.

Identical framing, boundary contract, and execution semantics to the #76
last-stage service. Differences (all evidence-side, no execution math):

- at EVERY prefill the final-row FP32 consumer logits are retained as
  ``<out_dir>/<case_id>/decision-<capture_step>.f32`` (the #88 chain
  runner sets ``capture_step`` on every decision, so the file index IS
  the decision index) with sha256 + element count + a frozen-rule
  executor proof returned in the TOKEN_RESULT response;
- ``--allow-producer`` records the #88 frozen producer explicitly (the
  plan file carries the historical R6 producer; #88 is an authorized
  execution campaign under a NEW frozen producer derived from the same
  base).

The service speaks the same request ops as the #76 service:
  CASE_BEGIN {case_id}     -> swaps in a fresh sink for the case
  CASE_SAVE  {tag}         -> persists the case bundle, returns manifest
"""

from __future__ import annotations

import argparse
import hashlib
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
    allow_producer: str | None = None,
    out_dir: str,
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
         "rev-parse", "HEAD"], text=True,
    ).strip()
    plan = json.loads(Path(participant_plan).read_text())
    plan_producer = plan.get("provenance", {}).get("r6", {}).get("producer_sha")
    if plan_producer and running_sha != plan_producer:
        if allow_producer and allow_producer == running_sha:
            producer_check = {
                "mode": "EXPLICIT_OVERRIDE_ISSUE88_EXECUTION",
                "plan_frozen_producer": plan_producer,
                "running_producer": running_sha,
                "reason": "issue #88 authorized execution campaign under a "
                          "new frozen producer derived from the same base",
            }
        else:
            raise RuntimeError(
                f"last-stage running producer {running_sha!r} != plan's frozen "
                f"producer {plan_producer!r}; pass --allow-producer "
                f"{running_sha!r} for the #88 execution campaign"
            )
    else:
        producer_check = {
            "mode": "PLAN_FROZEN",
            "plan_frozen_producer": plan_producer,
            "running_producer": running_sha,
        }

    from benchmarks.inferswarm_76.capture import RowPruningSink, arm_full_capture
    from benchmarks.inferswarm_r6.stage_runtime import GemmaDenseStage

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

    sink: RowPruningSink | None = None
    case_dir_root = Path(out_dir)
    case_dir_root.mkdir(parents=True, exist_ok=True)
    current_case_dir: Path | None = None
    current_case_id: str | None = None

    def _arm(case_id: str) -> None:
        nonlocal sink, current_case_dir, current_case_id
        current_case_id = case_id
        current_case_dir = case_dir_root / case_id
        current_case_dir.mkdir(parents=True, exist_ok=True)
        sink = RowPruningSink(role="last", gpu_uuid=gpu_uuid)
        runtime._capture_sink = sink
        runtime._capture_after_layers = frozenset()

    buffer_bytes = MAX_TOKEN_COUNT * ROW_WIDTH * 2
    host_u8 = torch.empty(buffer_bytes, dtype=torch.uint8)
    stats = {"boundaries_served": 0, "activation_bytes_rx": 0,
             "result_bytes_tx": 0, "cases_saved": 0,
             "decision_rows_retained": 0}

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((listen_host, listen_port))
    server.listen(1)
    if ready_file:
        Path(ready_file).write_text(json.dumps({
            "plan_digest": plan.get("digest"),
            "stage": "last",
            "gpu_uuid": gpu_uuid,
            "listen": [listen_host, listen_port],
            "pid": os.getpid(),
            "producer_freetoken_sha": running_sha,
            "producer_check": producer_check,
            "runtime": runtime.report("P4_ready_for_resident_execution"),
        }))
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
                    "kind": "response", "protocol": WIRE_PROTOCOL_ID,
                    "experiment_id": experiment_id, "session_id": session_id,
                    "op": "HELLO_ACK", "runtime_ready": True,
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
                    "kind": "response", "protocol": WIRE_PROTOCOL_ID,
                    "experiment_id": experiment_id, "session_id": session_id,
                    "op": "SESSION_ACK",
                }
                send_exact(conn, encode_frame(response))
                continue
            if op == "CASE_BEGIN":
                _arm(header["case_id"])
                runtime.reset_session_state()
                response = {
                    "kind": "response", "protocol": WIRE_PROTOCOL_ID,
                    "experiment_id": experiment_id, "session_id": session_id,
                    "op": "CASE_ACK", "case_id": current_case_id,
                }
                send_exact(conn, encode_frame(response))
                continue
            if op == "CASE_SAVE":
                if sink is None or current_case_dir is None:
                    raise WireError("CASE_SAVE before CASE_BEGIN")
                manifest = sink.save(str(current_case_dir), header["tag"])
                stats["cases_saved"] += 1
                response = {
                    "kind": "response", "protocol": WIRE_PROTOCOL_ID,
                    "experiment_id": experiment_id, "session_id": session_id,
                    "op": "SAVE_ACK", "case_id": current_case_id,
                    "manifest": manifest,
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
            capture_step = header.get("capture_step")
            if capture_step is not None:
                runtime._capture_step = int(capture_step)
            if header["operation"] == "prefill":
                token, logits = runtime.prefill(None, hidden, int(header["position"]))
            else:
                if token_count != 1:
                    raise WireError("decode boundary requires exactly one token")
                token, logits = runtime.decode(hidden, int(header["position"]))
            runtime._capture_step = None
            row = logits
            nan_inf = int(torch.isnan(row).sum().item()
                          + torch.isinf(row).sum().item())
            top2 = torch.topk(row, 2)

            # --- #88 v3: retain EVERY decision's FP32 row + rule proof ---
            rule_proof = None
            row_sha = None
            row_count = None
            if header["operation"] == "prefill" and capture_step is not None:
                from benchmarks.inferswarm_88 import executor_rule_proof

                host_row = row.detach().to("cpu", torch.float32).contiguous()
                values = host_row.tolist()
                rule_proof = executor_rule_proof(values, int(token))
                row_bytes = host_row.view(torch.uint8).numpy().tobytes()
                row_sha = hashlib.sha256(row_bytes).hexdigest()
                row_count = len(values)
                if current_case_dir is None:
                    raise WireError("decision row before CASE_BEGIN")
                row_path = current_case_dir / f"decision-{int(capture_step)}.f32"
                if row_path.exists():
                    raise WireError(f"refusing to overwrite {row_path}")
                row_path.write_bytes(row_bytes)
                stats["decision_rows_retained"] += 1
                del host_row, values, row_bytes

            stats["boundaries_served"] += 1
            stats["activation_bytes_rx"] += len(payload)
            response = {
                "kind": "response", "protocol": WIRE_PROTOCOL_ID,
                "experiment_id": experiment_id, "session_id": session_id,
                "op": "TOKEN_RESULT",
                "token_id": int(token),
                "top1_index": int(top2.indices[0].item()),
                "top1_value_hex": float(top2.values[0].item()).hex(),
                "top2_index": int(top2.indices[1].item()),
                "top2_value_hex": float(top2.values[1].item()).hex(),
                "margin_hex": float(
                    top2.values[0].item() - top2.values[1].item()).hex(),
                "nan_inf_count": nan_inf,
                "row_f32_sha256": row_sha,
                "row_element_count": row_count,
                "rule_proof": rule_proof,
                "consumer_sha256": payload_checksum(payload),
                "compute_ns": 0,
            }
            frame = encode_frame(response)
            send_exact(conn, frame)
            stats["result_bytes_tx"] += len(frame)
            del hidden
    finally:
        try:
            Path(os.environ.get(
                "I88_LAST_STAGE_FINAL_REPORT",
                "/tmp/i88-last-stage.json")).write_text(json.dumps({
                    "schema": "inferswarm.issue88.last-stage-final-report/1",
                    "plan_digest": plan.get("digest"),
                    "producer_freetoken_sha": running_sha,
                    "producer_check": producer_check,
                    "stats": stats,
                    "runtime": runtime.report("P5_post_run"),
                }))
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
    parser.add_argument("--allow-producer", default=None)
    parser.add_argument("--out-dir", required=True,
                        help="root dir for per-case capture bundles + rows")
    args = parser.parse_args(argv)
    serve(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        participant_plan=args.participant_plan,
        model_path=args.model,
        gpu_uuid=args.gpu_uuid,
        diagnostic=args.diagnostic,
        ready_file=args.ready_file,
        allow_producer=args.allow_producer,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
