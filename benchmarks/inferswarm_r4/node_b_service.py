"""Node B persistent service for the R4 two-node boundary.

Runs on inferswarm03.  Owns exactly one resident captured Block B runtime
(accepted R2 substrate, unchanged), plus a bounded registered host activation
buffer.  Serves one persistent full-duplex TCP connection from the Node A
coordinator: receives framed boundary activations, H2D-copies into the
captured decode graph or prefill path, executes Block B, and returns a small
framed result (token + bounded metadata; diagnostics only in the diagnostic
arm).

Research-internal: not a public daemon API.  Fail-closed on every wire
violation; no retries/reconnect/failover.
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
MAX_TOKEN_COUNT = 64  # one prefill chunk row bound
BOUNDARY_CONTRACT = {
    "dtype": "bfloat16",
    "layout": "plane-major-contiguous",
    "planes": 2,
    "row_width": 2048,
    "element_bytes": 2,
    "max_token_count": MAX_TOKEN_COUNT,
}


def _register(tensor, byte_count: int) -> None:
    import torch

    result = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), byte_count, 0)
    code = result[0] if isinstance(result, tuple) else result
    if int(code) != 0:
        raise RuntimeError(f"cudaHostRegister failed with CUDA error {int(code)}")


def _unregister(tensor) -> None:
    import torch

    result = torch.cuda.cudart().cudaHostUnregister(tensor.data_ptr())
    code = result[0] if isinstance(result, tuple) else result
    if int(code) != 0:
        raise RuntimeError(f"cudaHostUnregister failed with CUDA error {int(code)}")


def serve(
    *,
    listen_host: str,
    listen_port: int,
    plan_path: str,
    model_path: str,
    diagnostic: bool,
    host_staging_policy: str = "release_after_final_residency",
    expected_coordinator: str | None = None,
    ready_file: str | None = None,
) -> None:
    from benchmarks.inferswarm_r2.preflight_transport import _register, _unregister
    from benchmarks.inferswarm_r4.r4_plan import (
        GPU_B_UUID,
        MODEL_REVISION,
        load_r4_plan,
    )
    from benchmarks.inferswarm_r4.node_preflight import (
        require_gpu,
        verify_checkpoint_revision,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_B_UUID
    import torch  # noqa: F401  (device pinned before CUDA init)
    from freetoken.research.r1_frozen_plan import realize_frozen_plan
    from freetoken.research.r2_local_split import (
        validate_boundary_payload,
        validate_participant,
    )
    from benchmarks.inferswarm_r2.qwen_split_adapter import (
        BOUNDARY_PLANES,
        HIDDEN_SIZE,
        QwenSplitResearchAdapter,
        tensor_sha256,
    )

    plan = load_r4_plan(Path(plan_path))
    # fail closed: the running producer must be the plan's frozen producer
    repo_root = Path(__file__).resolve().parents[2]
    running_sha = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "-C",
            str(repo_root),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    plan_producer = plan.get("provenance", {}).get("r4", {}).get("producer_sha")
    if plan_producer and running_sha != plan_producer:
        raise RuntimeError(
            f"Node B running producer {running_sha!r} != plan's frozen producer "
            f"{plan_producer!r}; canonical execution refuses to proceed"
        )
    execution_id = "exec.block-b"
    r1_plan = plan["participant_r1_plans"][execution_id]
    validate_participant(
        plan,
        execution_id=execution_id,
        plan_digest_value=plan["digest"],
        stable_device_id=GPU_B_UUID,
        materialization_ids=[item["id"] for item in r1_plan["materializations"]],
    )
    verify_checkpoint_revision(model_path, MODEL_REVISION)
    gpu_row = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    found = any(GPU_B_UUID in line for line in gpu_row.splitlines())
    if not found:
        raise RuntimeError(f"frozen GPU {GPU_B_UUID} not visible on Node B")
    # frozen GPU BDF: refuse silent card selection / PCI drift
    frozen_identity = plan.get("frozen_gpu_identity", {}).get("node.inferswarm03")
    if frozen_identity:
        actual_bdf = None
        for line in gpu_row.splitlines():
            if GPU_B_UUID in line:
                actual_bdf = line.rsplit(",", 1)[1].strip()
        if actual_bdf != frozen_identity.get("pci_bdf"):
            raise RuntimeError(
                f"frozen GPU BDF drift on Node B: {actual_bdf!r} != "
                f"frozen {frozen_identity.get('pci_bdf')!r}"
            )
    # issue #57: MemAvailable >= 12 GiB and DMI physical RAM >= 16 GiB
    # measured immediately before Block-B heavyweight realization
    from benchmarks.inferswarm_r4.r4_preflight_gate import (
        read_process_vm_swap_kib,
        require_block_b_host_ram,
        require_no_swap_reliance,
    )
    from benchmarks.inferswarm_r4.node_preflight import _memory as _probe_memory

    pre_realization_memory = _probe_memory()
    host_ram_gate = require_block_b_host_ram(pre_realization_memory)
    environment = {
        "model_repository": plan["model"]["repository"],
        "model_revision": plan["model"]["revision"],
        "resources": r1_plan["resources"],
    }
    adapter = QwenSplitResearchAdapter(
        role="b",
        model_path=model_path,
        host_staging_policy=host_staging_policy,
    )
    realized = realize_frozen_plan(r1_plan, environment, adapter)
    runtime = adapter.runtime
    if runtime is None:
        raise RuntimeError("Node B realizer did not construct its runtime")

    buffer_bytes = 2 * MAX_TOKEN_COUNT * HIDDEN_SIZE * 2
    host_u8 = torch.empty(buffer_bytes, dtype=torch.uint8)
    _register(host_u8, buffer_bytes)
    session_open = False
    session_id = None
    experiment_id = plan["digest"]
    stats = {
        "boundaries_served": 0,
        "activation_bytes_rx": 0,
        "result_bytes_tx": 0,
        "checkum_mismatches": 0,
    }

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((listen_host, listen_port))
    server.listen(1)
    service_markers = {"last": time.perf_counter_ns()}
    if ready_file:
        Path(ready_file).write_text(
            json.dumps(
                {
                    "plan_digest": plan["digest"],
                    "execution_id": execution_id,
                    "gpu_uuid": GPU_B_UUID,
                    "listen": [listen_host, listen_port],
                    "diagnostic": diagnostic,
                    "pid": os.getpid(),
                    "producer_freetoken_sha": running_sha,
                    "host_ram_gate": host_ram_gate,
                    "realization": {
                        "validation": realized.validation,
                        "reconciliation": realized.reconciliation,
                        "materializations": realized.observed_materializations,
                        "execution": realized.observed_execution,
                        "authorities": realized.observed_authorities,
                    },
                    "runtime": runtime.report("P4_ready_for_resident_execution"),
                }
            )
        )
    try:
        conn, addr = server.accept()
        if expected_coordinator and addr[0] != expected_coordinator:
            raise WireError(f"unexpected coordinator address {addr[0]}")
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        nodelay_state = conn.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
        identity_sent = False
        while True:
            identity = {"protocol": WIRE_PROTOCOL_ID, "experiment_id": experiment_id}
            header, payload = recv_frame(conn, identity)
            kind = header.get("kind")
            if kind == "hello":
                # A hello re-establishes the session over the same persistent
                # connection; the new id becomes the enforced session.
                session_id = header["session_id"]
                runtime.reset_session_state()
                session_open = True
                response = {
                    "kind": "response",
                    "protocol": WIRE_PROTOCOL_ID,
                    "experiment_id": experiment_id,
                    "session_id": session_id,
                    "op": "HELLO_ACK",
                    "tcp_nodelay": bool(nodelay_state),
                    "runtime_ready": True,
                }
                frame = encode_frame(response)
                send_exact(conn, frame)
                stats["result_bytes_tx"] += len(frame)
                identity_sent = True
                continue
            if kind != "request" or not session_open:
                raise WireError(f"unexpected frame kind {kind!r} before session")
            if header.get("session_id") != session_id:
                raise WireError(
                    f"session_id mismatch: {header.get('session_id')!r} != {session_id!r}"
                )
            op = header["op"]
            if op not in ("OPEN_SESSION", "BOUNDARY"):
                raise WireError(f"unsupported request op {op!r}")
            if op == "OPEN_SESSION":
                runtime.reset_session_state()
                response = {
                    "kind": "response",
                    "protocol": WIRE_PROTOCOL_ID,
                    "experiment_id": experiment_id,
                    "session_id": session_id,
                    "op": "SESSION_ACK",
                }
                frame = encode_frame(response)
                send_exact(conn, frame)
                stats["result_bytes_tx"] += len(frame)
                continue
            # BOUNDARY request
            token_count = int(header["token_count"])
            validate_request(
                header,
                contract=BOUNDARY_CONTRACT,
                checksum=payload_checksum(payload) if diagnostic else None,
                payload=payload,
            )
            validate_boundary_payload(
                plan,
                producer_execution_id="exec.block-a",
                consumer_execution_id=execution_id,
                dtype=header["dtype"],
                layout=header["layout"],
                token_count=token_count,
                payload_bytes=int(header["payload_len"]),
            )
            receive_done = time.perf_counter_ns()
            host_u8[: len(payload)] = torch.frombuffer(
                payload, dtype=torch.uint8
            )
            host = (
                host_u8[: len(payload)]
                .view(torch.bfloat16)
                .reshape(BOUNDARY_PLANES, token_count, HIDDEN_SIZE)
            )
            hidden = torch.empty(
                (token_count, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda:0"
            )
            residual = torch.empty_like(hidden)
            h2d_started = time.perf_counter_ns()
            hidden.copy_(host[0], non_blocking=True)
            residual.copy_(host[1], non_blocking=True)
            torch.cuda.synchronize(0)
            h2d_ended = time.perf_counter_ns()
            consumer_digest = None
            if diagnostic:
                consumer_digest = payload_checksum(
                    torch.stack((hidden, residual))
                    .contiguous()
                    .view(torch.uint8)
                    .cpu()
                    .numpy()
                    .tobytes()
                )
                if consumer_digest != header["payload_sha256"]:
                    stats["checkum_mismatches"] += 1
                    raise WireError("boundary payload checksum mismatch on Node B")
            compute_started = time.perf_counter_ns()
            if header["operation"] == "prefill":
                token, logits, execution_diagnostic = runtime.prefill_b(
                    hidden,
                    residual,
                    int(header["position"]),
                    capture_diagnostics=bool(header.get("capture_state")),
                )
            else:
                if token_count != 1:
                    raise WireError("decode boundary requires exactly one token")
                token, logits = runtime.decode_b(
                    hidden, residual, int(header["position"])
                )
                execution_diagnostic = None
            torch.cuda.synchronize(0)
            compute_ended = time.perf_counter_ns()
            stats["boundaries_served"] += 1
            stats["activation_bytes_rx"] += len(payload)
            response = {
                "kind": "response",
                "protocol": WIRE_PROTOCOL_ID,
                "experiment_id": experiment_id,
                "session_id": session_id,
                "op": "TOKEN_RESULT",
                "token_id": token,
                "consumer_sha256": consumer_digest,
                "h2d_ns": h2d_ended - h2d_started,
                "compute_ns": compute_ended - compute_started,
                "service_ns": receive_done - service_markers["last"],
            }
            if diagnostic and header.get("capture_logits"):
                # Frozen R2-v2 comparator is float32 hash identity; the wire
                # carries only the bounded record, never full logit values.
                if logits is None:
                    raise WireError("logit evidence requested but not produced")
                values = logits.detach().float()
                response["logits"] = {
                    "shape": list(values.shape),
                    "float32_sha256": tensor_sha256(values),
                    "argmax": int(values.argmax(dim=-1).item()),
                    "nan_count": int(torch.isnan(values).sum().item()),
                    "inf_count": int(torch.isinf(values).sum().item()),
                }
            if diagnostic and header.get("capture_state"):
                response["consumer_tensors"] = {
                    "hidden": _tensor_record(hidden),
                    "residual": _tensor_record(residual),
                }
                response["consumer_state"] = runtime.logical_state_records(
                    int(header["position"]) + token_count
                )
                response["execution_diagnostic"] = execution_diagnostic
            frame = encode_frame(response)
            send_exact(conn, frame)
            stats["result_bytes_tx"] += len(frame)
            service_markers["last"] = time.perf_counter_ns()
            del hidden, residual, host, logits
    finally:
        # Compute every evidence field BEFORE the write; a failure in any
        # check must not silently swallow the report (the previous
        # `except Exception: pass` hid a raised check and skipped the
        # write).  The swap-reliance criterion is process-scoped (VmSwap
        # of this staging process); system-wide deltas are informational.
        try:
            post_release_memory = _probe_memory()
            vm_swap_kib = read_process_vm_swap_kib()
            memory_lifecycle = {
                "pre_realization": pre_realization_memory,
                "post_release": post_release_memory,
                "staging_process_vm_swap_kib": vm_swap_kib,
                "swap_reliance": require_no_swap_reliance(
                    pre_realization_memory["swap_activity"],
                    post_release_memory["swap_activity"],
                    staging_process_vm_swap_kib=vm_swap_kib,
                ),
            }
            runtime_post = runtime.report("P5_post_run")
        except Exception:  # noqa: BLE001 - never mask the original error
            memory_lifecycle = {"error": "lifecycle evidence collection failed"}
            runtime_post = None
        final_report = {
            "schema": "inferswarm.r4.node-b-final-report/1",
            "plan_digest": plan["digest"],
            "producer_freetoken_sha": running_sha,
            "diagnostic": diagnostic,
            "stats": stats,
            "host_ram_gate": host_ram_gate,
            "memory_lifecycle": memory_lifecycle,
            "runtime": runtime_post,
        }
        try:
            Path(
                os.environ.get("R4_NODE_B_FINAL_REPORT", "/tmp/r4-node-b-final.json")
            ).write_text(json.dumps(final_report))
        except Exception:  # noqa: BLE001 - never mask the original error
            pass
        _unregister(host_u8)
        server.close()


def _tensor_record(tensor) -> dict:
    from benchmarks.inferswarm_r2.correctness_support import tensor_record

    return tensor_record(tensor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="10.0.0.219")
    parser.add_argument("--listen-port", type=int, default=18485)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--ready-file", default=None)
    parser.add_argument("--expected-coordinator", default="10.0.0.141")
    args = parser.parse_args(argv)
    serve(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        plan_path=args.plan,
        model_path=args.model,
        diagnostic=args.diagnostic,
        ready_file=args.ready_file,
        expected_coordinator=args.expected_coordinator,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
