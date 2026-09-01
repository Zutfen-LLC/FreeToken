"""Spawn-only process service for one R2 execution block.

This is a narrow local research control protocol, not the future R4 wire protocol.
Bulk activations use one registered shared host buffer; the connection carries
only control, metadata, token IDs, and optional diagnostic logits.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from multiprocessing import shared_memory
from pathlib import Path

CONTROL_PROTOCOL = "inferswarm.r2.local-control/1"


def _stable_gpu_record(uuid: str) -> dict:
    fields = "index,uuid,name,memory.total,pci.bus_id,pcie.link.gen.current,pcie.link.width.current"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if values[1] == uuid:
            return dict(zip(fields.split(","), values, strict=True))
    raise RuntimeError(f"planned GPU UUID {uuid} is absent")


def service_entry(
    *,
    role: str,
    stable_gpu_uuid: str,
    plan_path: str,
    model_path: str,
    shared_name: str,
    shared_bytes: int,
    connection,
    diagnostic: bool,
    host_staging_policy: str = "release_after_final_residency",
    host_release_barrier=None,
) -> None:
    # The process is spawned before CUDA exists and pins its sole visible device by UUID.
    os.environ["CUDA_VISIBLE_DEVICES"] = stable_gpu_uuid
    import torch
    from freetoken.research.r1_frozen_plan import realize_frozen_plan
    from freetoken.research.r2_local_split import (
        validate_boundary_payload,
        validate_participant,
    )

    from benchmarks.inferswarm_r2.correctness_support import tensor_record
    from benchmarks.inferswarm_r2.preflight_transport import _register, _unregister
    from benchmarks.inferswarm_r2.qwen_split_adapter import (
        BOUNDARY_PLANES,
        HIDDEN_SIZE,
        QwenSplitResearchAdapter,
        tensor_sha256,
    )

    plan = json.loads(Path(plan_path).read_text())
    execution_id = f"exec.block-{role}"
    r1_plan = plan["participant_r1_plans"][execution_id]
    materialization_ids = [item["id"] for item in r1_plan["materializations"]]
    validate_participant(
        plan,
        execution_id=execution_id,
        plan_digest_value=plan["digest"],
        stable_device_id=stable_gpu_uuid,
        materialization_ids=materialization_ids,
    )
    gpu = _stable_gpu_record(stable_gpu_uuid)
    environment = {
        "model_repository": plan["model"]["repository"],
        "model_revision": plan["model"]["revision"],
        "resources": r1_plan["resources"],
    }
    adapter = QwenSplitResearchAdapter(
        role=role,
        model_path=model_path,
        host_staging_policy=host_staging_policy,
        host_release_barrier=host_release_barrier,
    )
    realized = realize_frozen_plan(r1_plan, environment, adapter)
    runtime = adapter.runtime
    if runtime is None:
        raise RuntimeError(
            "participant realizer did not construct its execution runtime"
        )
    segment = shared_memory.SharedMemory(name=shared_name)
    host_u8 = torch.frombuffer(segment.buf, dtype=torch.uint8, count=shared_bytes)
    _register(host_u8, shared_bytes)
    session_open = False
    session_id = None
    boundary_count = 0
    boundary_bytes = 0
    control_rx_bytes = 0
    control_tx_bytes = 0
    try:
        ready = {
            "protocol": CONTROL_PROTOCOL,
            "op": "READY",
            "role": role,
            "execution_id": execution_id,
            "pid": os.getpid(),
            "gpu": gpu,
            "plan_digest": plan["digest"],
            "r1_plan_digest": r1_plan["digest"],
            "realization": {
                "validation": realized.validation,
                "reconciliation": realized.reconciliation,
                "materializations": realized.observed_materializations,
                "execution": realized.observed_execution,
                "authorities": realized.observed_authorities,
            },
            "runtime": runtime.report("P4_ready_for_resident_execution"),
        }
        connection.send(ready)
        while True:
            message = connection.recv()
            control_rx_bytes += len(
                json.dumps(message, separators=(",", ":"), default=str).encode()
            )
            if message.get("protocol") != CONTROL_PROTOCOL:
                raise RuntimeError("local control protocol version mismatch")
            op = message["op"]
            if op == "OPEN_SESSION":
                if session_open:
                    raise RuntimeError("session already open")
                runtime.reset_session_state()
                session_open = True
                session_id = message["session_id"]
                response = {"op": "ACK", "session_id": session_id}
            elif op == "RESET_SESSION":
                if not session_open or message["session_id"] != session_id:
                    raise RuntimeError("cannot reset unknown session")
                runtime.reset_session_state()
                response = {"op": "ACK", "session_id": session_id}
            elif op == "CLOSE_SESSION":
                if not session_open or message["session_id"] != session_id:
                    raise RuntimeError("cannot close unknown session")
                runtime.reset_session_state()
                session_open = False
                session_id = None
                response = {"op": "ACK"}
            elif op in ("PREFILL", "DECODE") and role == "a":
                if not session_open or message["session_id"] != session_id:
                    raise RuntimeError("execution without matching open session")
                token_count = len(message["token_ids"])
                compute_started = time.perf_counter_ns()
                if op == "PREFILL":
                    hidden, residual = runtime.prefill_a(
                        message["token_ids"], message["position"]
                    )
                else:
                    if token_count != 1:
                        raise RuntimeError("decode requires one token")
                    hidden, residual = runtime.decode_a(
                        message["token_ids"][0], message["position"]
                    )
                torch.cuda.synchronize(0)
                compute_ended = time.perf_counter_ns()
                payload_bytes = token_count * BOUNDARY_PLANES * HIDDEN_SIZE * 2
                validate_boundary_payload(
                    plan,
                    producer_execution_id=execution_id,
                    consumer_execution_id="exec.block-b",
                    dtype="bfloat16",
                    layout="plane-major-contiguous",
                    token_count=token_count,
                    payload_bytes=payload_bytes,
                )
                host = (
                    host_u8[:payload_bytes]
                    .view(torch.bfloat16)
                    .reshape(BOUNDARY_PLANES, token_count, HIDDEN_SIZE)
                )
                transfer_started = time.perf_counter_ns()
                host[0].copy_(hidden, non_blocking=True)
                host[1].copy_(residual, non_blocking=True)
                torch.cuda.synchronize(0)
                transfer_ended = time.perf_counter_ns()
                digest = (
                    hashlib.sha256(
                        host_u8[:payload_bytes].numpy().tobytes()
                    ).hexdigest()
                    if diagnostic
                    else None
                )
                boundary_count += 1
                boundary_bytes += payload_bytes
                response = {
                    "op": "BOUNDARY_READY",
                    "session_id": session_id,
                    "operation": op.lower(),
                    "position": message["position"],
                    "token_count": token_count,
                    "dtype": "bfloat16",
                    "shape": [2, token_count, HIDDEN_SIZE],
                    "layout": "plane-major-contiguous",
                    "payload_bytes": payload_bytes,
                    "producer_sha256": digest,
                    "compute_ns": compute_ended - compute_started,
                    "d2h_ns": transfer_ended - transfer_started,
                }
                if diagnostic and message.get("capture_state"):
                    response["producer_tensors"] = {
                        "hidden": tensor_record(hidden),
                        "residual": tensor_record(residual),
                    }
                    response["producer_state"] = runtime.logical_state_records(
                        message["position"] + token_count
                    )
                del hidden, residual, host
            elif op == "CONSUME_BOUNDARY" and role == "b":
                if not session_open or message["session_id"] != session_id:
                    raise RuntimeError("execution without matching open session")
                token_count = message["token_count"]
                payload_bytes = message["payload_bytes"]
                validate_boundary_payload(
                    plan,
                    producer_execution_id="exec.block-a",
                    consumer_execution_id=execution_id,
                    dtype=message["dtype"],
                    layout=message["layout"],
                    token_count=token_count,
                    payload_bytes=payload_bytes,
                )
                host = (
                    host_u8[:payload_bytes]
                    .view(torch.bfloat16)
                    .reshape(BOUNDARY_PLANES, token_count, HIDDEN_SIZE)
                )
                hidden = torch.empty(
                    (token_count, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda:0"
                )
                residual = torch.empty_like(hidden)
                transfer_started = time.perf_counter_ns()
                hidden.copy_(host[0], non_blocking=True)
                residual.copy_(host[1], non_blocking=True)
                torch.cuda.synchronize(0)
                transfer_ended = time.perf_counter_ns()
                consumer_digest = None
                if diagnostic:
                    consumer_digest = hashlib.sha256(
                        torch.stack((hidden, residual))
                        .view(torch.uint8)
                        .cpu()
                        .numpy()
                        .tobytes()
                    ).hexdigest()
                    if consumer_digest != message["producer_sha256"]:
                        raise RuntimeError(
                            "activation checksum changed across transport"
                        )
                compute_started = time.perf_counter_ns()
                if message["operation"] == "prefill":
                    token, logits, execution_diagnostic = runtime.prefill_b(
                        hidden,
                        residual,
                        message["position"],
                        capture_diagnostics=bool(message.get("capture_state")),
                    )
                else:
                    token, logits = runtime.decode_b(
                        hidden, residual, message["position"]
                    )
                    execution_diagnostic = None
                torch.cuda.synchronize(0)
                compute_ended = time.perf_counter_ns()
                boundary_count += 1
                boundary_bytes += payload_bytes
                response = {
                    "op": "TOKEN_RESULT",
                    "session_id": session_id,
                    "token_id": token,
                    "consumer_sha256": consumer_digest,
                    "h2d_ns": transfer_ended - transfer_started,
                    "compute_ns": compute_ended - compute_started,
                }
                if diagnostic and message.get("capture_logits"):
                    values = logits.detach().float()
                    response["logits"] = {
                        "shape": list(values.shape),
                        "float32_sha256": tensor_sha256(values),
                        "argmax": int(values.argmax(dim=-1).item()),
                        "nan_count": int(torch.isnan(values).sum().item()),
                        "inf_count": int(torch.isinf(values).sum().item()),
                        "full_logits": values.cpu().tolist(),
                    }
                if diagnostic and message.get("capture_state"):
                    response["consumer_tensors"] = {
                        "hidden": tensor_record(hidden),
                        "residual": tensor_record(residual),
                    }
                    response["consumer_state"] = runtime.logical_state_records(
                        message["position"] + token_count
                    )
                    response["execution_diagnostic"] = execution_diagnostic
                del hidden, residual, host, logits
            elif op == "REPORT":
                response = {
                    "op": "REPORT",
                    "runtime": runtime.report(),
                    "boundary_count": boundary_count,
                    "activation_bytes": boundary_bytes,
                    "control_rx_bytes": control_rx_bytes,
                    "control_tx_bytes": control_tx_bytes,
                }
            elif op == "SHUTDOWN":
                from freetoken.research.host_reclamation import snapshot_host_memory

                connection.send(
                    {
                        "op": "ACK",
                        "P6_pre_worker_shutdown": snapshot_host_memory(),
                    }
                )
                break
            else:
                raise RuntimeError(f"unsupported {role} operation {op!r}")
            control_tx_bytes += len(
                json.dumps(response, separators=(",", ":"), default=str).encode()
            )
            connection.send(response)
    except Exception as exc:  # noqa: BLE001 - serialize child failures to coordinator
        try:
            connection.send(
                {"op": "ERROR", "type": type(exc).__name__, "message": str(exc)}
            )
        finally:
            raise
    finally:
        _unregister(host_u8)
        del host_u8
        segment.close()


__all__ = ["CONTROL_PROTOCOL", "service_entry"]
