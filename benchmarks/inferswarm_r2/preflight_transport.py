"""Measure the exact registered-host activation path before freezing R2."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import statistics
import subprocess
import time
from multiprocessing import shared_memory
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha


def _cuda_result_ok(result, operation: str) -> None:
    code = result[0] if isinstance(result, tuple) else result
    if int(code) != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {int(code)}")


def _register(tensor, byte_count: int) -> None:
    import torch

    _cuda_result_ok(
        torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), byte_count, 0),
        "cudaHostRegister",
    )


def _unregister(tensor) -> None:
    import torch

    _cuda_result_ok(
        torch.cuda.cudart().cudaHostUnregister(tensor.data_ptr()),
        "cudaHostUnregister",
    )


def _participant(
    role: str,
    device_index: int,
    shared_name: str,
    max_bytes: int,
    connection,
) -> None:
    import torch

    torch.cuda.set_device(device_index)
    segment = shared_memory.SharedMemory(name=shared_name)
    host = torch.frombuffer(segment.buf, dtype=torch.uint8, count=max_bytes)
    _register(host, max_bytes)
    device = torch.empty(max_bytes, dtype=torch.uint8, device=f"cuda:{device_index}")
    try:
        connection.send({"ready": True, "pid": os.getpid(), "role": role})
        while True:
            message = connection.recv()
            if message["op"] == "shutdown":
                break
            size = message["size"]
            if role == "producer":
                seed = message["seed"]
                pattern = (
                    torch.arange(size, dtype=torch.int64, device=device.device) + seed
                ).to(torch.uint8)
                device[:size].copy_(pattern)
                torch.cuda.synchronize(device.device)
                started = time.perf_counter_ns()
                host[:size].copy_(device[:size], non_blocking=True)
                torch.cuda.synchronize(device.device)
                ended = time.perf_counter_ns()
                connection.send({"copy_ns": ended - started})
            else:
                started = time.perf_counter_ns()
                device[:size].copy_(host[:size], non_blocking=True)
                torch.cuda.synchronize(device.device)
                ended = time.perf_counter_ns()
                digest = hashlib.sha256(
                    device[:size].cpu().numpy().tobytes()
                ).hexdigest()
                connection.send({"copy_ns": ended - started, "sha256": digest})
    finally:
        _unregister(host)
        del host
        segment.close()


def _gpu_records() -> list[dict[str, str]]:
    fields = (
        "index,uuid,name,memory.total,pci.bus_id,pcie.link.gen.current,"
        "pcie.link.width.current,compute_cap,driver_version"
    )
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [
        dict(
            zip(
                fields.split(","),
                (value.strip() for value in line.split(",")),
                strict=True,
            )
        )
        for line in output.splitlines()
    ]


def run_preflight(
    *, producer_index: int, consumer_index: int, sizes: list[int], repetitions: int
) -> dict:
    import torch

    # The parent deliberately does not call any torch.cuda operation before spawn.
    context = multiprocessing.get_context("spawn")
    maximum = max(sizes)
    segment = shared_memory.SharedMemory(create=True, size=maximum)
    parent_a, child_a = context.Pipe()
    parent_b, child_b = context.Pipe()
    producer = context.Process(
        target=_participant,
        args=("producer", producer_index, segment.name, maximum, child_a),
    )
    consumer = context.Process(
        target=_participant,
        args=("consumer", consumer_index, segment.name, maximum, child_b),
    )
    producer.start()
    consumer.start()
    ready = [parent_a.recv(), parent_b.recv()]
    rows = []
    try:
        for size in sizes:
            samples = []
            for repetition in range(repetitions + 1):
                seed = 17 + repetition
                expected = bytes((index + seed) & 0xFF for index in range(size))
                started = time.perf_counter_ns()
                parent_a.send({"op": "copy", "size": size, "seed": seed})
                d2h = parent_a.recv()
                parent_b.send({"op": "copy", "size": size})
                h2d = parent_b.recv()
                ended = time.perf_counter_ns()
                if h2d["sha256"] != hashlib.sha256(expected).hexdigest():
                    raise RuntimeError(f"transport corruption for {size} bytes")
                if repetition:
                    samples.append(
                        {
                            "d2h_ns": d2h["copy_ns"],
                            "h2d_ns": h2d["copy_ns"],
                            "end_to_end_ns": ended - started,
                            "sha256": h2d["sha256"],
                        }
                    )
            med = statistics.median(item["end_to_end_ns"] for item in samples)
            rows.append(
                {
                    "payload_bytes": size,
                    "retained_repetitions": samples,
                    "median_end_to_end_ns": med,
                    "median_effective_gib_per_second": size / med * 1e9 / (1024**3),
                    "median_d2h_ns": statistics.median(
                        item["d2h_ns"] for item in samples
                    ),
                    "median_h2d_ns": statistics.median(
                        item["h2d_ns"] for item in samples
                    ),
                    "correct": True,
                }
            )
    finally:
        for connection in (parent_a, parent_b):
            connection.send({"op": "shutdown"})
        producer.join(30)
        consumer.join(30)
        segment.close()
        segment.unlink()
    if producer.exitcode or consumer.exitcode:
        raise RuntimeError(
            f"transport participant exit codes: {producer.exitcode}, {consumer.exitcode}"
        )
    records = _gpu_records()
    return {
        "schema": "inferswarm.r2.transport-preflight/1",
        "status": "PREFLIGHT_PASS",
        "transport": {
            "kind": "registered-pinned-host-staging",
            "physical_path": "producer VRAM -> registered system-RAM buffer -> consumer VRAM",
            "peer_access_available": False,
            "control_path": "multiprocessing Pipe during preflight",
            "buffer_bytes": maximum,
        },
        "participants": ready,
        "producer_gpu": records[producer_index],
        "consumer_gpu": records[consumer_index],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-index", type=int, default=0)
    parser.add_argument("--consumer-index", type=int, default=1)
    parser.add_argument("--sizes", default="8192,524288")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = run_preflight(
        producer_index=args.producer_index,
        consumer_index=args.consumer_index,
        sizes=[int(value) for value in args.sizes.split(",")],
        repetitions=args.repetitions,
    )
    if args.out:
        write_json_with_sha(args.out, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
