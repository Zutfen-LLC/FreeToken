"""Isolated captured worker-kernel duration versus useful D5 route count."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from freetoken.engine.engine import Engine
from freetoken.gpu_select import set_assigned_gpu
from freetoken.moe.offload_cache import iter_offload_moe_layers
from freetoken.server.args import parse_args
from freetoken.server.launch import _resolve_server_gpu_args

PRIMARY = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
A = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
B = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"
COUNTS = (0, 1, 2, 4, 6, 8)


def config(ns):
    argv = ["--model", ns.model, "--gpu", PRIMARY,
            "--inferswarm-experimental-d5-compact-routes",
            "--inferswarm-experimental-d5-resident-loader",
            "--inferswarm-d5-loader-cpu-workers", "8",
            "--inferswarm-d3-active-workers", "ab",
            "--inferswarm-d3-placement", ns.placement,
            "--inferswarm-d3-worker-a-gpu", A, "--inferswarm-d3-worker-b-gpu", B,
            "--moe-backend", "offload", "--moe-cpu-layers", "0",
            "--nvfp4-backend", "triton", "--moe-cache-size", "3774",
            "--max-running-requests", "1", "--cuda-graph-max-bs", "1",
            "--sampling-defaults", "none"]
    parsed, _ = parse_args(argv); return _resolve_server_gpu_args(parsed)


def curve(engine, executor, layer, label: str, compact: bool, replays: int) -> list[dict]:
    device = getattr(executor, f"worker_{label}_torch_device")
    stream = getattr(executor, f"worker_{label}_stream")
    bank = getattr(executor.resident_banks, f"worker_{label}")
    activation = getattr(executor, f"worker_{label}_activation")
    slots = getattr(executor, f"worker_{label}_slots")
    weights = getattr(executor, f"worker_{label}_weights")
    count = getattr(executor, f"worker_{label}_count")
    routes = getattr(executor, f"worker_{label}_routes")
    gate_up = getattr(executor, f"worker_{label}_gate_up")
    activation_out = getattr(executor, f"worker_{label}_activation_out")
    torch.cuda.set_device(device)
    with torch.cuda.stream(stream):
        activation.normal_(); slots.copy_(torch.arange(executor.top_k, dtype=torch.int32, device=device).reshape(1, -1))
        weights.fill_(1.0 / executor.top_k); count.fill_(executor.top_k)
        layer._expert_route_contributions(engine.moe_offload_cache, activation, weights, slots,
            views=bank.bank_views(), alphas=bank.alpha_views(), out=routes,
            gate_up_out=gate_up, activation_out=activation_out,
            active_count=count if compact else None)
    stream.synchronize(); graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        layer._expert_route_contributions(engine.moe_offload_cache, activation, weights, slots,
            views=bank.bank_views(), alphas=bank.alpha_views(), out=routes,
            gate_up_out=gate_up, activation_out=activation_out,
            active_count=count if compact else None)
    rows = []
    for useful in COUNTS:
        with torch.cuda.stream(stream):
            weights.zero_(); weights[0, :useful] = 1.0 / max(useful, 1); count.fill_(useful)
            for _ in range(10): graph.replay()
        stream.synchronize(); timings = []
        for _ in range(replays):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream): start.record(stream); graph.replay(); end.record(stream)
            end.synchronize(); timings.append(start.elapsed_time(end))
        if compact:
            tail_nonzero = int(torch.count_nonzero(routes[0, useful:]).item())
            if tail_nonzero: raise RuntimeError(f"compact {label} stale tail at count {useful}")
        rows.append({"useful_routes": useful, "median_ms": statistics.median(timings),
                     "min_ms": min(timings), "max_ms": max(timings), "replays": replays})
    full = rows[-1]["median_ms"]
    for row in rows: row["duration_over_k"] = row["median_ms"] / full
    return rows


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True)
    p.add_argument("--placement", required=True); p.add_argument("--fixed-output", required=True)
    p.add_argument("--compact-output", required=True); p.add_argument("--replays", type=int, default=100)
    ns = p.parse_args(); started = time.monotonic(); args = config(ns)
    set_assigned_gpu((args.gpu_assigned or args.gpu)[0]); engine = Engine(args)
    try:
        executor = engine.inferswarm_d5_compact_routes
        layer = next(iter(iter_offload_moe_layers(engine.model)))
        common = {"freetoken_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "model": ns.model, "placement_sha256": executor.configuration_report()["placement_sha256"],
                  "layer_id": int(layer.layer_id), "counts": list(COUNTS),
                  "same_device_cuda_events": True, "fixed_capacity": executor.top_k,
                  "startup_s": time.monotonic() - started, "status": "complete"}
        fixed = {"schema": "inferswarm.d5.fixed-width-route-count/1", **common,
                 "workers": {x: curve(engine, executor, layer, x, False, ns.replays) for x in "ab"}}
        compact = {"schema": "inferswarm.d5.compact-route-count/1", **common,
                   "workers": {x: curve(engine, executor, layer, x, True, ns.replays) for x in "ab"}}
        Path(ns.fixed_output).write_text(json.dumps(fixed, indent=2) + "\n")
        Path(ns.compact_output).write_text(json.dumps(compact, indent=2) + "\n")
        print(json.dumps({"fixed": fixed["workers"], "compact": compact["workers"]}, indent=2))
    finally: engine.shutdown()
    return 0


if __name__ == "__main__": raise SystemExit(main())
