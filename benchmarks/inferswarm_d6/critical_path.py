"""Captured one-layer D6-A critical-path and fixed-byte decomposition."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import torch

from freetoken.engine.engine import Engine
from freetoken.gpu_select import set_assigned_gpu
from freetoken.moe.inferswarm_d3_placement import load_d3_placement
from freetoken.moe.offload_cache import iter_offload_moe_layers
from freetoken.server.args import parse_args
from freetoken.server.launch import _resolve_server_gpu_args
from inferswarm_d3.primitive import _delta, _vm

PRIMARY = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
A = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
B = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"
COUNTS = (0, 1, 2, 4, 6, 8)


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values); pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _dist(values: list[float]) -> dict:
    return {"n": len(values), "median_us": statistics.median(values),
            "p05_us": _pct(values, .05), "p95_us": _pct(values, .95),
            "min_us": min(values), "max_us": max(values)}


def _config(ns):
    argv = ["--model", ns.model, "--gpu", PRIMARY,
            "--inferswarm-experimental-d5-compact-routes",
            "--inferswarm-experimental-d5-resident-loader",
            "--inferswarm-d5-loader-cpu-workers", "8",
            "--inferswarm-d3-active-workers", ns.shape,
            "--inferswarm-d3-placement", ns.placement,
            "--moe-backend", "offload", "--moe-cpu-layers", "0",
            "--nvfp4-backend", "triton", "--moe-cache-size", "3774",
            "--max-running-requests", "1", "--cuda-graph-max-bs", "1",
            "--sampling-defaults", "none"]
    if "a" in ns.shape: argv += ["--inferswarm-d3-worker-a-gpu", A]
    if "b" in ns.shape: argv += ["--inferswarm-d3-worker-b-gpu", B]
    parsed, _ = parse_args(argv)
    return _resolve_server_gpu_args(parsed)


def _route_ids(placement, layer_id: int, b_count: int) -> list[int]:
    a = list(placement.worker_a.per_layer[layer_id].expert_ids)
    b = list(placement.worker_b.per_layer[layer_id].expert_ids)
    local = [x for x in range(placement.worker_a.num_experts)
             if x not in set(a) and x not in set(b)]
    # Equal AB sees b_count B routes and the balance on A. B-only sees the same raw
    # identities, with A identities becoming local. This freezes payload identity.
    if len(a) < 8 or len(b) < 8: raise RuntimeError("diagnostic layer lacks eight A/B identities")
    return b[:b_count] + a[:8-b_count]


def _bytes(count: int, hidden_size: int, element_size: int) -> dict:
    activation = hidden_size * element_size
    slots = weights = 8 * 4; active_count = 4
    returned_capacity = 8 * hidden_size * element_size
    returned_useful = count * hidden_size * element_size
    inbound_capacity = activation + slots + weights + active_count
    inbound_useful = activation + count * 8 + active_count
    total_capacity = inbound_capacity + 2 * returned_capacity
    total_useful = inbound_useful + 2 * returned_useful
    return {"activation_inbound": {"useful": activation, "capacity": activation, "padding": 0},
            "route_slots_inbound": {"useful": count * 4, "capacity": slots, "padding": (8-count)*4},
            "route_weights_inbound": {"useful": count * 4, "capacity": weights, "padding": (8-count)*4},
            "active_count_inbound": {"useful": active_count, "capacity": active_count, "padding": 0},
            "route_positions_inbound": {"useful": 0, "capacity": 0, "padding": 0},
            "returned_contribution_d2h": {"useful": returned_useful, "capacity": returned_capacity,
                                           "padding": returned_capacity-returned_useful},
            "returned_contribution_h2d_gpu0": {"useful": returned_useful, "capacity": returned_capacity,
                                                "padding": returned_capacity-returned_useful},
            "total_path": {"useful": total_useful, "capacity": total_capacity,
                           "padding": total_capacity-total_useful,
                           "transport_efficiency": total_useful / total_capacity}}


def _capture(executor, layer, cache, stream, hidden, weights, ids, diagnostic: bool):
    executor.set_d6_diagnostic(diagnostic)
    executor.decode(layer, cache, hidden, weights, ids); torch.cuda.synchronize(executor.primary_device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream): executor.decode(layer, cache, hidden, weights, ids)
    return graph


def _wall(graph, device, stream, repetitions: int) -> list[float]:
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    for _ in range(10): graph.replay()
    torch.cuda.synchronize(device); values = []
    for _ in range(repetitions):
        start.record(stream); graph.replay(); end.record(stream); end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0)
    return values


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True); p.add_argument("--placement", required=True)
    p.add_argument("--shape", choices=("b", "ab"), required=True)
    p.add_argument("--output", required=True); p.add_argument("--replays", type=int, default=200)
    ns = p.parse_args(); before = _vm(); started = time.monotonic(); args = _config(ns)
    set_assigned_gpu((args.gpu_assigned or args.gpu)[0]); engine = Engine(args)
    try:
        executor = engine.inferswarm_d5_compact_routes; cache = engine.moe_offload_cache
        layer = next(iter(iter_offload_moe_layers(engine.model))); layer_id = int(layer.layer_id)
        placement = load_d3_placement(ns.placement); device = engine.device
        hidden = torch.randn((1, executor.hidden_size), dtype=executor.hidden_dtype, device=device,
                             generator=torch.Generator(device=device).manual_seed(8606))
        weights = torch.full((1, 8), 1/8, dtype=torch.float32, device=device)
        ids = torch.tensor([_route_ids(placement, layer_id, 4)], dtype=torch.int32, device=device)
        cache.reset(); uninstrumented = _capture(executor, layer, cache, engine.stream, hidden, weights, ids, False)
        uninstrumented_wall = _wall(uninstrumented, device, engine.stream, ns.replays)
        diagnostic = _capture(executor, layer, cache, engine.stream, hidden, weights, ids, True)
        diagnostic_wall = _wall(diagnostic, device, engine.stream, ns.replays)
        perturbation = statistics.median(uninstrumented_wall) / statistics.median(diagnostic_wall)
        cases = []
        for useful_b in COUNTS:
            ids.copy_(torch.tensor([_route_ids(placement, layer_id, useful_b)], dtype=torch.int32, device=device))
            for _ in range(10): diagnostic.replay()
            torch.cuda.synchronize(device); rows = []
            for _ in range(ns.replays):
                diagnostic.replay(); torch.cuda.synchronize(device)
                snap = executor.d6_diagnostic_snapshot(layer_id)
                rows.append(snap)
            gpu0 = {name: _dist([r["gpu0_ms"][name] * 1000 for r in rows]) for name in rows[0]["gpu0_ms"]}
            workers = {x: {name: _dist([r["workers_ms"][x][name] * 1000 for r in rows])
                           for name in rows[0]["workers_ms"][x]} for x in executor.active_workers}
            counts = rows[-1]["route_counts"]
            cases.append({"requested_b_routes": useful_b, "route_counts": counts,
                          "gpu0": gpu0, "workers": workers,
                          "fixed_transport_bytes": {x: _bytes(counts[x], executor.hidden_size,
                                                               torch.empty((), dtype=executor.hidden_dtype).element_size())
                                                    for x in executor.active_workers}})
        report = executor.configuration_report(); after = _vm()
        result = {"schema": "inferswarm.d6.critical-path/1", "phase": "D6-A_FROZEN",
                  "freetoken_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "model": ns.model, "revision": ns.revision, "placement_sha256": report["placement_sha256"],
                  "shape": ns.shape, "layer_id": layer_id, "replays_after_warmup": ns.replays,
                  "same_device_elapsed_only": True, "events_preallocated_before_capture": True,
                  "host_reads_inside_replay": False, "diagnostic_mode_separate": True,
                  "instrumentation_perturbation": {"uninstrumented_wall": _dist(uninstrumented_wall),
                                                   "instrumented_wall": _dist(diagnostic_wall),
                                                   "instrumented_over_uninstrumented_throughput": perturbation,
                                                   "threshold": .97, "trusted": perturbation >= .97},
                  "representative_case": cases[COUNTS.index(4)], "controlled_cases": cases,
                  "graph_contract": report, "paging_delta": _delta(before, after),
                  "startup_s": time.monotonic() - started, "status": "complete"}
        Path(ns.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"shape": ns.shape, "perturbation": result["instrumentation_perturbation"],
                          "representative": result["representative_case"]}, indent=2)); return 0
    finally:
        engine.shutdown()


if __name__ == "__main__": raise SystemExit(main())
