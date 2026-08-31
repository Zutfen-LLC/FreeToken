"""D6 physical correctness, dynamic-count transport, and graph primitive gate."""
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
from freetoken.moe.inferswarm_d6_count_aware_transport import transport_byte_geometry
from freetoken.moe.offload_cache import iter_offload_moe_layers
from freetoken.server.args import parse_args
from freetoken.server.launch import _resolve_server_gpu_args
from inferswarm_d3.primitive import _delta, _one_layer_diagnostics, _vm

PRIMARY = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
A = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
B = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"
COUNTS = (0, 1, 2, 4, 6, 8)


def _pct(values, q):
    values = sorted(values); pos = (len(values)-1)*q; lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo]*(hi-pos)+values[hi]*(pos-lo)


def _dist(values):
    return {"n": len(values), "median_us": statistics.median(values),
            "p05_us": _pct(values, .05), "p95_us": _pct(values, .95),
            "min_us": min(values), "max_us": max(values)}


def _config(ns):
    argv = ["--model", ns.model, "--gpu", PRIMARY,
            "--inferswarm-experimental-d6-count-aware-transport",
            "--inferswarm-experimental-d5-resident-loader", "--inferswarm-d5-loader-cpu-workers", "8",
            "--inferswarm-d3-active-workers", "ab", "--inferswarm-d3-placement", ns.placement,
            "--inferswarm-d3-worker-a-gpu", A, "--inferswarm-d3-worker-b-gpu", B,
            "--moe-backend", "offload", "--moe-cpu-layers", "0", "--nvfp4-backend", "triton",
            "--moe-cache-size", "3774", "--max-running-requests", "1", "--cuda-graph-max-bs", "1",
            "--sampling-defaults", "none"]
    parsed, _ = parse_args(argv); return _resolve_server_gpu_args(parsed)


def _ids(placement, layer_id, b_count):
    a = list(placement.worker_a.per_layer[layer_id].expert_ids)
    b = list(placement.worker_b.per_layer[layer_id].expert_ids)
    return b[:b_count] + a[:8-b_count]


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True); p.add_argument("--placement", required=True)
    p.add_argument("--output", required=True); p.add_argument("--replays", type=int, default=200)
    ns = p.parse_args(); before = _vm(); started = time.monotonic(); args = _config(ns)
    set_assigned_gpu((args.gpu_assigned or args.gpu)[0]); engine = Engine(args)
    try:
        executor = engine.inferswarm_d6_count_aware_transport
        report = executor.configuration_report(); required = {
            "enabled": True, "graph_active": True, "eager_fallback": False,
            "fallback_count": 0, "failure_count": 0, "graph_recapture_count": 0,
            "steady_state_host_sync_count": 0, "count_aware_return_transport": True,
        }
        graph_checks = {key: report.get(key) == value for key, value in required.items()}
        if not all(graph_checks.values()): raise RuntimeError(f"D6 graph contract failed: {graph_checks}")
        diagnostics = _one_layer_diagnostics(engine, executor, ns.placement, "ab", 50)
        layer = next(iter(iter_offload_moe_layers(engine.model))); layer_id = int(layer.layer_id)
        placement = load_d3_placement(ns.placement); device = engine.device
        hidden = torch.randn((1, executor.hidden_size), dtype=executor.hidden_dtype, device=device,
                             generator=torch.Generator(device=device).manual_seed(8616))
        weights = torch.full((1, 8), 1/8, dtype=torch.float32, device=device)
        ids = torch.tensor([_ids(placement, layer_id, 4)], dtype=torch.int32, device=device)
        executor.set_d6_diagnostic(True, worker_markers={"branch_start", "branch_end"})
        engine.moe_offload_cache.reset(); executor.decode(layer, engine.moe_offload_cache, hidden, weights, ids)
        torch.cuda.synchronize(device); graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=engine.stream):
            executor.decode(layer, engine.moe_offload_cache, hidden, weights, ids)
        executor.reset_counters(); curves = []
        for b_count in COUNTS:
            ids.copy_(torch.tensor([_ids(placement, layer_id, b_count)], dtype=torch.int32, device=device))
            for _ in range(10): graph.replay()
            torch.cuda.synchronize(device); samples = {"a": [], "b": []}
            for _ in range(ns.replays):
                graph.replay(); torch.cuda.synchronize(device)
                row = executor.d6_interval_snapshot("worker", "complete_worker_branch", layer_id)
                for x in "ab": samples[x].append(row[x] * 1000)
            counts = {"a": 8-b_count, "b": b_count, "local": 0}
            curves.append({"route_counts": counts, "workers": {x: _dist(samples[x]) for x in "ab"},
                           "bytes": {x: transport_byte_geometry(counts[x], top_k=8,
                                                                  hidden_size=executor.hidden_size,
                                                                  element_size=2) for x in "ab"}})
        torch.cuda.synchronize(device); snapshot = executor.snapshot(); transport = snapshot["transport"]
        counter_checks = {
            "ownership_exact": snapshot["ownership"]["selection_arithmetic_exact"],
            "physical_equals_owned": snapshot["physical"]["physical_worker_invocations_equal_owned_remote_routes"],
            "actual_return_equals_useful_a": transport["workers"]["a"]["actual_returned_bytes_d2h"] == transport["workers"]["a"]["useful_returned_contribution_bytes"],
            "actual_return_equals_useful_b": transport["workers"]["b"]["actual_returned_bytes_d2h"] == transport["workers"]["b"]["useful_returned_contribution_bytes"],
            "bytes_saved_positive": transport["returned_bytes_saved_vs_d5"] > 0,
            "zero_route_a_observed": transport["workers"]["a"]["zero_route_layers"] > 0,
            "zero_route_b_observed": transport["workers"]["b"]["zero_route_layers"] > 0,
        }
        passed = diagnostics["all_cases_passed"] and all(graph_checks.values()) and all(counter_checks.values())
        result = {"schema": "inferswarm.d6.transport-primitive/1",
                  "freetoken_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "model": ns.model, "revision": ns.revision, "placement_sha256": report["placement_sha256"],
                  "startup_s": time.monotonic()-started, "graph_checks": graph_checks,
                  "one_layer": diagnostics, "controlled_route_counts": curves,
                  "counter_checks": counter_checks, "snapshot": snapshot,
                  "paging_delta": _delta(before, _vm()),
                  "classification": "D6_TRANSPORT_PRIMITIVE_PASS" if passed else "D6_INVALID"}
        Path(ns.output).write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({"classification": result["classification"], "graph_checks": graph_checks,
                          "counter_checks": counter_checks, "curves": curves}, indent=2)); return 0 if passed else 2
    finally: engine.shutdown()


if __name__ == "__main__": raise SystemExit(main())
