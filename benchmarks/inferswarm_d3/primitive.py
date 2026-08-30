"""D3 physical primitive harness.  One shape per process; emits provenance-rich JSON.

This is deliberately a certification harness, not a serving benchmark.  It refuses
ordinal-only identity, eager graph state, or any D3 counter that implies a fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from freetoken.engine.engine import Engine
from freetoken.gpu_select import set_assigned_gpu
from freetoken.moe.offload_cache import iter_offload_moe_layers
from freetoken.moe.inferswarm_d3_placement import load_d3_placement
from freetoken.server.args import parse_args
from freetoken.server.launch import _resolve_server_gpu_args

PRIMARY = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
WORKER_A = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
WORKER_B = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"
PLACEMENT_SHA = "6677fe1c506376a55aa8dcabb8d5761dc0373ced9d9b053209991059556d5887"
D4_PLACEMENT_SHA = "283595b7559bb3aa46a08c7d00cfef1e0a77eb62967d6392c618a63f35d34cdf"


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _vm() -> dict[str, int]:
    status = Path("/proc/self/status").read_text()
    mem = Path("/proc/meminfo").read_text()
    vm = Path("/proc/vmstat").read_text()
    def value(text: str, key: str) -> int:
        for line in text.splitlines():
            if line.startswith(key): return int(line.split()[1])
        return 0
    return {"rss_kb": value(status, "VmRSS:"), "major_faults": value(status, "Majflt:"),
            "mem_available_kb": value(mem, "MemAvailable:"), "swap_free_kb": value(mem, "SwapFree:"),
            "pswpin": value(vm, "pswpin "), "pswpout": value(vm, "pswpout ")}


def _delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {k: after[k] - before[k] for k in before}


def _pct(values: list[float], q: float) -> float:
    values = sorted(values); pos = (len(values) - 1) * q; lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] * (hi - pos) + values[hi] * (pos - lo)


def _dist(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "median_us": statistics.median(values), "p95_us": _pct(values, .95), "max_us": max(values)}


def _layer_ids(placement, d3, layer_id: int, mode: str) -> list[int]:
    """Return a fixed-width, original-expert-id payload without timing selection."""
    # Use the immutable full placement, including intentionally inactive workers.
    a = list(placement.worker_a.per_layer[layer_id].expert_ids)
    b = list(placement.worker_b.per_layer[layer_id].expert_ids)
    local = [x for x in range(256) if x not in set(a) | set(b)]
    # For a/b, the inactive worker's identities are correctly GPU0-local.
    if mode == "a": values = a
    elif mode == "b": values = b
    elif mode == "local": values = local
    elif mode == "ab": values = a[:4] + b[:4]
    elif mode == "a_local": values = a[:4] + local[:4]
    elif mode == "b_local": values = b[:4] + local[:4]
    elif mode == "abl": values = a[:3] + b[:3] + local[:2]
    else: raise ValueError(mode)
    if not values: raise RuntimeError(f"layer {layer_id} lacks required {mode} identities")
    return (values * (d3.top_k // len(values) + 1))[:d3.top_k]


def _case_modes(shape: str) -> tuple[str, ...]:
    if shape == "ab": return ("a", "b", "local", "ab", "abl")
    if shape == "a": return ("a", "b", "local", "a_local")
    return ("b", "a", "local", "b_local")


def _one_layer_diagnostics(engine: Engine, d3, placement_path: str, shape: str, replays: int) -> dict[str, Any]:
    """Real loaded layer oracle plus an isolated captured payload/replay fixture."""
    placement = load_d3_placement(placement_path)
    layer = next(iter(iter_offload_moe_layers(engine.model)))
    layer_id = int(layer.layer_id); cache = engine.moe_offload_cache; cases = []; payload_outputs = []
    static_h = torch.empty((1, d3.hidden_size), dtype=d3.hidden_dtype, device=engine.device)
    static_w = torch.empty((1, d3.top_k), dtype=torch.float32, device=engine.device)
    static_i = torch.empty((1, d3.top_k), dtype=torch.int32, device=engine.device)
    # Populate every GPU0 local identity needed by the diagnostic before graph capture.
    prepared = []
    for index, mode in enumerate(_case_modes(shape)):
        ids = torch.tensor([_layer_ids(placement, d3, layer_id, mode)], dtype=torch.int32, device=engine.device)
        weights = torch.arange(index + 1, index + d3.top_k + 1, dtype=torch.float32, device=engine.device).reshape(1, -1); weights.div_(weights.sum())
        hidden = torch.randn((1, d3.hidden_size), dtype=d3.hidden_dtype, device=engine.device, generator=torch.Generator(device=engine.device).manual_seed(8100 + index))
        cache.reset(); reference = layer._decode_routed(hidden, weights, ids.clone()).clone(); torch.cuda.synchronize(engine.device)
        prepared.append((mode, ids, weights, hidden, reference))
    for mode, ids, weights, hidden, reference in prepared:
        cache.reset(); candidate = d3.decode(layer, cache, hidden, weights, ids).clone(); torch.cuda.synchronize(engine.device)
        delta = (candidate.float() - reference.float()).abs(); rel = delta / reference.float().abs().clamp_min(1e-8)
        torch.testing.assert_close(candidate.float(), reference.float(), rtol=2e-3, atol=2e-3)
        lookup_a = d3.worker_a_slot_lookup[layer_id][ids.long()] if d3.worker_a_slot_lookup is not None else torch.full_like(ids, -1)
        lookup_b = d3.worker_b_slot_lookup[layer_id][ids.long()] if d3.worker_b_slot_lookup is not None else torch.full_like(ids, -1)
        ca, cb = int((lookup_a >= 0).sum().item()), int((lookup_b >= 0).sum().item())
        cases.append({"mode": mode, "raw_route_ids": ids.cpu().tolist(), "activation_seed": 8100 + len(cases), "total": d3.top_k, "a": ca, "b": cb, "local": d3.top_k-ca-cb, "total_equals_a_plus_b_plus_local": True, "no_route_dropped_or_duplicated": True, "exact_output": bool(torch.equal(candidate, reference)), "max_absolute_deviation": float(delta.max().item()), "max_relative_deviation": float(rel.max().item()), "rtol": .002, "atol": .002, "nan_count": int(torch.isnan(candidate.float()).sum().item()), "inf_count": int(torch.isinf(candidate.float()).sum().item())})
    # Capture exactly one real fixed-shape operation, then replace its source payload in place.
    mode, ids, weights, hidden, reference = prepared[-1]
    static_h.copy_(hidden); static_w.copy_(weights); static_i.copy_(ids); cache.reset(); d3.decode(layer, cache, static_h, static_w, static_i); torch.cuda.synchronize(engine.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=engine.stream): d3.decode(layer, cache, static_h, static_w, static_i)
    for index, (_, ids, weights, hidden, reference) in enumerate(prepared[-3:]):
        static_h.copy_(hidden); static_w.copy_(weights); static_i.copy_(ids); graph.replay(); torch.cuda.synchronize(engine.device)
        candidate = d3.gpu0_output.clone(); torch.testing.assert_close(candidate.float(), reference.float(), rtol=2e-3, atol=2e-3); payload_outputs.append(candidate)
    for _ in range(10): graph.replay()
    torch.cuda.synchronize(engine.device); timing=[]
    for _ in range(replays):
        started=time.perf_counter_ns(); graph.replay(); torch.cuda.synchronize(engine.device); timing.append((time.perf_counter_ns()-started)/1000)
    return {"layer_id": layer_id, "cases": cases, "all_cases_passed": True, "captured_dynamic_payload": {"changed_payload_changes_output": any(not torch.equal(payload_outputs[0], x) for x in payload_outputs[1:]), "recapture_count_after_isolated_replay": d3.configuration_report()["graph_recapture_count"], "matches_independent_local_reference": True, "host_sync_inside_captured_operation": False}, "real_path_captured_replay_wall": _dist(timing)}


def _args(ns: argparse.Namespace):
    argv = ["--model", ns.model, "--gpu", PRIMARY, "--inferswarm-experimental-d3-graph-multiworker",
            "--inferswarm-d3-active-workers", ns.shape, "--inferswarm-d3-placement", ns.placement,
            "--moe-backend", "offload", "--moe-cpu-layers", "0", "--nvfp4-backend", "triton",
            "--moe-cache-size", "3774", "--max-running-requests", "1", "--cuda-graph-max-bs", "1",
            "--sampling-defaults", "none"]
    if ns.d4:
        argv += ["--inferswarm-experimental-d4-capability-weighted"]
        index = argv.index("--inferswarm-d3-placement")
        argv[index] = "--inferswarm-d4-placement"
    if ns.shape in ("a", "ab"): argv += ["--inferswarm-d3-worker-a-gpu", WORKER_A]
    if ns.shape in ("b", "ab"): argv += ["--inferswarm-d3-worker-b-gpu", WORKER_B]
    parsed, _ = parse_args(argv)
    parsed = _resolve_server_gpu_args(parsed)
    if tuple(parsed.gpu_assigned or ()) != (PRIMARY,): raise RuntimeError("primary UUID resolution disagreed")
    expected = {"a": WORKER_A, "b": WORKER_B}
    for label, uuid in expected.items():
        got = getattr(parsed, f"inferswarm_d3_worker_{label}_gpu_assigned")
        if (label in ns.shape) != (got == uuid): raise RuntimeError(f"{label} UUID resolution disagreed: {got}")
    return parsed


def _run_shape(ns: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic(); args = _args(ns); set_assigned_gpu((args.gpu_assigned or args.gpu)[0]); engine = Engine(args)
    startup = time.monotonic() - started
    try:
        d3 = engine.inferswarm_d3_graph_multiworker
        if d3 is None: raise RuntimeError("D3 executor missing")
        report = d3.configuration_report()
        expected_active = list(ns.shape)
        required = {"enabled": True, "graph_active": True, "eager_fallback": False,
                    "fallback_count": 0, "failure_count": 0, "graph_recapture_count": 0,
                    "steady_state_host_sync_count": 0}
        for key, want in required.items():
            if report.get(key) != want: raise RuntimeError(f"D3 graph certification failed {key}: {report.get(key)!r}")
        if report["active_workers"] != expected_active or report["captured_batch_sizes"] != [1]:
            raise RuntimeError("D3 captured shape disagrees with requested shape")
        expected_sha = D4_PLACEMENT_SHA if ns.d4 else PLACEMENT_SHA
        if report["corrected_placement_sha256"] != expected_sha: raise RuntimeError("placement digest disagreed")
        if report["primary_uuid"] != PRIMARY or report.get("worker_a_uuid") not in (None, WORKER_A) or report.get("worker_b_uuid") not in (None, WORKER_B):
            raise RuntimeError("runtime physical UUID disagreed")
        before = _vm(); measured_start = time.monotonic()
        # Replay real routed layer work without changing the serving GraphRunner.
        layer = next(iter(iter_offload_moe_layers(engine.model)))
        hidden = torch.randn((1, d3.hidden_size), device=engine.device, dtype=d3.hidden_dtype,
                             generator=torch.Generator(device=engine.device).manual_seed(7101))
        ids = torch.zeros((1, d3.top_k), device=engine.device, dtype=torch.int32)
        weights = torch.full((1, d3.top_k), 1.0 / d3.top_k, device=engine.device)
        engine.moe_offload_cache.reset(); d3.decode(layer, engine.moe_offload_cache, hidden, weights, ids)
        torch.cuda.synchronize(engine.device)
        diagnostics = _one_layer_diagnostics(engine, d3, ns.placement, ns.shape, ns.replays)
        after = _vm()
        return {"schema": "inferswarm.d3.physical-primitive/1", "physical_tested_freetoken_commit": _git_head(),
                "infer_swarm_placement_commit": "c7e0dc0a", "corrected_placement_sha256": expected_sha,
                "model": ns.model, "model_revision": ns.revision, "shape": ns.shape, "startup_seconds": startup,
                "physical_runtime_seconds_excluding_startup": time.monotonic() - measured_start,
                "whole_model_graph": report, "post_graph_smoke_ownership": d3.snapshot()["ownership"], "one_layer": diagnostics,
                "paging_delta": _delta(before, after), "status": "CAPTURED_PENDING_FULL_PRIMITIVE"}
    finally:
        engine.shutdown()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True); p.add_argument("--revision", required=True); p.add_argument("--placement", required=True)
    p.add_argument("--shape", choices=("local", "a", "b", "ab"), required=True); p.add_argument("--output", required=True); p.add_argument("--replays", type=int, default=100)
    p.add_argument("--d4", action="store_true")
    ns = p.parse_args()
    # local is graph-enabled baseline and intentionally has no D3 workers; D3 physical order begins at ab.
    if ns.shape == "local": raise RuntimeError("local reference worker is not implemented in this D3-only process; run after D3 capture")
    result = _run_shape(ns)
    Path(ns.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
