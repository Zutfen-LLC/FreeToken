"""D7 placement/residency/correctness primitive on the unchanged D6 executor."""
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
from freetoken.moe.inferswarm_d7_placement import D7_ARTIFACT_SHA256, load_d7_placement
from freetoken.moe.offload_cache import iter_offload_moe_layers
from freetoken.server.args import parse_args
from freetoken.server.launch import _resolve_server_gpu_args
from inferswarm_d3.primitive import _delta, _vm

PRIMARY = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
A = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
B = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"


def _pct(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] if lo == hi else values[lo] * (hi - position) + values[hi] * (position - lo)


def _dist(values):
    return {"n": len(values), "median_us": statistics.median(values), "p95_us": _pct(values, .95),
            "min_us": min(values), "max_us": max(values)}


def _config(ns):
    argv = ["--model", ns.model, "--gpu", PRIMARY,
            "--inferswarm-experimental-d6-count-aware-transport",
            "--inferswarm-d7-fanin-sparse-placement", "--inferswarm-d7-participation-diagnostics",
            "--inferswarm-experimental-d5-resident-loader", "--inferswarm-d5-loader-cpu-workers", "8",
            "--inferswarm-d3-active-workers", "ab", "--inferswarm-d7-placement", ns.placement,
            "--inferswarm-d3-worker-a-gpu", A, "--inferswarm-d3-worker-b-gpu", B,
            "--moe-backend", "offload", "--moe-cpu-layers", "0", "--nvfp4-backend", "triton",
            "--moe-cache-size", "3774", "--max-running-requests", "1", "--cuda-graph-max-bs", "1",
            "--sampling-defaults", "none"]
    parsed, _ = parse_args(argv)
    return _resolve_server_gpu_args(parsed)


def _ids(placement, layer_id: int, owner: str, remote_count: int) -> list[int]:
    remote = list(getattr(placement, f"worker_{owner}").per_layer[layer_id].expert_ids)
    a = set(placement.worker_a.per_layer[layer_id].expert_ids)
    b = set(placement.worker_b.per_layer[layer_id].expert_ids)
    local = [expert for expert in range(256) if expert not in a | b]
    if len(remote) < remote_count or len(local) < 8 - remote_count:
        raise RuntimeError("D7 layer lacks requested diagnostic geometry")
    return remote[:remote_count] + local[:8 - remote_count]


def _oracle_case(engine, executor, layer, ids_values, seed: int, label: str):
    ids = torch.tensor([ids_values], dtype=torch.int32, device=engine.device)
    weights = torch.arange(1, 9, dtype=torch.float32, device=engine.device).reshape(1, 8)
    weights.div_(weights.sum())
    hidden = torch.randn((1, executor.hidden_size), dtype=executor.hidden_dtype, device=engine.device,
                         generator=torch.Generator(device=engine.device).manual_seed(seed))
    cache = engine.moe_offload_cache
    cache.reset()
    reference = layer._decode_routed(hidden, weights, ids.clone()).clone()
    torch.cuda.synchronize(engine.device)
    cache.reset()
    candidate = executor.decode(layer, cache, hidden, weights, ids).clone()
    torch.cuda.synchronize(engine.device)
    error = (candidate.float() - reference.float()).abs()
    torch.testing.assert_close(candidate.float(), reference.float(), rtol=2e-3, atol=2e-3)
    return {"label": label, "layer": int(layer.layer_id), "route_ids": ids_values,
            "exact_output": bool(torch.equal(candidate, reference)),
            "max_absolute_deviation": float(error.max().item()), "rtol": .002, "atol": .002}


def _wall(engine, executor, layer, ids_values, seed: int, replays: int):
    ids = torch.tensor([ids_values], dtype=torch.int32, device=engine.device)
    weights = torch.full((1, 8), 1 / 8, dtype=torch.float32, device=engine.device)
    hidden = torch.randn((1, executor.hidden_size), dtype=executor.hidden_dtype, device=engine.device,
                         generator=torch.Generator(device=engine.device).manual_seed(seed))
    cache = engine.moe_offload_cache
    cache.reset()
    executor.decode(layer, cache, hidden, weights, ids)
    torch.cuda.synchronize(engine.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=engine.stream):
        executor.decode(layer, cache, hidden, weights, ids)
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize(engine.device)
    values = []
    for _ in range(replays):
        started = time.perf_counter_ns()
        graph.replay()
        torch.cuda.synchronize(engine.device)
        values.append((time.perf_counter_ns() - started) / 1000)
    return _dist(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replays", type=int, default=100)
    ns = parser.parse_args()
    ns.output.parent.mkdir(parents=True, exist_ok=True)
    before = _vm()
    started = time.monotonic()
    args = _config(ns)
    set_assigned_gpu((args.gpu_assigned or args.gpu)[0])
    engine = Engine(args)
    try:
        executor = engine.inferswarm_d6_count_aware_transport
        placement = load_d7_placement(ns.placement)
        report = executor.configuration_report()
        layers = {int(layer.layer_id): layer for layer in iter_offload_moe_layers(engine.model)}
        a_layer = next(layer for layer in range(40) if placement.worker_a.per_layer[layer].expert_ids)
        b_layer = next(layer for layer in range(40) if placement.worker_b.per_layer[layer].expert_ids)
        graph_checks = {
            "placement_sha": report["placement_sha256"] == D7_ARTIFACT_SHA256,
            "graph_bs1": report["graph_active"] and report["captured_batch_sizes"] == [1],
            "d6_executor": report["count_aware_expert_compute"] and report["count_aware_return_transport"],
            "joint_diagnostic": report["d7_participation_diagnostics"] is True,
            "fallback_zero": report["fallback_count"] == 0,
            "failure_zero": report["failure_count"] == 0,
            "recapture_zero": report["graph_recapture_count"] == 0,
            "host_sync_zero": report["steady_state_host_sync_count"] == 0,
        }
        banks = engine.inferswarm_d3_resident_banks
        residency_checks = {
            "a_3000": banks.worker_a.report.placement.remote_slots == 3000,
            "b_3000": banks.worker_b.report.placement.remote_slots == 3000,
            "a_bytes": banks.worker_a.report.placement.remote_resident_bytes == 5326848000,
            "b_bytes": banks.worker_b.report.placement.remote_resident_bytes == 5326848000,
            "loader_bulk_concurrent": banks.loader_profile["mode"] == "bulk" and banks.loader_profile["concurrent_ab"],
            "no_overlap": not bool(set(placement.worker_a.flat_ids_in_rank_order) & set(placement.worker_b.flat_ids_in_rank_order)),
        }
        executor.reset_counters()
        cases = []
        sequence = ((a_layer, "a", 8, "a_only_8"), (a_layer, "a", 4, "local_plus_a"),
                    (a_layer, "a", 1, "a_stale_tail_1"), (a_layer, "a", 0, "a_zero_route_transition"),
                    (b_layer, "b", 8, "b_only_8"), (b_layer, "b", 4, "local_plus_b"),
                    (b_layer, "b", 1, "b_stale_tail_1"), (b_layer, "b", 0, "b_zero_route_transition"))
        for index, (layer_id, owner, count, label) in enumerate(sequence):
            cases.append(_oracle_case(engine, executor, layers[layer_id], _ids(placement, layer_id, owner, count),
                                      9700 + index, label))
        timing = {
            "d7_a_only_layer_wall": _wall(engine, executor, layers[a_layer], _ids(placement, a_layer, "a", 4), 9801, ns.replays),
            "d7_b_only_layer_wall": _wall(engine, executor, layers[b_layer], _ids(placement, b_layer, "b", 4), 9802, ns.replays),
            "equal_ab_reference": {"median_us": 244.736,
                                   "source": "/home/zutfen/inferswarm-evidence/architecture-search/d6-count-aware-transport/d6-critical-path.json",
                                   "note": "accepted D6 uninstrumented equal A+B layer wall"},
            "d7_remaining_ab": None,
        }
        snapshot = executor.snapshot()
        counter_checks = {
            "ownership_exact": snapshot["ownership"]["selection_arithmetic_exact"],
            "physical_equals_owned": snapshot["physical"]["physical_worker_invocations_equal_owned_remote_routes"],
            "zero_route_a_observed": snapshot["transport"]["workers"]["a"]["zero_route_layers"] > 0,
            "zero_route_b_observed": snapshot["transport"]["workers"]["b"]["zero_route_layers"] > 0,
            "both_event_zero": snapshot["d7_participation"]["counts"]["both"] == 0,
            "returned_zero_route_payload_contract": report["zero_route_return_payload_bytes"] == 0,
            "weight_movement_zero": report["steady_state_expert_weight_bytes_host_to_worker_a"] == 0
                                    and report["steady_state_expert_weight_bytes_host_to_worker_b"] == 0,
        }
        split_path = {"required": False, "reason": "frozen D7 placement has zero split layers"}
        passed = all(graph_checks.values()) and all(residency_checks.values()) and all(counter_checks.values())
        result = {
            "schema": "inferswarm.d7.primitive/1",
            "freetoken_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "model": ns.model, "revision": ns.revision, "placement_sha256": D7_ARTIFACT_SHA256,
            "startup_s": time.monotonic() - started, "graph_checks": graph_checks,
            "residency_checks": residency_checks, "one_layer_oracles": cases,
            "split_layer_path": split_path, "stale_tail_protection": True,
            "representative_layer_walls_us": timing, "counter_checks": counter_checks,
            "snapshot": snapshot, "paging_delta": _delta(before, _vm()),
            "classification": "D7_FANIN_SPARSE_PRIMITIVE_PASS" if passed else "D7_INVALID",
        }
        ns.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"classification": result["classification"], "graph_checks": graph_checks,
                          "residency_checks": residency_checks, "counter_checks": counter_checks,
                          "timing": timing}, indent=2))
        return 0 if passed else 2
    finally:
        engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
