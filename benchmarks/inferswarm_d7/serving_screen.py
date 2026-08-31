"""Frozen-order D7 L0/L1/L2 W4 serving screen and classification."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferswarm_phase0.client import fetch_instrumentation, free_port, measure_generation, prefill_seq_floor, start_server, stop_server
from inferswarm_phase0.manifest import CANONICAL_GREEDY_SAMPLING, load_manifest
from inferswarm_phase1.campaign import moe_instrumentation
from inferswarm_d3.serving_screen import delta, gpu_state, vm

GPU0 = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
A = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
B = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"
D3_SHA = "6677fe1c506376a55aa8dcabb8d5761dc0373ced9d9b053209991059556d5887"
D7_SHA = "c360cad506fa4dbc2f768b24d8ad5dfd1d10956aafc88a5fa6e2736dfe0581d1"
KV = 17075
ORDER = ("L0", "L1", "L2")


def _major_faults(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[9])


def command(root: Path, model: str, port: int, arm: str, d3_placement: str, d7_placement: str) -> list[str]:
    shape = "b" if arm == "L0" else "ab"
    sparse = arm == "L2"
    cmd = [str(root / ".venv/bin/python"), "-m", "freetoken.cli", "serve", "--model", model,
           "--host", "127.0.0.1", "--port", str(port), "--gpu", GPU0,
           "--moe-backend", "offload", "--moe-cpu-layers", "0", "--nvfp4-backend", "triton",
           "--moe-cache-size", "3774", "--kv-reserve-tokens", str(KV), "--num-tokens", str(KV),
           "--memory-ratio", "0.85", "--max-running-requests", "1", "--cuda-graph-max-bs", "1",
           "--sampling-defaults", "none", "--inferswarm-experimental-d5-resident-loader",
           "--inferswarm-d5-loader-cpu-workers", "8", "--inferswarm-experimental-d6-count-aware-transport",
           "--inferswarm-d7-participation-diagnostics", "--inferswarm-d3-active-workers", shape]
    if sparse:
        cmd += ["--inferswarm-d7-fanin-sparse-placement", "--inferswarm-d7-placement", d7_placement]
    else:
        cmd += ["--inferswarm-d3-placement", d3_placement]
    if "a" in shape:
        cmd += ["--inferswarm-d3-worker-a-gpu", A]
    if "b" in shape:
        cmd += ["--inferswarm-d3-worker-b-gpu", B]
    return cmd


def contract(runtime: dict[str, Any], arm: str) -> dict[str, Any]:
    shape = "b" if arm == "L0" else "ab"
    expected_sha = D7_SHA if arm == "L2" else D3_SHA
    rt, moe, cache = runtime["runtime"], runtime["moe"], runtime["cache"]
    d6 = runtime["inferswarm_d6_count_aware_transport"]
    checks = {
        "gpu_decode": moe["decode_target"] == "gpu",
        "zero_cpu_moe": moe["cpu_layers_resolved"] == [],
        "triton_nvfp4": runtime["nvfp4"]["resolved"] == "triton",
        "cache_3774": cache["resolved_slots"] == 3774,
        "kv_17075": rt["num_pages"] == KV,
        "max_requests_1": rt["max_running_req"] == 1,
        "graph_bs1": rt["cuda_graph_capture_happened"] is True and rt["cuda_graph_captured_bs"] == [1],
        "d6_enabled": d6["enabled"] is True,
        "d6_graph": d6["graph_active"] is True,
        "active_workers": d6["active_workers"] == list(shape),
        "placement_sha": d6["placement_sha256"] == expected_sha,
        "count_aware_compute": d6["count_aware_expert_compute"] is True,
        "count_aware_metadata": d6["count_aware_inbound_metadata"] is True,
        "count_aware_return": d6["count_aware_return_transport"] is True,
        "d7_participation_diagnostics": d6["d7_participation_diagnostics"] is True,
        "fallback_zero": d6["fallback_count"] == 0,
        "failure_zero": d6["failure_count"] == 0,
        "recapture_zero": d6["graph_recapture_count"] == 0,
        "host_sync_zero": d6["steady_state_host_sync_count"] == 0,
        "weight_a_zero": d6["steady_state_expert_weight_bytes_host_to_worker_a"] == 0,
        "weight_b_zero": d6["steady_state_expert_weight_bytes_host_to_worker_b"] == 0,
        "loader_bulk_8": runtime["inferswarm_d5_resident_loader"]["mode"] == "bulk"
                         and runtime["inferswarm_d5_resident_loader"]["cpu_workers"] == 8,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _mechanism(selected: dict[str, Any]) -> dict[str, Any]:
    joint = selected["d7_participation"]
    counts = joint["counts"]
    total = joint["total_layer_events"]
    workers = selected["transport"]["workers"]
    ownership, physical = selected["ownership"], selected["physical"]
    return {
        "total_layer_token_events": total,
        "event_counts": counts,
        "event_percentages": {key: 100 * value / total for key, value in counts.items()},
        "mean_remote_workers_active": joint["mean_remote_workers_active"],
        "routes": {"a": ownership["worker_a_active_count"], "b": ownership["worker_b_active_count"],
                   "local": ownership["local_active_count"], "total": ownership["original_topk_selections"]},
        "route_shares": {key: ownership[name] / ownership["original_topk_selections"]
                         for key, name in (("a", "worker_a_active_count"), ("b", "worker_b_active_count"),
                                           ("local", "local_active_count"))},
        "physical": {
            "represented_branches": {label: row["layer_calls"] for label, row in workers.items()},
            "route_active_branches": {label: row["layer_calls"] - row["zero_route_layers"] for label, row in workers.items()},
            "zero_route_branches": {label: row["zero_route_layers"] for label, row in workers.items()},
            "expert_route_invocations": {"a": physical["worker_a_expert_invocations"],
                                         "b": physical["worker_b_expert_invocations"],
                                         "local": physical["local_expert_invocations"]},
            "returned_payload_bytes_d2h": {label: row["actual_returned_bytes_d2h"] for label, row in workers.items()},
        },
    }


def run_arm(ns, arm: str) -> dict[str, Any]:
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    started = time.monotonic()
    cmd = command(ns.repo, ns.model, port, arm, ns.d3_placement, ns.d7_placement)
    log_path = ns.output_dir / f"d7-{arm.lower()}.server.log"
    handle = start_server(cmd, origin, str(log_path), env_overrides={"PYTHONPATH": "python:benchmarks"},
                          ready_timeout=900, echo=True)
    try:
        ready = time.monotonic()
        runtime = fetch_instrumentation(origin)["runtime_config"]
        gate = contract(runtime, arm)
        if not gate["passed"]:
            raise RuntimeError(f"{arm} runtime contract failed: {gate}")
        workload = load_manifest(ns.manifest, canonical=True).by_class()["W4"]
        body = workload.request_body(ns.revision, sampling_override=CANONICAL_GREEDY_SAMPLING)
        gpu_before = gpu_state()
        floor = prefill_seq_floor(origin)
        warmup = measure_generation(origin, body, prefill_seq_floor=floor, store_text=False)
        reset = moe_instrumentation(origin, "reset", timeout=300)
        generations = [{"phase": "warmup", "repetition": 0, **warmup}]
        for repetition in range(3):
            before = vm(handle.proc.pid)
            major_before = _major_faults(handle.proc.pid)
            floor = prefill_seq_floor(origin)
            generated = measure_generation(origin, body, prefill_seq_floor=floor, store_text=False)
            after = vm(handle.proc.pid)
            major_after = _major_faults(handle.proc.pid)
            generated.update(phase="retained", repetition=repetition, paging_before=before, paging_after=after,
                             paging_delta=delta(before, after), major_faults_delta=major_after - major_before)
            if not generated["completion_matches_request"]:
                raise RuntimeError(f"{arm} retained output length mismatch")
            if any(generated["paging_delta"][key] != 0 for key in ("pswpin", "pswpout")) or generated["major_faults_delta"] != 0:
                raise RuntimeError(f"{arm} paging contamination")
            generations.append(generated)
        snapshot = moe_instrumentation(origin, "snapshot", timeout=300)
        gpu_after = gpu_state()
        selected = snapshot["inferswarm_d6_count_aware_transport"]
        ownership = selected["ownership"]
        if not all(ownership[key] for key in ("selection_arithmetic_exact", "no_route_dropped", "no_route_duplicated")):
            raise RuntimeError(f"{arm} ownership arithmetic failed")
        dirty = ("fallback_count", "failure_count", "graph_recapture_count", "steady_state_host_sync_count",
                 "steady_state_expert_weight_bytes_host_to_worker_a", "steady_state_expert_weight_bytes_host_to_worker_b")
        if any(selected[key] for key in dirty):
            raise RuntimeError(f"{arm} executor counters dirty")
        if not selected["physical"]["physical_worker_invocations_equal_owned_remote_routes"]:
            raise RuntimeError(f"{arm} physical invocation mismatch")
        retained = generations[1:]
        rates = [float(row["decode_tok_s"]) for row in retained]
        result = {
            "schema": "inferswarm.d7.serving-arm/1", "arm": arm, "order": list(ORDER),
            "freetoken_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ns.repo, text=True).strip(),
            "model": ns.model, "revision": ns.revision, "command": cmd,
            "startup_s": ready - started, "runtime_contract": gate, "resolved_runtime": runtime,
            "loader": runtime["inferswarm_d5_resident_loader"], "generations": generations,
            "reset": reset, "counters": selected, "ownership": ownership, "physical": selected["physical"],
            "transport": selected["transport"], "mechanism": _mechanism(selected),
            "gpu_before": gpu_before, "gpu_after": gpu_after,
            "analysis": {"decode_tok_s": {"each_retained": rates, "median": statistics.median(rates),
                                             "min": min(rates), "max": max(rates)},
                         "paging_valid": True},
            "status": "complete",
        }
        (ns.output_dir / f"d7-{arm.lower()}.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        stop_server(handle)


def classify(affinity_gain: float, d7_e3: float, pab_reduction: float) -> str:
    if affinity_gain >= 1.08 and d7_e3 >= .90 and pab_reduction >= .50:
        return "D7_FANIN_SPARSE_STRONG"
    if affinity_gain >= 1.03 and d7_e3 >= .85:
        return "D7_FANIN_SPARSE_PROMISING"
    if affinity_gain >= 1.03:
        return "D7_FANIN_SPARSE_PARTIAL"
    if affinity_gain >= .97:
        return "D7_FANIN_SPARSE_NOT_SUPPORTED"
    return "D7_FANIN_SPARSE_HARMFUL"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--d3-placement", required=True)
    parser.add_argument("--d7-placement", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    ns = parser.parse_args()
    ns.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_arm(ns, arm) for arm in ORDER]
    by_arm = {row["arm"]: row for row in rows}
    rates = {arm: by_arm[arm]["analysis"]["decode_tok_s"]["median"] for arm in ORDER}
    t_b, t_equal, t_sparse = rates["L0"], rates["L1"], rates["L2"]
    affinity_gain = t_sparse / t_equal
    d7_e3 = t_sparse / t_b
    equal_mechanism, sparse_mechanism = by_arm["L1"]["mechanism"], by_arm["L2"]["mechanism"]
    equal_pab = equal_mechanism["event_counts"]["both"] / equal_mechanism["total_layer_token_events"]
    sparse_pab = sparse_mechanism["event_counts"]["both"] / sparse_mechanism["total_layer_token_events"]
    pab_reduction = 1 - sparse_pab / equal_pab
    participation_reduction = 1 - sparse_mechanism["mean_remote_workers_active"] / equal_mechanism["mean_remote_workers_active"]
    verdict = classify(affinity_gain, d7_e3, pab_reduction)
    analysis = {
        "schema": "inferswarm.d7.serving-analysis/1",
        "predeclared_thresholds_frozen_before_performance": True,
        "order": list(ORDER), "median_decode_tok_s": rates,
        "AFFINITY_GAIN": affinity_gain, "D7_E3": d7_e3,
        "E3_GAIN": d7_e3 / (t_equal / t_b),
        "P_BOTH_SERVING": {"equal": equal_pab, "sparse": sparse_pab},
        "MEAN_WORKERS_SERVING": {"equal": equal_mechanism["mean_remote_workers_active"],
                                 "sparse": sparse_mechanism["mean_remote_workers_active"]},
        "P_BOTH_REDUCTION_SERVING": pab_reduction,
        "PARTICIPATION_REDUCTION_SERVING": participation_reduction,
        "classification": verdict,
        "cleared_0_85": d7_e3 >= .85, "cleared_0_90": d7_e3 >= .90,
        "fourth_worker_recommended": d7_e3 >= .85 and verdict in ("D7_FANIN_SPARSE_STRONG", "D7_FANIN_SPARSE_PROMISING"),
        "fourth_worker_started": False,
    }
    (ns.output_dir / "d7-analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
