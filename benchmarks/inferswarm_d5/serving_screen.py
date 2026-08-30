"""Frozen-order D5 F0/C1/C2/C3 W4 serving screen."""
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
from inferswarm_d3.serving_screen import vm, delta, gpu_state, timing

GPU0 = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
A = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
B = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"
D3_SHA = "6677fe1c506376a55aa8dcabb8d5761dc0373ced9d9b053209991059556d5887"
D4_SHA = "283595b7559bb3aa46a08c7d00cfef1e0a77eb62967d6392c618a63f35d34cdf"
KV = 17075


def command(root: Path, model: str, port: int, arm: str, equal: str, weighted: str):
    compact = arm != "F0"; shape = "b" if arm == "C1" else "ab"
    placement = weighted if arm == "C3" else equal
    cmd = [str(root / ".venv/bin/python"), "-m", "freetoken.cli", "serve", "--model", model,
           "--host", "127.0.0.1", "--port", str(port), "--gpu", GPU0,
           "--moe-backend", "offload", "--moe-cpu-layers", "0", "--nvfp4-backend", "triton",
           "--moe-cache-size", "3774", "--kv-reserve-tokens", str(KV), "--num-tokens", str(KV),
           "--memory-ratio", "0.85", "--max-running-requests", "1", "--cuda-graph-max-bs", "1",
           "--sampling-defaults", "none", "--moe-layer-timing-role", "candidate",
           "--moe-layer-timing-max-steps", "400", "--inferswarm-experimental-d5-resident-loader",
           "--inferswarm-d5-loader-cpu-workers", "8", "--inferswarm-d3-active-workers", shape]
    if compact: cmd += ["--inferswarm-experimental-d5-compact-routes"]
    else: cmd += ["--inferswarm-experimental-d3-graph-multiworker"]
    if arm == "C3": cmd += ["--inferswarm-d5-weighted-placement", "--inferswarm-d4-placement", placement]
    else: cmd += ["--inferswarm-d3-placement", placement]
    if "a" in shape: cmd += ["--inferswarm-d3-worker-a-gpu", A]
    if "b" in shape: cmd += ["--inferswarm-d3-worker-b-gpu", B]
    return cmd


def contract(runtime: dict[str, Any], arm: str):
    compact = arm != "F0"; shape = "b" if arm == "C1" else "ab"
    rt, moe, cache = runtime["runtime"], runtime["moe"], runtime["cache"]
    d3, d5 = runtime["inferswarm_d3_graph_multiworker"], runtime["inferswarm_d5_compact_routes"]
    expected_sha = D4_SHA if arm == "C3" else D3_SHA
    selected = d5 if compact else d3
    checks = {"gpu_decode": moe["decode_target"] == "gpu", "zero_cpu_moe": moe["cpu_layers_resolved"] == [],
              "triton": runtime["nvfp4"]["resolved"] == "triton", "cache_3774": cache["resolved_slots"] == 3774,
              "kv_17075": rt["num_pages"] == KV, "graph_bs1": rt["cuda_graph_captured_bs"] == [1],
              "selected_enabled": selected["enabled"] is True, "selected_graph": selected["graph_active"] is True,
              "fallback_zero": selected["fallback_count"] == 0, "failure_zero": selected["failure_count"] == 0,
              "recapture_zero": selected["graph_recapture_count"] == 0,
              "host_sync_zero": selected["steady_state_host_sync_count"] == 0,
              "active_workers": selected["active_workers"] == list(shape),
              "placement_sha": selected.get("placement_sha256", selected.get("corrected_placement_sha256")) == expected_sha,
              "executor_isolated": (not d3["enabled"] if compact else not d5["enabled"]),
              "loader_frozen": runtime["inferswarm_d5_resident_loader"]["mode"] == "bulk"
                               and runtime["inferswarm_d5_resident_loader"]["cpu_workers"] == 8}
    return {"passed": all(checks.values()), "checks": checks}


def run_arm(ns, arm: str):
    port = free_port(); origin = f"http://127.0.0.1:{port}"; started = time.monotonic()
    cmd = command(ns.repo, ns.model, port, arm, ns.equal_placement, ns.weighted_placement)
    handle = start_server(cmd, origin, str(ns.output_dir / f"d5-{arm.lower()}.server.log"),
                          env_overrides={"PYTHONPATH": "python:benchmarks"}, ready_timeout=900, echo=True)
    try:
        ready = time.monotonic(); runtime = fetch_instrumentation(origin)["runtime_config"]
        gate = contract(runtime, arm)
        if not gate["passed"]: raise RuntimeError(f"{arm} runtime contract failed: {gate}")
        workload = load_manifest(ns.manifest, canonical=True).by_class()["W4"]
        body = workload.request_body(ns.revision, sampling_override=CANONICAL_GREEDY_SAMPLING)
        gpu_before = gpu_state(); floor = prefill_seq_floor(origin)
        warmup = measure_generation(origin, body, prefill_seq_floor=floor, store_text=False)
        reset = moe_instrumentation(origin, "reset", timeout=300)
        rows = [{"phase": "warmup", "repetition": 0, **warmup}]
        for repetition in range(3):
            before = vm(handle.proc.pid); floor = prefill_seq_floor(origin)
            generated = measure_generation(origin, body, prefill_seq_floor=floor, store_text=False)
            after = vm(handle.proc.pid); generated.update(phase="retained", repetition=repetition,
                paging_before=before, paging_after=after, paging_delta=delta(before, after))
            if not generated["completion_matches_request"]: raise RuntimeError(f"{arm} output length mismatch")
            if any(generated["paging_delta"][k] != 0 for k in ("pswpin", "pswpout")):
                raise RuntimeError(f"{arm} paging contamination")
            rows.append(generated)
        snapshot = moe_instrumentation(origin, "snapshot", timeout=300); gpu_after = gpu_state()
        selected = snapshot["inferswarm_d5_compact_routes"] if arm != "F0" else snapshot["inferswarm_d3_graph_multiworker"]
        ownership = selected["ownership"]
        if not ownership["selection_arithmetic_exact"] or not ownership["no_route_dropped"] or not ownership["no_route_duplicated"]:
            raise RuntimeError(f"{arm} ownership invalid")
        if any(selected[k] for k in ("fallback_count", "failure_count", "graph_recapture_count", "steady_state_host_sync_count")):
            raise RuntimeError(f"{arm} graph counters dirty")
        if arm != "F0" and not selected["physical"]["physical_worker_invocations_equal_owned_remote_routes"]:
            raise RuntimeError(f"{arm} physical count mismatch")
        measured = rows[1:]; rates = [float(x["decode_tok_s"]) for x in measured]
        result = {"schema": "inferswarm.d5.serving-arm/1", "arm": arm, "command": cmd,
                  "freetoken_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ns.repo, text=True).strip(),
                  "startup_s": ready - started, "runtime_contract": gate, "resolved_runtime": runtime,
                  "loader": runtime["inferswarm_d5_resident_loader"], "generations": rows,
                  "reset": reset, "counters": selected, "ownership": ownership,
                  "physical": selected.get("physical"), "gpu_before": gpu_before, "gpu_after": gpu_after,
                  "analysis": {"decode_tok_s": {"each_retained": rates, "median": statistics.median(rates),
                                                 "min": min(rates), "max": max(rates)},
                               "timing": timing(snapshot, measured)}, "status": "complete"}
        (ns.output_dir / f"d5-{arm.lower()}.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally: stop_server(handle)


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--model", required=True); p.add_argument("--revision", required=True)
    p.add_argument("--manifest", required=True); p.add_argument("--equal-placement", required=True)
    p.add_argument("--weighted-placement", required=True); p.add_argument("--output-dir", type=Path, required=True)
    ns = p.parse_args(); ns.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_arm(ns, arm) for arm in ("F0", "C1", "C2", "C3")]
    t = {row["arm"]: row["analysis"]["decode_tok_s"]["median"] for row in rows}
    equal_gain, e3_equal = t["C2"] / t["F0"], t["C2"] / t["C1"]
    weighting_gain, e3_weighted = t["C3"] / t["C2"], t["C3"] / t["C1"]
    classification = ("D5_DUMMY_TAX_CONFIRMED" if equal_gain >= 1.10 else
                      "D5_DUMMY_TAX_PARTIAL" if equal_gain >= 1.03 else
                      "D5_DUMMY_TAX_NOT_SUPPORTED" if equal_gain >= .97 else "D5_COMPACTION_HARMFUL")
    analysis = {"schema": "inferswarm.d5.serving-analysis/1", "order": ["F0", "C1", "C2", "C3"],
                "median_decode_tok_s": t, "COMPACT_EQUAL_GAIN": equal_gain,
                "COMPACT_E3_EQUAL": e3_equal, "COMPACT_WEIGHTING_GAIN": weighting_gain,
                "COMPACT_E3_WEIGHTED": e3_weighted,
                "historical": {"compact_equal_over_d3_s3": t["C2"] / 51.2377,
                               "compact_weighted_over_d3_s3": t["C3"] / 51.2377,
                               "compact_weighted_over_d3_s2b": t["C3"] / 67.8157},
                "classification": classification}
    (ns.output_dir / "d5-analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
