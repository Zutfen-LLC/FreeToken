"""Frozen-order D6 T0/T1/T2/T3 W4 serving screen and classification."""
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
from inferswarm_d3.serving_screen import delta, gpu_state, timing, vm

GPU0 = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
A = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
B = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"
D3_SHA = "6677fe1c506376a55aa8dcabb8d5761dc0373ced9d9b053209991059556d5887"
HISTORICAL_C3 = 57.93277996862942
KV = 17075


def _major_faults(pid: int) -> int:
    # /proc/PID/stat field 12 (majflt); split after the parenthesized comm field.
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[9])


def command(root: Path, model: str, port: int, arm: str, placement: str):
    d6 = arm in ("T2", "T3"); shape = "b" if arm in ("T0", "T2") else "ab"
    cmd = [str(root / ".venv/bin/python"), "-m", "freetoken.cli", "serve", "--model", model,
           "--host", "127.0.0.1", "--port", str(port), "--gpu", GPU0,
           "--moe-backend", "offload", "--moe-cpu-layers", "0", "--nvfp4-backend", "triton",
           "--moe-cache-size", "3774", "--kv-reserve-tokens", str(KV), "--num-tokens", str(KV),
           "--memory-ratio", "0.85", "--max-running-requests", "1", "--cuda-graph-max-bs", "1",
           "--sampling-defaults", "none", "--inferswarm-experimental-d5-resident-loader",
           "--inferswarm-d5-loader-cpu-workers", "8", "--inferswarm-d3-active-workers", shape,
           "--inferswarm-d3-placement", placement,
           "--inferswarm-experimental-d6-count-aware-transport" if d6 else "--inferswarm-experimental-d5-compact-routes"]
    if "a" in shape: cmd += ["--inferswarm-d3-worker-a-gpu", A]
    if "b" in shape: cmd += ["--inferswarm-d3-worker-b-gpu", B]
    return cmd


def contract(runtime: dict[str, Any], arm: str):
    d6 = arm in ("T2", "T3"); shape = "b" if arm in ("T0", "T2") else "ab"
    rt, moe, cache = runtime["runtime"], runtime["moe"], runtime["cache"]
    d5, d6rt = runtime["inferswarm_d5_compact_routes"], runtime["inferswarm_d6_count_aware_transport"]
    selected, absent = (d6rt, d5) if d6 else (d5, d6rt)
    checks = {"gpu_decode": moe["decode_target"] == "gpu", "zero_cpu_moe": moe["cpu_layers_resolved"] == [],
              "triton": runtime["nvfp4"]["resolved"] == "triton", "cache_3774": cache["resolved_slots"] == 3774,
              "kv_17075": rt["num_pages"] == KV, "graph_bs1": rt["cuda_graph_captured_bs"] == [1],
              "selected_enabled": selected["enabled"] is True, "selected_graph": selected["graph_active"] is True,
              "fallback_zero": selected["fallback_count"] == 0, "failure_zero": selected["failure_count"] == 0,
              "recapture_zero": selected["graph_recapture_count"] == 0,
              "host_sync_zero": selected["steady_state_host_sync_count"] == 0,
              "active_workers": selected["active_workers"] == list(shape),
              "placement_sha": selected["placement_sha256"] == D3_SHA,
              "other_executor_absent": absent["enabled"] is False,
              "loader_frozen": runtime["inferswarm_d5_resident_loader"]["mode"] == "bulk"
                               and runtime["inferswarm_d5_resident_loader"]["cpu_workers"] == 8}
    if d6:
        checks.update(count_aware_return=selected["count_aware_return_transport"] is True,
                      count_aware_metadata=selected["count_aware_inbound_metadata"] is True)
    return {"passed": all(checks.values()), "checks": checks}


def run_arm(ns, arm: str):
    port = free_port(); origin = f"http://127.0.0.1:{port}"; started = time.monotonic()
    cmd = command(ns.repo, ns.model, port, arm, ns.placement)
    handle = start_server(cmd, origin, str(ns.output_dir / f"d6-{arm.lower()}.server.log"),
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
            before = vm(handle.proc.pid); major_before = _major_faults(handle.proc.pid)
            floor = prefill_seq_floor(origin)
            generated = measure_generation(origin, body, prefill_seq_floor=floor, store_text=False)
            after = vm(handle.proc.pid); major_after = _major_faults(handle.proc.pid)
            generated.update(phase="retained", repetition=repetition, paging_before=before,
                             paging_after=after, paging_delta=delta(before, after),
                             major_faults_delta=major_after-major_before)
            if not generated["completion_matches_request"]: raise RuntimeError(f"{arm} output length mismatch")
            if any(generated["paging_delta"][key] != 0 for key in ("pswpin", "pswpout")) or generated["major_faults_delta"] != 0:
                raise RuntimeError(f"{arm} paging contamination: {generated['paging_delta']}, major={generated['major_faults_delta']}")
            rows.append(generated)
        snapshot = moe_instrumentation(origin, "snapshot", timeout=300); gpu_after = gpu_state()
        selected = snapshot["inferswarm_d6_count_aware_transport"] if arm in ("T2", "T3") else snapshot["inferswarm_d5_compact_routes"]
        ownership = selected["ownership"]
        if not all(ownership[k] for k in ("selection_arithmetic_exact", "no_route_dropped", "no_route_duplicated")):
            raise RuntimeError(f"{arm} ownership invalid")
        if any(selected[k] for k in ("fallback_count", "failure_count", "graph_recapture_count", "steady_state_host_sync_count")):
            raise RuntimeError(f"{arm} graph counters dirty")
        if not selected["physical"]["physical_worker_invocations_equal_owned_remote_routes"]:
            raise RuntimeError(f"{arm} physical count mismatch")
        measured = rows[1:]; rates = [float(row["decode_tok_s"]) for row in measured]
        result = {"schema": "inferswarm.d6.serving-arm/1", "arm": arm, "command": cmd,
                  "freetoken_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ns.repo, text=True).strip(),
                  "startup_s": ready-started, "runtime_contract": gate, "resolved_runtime": runtime,
                  "loader": runtime["inferswarm_d5_resident_loader"], "generations": rows,
                  "reset": reset, "counters": selected, "ownership": ownership,
                  "physical": selected["physical"], "transport": selected.get("transport"),
                  "gpu_before": gpu_before, "gpu_after": gpu_after,
                  "analysis": {"decode_tok_s": {"each_retained": rates, "median": statistics.median(rates),
                                                  "min": min(rates), "max": max(rates)},
                               "paging_valid": True}, "status": "complete"}
        (ns.output_dir / f"d6-{arm.lower()}.json").write_text(json.dumps(result, indent=2)+"\n")
        return result
    finally: stop_server(handle)


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--model", required=True); p.add_argument("--revision", required=True)
    p.add_argument("--manifest", required=True); p.add_argument("--placement", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    ns = p.parse_args(); ns.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_arm(ns, arm) for arm in ("T0", "T1", "T2", "T3")]
    rates = {row["arm"]: row["analysis"]["decode_tok_s"]["median"] for row in rows}
    b_gain, ab_gain = rates["T2"]/rates["T0"], rates["T3"]/rates["T1"]
    e3 = rates["T3"]/rates["T2"]; prior_e3 = rates["T1"]/rates["T0"]
    classification = ("D6_TRANSPORT_TAX_CONFIRMED" if ab_gain >= 1.08 and e3 >= .82 else
                      "D6_TRANSPORT_TAX_PARTIAL" if ab_gain >= 1.03 else
                      "D6_TRANSPORT_TAX_NOT_SUPPORTED" if ab_gain >= .97 else "D6_TRANSPORT_HARMFUL")
    analysis = {"schema": "inferswarm.d6.serving-analysis/1", "predeclared_thresholds_frozen": True,
                "order": ["T0", "T1", "T2", "T3"], "median_decode_tok_s": rates,
                "B_TRANSPORT_GAIN": b_gain, "AB_TRANSPORT_GAIN": ab_gain,
                "D6_E3": e3, "D6_VS_D5_E3": e3/prior_e3,
                "AB6_over_historical_C3_weighted": rates["T3"]/HISTORICAL_C3,
                "marginal_retention": "strong" if e3 >= .90 else "promising" if e3 >= .75 else "weak",
                "fourth_worker_recommended": e3 >= .85, "classification": classification}
    (ns.output_dir / "d6-analysis.json").write_text(json.dumps(analysis, indent=2)+"\n")
    print(json.dumps(analysis, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
