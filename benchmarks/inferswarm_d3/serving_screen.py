"""Frozen D3 S1/S2A/S2B/S3 W4 serving screen (one warmup, three retained)."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferswarm_phase0.client import (fetch_instrumentation, free_port, measure_generation,
                                      prefill_seq_floor, start_server, stop_server)
from inferswarm_phase0.manifest import CANONICAL_GREEDY_SAMPLING, load_manifest
from inferswarm_phase1.campaign import moe_instrumentation

GPU0 = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
A = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
B = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"
PLACEMENT_SHA = "6677fe1c506376a55aa8dcabb8d5761dc0373ced9d9b053209991059556d5887"
KV = 17075


def vm(pid: int) -> dict[str, int]:
    def value(path: str, key: str) -> int:
        for line in Path(path).read_text().splitlines():
            if line.startswith(key): return int(line.split()[1])
        return 0
    return {"process_rss_kb": value(f"/proc/{pid}/status", "VmRSS:"),
            "process_major_faults": value(f"/proc/{pid}/status", "Majflt:"),
            "mem_available_kb": value("/proc/meminfo", "MemAvailable:"),
            "swap_free_kb": value("/proc/meminfo", "SwapFree:"),
            "pswpin": value("/proc/vmstat", "pswpin "),
            "pswpout": value("/proc/vmstat", "pswpout ")}


def delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: after[key] - before[key] for key in before}


def gpu_state() -> list[dict[str, str]]:
    fields = "uuid,pci.bus_id,temperature.gpu,power.draw,clocks.sm,pcie.link.gen.current,pcie.link.width.current"
    output = subprocess.check_output(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"], text=True)
    keys = fields.split(",")
    return [dict(zip(keys, (part.strip() for part in line.split(",")))) for line in output.splitlines()]


def pct(values: list[float], q: float) -> float:
    values = sorted(values); pos = (len(values) - 1) * q; lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] * (hi - pos) + values[hi] * (pos - lo)


def dist(values: list[float]) -> dict[str, float | int]:
    return {"n": len(values), "min": min(values), "median": statistics.median(values), "p95": pct(values, .95), "max": max(values)}


def command(root: Path, model: str, port: int, shape: str, placement: str) -> list[str]:
    cmd = [str(root / ".venv/bin/python"), "-m", "freetoken.cli", "serve", "--model", model,
           "--host", "127.0.0.1", "--port", str(port), "--gpu", GPU0, "--moe-backend", "offload",
           "--moe-cpu-layers", "0", "--nvfp4-backend", "triton", "--moe-cache-size", "3774",
           "--kv-reserve-tokens", str(KV), "--num-tokens", str(KV), "--memory-ratio", "0.85",
           "--max-running-requests", "1", "--cuda-graph-max-bs", "1", "--sampling-defaults", "none",
           "--moe-layer-timing-role", "candidate", "--moe-layer-timing-max-steps", "400"]
    if shape != "local":
        cmd += ["--inferswarm-experimental-d3-graph-multiworker", "--inferswarm-d3-active-workers", shape,
                "--inferswarm-d3-placement", placement]
        if "a" in shape: cmd += ["--inferswarm-d3-worker-a-gpu", A]
        if "b" in shape: cmd += ["--inferswarm-d3-worker-b-gpu", B]
    return cmd


def contract(runtime: dict[str, Any], shape: str) -> dict[str, Any]:
    rt, moe, cache = runtime["runtime"], runtime["moe"], runtime["cache"]
    d2, d3 = runtime["inferswarm_d2_graph_remote"], runtime["inferswarm_d3_graph_multiworker"]
    # The runtime report intentionally does not echo the primary UUID.  The frozen
    # --gpu UUID is retained verbatim in serve_command and inventory is captured per arm.
    checks: dict[str, bool] = {"gpu0_exact_uuid_cli": True, "gpu_decode": moe["decode_target"] == "gpu",
      "zero_cpu_moe_layers": moe["cpu_layers_resolved"] == [], "triton": runtime["nvfp4"]["resolved"] == "triton",
      "cache_3774": cache["resolved_slots"] == 3774, "kv_17075": rt["num_pages"] == KV,
      "max_running_1": rt["max_running_req"] == 1, "graph_bs1": rt["cuda_graph_capture_happened"] is True and rt["cuda_graph_captured_bs"] == [1],
      "no_d2": d2["enabled"] is False, "no_canonical_remote": runtime["inferswarm_remote_decode"]["enabled"] is False}
    if shape == "local": checks.update(no_d3=d3["enabled"] is False, no_secondary=runtime["inferswarm_secondary_device"]["configured"] is False)
    else:
        checks.update(d3_enabled=d3["enabled"] is True, graph_active=d3["graph_active"] is True,
          active_workers=d3["active_workers"] == list(shape), placement_sha=d3["corrected_placement_sha256"] == PLACEMENT_SHA,
          eager_fallback_false=d3["eager_fallback"] is False, fallback_zero=d3["fallback_count"] == 0,
          failure_zero=d3["failure_count"] == 0, recapture_zero=d3["graph_recapture_count"] == 0,
          host_sync_zero=d3["steady_state_host_sync_count"] == 0,
          weight_a_zero=d3["steady_state_expert_weight_bytes_host_to_worker_a"] == 0,
          weight_b_zero=d3["steady_state_expert_weight_bytes_host_to_worker_b"] == 0,
          inactive_absent=(d3["worker_b_uuid"] is None if shape == "a" else d3["worker_a_uuid"] is None if shape == "b" else True))
    return {"passed": all(checks.values()), "checks": checks}


def timing(snapshot: dict[str, Any], measured: list[dict[str, Any]]) -> dict[str, Any]:
    records = snapshot["moe_layer_timing"]["records"]
    def get(record: dict[str, Any], *path: str) -> float | None:
        value: Any = record
        for key in path: value = value[key]
        return float(value["value_ms"]) if value.get("status") == "valid" else None
    complete = [x for r in records if (x := get(r, "durations", "complete_layer")) is not None]
    local = [x for r in records if (x := get(r, "durations", "gpu0_branch", "complete_local_branch")) is not None]
    join = [x for r in records if (x := get(r, "durations", "join_reconstruct_reduce", "route_reconstruction")) is not None]
    return {"complete_moe_layer_ms": dist(complete) if complete else None,
            "gpu0_local_branch_ms": dist(local) if local else None,
            "join_reconstruction_ms": dist(join) if join else None,
            "inter_token_ms": dist([float(x) for g in measured for x in g["inter_token_ms"]]),
            "records": len(records), "warning": "overlapping components are non-additive; end-to-end decode throughput is authoritative"}


def run_arm(root: Path, out: Path, model: str, revision: str, manifest: str, placement: str, arm: str, shape: str) -> dict[str, Any]:
    port = free_port(); origin = f"http://127.0.0.1:{port}"; started = time.monotonic()
    handle = start_server(command(root, model, port, shape, placement), origin, str(out / f"d3-{arm.lower()}.server.log"),
                          env_overrides={"PYTHONPATH": "python:benchmarks", "FREETOKEN_INSTRUMENT_PREFILL": "1"}, ready_timeout=900, echo=True)
    try:
        ready = time.monotonic(); runtime = fetch_instrumentation(origin)["runtime_config"]; gate = contract(runtime, shape)
        if not gate["passed"]: raise RuntimeError(f"{arm} runtime contract failed: {gate}")
        workload = load_manifest(manifest, canonical=True).by_class()["W4"]
        body = workload.request_body(revision, sampling_override=CANONICAL_GREEDY_SAMPLING)
        before_arm = gpu_state(); floor = prefill_seq_floor(origin); warmup = measure_generation(origin, body, prefill_seq_floor=floor, store_text=False)
        reset = moe_instrumentation(origin, "reset", timeout=300); rows = [{"phase": "warmup", "repetition": 0, **warmup}]
        for repetition in range(3):
            before = vm(handle.proc.pid); floor = prefill_seq_floor(origin)
            generated = measure_generation(origin, body, prefill_seq_floor=floor, store_text=False); after = vm(handle.proc.pid)
            generated.update({"phase": "retained", "repetition": repetition, "paging_before": before, "paging_after": after, "paging_delta": delta(before, after)})
            if not generated["completion_matches_request"]: raise RuntimeError(f"{arm} retained {repetition} output length mismatch")
            if any(generated["paging_delta"][k] != 0 for k in ("pswpin", "pswpout")): raise RuntimeError(f"{arm} retained {repetition} paging contamination")
            rows.append(generated)
        snapshot = moe_instrumentation(origin, "snapshot", timeout=300); after_arm = gpu_state(); measured = rows[1:]
        d3 = snapshot["inferswarm_d3_graph_multiworker"]; ownership = d3.get("ownership")
        if shape != "local":
            bad = (not ownership["selection_arithmetic_exact"] or not ownership["no_route_dropped"] or not ownership["no_route_duplicated"] or
                   any(d3[k] != 0 for k in ("fallback_count", "failure_count", "graph_recapture_count", "steady_state_host_sync_count", "steady_state_expert_weight_bytes_host_to_worker_a", "steady_state_expert_weight_bytes_host_to_worker_b")))
            if bad: raise RuntimeError(f"{arm} retained D3 counter contract failed: {d3}")
            if (shape == "a" and not (ownership["executed_on_worker_a"] > 0 and ownership["executed_on_worker_b"] == 0)) or (shape == "b" and not (ownership["executed_on_worker_b"] > 0 and ownership["executed_on_worker_a"] == 0)) or (shape == "ab" and not (ownership["executed_on_worker_a"] > 0 and ownership["executed_on_worker_b"] > 0)): raise RuntimeError(f"{arm} ownership shape failed: {ownership}")
        rates = [float(x["decode_tok_s"]) for x in measured]
        result = {"schema": "inferswarm.d3.serving-arm/1", "arm": arm, "shape": shape, "serve_command": command(root, model, port, shape, placement),
          "model": model, "model_revision": revision, "placement_sha256": PLACEMENT_SHA, "gpu_roles": {"gpu0": GPU0, "worker_a": A, "worker_b": B},
          "startup_duration_s": ready-started, "runtime_contract": gate, "resolved_runtime": runtime, "request": {"workload": "W4", "content_sha256": workload.content_sha256, "max_tokens": body["max_tokens"], "ignore_eos": body["ignore_eos"], "sampling": {k: body[k] for k in ("temperature", "top_p", "top_k")}, "batch_size": 1},
          "reset": reset, "generations": rows, "snapshot": snapshot, "d3_counters": d3, "ownership": ownership, "gpu_before": before_arm, "gpu_after": after_arm,
          "analysis": {"decode_tok_s": {"each_retained": rates, "min": min(rates), "median": statistics.median(rates), "max": max(rates)}, "ttft_ms": {"median": statistics.median([float(x["ttft_ms"]) for x in measured])}, "timing": timing(snapshot, measured)},
          "physical_generation_wall_seconds": sum(float(x["wall_total_ms"]) for x in rows) / 1000, "status": "complete"}
        (out / f"d3-{arm.lower()}.json").write_text(json.dumps(result, indent=2) + "\n"); return result
    finally: stop_server(handle)


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--repo", required=True); p.add_argument("--model", required=True); p.add_argument("--revision", required=True); p.add_argument("--manifest", required=True); p.add_argument("--placement", required=True); p.add_argument("--output-dir", required=True); ns = p.parse_args()
    root, out = Path(ns.repo), Path(ns.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = [run_arm(root, out, ns.model, ns.revision, ns.manifest, ns.placement, arm, shape) for arm, shape in (("S1", "local"), ("S2A", "a"), ("S2B", "b"), ("S3", "ab"))]
    by = {row["arm"]: row for row in rows}; t = {arm: by[arm]["analysis"]["decode_tok_s"]["median"] for arm in by}; best = max(t["S2A"], t["S2B"]); e3 = t["S3"] / best
    verdict = "D3_SCALING_STRONG" if e3 >= .90 else "D3_SCALING_PROMISING" if e3 >= .75 else "D3_SCALING_WEAK"
    analysis = {"schema": "inferswarm.d3.serving-analysis/1", "throughput_median_decode_tok_s": t, "E2A": t["S2A"]/t["S1"], "E2B": t["S2B"]/t["S1"], "BEST_SINGLE": best, "E3": e3, "S3_vs_S1": t["S3"]/t["S1"], "S3_vs_S2A": t["S3"]/t["S2A"], "S3_vs_S2B": t["S3"]/t["S2B"], "classification": verdict, "COMPOUNDING_MARGINAL_PENALTY": e3 < .50, "physical_generation_wall_seconds": sum(x["physical_generation_wall_seconds"] for x in rows), "startup_seconds": sum(x["startup_duration_s"] for x in rows)}
    (out / "d3-analysis.json").write_text(json.dumps(analysis, indent=2) + "\n"); print(json.dumps(analysis, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
