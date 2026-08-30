"""Stage one immutable tmpfs model source and time three fresh D3 S3 engines."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferswarm_d4.host_source import fresh_arm_command, fresh_arm_env, safety_contract, stage_read_only_tmpfs
from inferswarm_phase0.client import fetch_instrumentation, free_port, start_server, stop_server

GPU0 = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
A = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
B = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"


def command(root: Path, source_model: str, placement: str, port: int) -> list[str]:
    return [str(root / ".venv/bin/python"), "-m", "freetoken.cli", "serve", "--model", source_model,
            "--host", "127.0.0.1", "--port", str(port), "--gpu", GPU0, "--moe-backend", "offload",
            "--moe-cpu-layers", "0", "--nvfp4-backend", "triton", "--moe-cache-size", "3774",
            "--kv-reserve-tokens", "17075", "--num-tokens", "17075", "--memory-ratio", "0.85",
            "--max-running-requests", "1", "--cuda-graph-max-bs", "1", "--sampling-defaults", "none",
            "--inferswarm-experimental-d3-graph-multiworker", "--inferswarm-d3-active-workers", "ab",
            "--inferswarm-d3-placement", placement, "--inferswarm-d3-worker-a-gpu", A,
            "--inferswarm-d3-worker-b-gpu", B]


def logged_phase_seconds(path: Path) -> dict[str, float]:
    patterns = {
        "gpu_runtime_ready": "Free memory before loading model:",
        "expert_bank_start": "expert banks: slow path",
        "expert_bank_complete": "NVFP4 expert backend:",
        "resident_workers_complete": "InferSwarm D3 graph attached:",
        "gpu_initialization_complete": "Free memory after initialization:",
        "graph_capture_start": "Start capturing CUDA graphs",
        "graph_capture_complete": "Free GPU memory after capturing CUDA graphs:",
        "ready": "API server is ready to serve",
    }
    stamps = {}
    for line in path.read_text(errors="replace").splitlines():
        match = re.search(r"\[(\d{4}-\d\d-\d\d\|\d\d:\d\d:\d\d)(?:\|[^]]+)?\]", line)
        if not match:
            continue
        stamp = dt.datetime.strptime(match.group(1), "%Y-%m-%d|%H:%M:%S")
        for name, text in patterns.items():
            if name not in stamps and text in line:
                stamps[name] = stamp
    if "gpu_runtime_ready" not in stamps or "ready" not in stamps:
        return {}
    def span(a: str, b: str) -> float:
        return (stamps[b] - stamps[a]).total_seconds()
    return {"non_expert_model_loading_s": span("gpu_runtime_ready", "expert_bank_start"),
            "normalized_host_expert_bank_materialization_s": span("expert_bank_start", "expert_bank_complete"),
            "resident_worker_loading_s": span("expert_bank_complete", "resident_workers_complete"),
            "cuda_graph_capture_s": span("graph_capture_start", "graph_capture_complete"),
            "post_capture_to_ready_s": span("graph_capture_complete", "ready")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--staged", required=True, type=Path)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-seconds", required=True, type=float)
    ns = parser.parse_args(); ns.output_dir.mkdir(parents=True, exist_ok=True)
    stage_start = time.monotonic(); source = stage_read_only_tmpfs(ns.source, ns.staged)
    stage_seconds = time.monotonic() - stage_start
    base = command(ns.repo, str(ns.source), ns.placement, 0)
    rows = []
    for index in range(1, 4):
        port = free_port(); base[base.index("--port") + 1] = str(port)
        cmd = fresh_arm_command(base, source); origin = f"http://127.0.0.1:{port}"
        log = ns.output_dir / f"d4-startup-reuse-arm-{index}.server.log"
        started = time.monotonic()
        handle = start_server(cmd, origin, str(log), ready_timeout=900, echo=True,
                              env_overrides={"PYTHONPATH": "python:benchmarks", **fresh_arm_env(source, f"startup-{index}")})
        try:
            startup = time.monotonic() - started
            runtime = fetch_instrumentation(origin)["runtime_config"]
            d3 = runtime["inferswarm_d3_graph_multiworker"]
            checks = {"fresh_frontend_pid": handle.proc.pid not in [r["frontend_pid"] for r in rows],
                      "graph_captured_fresh": runtime["runtime"]["cuda_graph_capture_happened"] is True,
                      "gpu0_cache_fresh": runtime["cache"]["resolved_slots"] == 3774,
                      "worker_banks_fresh": d3["active_worker_count"] == 2
                                            and d3["worker_a_resident_slots"] == 3000
                                            and d3["worker_b_resident_slots"] == 3000,
                      "counters_fresh": all(d3[key] == 0 for key in ("fallback_count", "failure_count", "graph_recapture_count",
                                                                       "steady_state_host_sync_count",
                                                                       "steady_state_expert_weight_bytes_host_to_worker_a",
                                                                       "steady_state_expert_weight_bytes_host_to_worker_b"))}
            if not all(checks.values()):
                raise RuntimeError(f"fresh-engine boundary failed: {checks}")
            rows.append({"arm": index, "frontend_pid": handle.proc.pid, "startup_s": startup,
                         "remaining_phases": logged_phase_seconds(log), "fresh_state_checks": checks})
        finally:
            stop_server(handle)
    startups = [row["startup_s"] for row in rows]
    result = {"schema": "inferswarm.d4.startup-reuse/1", "freetoken_sha": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ns.repo, text=True).strip(),
              "source": str(source.source), "staged_source": str(source.staged),
              "source_manifest_sha256": source.manifest_sha256, "staged_bytes": source.staged_bytes,
              "stage_once_seconds": stage_seconds, "baseline_startup_s": ns.baseline_seconds,
              "fresh_engine_arms": rows, "first_reused_source_startup_s": startups[0],
              "subsequent_reused_source_startup_s": startups[1:],
              "median_reused_source_startup_s": sorted(startups)[1],
              "startup_speedup_factor": ns.baseline_seconds / sorted(startups)[1],
              "median_startup_saved_s": ns.baseline_seconds - sorted(startups)[1],
              "safety_contract": safety_contract(), "status": "complete"}
    (ns.output_dir / "d4-startup-reuse.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
