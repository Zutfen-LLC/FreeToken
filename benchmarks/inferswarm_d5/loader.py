"""Fresh-engine D5 resident-loader profiler (startup evidence, never serving evidence)."""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferswarm_phase0.client import fetch_instrumentation, free_port, start_server, stop_server

GPU0 = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
A = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
B = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"


def command(root: Path, model: str, placement: str, port: int, shape: str,
            mode: str, workers: int) -> list[str]:
    cmd = [str(root / ".venv/bin/python"), "-m", "freetoken.cli", "serve",
           "--model", model, "--host", "127.0.0.1", "--port", str(port),
           "--gpu", GPU0, "--moe-backend", "offload", "--moe-cpu-layers", "0",
           "--nvfp4-backend", "triton", "--moe-cache-size", "3774",
           "--kv-reserve-tokens", "17075", "--num-tokens", "17075",
           "--memory-ratio", "0.85", "--max-running-requests", "1",
           "--cuda-graph-max-bs", "1", "--sampling-defaults", "none",
           "--inferswarm-experimental-d3-graph-multiworker",
           "--inferswarm-d3-active-workers", shape,
           "--inferswarm-d3-placement", placement]
    if "a" in shape: cmd += ["--inferswarm-d3-worker-a-gpu", A]
    if "b" in shape: cmd += ["--inferswarm-d3-worker-b-gpu", B]
    if mode == "bulk":
        cmd += ["--inferswarm-experimental-d5-resident-loader",
                "--inferswarm-d5-loader-cpu-workers", str(workers)]
    return cmd


def phases(log: Path) -> dict[str, float]:
    needles = {"gpu_runtime_ready": "Free memory before loading model:",
               "expert_bank_start": "expert banks: slow path",
               "expert_bank_complete": "NVFP4 expert backend:",
               "resident_complete": "InferSwarm D3 graph attached:",
               "graph_start": "Start capturing CUDA graphs",
               "graph_complete": "Free GPU memory after capturing CUDA graphs:",
               "ready": "API server is ready to serve"}
    found: dict[str, datetime] = {}
    for line in log.read_text(errors="replace").splitlines():
        match = re.search(r"\[(\d{4}-\d\d-\d\d\|\d\d:\d\d:\d\d)", line)
        if not match: continue
        stamp = datetime.strptime(match.group(1), "%Y-%m-%d|%H:%M:%S")
        for key, needle in needles.items():
            if key not in found and needle in line: found[key] = stamp
    def span(a: str, b: str) -> float: return (found[b] - found[a]).total_seconds()
    return {"gpu_runtime_initialization_s": span("gpu_runtime_ready", "expert_bank_start"),
            "normalized_expert_bank_s": span("expert_bank_start", "expert_bank_complete"),
            "resident_worker_loading_s": span("expert_bank_complete", "resident_complete"),
            "cuda_graph_capture_s": span("graph_start", "graph_complete")}


def run(ns: argparse.Namespace, index: int) -> dict:
    port = free_port(); log = ns.output_dir / f"d5-loader-{ns.mode}-{ns.shape}-w{ns.workers}-{index}.server.log"
    cmd = command(ns.repo, ns.model, ns.placement, port, ns.shape, ns.mode, ns.workers)
    env = {"PYTHONPATH": "python:benchmarks"}
    if ns.mode == "legacy-profile": env["FREETOKEN_D5_PROFILE_LEGACY_LOADER"] = "1"
    started = time.monotonic()
    handle = start_server(cmd, f"http://127.0.0.1:{port}", str(log), ready_timeout=1200,
                          echo=True, env_overrides=env)
    try:
        startup = time.monotonic() - started
        runtime = fetch_instrumentation(f"http://127.0.0.1:{port}")["runtime_config"]
        profile = runtime["inferswarm_d5_resident_loader"]
        return {"repetition": index, "startup_s": startup, "phases": phases(log),
                "loader_profile": profile,
                "graph": runtime["inferswarm_d3_graph_multiworker"]}
    finally:
        stop_server(handle)


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--model", required=True); p.add_argument("--placement", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("legacy-profile", "bulk"), required=True)
    p.add_argument("--shape", choices=("a", "b", "ab"), required=True)
    p.add_argument("--workers", type=int, choices=(1, 2, 4, 8), default=4)
    p.add_argument("--repetitions", type=int, default=1); p.add_argument("--output", required=True)
    ns = p.parse_args(); ns.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run(ns, index + 1) for index in range(ns.repetitions)]
    result = {"schema": "inferswarm.d5.resident-loader/1", "kind": "startup_only",
              "freetoken_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ns.repo, text=True).strip(),
              "mode": ns.mode, "shape": ns.shape, "cpu_workers": ns.workers,
              "runs": rows, "median_startup_s": statistics.median(x["startup_s"] for x in rows),
              "median_resident_wall_s": statistics.median(x["loader_profile"]["wall_s"] for x in rows),
              "status": "complete"}
    Path(ns.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
