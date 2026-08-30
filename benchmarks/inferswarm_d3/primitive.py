"""D3 physical primitive harness.  One shape per process; emits provenance-rich JSON.

This is deliberately a certification harness, not a serving benchmark.  It refuses
ordinal-only identity, eager graph state, or any D3 counter that implies a fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from freetoken.engine.engine import Engine
from freetoken.gpu_select import set_assigned_gpu
from freetoken.moe.offload_cache import iter_offload_moe_layers
from freetoken.server.args import parse_args
from freetoken.server.launch import _resolve_server_gpu_args

PRIMARY = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
WORKER_A = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
WORKER_B = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"
PLACEMENT_SHA = "6677fe1c506376a55aa8dcabb8d5761dc0373ced9d9b053209991059556d5887"


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


def _args(ns: argparse.Namespace):
    argv = ["--model", ns.model, "--gpu", PRIMARY, "--inferswarm-experimental-d3-graph-multiworker",
            "--inferswarm-d3-active-workers", ns.shape, "--inferswarm-d3-placement", ns.placement,
            "--moe-backend", "offload", "--moe-cpu-layers", "0", "--nvfp4-backend", "triton",
            "--moe-cache-size", "3774", "--max-running-requests", "1", "--cuda-graph-max-bs", "1",
            "--sampling-defaults", "none"]
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
        if report["corrected_placement_sha256"] != PLACEMENT_SHA: raise RuntimeError("placement digest disagreed")
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
        after = _vm()
        return {"schema": "inferswarm.d3.physical-primitive/1", "physical_tested_freetoken_commit": _git_head(),
                "infer_swarm_placement_commit": "5c916e799aeb237ab518b17639fa948e6b00ff4d", "corrected_placement_sha256": PLACEMENT_SHA,
                "model": ns.model, "model_revision": ns.revision, "shape": ns.shape, "startup_seconds": startup,
                "physical_runtime_seconds_excluding_startup": time.monotonic() - measured_start,
                "whole_model_graph": report, "post_graph_smoke_ownership": d3.snapshot()["ownership"],
                "paging_delta": _delta(before, after), "status": "CAPTURED_PENDING_FULL_PRIMITIVE"}
    finally:
        engine.shutdown()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True); p.add_argument("--revision", required=True); p.add_argument("--placement", required=True)
    p.add_argument("--shape", choices=("local", "a", "b", "ab"), required=True); p.add_argument("--output", required=True)
    ns = p.parse_args()
    # local is graph-enabled baseline and intentionally has no D3 workers; D3 physical order begins at ab.
    if ns.shape == "local": raise RuntimeError("local reference worker is not implemented in this D3-only process; run after D3 capture")
    result = _run_shape(ns)
    Path(ns.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
