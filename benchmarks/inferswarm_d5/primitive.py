"""D5 compact one-layer oracle, dynamic replay, counters, and graph certification."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from freetoken.engine.engine import Engine
from freetoken.gpu_select import set_assigned_gpu
from freetoken.server.args import parse_args
from freetoken.server.launch import _resolve_server_gpu_args
from inferswarm_d3.primitive import _one_layer_diagnostics, _vm, _delta

PRIMARY = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
A = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
B = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"


def config(ns):
    argv = ["--model", ns.model, "--gpu", PRIMARY,
            "--inferswarm-experimental-d5-compact-routes",
            "--inferswarm-experimental-d5-resident-loader",
            "--inferswarm-d5-loader-cpu-workers", "8",
            "--inferswarm-d3-active-workers", "ab", "--inferswarm-d3-placement", ns.placement,
            "--inferswarm-d3-worker-a-gpu", A, "--inferswarm-d3-worker-b-gpu", B,
            "--moe-backend", "offload", "--moe-cpu-layers", "0", "--nvfp4-backend", "triton",
            "--moe-cache-size", "3774", "--max-running-requests", "1",
            "--cuda-graph-max-bs", "1", "--sampling-defaults", "none"]
    parsed, _ = parse_args(argv); return _resolve_server_gpu_args(parsed)


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True); p.add_argument("--placement", required=True)
    p.add_argument("--output", required=True); p.add_argument("--replays", type=int, default=100)
    ns = p.parse_args(); started = time.monotonic(); args = config(ns)
    set_assigned_gpu((args.gpu_assigned or args.gpu)[0]); engine = Engine(args)
    try:
        executor = engine.inferswarm_d5_compact_routes
        report = executor.configuration_report()
        required = {"enabled": True, "graph_active": True, "eager_fallback": False,
                    "fallback_count": 0, "failure_count": 0, "graph_recapture_count": 0,
                    "steady_state_host_sync_count": 0, "stable_compaction": True,
                    "count_aware_expert_compute": True, "inactive_tail_zeroed": True,
                    "one_canonical_route_order_reduction": True}
        checks = {key: report.get(key) == value for key, value in required.items()}
        if not all(checks.values()): raise RuntimeError(f"D5 graph contract failed: {checks}")
        before = _vm(); diagnostics = _one_layer_diagnostics(engine, executor, ns.placement, "ab", ns.replays)
        after = _vm(); snapshot = executor.snapshot(); ownership = snapshot["ownership"]; physical = snapshot["physical"]
        counter_checks = {"ownership_exact": ownership["selection_arithmetic_exact"],
                          "no_drop": ownership["no_route_dropped"], "no_duplicate": ownership["no_route_duplicated"],
                          "physical_equals_remote_owned": physical["physical_worker_invocations_equal_owned_remote_routes"],
                          "remote_work_nonzero": physical["worker_expert_invocations"] > 0,
                          "tails_skipped_nonzero": physical["compact_tail_entries_skipped"] > 0}
        passed = diagnostics["all_cases_passed"] and all(checks.values()) and all(counter_checks.values())
        result = {"schema": "inferswarm.d5.compact-primitive/1",
                  "freetoken_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "model": ns.model, "model_revision": ns.revision,
                  "placement_sha256": report["placement_sha256"], "startup_s": time.monotonic() - started,
                  "graph_checks": checks, "counter_checks": counter_checks,
                  "one_layer": diagnostics, "snapshot": snapshot,
                  "loader": engine.inferswarm_d3_resident_banks.loader_profile,
                  "paging_delta": _delta(before, after),
                  "classification": "D5_COMPACT_PRIMITIVE_PASS" if passed else "D5_INVALID"}
        Path(ns.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"classification": result["classification"], "checks": counter_checks,
                          "cases": diagnostics["cases"]}, indent=2))
        return 0 if passed else 2
    finally: engine.shutdown()


if __name__ == "__main__": raise SystemExit(main())
