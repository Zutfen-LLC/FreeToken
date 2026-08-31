"""Short deterministic W4 equality: D6 equal placement versus D7 sparse."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferswarm_phase0.client import fetch_instrumentation, free_port, start_server, stop_server, stream_generation
from inferswarm_phase0.manifest import CANONICAL_GREEDY_SAMPLING, load_manifest
from inferswarm_phase1.campaign import moe_instrumentation
from inferswarm_d3.whole_model_correctness import _delta, _vm

PRIMARY = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
A = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
B = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"
D3_SHA = "6677fe1c506376a55aa8dcabb8d5761dc0373ced9d9b053209991059556d5887"
D7_SHA = "c360cad506fa4dbc2f768b24d8ad5dfd1d10956aafc88a5fa6e2736dfe0581d1"
OUTPUT_CAP = 32


def command(root, model, port, placement, sparse):
    result = [str(root / ".venv/bin/python"), "-m", "freetoken.cli", "serve", "--model", model,
              "--host", "127.0.0.1", "--port", str(port), "--gpu", PRIMARY,
              "--moe-backend", "offload", "--moe-cpu-layers", "0", "--nvfp4-backend", "triton",
              "--moe-cache-size", "3774", "--kv-reserve-tokens", "17075", "--num-tokens", "17075",
              "--max-running-requests", "1", "--cuda-graph-max-bs", "1", "--sampling-defaults", "none",
              "--inferswarm-correctness-diagnostics", "--inferswarm-experimental-d5-resident-loader",
              "--inferswarm-d5-loader-cpu-workers", "8", "--inferswarm-d3-active-workers", "ab",
              "--inferswarm-d3-worker-a-gpu", A, "--inferswarm-d3-worker-b-gpu", B,
              "--inferswarm-experimental-d6-count-aware-transport", "--inferswarm-d7-participation-diagnostics"]
    if sparse:
        result += ["--inferswarm-d7-fanin-sparse-placement", "--inferswarm-d7-placement", placement]
    else:
        result += ["--inferswarm-d3-placement", placement]
    return result


def run(ns, sparse):
    label = "d7-sparse" if sparse else "d6-equal"
    placement = ns.d7_placement if sparse else ns.d3_placement
    expected_sha = D7_SHA if sparse else D3_SHA
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    started = time.monotonic()
    handle = start_server(command(ns.repo, ns.model, port, placement, sparse), origin,
                          str(ns.output.parent / f"d7-correctness-{label}.server.log"),
                          env_overrides={"PYTHONPATH": "python:benchmarks"}, ready_timeout=900, echo=True)
    try:
        runtime = fetch_instrumentation(origin)["runtime_config"]
        selected = runtime["inferswarm_d6_count_aware_transport"]
        checks = {"graph_bs1": runtime["runtime"]["cuda_graph_captured_bs"] == [1],
                  "cache_3774": runtime["cache"]["resolved_slots"] == 3774,
                  "triton": runtime["nvfp4"]["resolved"] == "triton",
                  "zero_cpu_moe": runtime["moe"]["cpu_layers_resolved"] == [],
                  "d6_graph": selected["enabled"] and selected["graph_active"] and not selected["eager_fallback"],
                  "placement_sha": selected["placement_sha256"] == expected_sha,
                  "joint_diagnostic": selected["d7_participation_diagnostics"] is True}
        if not all(checks.values()):
            raise RuntimeError(f"{label} contract failed: {checks}")
        workload = load_manifest(ns.manifest, canonical=True).by_class()["W4"]
        body = workload.request_body(ns.revision, sampling_override=CANONICAL_GREEDY_SAMPLING)
        body["max_tokens"] = OUTPUT_CAP
        reset = moe_instrumentation(origin, "reset", timeout=300)
        before = _vm(handle.proc.pid)
        observation = stream_generation(origin, body, timeout=900)
        after = _vm(handle.proc.pid)
        snapshot = moe_instrumentation(origin, "snapshot", timeout=300)
        records = snapshot["inferswarm_correctness_diagnostics"]["records"]
        if len(records) != 1:
            raise RuntimeError("expected one correctness record")
        tokens = [int(value) for value in records[0]["generated_token_ids"]]
        if len(tokens) != OUTPUT_CAP:
            raise RuntimeError("short W4 token count mismatch")
        live = snapshot["inferswarm_d6_count_aware_transport"]
        if (not live["ownership"]["selection_arithmetic_exact"]
                or not live["physical"]["physical_worker_invocations_equal_owned_remote_routes"]
                or any(live[name] for name in ("fallback_count", "failure_count", "graph_recapture_count",
                                                "steady_state_host_sync_count"))):
            raise RuntimeError(f"{label} counters invalid")
        return {"arm": label, "startup_s": time.monotonic() - started, "runtime_checks": checks,
                "reset": reset, "token_ids": tokens,
                "token_sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
                "snapshot": live, "paging_delta": _delta(before, after), "response_usage": observation["usage"]}
    finally:
        stop_server(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--d3-placement", required=True)
    parser.add_argument("--d7-placement", required=True)
    parser.add_argument("--output", type=Path, required=True)
    ns = parser.parse_args()
    ns.output.parent.mkdir(parents=True, exist_ok=True)
    rows = [run(ns, False), run(ns, True)]
    passed = rows[0]["token_ids"] == rows[1]["token_ids"]
    result = {"schema": "inferswarm.d7.whole-model-correctness/1", "output_cap": OUTPUT_CAP,
              "arms": rows, "token_ids_byte_for_byte_identical": passed,
              "classification": "D7_FANIN_SPARSE_PRIMITIVE_PASS" if passed else "D7_INVALID"}
    ns.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"classification": result["classification"],
                      "hashes": [row["token_sha256"] for row in rows]}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
