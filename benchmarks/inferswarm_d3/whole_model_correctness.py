"""D3 completion: one graph-enabled W4 generation per fresh local/a/b/ab server.

This deliberately records exact scheduler token IDs, not serving throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

# The evidence command intentionally invokes this dedicated file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inferswarm_phase0.client import free_port, fetch_instrumentation, start_server, stop_server, stream_generation
from inferswarm_phase0.manifest import CANONICAL_GREEDY_SAMPLING, load_manifest
from inferswarm_phase1.campaign import moe_instrumentation

PRIMARY = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
WORKER_A = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
WORKER_B = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"
PLACEMENT_SHA = "6677fe1c506376a55aa8dcabb8d5761dc0373ced9d9b053209991059556d5887"
OUTPUT_CAP = 32


def token_hash(token_ids: list[int]) -> str:
    """SHA-256 of UTF-8 compact JSON token IDs (the evidence's canonical encoding)."""
    return hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode("utf-8")).hexdigest()


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return len(left) if len(left) != len(right) else None


def command(root: Path, model: str, port: int, shape: str, placement: str) -> list[str]:
    result = [str(root / ".venv/bin/python"), "-m", "freetoken.cli", "serve", "--model", model,
              "--host", "127.0.0.1", "--port", str(port), "--gpu", PRIMARY,
              "--moe-backend", "offload", "--moe-cpu-layers", "0", "--nvfp4-backend", "triton",
              "--moe-cache-size", "3774", "--max-running-requests", "1", "--cuda-graph-max-bs", "1",
              "--kv-reserve-tokens", "17075", "--num-tokens", "17075",
              "--sampling-defaults", "none", "--inferswarm-correctness-diagnostics"]
    if shape != "local":
        result += ["--inferswarm-experimental-d3-graph-multiworker", "--inferswarm-d3-active-workers", shape,
                   "--inferswarm-d3-placement", placement]
        if "a" in shape: result += ["--inferswarm-d3-worker-a-gpu", WORKER_A]
        if "b" in shape: result += ["--inferswarm-d3-worker-b-gpu", WORKER_B]
    return result


def _vm(pid: int) -> dict[str, int]:
    def field(path: str, key: str) -> int:
        for line in Path(path).read_text().splitlines():
            if line.startswith(key): return int(line.split()[1])
        return 0
    return {"process_rss_kb": field(f"/proc/{pid}/status", "VmRSS:"),
            "process_major_faults": field(f"/proc/{pid}/status", "Majflt:"),
            "mem_available_kb": field("/proc/meminfo", "MemAvailable:"),
            "swap_free_kb": field("/proc/meminfo", "SwapFree:"),
            "pswpin": field("/proc/vmstat", "pswpin "), "pswpout": field("/proc/vmstat", "pswpout ")}


def _delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: after[key] - before[key] for key in before}


def _post(origin: str, operation: str) -> dict[str, Any]:
    req = urllib.request.Request(f"{origin}/v1/moe/instrumentation", data=json.dumps({"operation": operation, "timeout": 300}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=310) as response:
        result = json.load(response)
    if result.get("status") != "ok": raise RuntimeError(f"instrumentation {operation} failed: {result}")
    return result


def _contract(runtime: dict[str, Any], shape: str) -> dict[str, Any]:
    common = runtime["runtime"], runtime["moe"], runtime["cache"]
    rt, moe, cache = common
    checks = {"gpu_decode": moe["decode_target"] == "gpu", "zero_cpu_moe_layers": moe["cpu_layers_resolved"] == [],
              "native_triton_nvfp4": runtime["nvfp4"]["resolved"] == "triton", "gpu0_cache_3774": cache["resolved_slots"] == 3774,
              "cuda_graph_capture_happened": rt["cuda_graph_capture_happened"] is True, "captured_bs_1": rt["cuda_graph_captured_bs"] == [1]}
    d2, d3, remote, secondary = runtime["inferswarm_d2_graph_remote"], runtime["inferswarm_d3_graph_multiworker"], runtime["inferswarm_remote_decode"], runtime["inferswarm_secondary_device"]
    checks.update(no_d2=d2["enabled"] is False, no_canonical_remote=remote["enabled"] is False)
    if shape == "local":
        checks.update(no_d3=d3["enabled"] is False, no_secondary=secondary["configured"] is False,
                      no_remote_execution=d3["active_workers"] == [])
    else:
        checks.update(d3_enabled=d3["enabled"] is True, d3_graph_active=d3["graph_active"] is True,
                      active_workers=d3["active_workers"] == list(shape), placement_sha=d3["corrected_placement_sha256"] == PLACEMENT_SHA,
                      fallback_zero=d3["fallback_count"] == 0, failure_zero=d3["failure_count"] == 0,
                      recapture_zero=d3["graph_recapture_count"] == 0, host_sync_zero=d3["steady_state_host_sync_count"] == 0,
                      inactive_worker_absent=(d3["worker_b_uuid"] is None if shape == "a" else d3["worker_a_uuid"] is None if shape == "b" else True))
    return {"passed": all(checks.values()), "checks": checks}


def run_shape(root: Path, model: str, revision: str, manifest_path: str, placement: str, shape: str, log_dir: Path) -> dict[str, Any]:
    port = free_port(); origin = f"http://127.0.0.1:{port}"; started = time.monotonic()
    handle = start_server(command(root, model, port, shape, placement), origin, str(log_dir / f"d3-whole-model-{shape}.server.log"), env_overrides={"PYTHONPATH": "python:benchmarks"}, ready_timeout=900, echo=True)
    try:
        runtime = fetch_instrumentation(origin)["runtime_config"]; contract = _contract(runtime, shape)
        if not contract["passed"]: raise RuntimeError(f"{shape} runtime contract failed: {contract}")
        workload = load_manifest(manifest_path, canonical=True).by_class()["W4"]
        body = workload.request_body(model, sampling_override=CANONICAL_GREEDY_SAMPLING); body["max_tokens"] = OUTPUT_CAP
        # The reset is the only boundary before the single generation; no warmup or repetition.
        reset = _post(origin, "reset") if shape != "local" else _post(origin, "reset")
        before = _vm(handle.proc.pid); generated_at = time.monotonic(); observation = stream_generation(origin, body, timeout=900); after = _vm(handle.proc.pid)
        snapshot = _post(origin, "snapshot")["snapshot"]
        diagnostics = snapshot["inferswarm_correctness_diagnostics"]
        records = diagnostics["records"]
        if diagnostics.get("truncated") or diagnostics.get("overflow_requests") or len(records) != 1: raise RuntimeError(f"invalid correctness diagnostics: {diagnostics}")
        tokens = [int(value) for value in records[0]["generated_token_ids"]]
        if len(tokens) != OUTPUT_CAP: raise RuntimeError(f"expected {OUTPUT_CAP} tokens, got {len(tokens)}")
        d3 = snapshot["inferswarm_d3_graph_multiworker"]
        ownership = d3.get("ownership")
        if shape != "local":
            if not ownership["selection_arithmetic_exact"] or not ownership["no_route_dropped"] or not ownership["no_route_duplicated"]: raise RuntimeError(f"{shape} routing accounting failed: {ownership}")
            if d3["fallback_count"] or d3["failure_count"] or d3["graph_recapture_count"] or d3["steady_state_host_sync_count"]: raise RuntimeError(f"{shape} D3 counters dirty: {d3}")
            if not all(d3[key] == 0 for key in ("steady_state_expert_weight_bytes_host_to_worker_a", "steady_state_expert_weight_bytes_host_to_worker_b")): raise RuntimeError(f"{shape} moved expert weights")
        return {"shape": shape, "startup_seconds": generated_at - started, "generation_wall_seconds": time.monotonic() - generated_at,
                "runtime_contract": contract, "resolved_runtime": runtime, "reset": reset, "request": {"workload": "W4", "content_sha256": workload.content_sha256, "model_revision": revision, "max_tokens": OUTPUT_CAP, "ignore_eos": body["ignore_eos"], "temperature": body["temperature"], "top_p": body["top_p"], "top_k": body["top_k"], "batch_size": 1},
                "token_ids": tokens, "token_count": len(tokens), "token_id_hash_sha256": token_hash(tokens), "token_hash_encoding": "sha256(json.dumps(token_ids, separators=(',', ':')).encode('utf-8'))", "ownership": ownership, "d3_counters": d3, "paging_delta": _delta(before, after), "response_usage": observation["usage"]}
    finally: stop_server(handle)


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--repo", required=True); p.add_argument("--model", required=True); p.add_argument("--revision", required=True); p.add_argument("--manifest", required=True); p.add_argument("--placement", required=True); p.add_argument("--output", required=True); ns = p.parse_args()
    root, output = Path(ns.repo), Path(ns.output); output.parent.mkdir(parents=True, exist_ok=True)
    rows = [run_shape(root, ns.model, ns.revision, ns.manifest, ns.placement, shape, output.parent) for shape in ("local", "a", "b", "ab")]
    by_shape = {row["shape"]: row for row in rows}; reference = by_shape["local"]["token_ids"]
    differences = {shape: first_difference(reference, by_shape[shape]["token_ids"]) for shape in ("a", "b", "ab")}
    passed = all(value is None for value in differences.values())
    result = {"schema": "inferswarm.d3.whole-model-correctness/1", "previous_physical_runtime_sha": "38e8b20327a5a01f00c84ad542111cf746dcb252", "production_d3_runtime_changed": False, "model": ns.model, "model_revision": ns.revision, "placement_sha256": PLACEMENT_SHA, "gpu_roles": {"gpu0": PRIMARY, "worker_a": WORKER_A, "worker_b": WORKER_B}, "output_cap": OUTPUT_CAP, "shapes": rows, "token_ids_byte_for_byte_identical": passed, "first_differing_token_index_vs_local": differences, "classification": "D3_PRIMITIVE_PASS_OVERLAP_CONFIRMED" if passed else "D3_PRIMITIVE_FAIL_CORRECTNESS", "serving_throughput_screen_occurred": False}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps({"classification": result["classification"], "hashes": {x: by_shape[x]["token_id_hash_sha256"] for x in by_shape}}, indent=2)); return 0 if passed else 2


if __name__ == "__main__": raise SystemExit(main())
