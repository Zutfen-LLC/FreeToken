"""One-session, one-workload D2 G0/G1 serving screen (1 warmup + 3 retained)."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from inferswarm_phase0.client import (
    fetch_instrumentation,
    free_port,
    measure_generation,
    prefill_seq_floor,
    start_server,
    stop_server,
)
from inferswarm_phase0.manifest import CANONICAL_GREEDY_SAMPLING, load_manifest
from inferswarm_phase1.campaign import moe_instrumentation

GPU0 = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
GPU1 = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
PLACEMENT_SHA = "2f62bb84df40d4cc5649e940a39cb53d2975eadecbc320fb97d2b037d4e005f4"


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _dist(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _component(records: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, Any]:
    entries = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        entries.append(value)
    statuses = Counter(str(entry.get("status")) for entry in entries)
    valid = [float(entry["value_ms"]) for entry in entries if entry.get("status") == "valid"]
    return {
        "semantic_boundary": ".".join(path),
        "statuses": dict(statuses),
        "distribution_ms": _dist(valid) if valid else None,
        "unavailable_reason": next(
            (entry.get("reason") for entry in entries if entry.get("status") == "unavailable"),
            None,
        ),
    }


def _analyze(measured: list[dict[str, Any]], snapshot: dict[str, Any], *, d2: bool):
    timing = snapshot["moe_layer_timing"]
    records = timing["records"]
    complete = [
        float(r["durations"]["complete_layer"]["value_ms"])
        for r in records
        if r["durations"]["complete_layer"]["status"] == "valid"
    ]
    by_step: dict[int, dict[int, float]] = defaultdict(dict)
    for record in records:
        step = int(record["identity"]["decode_step"])
        layer = int(record["identity"]["layer_id"])
        duration = record["durations"]["complete_layer"]
        if duration["status"] == "valid":
            by_step[step][layer] = float(duration["value_ms"])
    malformed = {
        str(step): sorted(layers)
        for step, layers in by_step.items()
        if len(layers) != 40 or set(layers) != set(range(40))
    }
    moe_token = [
        sum(layers.values())
        for step, layers in sorted(by_step.items())
        if str(step) not in malformed
    ]
    gaps = [float(v) for generation in measured for v in generation["inter_token_ms"]]
    rates = [float(g["decode_tok_s"]) for g in measured]
    ttft = [float(g["ttft_ms"]) for g in measured]
    analysis = {
        "decode_tok_s": {"each_repetition": rates, "median": statistics.median(rates)},
        "ttft_ms": {"each_repetition": ttft, "median": statistics.median(ttft)},
        "inter_token_ms": _dist(gaps),
        "complete_layer_ms": _dist(complete),
        "moe_only_token_wall_ms": {
            **_dist(moe_token),
            "median_inter_token_ms": statistics.median(gaps),
            "fraction_of_median_inter_token": (
                statistics.median(moe_token) / statistics.median(gaps)
            ),
        },
        "moe_only_step_validation": {
            "all_steps_exactly_40_layers": not malformed and len(by_step) == 381,
            "malformed_steps": malformed,
        },
        "timing_population": {
            "steps_observed": timing["steps_observed"],
            "records_retained": len(records),
            "truncated": timing["truncated"],
            "overflow_layer_calls": timing["overflow_layer_calls"],
            "layer_steps_observed": timing["layer_steps_observed"],
        },
    }
    if d2:
        analysis["remote_components_nonadditive"] = {
            "classification_control": {
                "semantic_boundary": "captured GPU0 device lookup/mask; no host control read",
                "status": "unavailable_separately",
            },
            "gpu0_to_remote_payload": {
                "semantic_boundary": "captured pinned GPU0 D2H plus GPU1 H2D before resident execution",
                "status": "unavailable_separately",
            },
            "host_launch_replay_submission": {
                "semantic_boundary": "one whole-model multi-device graph replay per token",
                "source": "d2-part1-replay-benchmark.json",
            },
            "gpu1_execution": {
                "semantic_boundary": "captured resident NVFP4 route-contribution work",
                "status": "unavailable_separately",
            },
            "return_movement": _component(
                records,
                (
                    "durations",
                    "join_reconstruct_reduce",
                    "host_to_gpu0_returned_route_contributions",
                ),
            ),
            "join_dependency_delay": {
                "semantic_boundary": "captured internal cross-device event edge",
                "status": "not_applicable_host_wait_eliminated",
            },
            "reconstruction": _component(
                records, ("durations", "join_reconstruct_reduce", "route_reconstruction")
            ),
            "final_reduction": _component(
                records, ("durations", "join_reconstruct_reduce", "final_moe_sum_reduce")
            ),
            "gpu0_local_branch": _component(
                records, ("durations", "gpu0_branch", "complete_local_branch")
            ),
            "overlap_warning": "components overlap and must not be summed as complete wall",
        }
    return analysis


def _command(root: Path, model: str, port: int, role: str, placement: str | None):
    command = [
        str(root / ".venv/bin/python"), "-m", "freetoken.cli", "serve",
        "--model", model,
        "--host", "127.0.0.1", "--port", str(port),
        "--gpu", GPU0,
        "--moe-backend", "offload",
        "--moe-cpu-layers", "0",
        "--nvfp4-backend", "triton",
        "--moe-cache-size", "3774",
        "--kv-reserve-tokens", "17075",
        "--num-tokens", "17075",
        "--memory-ratio", "0.85",
        "--max-running-requests", "1",
        "--cuda-graph-max-bs", "1",
        "--sampling-defaults", "none",
        "--moe-layer-timing-role", role,
        "--moe-layer-timing-max-steps", "400",
    ]
    if placement is not None:
        command.extend(
            [
                "--inferswarm-secondary-gpu", GPU1,
                "--inferswarm-placement", placement,
                "--inferswarm-experimental-d2-graph-remote",
            ]
        )
    return command


def _contract(runtime: dict[str, Any], *, d2: bool) -> dict[str, Any]:
    rt, moe, cache = runtime["runtime"], runtime["moe"], runtime["cache"]
    checks = {
        "gpu_decode": moe["decode_target"] == "gpu",
        "zero_cpu_layers": moe["cpu_layers_resolved"] == [],
        "triton": runtime["nvfp4"]["resolved"] == "triton",
        "cache_slots_3774": cache["resolved_slots"] == 3774,
        "kv_capacity_17075": rt["num_pages"] == 17075,
        "max_running_requests_1": rt["max_running_req"] == 1,
        "cuda_graph_bs1_captured": rt["cuda_graph_capture_happened"] is True and rt["cuda_graph_captured_bs"] == [1],
    }
    d2_block = runtime["inferswarm_d2_graph_remote"]
    if d2:
        checks.update(
            d2_enabled=d2_block["enabled"] is True,
            gpu0_graph_active=d2_block["gpu0_graph_active"] is True,
            gpu1_graph_active=d2_block["gpu1_graph_active"] is True,
            no_eager_gpu0_fallback=d2_block["eager_gpu0_fallback"] is False,
            zero_host_sync=d2_block["steady_state_host_sync_count"] == 0,
            frozen_placement=d2_block["placement_sha256"] == PLACEMENT_SHA,
        )
    else:
        checks.update(
            no_secondary=(
                runtime["inferswarm_secondary_device"]["configured"] is False
            ),
            d2_disabled=d2_block["enabled"] is False,
        )
    return {"passed": all(checks.values()), "checks": checks}


def run_arm(root: Path, out: Path, model: str, manifest_path: str, arm: str, placement: str | None):
    d2 = placement is not None
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    command = _command(root, model, port, "candidate" if d2 else "baseline", placement)
    launched = time.monotonic()
    handle = start_server(
        command,
        origin,
        str(out / f"d2-arm-{arm.lower()}.server.log"),
        env_overrides={"PYTHONPATH": "python:benchmarks", "FREETOKEN_INSTRUMENT_PREFILL": "1"},
        ready_timeout=600,
        echo=True,
    )
    try:
        ready = time.monotonic()
        instrumentation = fetch_instrumentation(origin, limit=1)
        runtime = instrumentation["runtime_config"]
        contract = _contract(runtime, d2=d2)
        if not contract["passed"]:
            raise RuntimeError(f"{arm} runtime contract failed: {contract}")
        workload = load_manifest(manifest_path, canonical=False).by_class()["W4"]
        body = workload.request_body(
            "491c2f1ea524c639598bf8fa787a93fed5a6fbce",
            sampling_override=CANONICAL_GREEDY_SAMPLING,
        )
        generations = []
        floor = prefill_seq_floor(origin)
        warmup = measure_generation(origin, body, prefill_seq_floor=floor, store_text=False)
        generations.append({"phase": "warmup", "repetition": 0, **warmup})
        reset_started = time.monotonic()
        reset = moe_instrumentation(origin, "reset", timeout=300)
        reset_duration = time.monotonic() - reset_started
        for repetition in range(3):
            floor = prefill_seq_floor(origin)
            generation = measure_generation(origin, body, prefill_seq_floor=floor, store_text=False)
            generations.append({"phase": "measured", "repetition": repetition, **generation})
        snapshot_started = time.monotonic()
        snapshot = moe_instrumentation(origin, "snapshot", timeout=300)
        snapshot_duration = time.monotonic() - snapshot_started
        measured = [g for g in generations if g["phase"] == "measured"]
        result = {
            "schema": "inferswarm.d2.serving-arm/1",
            "arm": arm,
            "serve_command": command,
            "startup_duration_s": ready - launched,
            "timing_snapshot_duration_s": snapshot_duration,
            "reset_duration_s": reset_duration,
            "resolved_runtime": runtime,
            "runtime_contract": contract,
            "request": {
                "workload": "W4",
                "prompt_sha256": workload.content_sha256,
                "max_tokens": body["max_tokens"],
                "ignore_eos": body["ignore_eos"],
                "sampling": {k: body[k] for k in ("temperature", "top_p", "top_k")},
                "batch_size": 1,
            },
            "generations": generations,
            "snapshot": snapshot,
            "analysis": _analyze(measured, snapshot, d2=d2),
            "physical_generation_wall_seconds": sum(float(g["wall_total_ms"]) for g in generations) / 1000.0,
            "status": "complete",
        }
        (out / f"d2-arm-{arm.lower()}.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
        return result
    finally:
        stop_server(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--output-dir", required=True)
    ns = parser.parse_args()
    root, out = Path(ns.repo), Path(ns.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    g0 = run_arm(root, out, ns.model, ns.manifest, "G0", None)
    g1 = run_arm(root, out, ns.model, ns.manifest, "G1", ns.placement)
    t0 = g0["analysis"]["decode_tok_s"]["median"]
    t1 = g1["analysis"]["decode_tok_s"]["median"]
    retention = t1 / t0
    if retention >= 0.75:
        conclusion = "GRAPH_DISTRIBUTED_STRONG"
    elif retention >= 0.50:
        conclusion = "GRAPH_DISTRIBUTED_PROMISING"
    elif retention >= 0.25:
        conclusion = "GRAPH_DISTRIBUTED_WEAK"
    elif retention >= 0.10:
        conclusion = "GRAPH_DISTRIBUTED_VERY_WEAK"
    else:
        conclusion = "GRAPH_DISTRIBUTED_CATASTROPHIC"
    analysis = {
        "schema": "inferswarm.d2.analysis/1",
        "authoritative_comparison": "direct branch-local graph-enabled G1 / graph-enabled G0",
        "T_G0_median_decode_tok_s": t0,
        "T_G1_median_decode_tok_s": t1,
        "NODE2_RETENTION": retention,
        "conclusion": conclusion,
        "FANOUT_SHAPE": g1["resolved_runtime"]["inferswarm_d2_graph_remote"]["fanout_shape"],
        "physical_gpu_experiment_runtime_seconds": (
            3.509098359994823
            + g0["physical_generation_wall_seconds"]
            + g1["physical_generation_wall_seconds"]
        ),
        "hard_firewall_seconds": 1800,
    }
    (out / "d2-analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
