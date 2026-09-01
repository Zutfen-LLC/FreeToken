"""Fresh matched baseline/candidate R2 benchmark with raw repetitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha

from .coordinator import LocalSplitCoordinator


def _summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p50": statistics.median(values),
        "p95": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
    }


def _prompt_ids(tokenizer, workload) -> list[int]:
    body = workload.greedy_reference_body("nvidia/Qwen3.6-35B-A3B-NVFP4")
    return list(
        tokenizer.apply_chat_template(
            body["messages"],
            tokenize=True,
            add_generation_prompt=True,
            **body["chat_template_kwargs"],
        )["input_ids"]
    )


def _gpu_state() -> list[dict]:
    fields = "uuid,temperature.gpu,power.draw,clocks.sm,pcie.link.gen.current,pcie.link.width.current"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [
        dict(
            zip(
                fields.split(","),
                (value.strip() for value in line.split(",")),
                strict=True,
            )
        )
        for line in output.splitlines()
    ]


def _candidate(
    *,
    plan_path: Path,
    model: str,
    manifest,
    tokenizer,
    reference,
    repetitions: int,
    classes: list[str],
) -> dict:
    plan = json.loads(plan_path.read_text())
    expected = {item["class_id"]: item for item in reference["workloads"]}
    rows = []
    started = time.perf_counter()
    with LocalSplitCoordinator(
        plan_path=str(plan_path), model_path=model, diagnostic=False
    ) as coordinator:
        ready_s = time.perf_counter() - started
        for class_index, class_id in enumerate(classes):
            prompt = _prompt_ids(tokenizer, manifest.by_class()[class_id])
            class_rows = []
            for repetition in range(repetitions + 1):
                row = coordinator.run_session(
                    session_id=10_000 + class_index * 100 + repetition,
                    prompt_ids=prompt,
                    max_new_tokens=32,
                    prefill_chunk=plan["runtime_capacity"]["prefill_chunk_tokens"],
                )
                row["phase"] = "warmup" if repetition == 0 else "retained"
                row["repetition"] = max(0, repetition - 1)
                row["exact_generated_sequence"] = (
                    row["generated_token_ids"]
                    == expected[class_id]["generated_token_ids"][:32]
                )
                if not row["exact_generated_sequence"]:
                    raise RuntimeError(
                        f"candidate {class_id} repetition {repetition} diverged"
                    )
                class_rows.append(row)
            rows.append(
                {
                    "class_id": class_id,
                    "runs": class_rows,
                    "summary": {
                        key: _summary([float(row[key]) for row in class_rows[1:]])
                        for key in (
                            "prefill_wall_ns",
                            "ttft_ns",
                            "decode_wall_ns",
                            "decode_tokens_per_second",
                            "total_request_wall_ns",
                            "boundary_transfer_ns",
                            "block_a_compute_ns",
                            "block_b_compute_ns",
                        )
                    },
                }
            )
        reports = coordinator.reports()
    return {
        "configuration": "candidate",
        "startup_materialization_seconds": ready_s,
        "workloads": rows,
        "participants": reports,
        "all_sequences_exact": all(
            run["exact_generated_sequence"] for row in rows for run in row["runs"]
        ),
        "gpu_state_after": _gpu_state(),
    }


def _baseline(
    *,
    repo: Path,
    config: dict,
    model: str,
    manifest,
    tokenizer,
    reference,
    repetitions: int,
    log_path: Path,
    classes: list[str],
) -> dict:
    from inferswarm_phase0.client import (
        fetch_instrumentation,
        free_port,
        measure_generation,
        prefill_seq_floor,
        start_server,
        stop_server,
    )
    from inferswarm_phase0.manifest import CANONICAL_GREEDY_SAMPLING

    server = config["server"]
    backend = config["backend"]
    uuid = config["compute_unit"]["stable_device_id"]
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    command = [
        str(repo / ".venv/bin/python"),
        "-m",
        "freetoken.cli",
        "serve",
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--gpu",
        uuid,
        "--moe-backend",
        backend["moe"],
        "--moe-cpu-layers",
        str(backend["moe_cpu_layers"]),
        "--nvfp4-backend",
        backend["nvfp4"],
        "--moe-cache-size",
        str(backend["moe_cache_slots"]),
        "--kv-reserve-tokens",
        str(server["kv_reserve_tokens"]),
        "--num-tokens",
        str(server["num_tokens"]),
        "--memory-ratio",
        str(server["memory_ratio"]),
        "--max-running-requests",
        str(server["max_running_requests"]),
        "--cuda-graph-max-bs",
        "1",
        "--sampling-defaults",
        server["sampling_defaults"],
        "--max-prefill-length",
        str(server["max_prefill_length"]),
        "--attention-backend",
        backend["attention"],
    ]
    started = time.perf_counter()
    handle = start_server(
        command,
        origin,
        str(log_path),
        env_overrides={"PYTHONPATH": "python:benchmarks"},
        ready_timeout=1200,
        echo=False,
    )
    try:
        ready_s = time.perf_counter() - started
        runtime = fetch_instrumentation(origin)
        rows = []
        expected_by_class = {item["class_id"]: item for item in reference["workloads"]}
        for class_id in classes:
            workload = manifest.by_class()[class_id]
            body = workload.request_body(
                reference["revision"], sampling_override=CANONICAL_GREEDY_SAMPLING
            )
            body["max_tokens"] = 32
            expected_text = tokenizer.decode(
                expected_by_class[class_id]["generated_token_ids"][:32],
                skip_special_tokens=True,
            )
            expected_hash = hashlib.sha256(expected_text.encode()).hexdigest()
            class_rows = []
            for repetition in range(repetitions + 1):
                floor = prefill_seq_floor(origin)
                row = measure_generation(
                    origin, body, prefill_seq_floor=floor, store_text=True, timeout=3600
                )
                row["phase"] = "warmup" if repetition == 0 else "retained"
                row["repetition"] = max(0, repetition - 1)
                row["expected_output_sha256"] = expected_hash
                row["exact_generated_text"] = row["output_sha256"] == expected_hash
                # Request-visible prefill wall is TTFT here; the server's GPU-only
                # prefill record may represent only one chunk and is retained
                # separately in ``prefill`` rather than misreported as whole prompt.
                row["prefill_wall_ms"] = row["ttft_ms"]
                if not row["exact_generated_text"]:
                    raise RuntimeError(
                        f"baseline {class_id} output differs from frozen token reference"
                    )
                row.pop("output_text", None)
                class_rows.append(row)
            rows.append(
                {
                    "class_id": class_id,
                    "runs": class_rows,
                    "summary": {
                        key: _summary([float(row[key]) for row in class_rows[1:]])
                        for key in (
                            "prefill_wall_ms",
                            "ttft_ms",
                            "decode_window_s",
                            "decode_tok_s",
                            "wall_total_ms",
                        )
                    },
                }
            )
        return {
            "configuration": "baseline",
            "command": command,
            "startup_materialization_seconds": ready_s,
            "resolved_runtime": runtime,
            "workloads": rows,
            "all_sequences_exact": True,
            "gpu_state_after": _gpu_state(),
        }
    finally:
        stop_server(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--classes", default="W1,W2,W3,W4")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from inferswarm_phase0.manifest import load_manifest
    from transformers import AutoTokenizer

    plan = json.loads(args.plan.read_text())
    baseline_config = json.loads(args.baseline_config.read_text())
    reference = json.loads(args.reference.read_text())
    manifest = load_manifest(args.manifest, canonical=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    classes = [value.strip() for value in args.classes.split(",") if value.strip()]
    before = _gpu_state()
    baseline = _baseline(
        repo=args.repo,
        config=baseline_config,
        model=args.model,
        manifest=manifest,
        tokenizer=tokenizer,
        reference=reference,
        repetitions=args.repetitions,
        log_path=args.out.with_suffix(".baseline.log"),
        classes=classes,
    )
    candidate = _candidate(
        plan_path=args.plan,
        model=args.model,
        manifest=manifest,
        tokenizer=tokenizer,
        reference=reference,
        repetitions=args.repetitions,
        classes=classes,
    )
    ratios = {}
    for baseline_row, candidate_row in zip(
        baseline["workloads"], candidate["workloads"], strict=True
    ):
        class_id = baseline_row["class_id"]
        b_rate = baseline_row["summary"]["decode_tok_s"]["median"]
        c_rate = candidate_row["summary"]["decode_tokens_per_second"]["median"]
        ratios[class_id] = {
            "candidate_over_baseline_decode_throughput": c_rate / b_rate,
            "candidate_over_baseline_ttft": (
                candidate_row["summary"]["ttft_ns"]["median"] / 1e6
            )
            / baseline_row["summary"]["ttft_ms"]["median"],
        }
    median_ratio = statistics.median(
        value["candidate_over_baseline_decode_throughput"] for value in ratios.values()
    )
    payload = {
        "schema": "inferswarm.r2.matched-benchmark/1",
        "plan_digest": plan["digest"],
        "warmup_repetitions": 1,
        "retained_repetitions": args.repetitions,
        "diagnostics_disabled": True,
        "gpu_state_before": before,
        "baseline": baseline,
        "candidate": candidate,
        "ratios": ratios,
        "placement_performance_result": "PERFORMANCE_POSITIVE"
        if median_ratio > 1
        else "PERFORMANCE_NEGATIVE",
        "median_candidate_over_baseline_decode_throughput": median_ratio,
        "passed": baseline["all_sequences_exact"] and candidate["all_sequences_exact"],
    }
    write_json_with_sha(args.out, payload)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "passed": payload["passed"],
                "assessment": payload["placement_performance_result"],
                "ratios": ratios,
            },
            indent=2,
        )
    )
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
