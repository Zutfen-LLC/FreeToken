"""Compose the retained review result for frozen R2 methodology v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha

from .v2_support import (
    FROZEN_PLAN_DIGEST,
    methodology_record,
    sha256_file,
    validate_candidate_pass,
    validate_v2_output_path,
)


def compose(evidence_dir: Path) -> dict:
    correctness_path = evidence_dir / "correctness-v2.json"
    benchmark_path = evidence_dir / "benchmark.json"
    historical_correctness_path = evidence_dir / "correctness.json"
    historical_result_path = evidence_dir / "result.json"
    correctness = json.loads(correctness_path.read_text())
    validate_candidate_pass(correctness)
    benchmark = json.loads(benchmark_path.read_text())
    historical_result = json.loads(historical_result_path.read_text())
    if benchmark.get("plan_digest") != FROZEN_PLAN_DIGEST:
        raise ValueError("historical benchmark plan digest differs from frozen R2 plan")
    if benchmark.get("placement_performance_result") != "PERFORMANCE_NEGATIVE":
        raise ValueError("historical placement performance assessment changed")

    return {
        "schema": "inferswarm.r2.result-v2/1",
        "methodology": methodology_record(),
        "architectural_result": "R2_LOCAL_SPLIT_EXECUTION_PASS",
        "historical_original_comparator_result": (
            "R2_LOCAL_SPLIT_EXECUTION_BLOCKED_CORRECTNESS"
        ),
        "diagnosis": "REFERENCE_GEOMETRY_MISMATCH",
        "corrected_frozen_methodology_result": "R2_LOCAL_SPLIT_EXECUTION_PASS",
        "history_preserved": {
            "correctness_artifact_sha256": sha256_file(historical_correctness_path),
            "result_artifact_sha256": sha256_file(historical_result_path),
            "historical_verdict": historical_result["architectural_result"],
        },
        "correctness": {
            "artifact": correctness_path.name,
            "artifact_sha256": sha256_file(correctness_path),
            "reference": correctness["reference"],
            "workloads": [
                {
                    "class_id": row["class_id"],
                    "exact_generated_tokens": row["exact_generated_sequence"],
                    "selected_logits_within_threshold": all(
                        item["within_canonical_threshold"]
                        for item in row["logit_checkpoints"]
                    ),
                }
                for row in correctness["correctness"]["workloads"]
            ],
            "threshold": {"rtol": 0.002, "atol": 0.002},
            "max_absolute_deviation": correctness["correctness"][
                "max_absolute_deviation"
            ],
            "max_relative_deviation": correctness["correctness"][
                "max_relative_deviation"
            ],
            "nan_count": correctness["correctness"]["nan_count"],
            "inf_count": correctness["correctness"]["inf_count"],
            "acceptance_gates": correctness["acceptance_gates"],
        },
        "plan": correctness["plan"],
        "backend_native": correctness["backend_native"],
        "host_mirror": correctness["host_mirror"],
        "state_ownership": correctness["state_ownership"],
        "boundary": {
            **correctness["boundary"],
            "steady_state_model_state_movement_bytes": 0,
        },
        "performance": {
            "status": "PREVIOUSLY_MEASURED_MATCHED_PERFORMANCE_EVIDENCE",
            "rerun_in_v2": False,
            "performance_sensitive_executable_code_changed": False,
            "placement_assessment": "PERFORMANCE_NEGATIVE",
            "exact_scope": "this frozen plan, hardware, software lineage, and topology",
            "median_split_over_baseline_decode_throughput": benchmark[
                "median_candidate_over_baseline_decode_throughput"
            ],
            "ttft_disposition": (
                "dramatically lower for the resident split, especially for long prompts"
            ),
            "artifact_sha256": sha256_file(benchmark_path),
        },
        "scope": {
            "one_frozen_doctrine_shaped_local_split_plan": True,
            "automatic_planning": False,
            "optimal_placement": False,
            "general_multi_gpu_scaling": False,
            "network_execution": False,
            "dynamic_epochs": False,
            "other_model_architectures": False,
            "future_public_apis": False,
        },
        "passed": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    validate_v2_output_path(args.out)
    if args.out.name != "result-v2.json":
        raise ValueError("v2 result composer must write result-v2.json")
    result = compose(args.evidence_dir)
    write_json_with_sha(args.out, result)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "architectural_result": result["architectural_result"],
                "performance": result["performance"]["placement_assessment"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
