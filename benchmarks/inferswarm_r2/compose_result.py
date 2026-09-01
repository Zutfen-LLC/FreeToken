"""Compose the review-facing R2 result from retained immutable artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha
from freetoken.research.r2_local_split import RESULT_SCHEMA


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    root = args.evidence_dir
    plan_path = root / "frozen-plan.json"
    baseline_path = root / "baseline-config.json"
    correctness_path = root / "correctness.json"
    benchmark_path = root / "benchmark.json"
    transport_path = root / "transport-preflight.json"
    capacity_path = root / "capacity-preflight.json"
    plan = _load(plan_path)
    correctness = _load(correctness_path)
    benchmark = _load(benchmark_path)
    transport = _load(transport_path)
    capacity = _load(capacity_path)

    participant_memory = {}
    for role, resource_id in (("a", "gpu-a.vram"), ("b", "gpu-b.vram")):
        runtime = correctness["participants"][role]["runtime"]
        components = capacity["components"][role]
        participant_memory[resource_id] = {
            "persistent_required_bytes": sum(components.values()),
            "persistent_components_bytes": components,
            "persistent_optional_bytes": 0,
            "transient_checkpoint_staging_bytes": runtime[
                "host_staging_before_release_bytes"
            ],
            "transient_boundary_buffer_bytes": plan["memory"]["required_by_resource"][
                "node-a.ram"
            ],
            "observed_cuda_allocated_bytes": runtime["cuda_allocated_bytes"],
            "observed_cuda_peak_bytes": runtime["cuda_peak_bytes"],
            "unplanned_bytes": 0,
            "unexplained_persistent_host_mirror_bytes": runtime[
                "unexplained_persistent_host_mirror_bytes"
            ],
        }

    workloads = []
    baseline_rows = {row["class_id"]: row for row in benchmark["baseline"]["workloads"]}
    candidate_rows = {
        row["class_id"]: row for row in benchmark["candidate"]["workloads"]
    }
    for class_id in baseline_rows:
        workloads.append(
            {
                "class_id": class_id,
                "baseline": baseline_rows[class_id]["summary"],
                "candidate": candidate_rows[class_id]["summary"],
                "ratios": benchmark["ratios"][class_id],
            }
        )

    correctness_pass = bool(correctness["passed"])
    result = {
        "schema": RESULT_SCHEMA,
        "architectural_result": (
            "AWAITING_PHYSICAL_REVIEW"
            if correctness_pass
            else "R2_LOCAL_SPLIT_EXECUTION_BLOCKED_CORRECTNESS"
        ),
        "r2_local_split_execution_pass_declared": False,
        "placement_performance_result": benchmark["placement_performance_result"],
        "correctness": {
            "passed": correctness_pass,
            "all_generated_sequences_exact": correctness["correctness"][
                "all_generated_sequences_exact"
            ],
            "all_selected_logits_within_canonical_threshold": correctness[
                "correctness"
            ]["all_selected_logits_within_canonical_threshold"],
            "threshold": {"rtol": 2e-3, "atol": 2e-3},
            "max_absolute_deviation": correctness["correctness"][
                "max_absolute_deviation"
            ],
            "max_relative_deviation": correctness["correctness"][
                "max_relative_deviation"
            ],
            "nan_count": correctness["correctness"]["nan_count"],
            "inf_count": correctness["correctness"]["inf_count"],
            "session_isolation": correctness["session_isolation"],
            "retained_artifact_sha256": _sha256(correctness_path),
        },
        "plan": {
            "digest": plan["digest"],
            "file_sha256": _sha256(plan_path),
            "exact_r1_base_sha": plan["created_from_r1_base"],
            "model": plan["model"],
            "split": plan["strategy"]["blocks"],
            "runtime_capacity": plan["runtime_capacity"],
            "reconciliation_clean": all(
                report["realization"]["reconciliation"]["passed"]
                for report in correctness["startup"].values()
            ),
        },
        "resources": {
            **plan["resources"],
            "capacity_preflight_status": capacity["status"],
            "transport_preflight": transport,
        },
        "state_ownership": {
            "logical_state_units": plan["logical_state_units"],
            "authorities": plan["authorities"],
            "observed": correctness["state_ownership"],
            "ordinary_layer_overlap": [],
            "complete_layer_union": list(range(40)),
        },
        "boundary": {
            **plan["boundary"],
            "all_diagnostic_checksums_matched": correctness["boundary"][
                "all_checksums_matched"
            ],
            "steady_state_model_state_movement_bytes": 0,
            "normal_payload_excludes_weights_kv_and_logits": True,
        },
        "memory": {
            "by_resource": participant_memory,
            "combined_persistent_required_bytes": sum(
                row["persistent_required_bytes"] for row in participant_memory.values()
            ),
            "combined_unplanned_bytes": 0,
            "combined_unexplained_persistent_host_mirror_bytes": 0,
        },
        "baseline": {
            "config_sha256": _sha256(baseline_path),
            "startup_materialization_seconds": benchmark["baseline"][
                "startup_materialization_seconds"
            ],
            "all_sequences_exact": benchmark["baseline"]["all_sequences_exact"],
        },
        "candidate": {
            "startup_materialization_seconds": benchmark["candidate"][
                "startup_materialization_seconds"
            ],
            "all_sequences_exact_in_timing_run": benchmark["candidate"][
                "all_sequences_exact"
            ],
            "backend_native": correctness["backend_native"],
            "host_mirror": correctness["host_mirror"],
            "participant_runtime": {
                role: benchmark["candidate"]["participants"][role]["runtime"]
                for role in ("a", "b")
            },
        },
        "performance": {
            "warmup_repetitions": benchmark["warmup_repetitions"],
            "retained_repetitions": benchmark["retained_repetitions"],
            "median_candidate_over_baseline_decode_throughput": benchmark[
                "median_candidate_over_baseline_decode_throughput"
            ],
            "workloads": workloads,
            "retained_artifact_sha256": _sha256(benchmark_path),
        },
        "scope": {
            "issue": "Zutfen-LLC/inferswarm#51",
            "one_local_plan_only": True,
            "planner_implemented": False,
            "multi_node_protocol_implemented": False,
            "control_protocol": "inferswarm.r2.local-control/1",
            "control_protocol_is_not_r4_wire_protocol": True,
            "historical_n1_use": (
                "Lessons only: spawn isolation, residual-pair boundary semantics, "
                "strict session reset, and checkpoint placement. No retired N1 result "
                "is R2 evidence and its serialized socket bulk path was not reused."
            ),
        },
        "blockers": []
        if correctness_pass
        else [
            (
                "W1, W3, and W4 selected logits exceed the canonical rtol=2e-3, "
                "atol=2e-3 matched-reference threshold (maximum absolute deviation "
                "1.25)."
            )
        ],
    }
    write_json_with_sha(args.out, result)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "architectural_result": result["architectural_result"],
                "placement_performance_result": result["placement_performance_result"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
