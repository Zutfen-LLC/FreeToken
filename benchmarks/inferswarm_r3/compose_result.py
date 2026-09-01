"""Compose the retained R3 gate result from immutable input and execution artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha


def _load(root: Path, name: str) -> dict:
    path = root / name
    expected = path.with_suffix(path.suffix + ".sha256").read_text().split()[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"checksum mismatch for {path}")
    return json.loads(path.read_text())


def _evaluation(decision: dict, candidate_id: str) -> dict:
    return next(
        item for item in decision["evaluations"] if item["id"] == candidate_id
    )


def _candidate_behavior(decisions: dict[str, dict], expected: dict[str, str]) -> dict:
    """Derive required A/B loser ranks and A's unused-resource explanation."""
    a_s0 = _evaluation(decisions["a"], expected["a"])
    a_s1 = _evaluation(decisions["a"], expected["b"])
    b_s0 = _evaluation(decisions["b"], expected["a"])
    b_s1 = _evaluation(decisions["b"], expected["b"])
    unused_gpu_b = next(
        (
            item
            for item in decisions["a"].get("unused_resources", [])
            if item.get("compute_unit_id") == "gpu-b"
        ),
        None,
    )
    a_s1_lower = (
        decisions["a"].get("selected_candidate_id") == expected["a"]
        and a_s1.get("technically_feasible") is True
        and a_s1.get("state") == "RANKED"
        and isinstance(a_s0.get("rank"), int)
        and isinstance(a_s1.get("rank"), int)
        and a_s1["rank"] > a_s0["rank"]
    )
    a_unused_explained = unused_gpu_b == {
        "compute_unit_id": "gpu-b",
        "reason": "not needed by the highest-ranked candidate",
    }
    b_s0_lower = (
        decisions["b"].get("selected_candidate_id") == expected["b"]
        and b_s0.get("technically_feasible") is True
        and b_s0.get("state") == "RANKED"
        and isinstance(b_s0.get("rank"), int)
        and isinstance(b_s1.get("rank"), int)
        and b_s0["rank"] > b_s1["rank"]
    )
    return {
        "a_s0": a_s0,
        "a_s1": a_s1,
        "a_unused_gpu_b": unused_gpu_b,
        "a_s1_technically_feasible_but_lower_ranked": a_s1_lower,
        "a_unused_gpu_b_explained": a_unused_explained,
        "b_s0": b_s0,
        "b_s1": b_s1,
        "b_s0_technically_feasible_but_lower_ranked": b_s0_lower,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("docs/inferswarm_r3"))
    parser.add_argument("--out", type=Path, default=Path("docs/inferswarm_r3/result.json"))
    args = parser.parse_args(argv)
    decisions = {key: _load(args.artifact_dir, f"decision-{key}.json") for key in "abc"}
    physical = {key: _load(args.artifact_dir, f"physical-{key}.json") for key in "ab"}
    tests = _load(args.artifact_dir, "test-summary.json")
    synthetic = _load(args.artifact_dir, "synthetic-planner-proof.json")
    implementation_commits = {
        artifact.get("implementation_commit")
        for artifact in (*decisions.values(), *physical.values(), tests, synthetic)
    }
    if len(implementation_commits) != 1 or None in implementation_commits:
        raise ValueError("R3 artifacts do not share one implementation commit")
    implementation_commit = implementation_commits.pop()
    expected = {
        "a": "s0.source-backed-single-offload[whole-model-slot=gpu-a]",
        "b": "s1.resident-two-slot-split[opaque-slot-a=gpu-a,opaque-slot-b=gpu-b]",
        "c": "s0.source-backed-single-offload[whole-model-slot=gpu-a]",
    }
    selection_pass = all(decisions[key]["selected_candidate_id"] == value for key, value in expected.items())
    behavior = _candidate_behavior(decisions, expected)
    split_c = _evaluation(decisions["c"], expected["b"])
    policy_pass = split_c["technically_feasible"] and split_c["state"] == "POLICY_EXCLUDED"
    uncertainty_pass = all(
        any(item["state"] == "FEASIBLE_UNRANKED" for item in decisions[key]["evaluations"])
        for key in "ab"
    )
    physical_pass = all(item["passed"] for item in physical.values())
    gates = {
        "automatic_objective_relative_selection": selection_pass,
        "scenario_a_s1_ranked_lower_than_selected_s0": behavior["a_s1_technically_feasible_but_lower_ranked"],
        "scenario_a_unused_gpu_b_truthfully_explained": behavior["a_unused_gpu_b_explained"],
        "scenario_b_s0_ranked_lower_than_selected_s1": behavior["b_s0_technically_feasible_but_lower_ranked"],
        "policy_separate_from_technical_feasibility": policy_pass,
        "unknown_evidence_preserved_as_feasible_unranked": uncertainty_pass,
        "decisions_frozen_before_materialization": all(item["decision_frozen_and_environment_validated_before_materialization"] for item in physical.values()),
        "both_selected_shapes_physically_correct": physical_pass,
        "synthetic_non_qwen_strategy_uses_generic_planner": synthetic["passed"],
        "ordinary_offload_source_semantics_preserved": physical["a"]["correctness"]["source_lifecycle"] == "RETAIN_REQUIRED_SOURCE_BACKING",
        "resident_release_and_steady_invariants_preserved": physical["b"]["correctness"]["source_lifecycle"] == "RELEASE_AFTER_FINAL_RESIDENCY" and physical["b"]["correctness"]["resident_invariants_passed"],
    }
    passed = all(gates.values())
    payload = {
        "schema": "inferswarm.r3.result/1",
        "implementation_commit": implementation_commit,
        "issue": "https://github.com/Zutfen-LLC/inferswarm/issues/55",
        "base": "2fc64ae7c79bdc494a52468da329ddafd0adb8ba",
        "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "revision": "491c2f1ea524c639598bf8fa787a93fed5a6fbce",
        "scenario_selections": {key.upper(): decisions[key]["selected_candidate_id"] for key in "abc"},
        "architectural_disposition": {
            "passed": passed,
            "statement": "R3 proves the minimum research-internal strategy/planner seam: a model-opaque generic planner selects from strategy-defined legal candidates before the strategy compiles the immutable selection into existing execution paths.",
        },
        "candidate_objective_behavior": {
            "A": {
                "selected": expected["a"],
                "reason": "highest applicable measured median warm decode throughput",
                "s0_rank": behavior["a_s0"].get("rank"),
                "s1_rank": behavior["a_s1"].get("rank"),
                "s1_technically_feasible_but_lower_ranked": behavior["a_s1_technically_feasible_but_lower_ranked"],
                "unused_gpu_b": behavior["a_unused_gpu_b"],
                "unused_gpu_b_explained": behavior["a_unused_gpu_b_explained"],
            },
            "B": {
                "selected": expected["b"],
                "reason": "lowest applicable measured median matched warm-request TTFT",
                "s1_rank": behavior["b_s1"].get("rank"),
                "s0_rank": behavior["b_s0"].get("rank"),
                "s0_technically_feasible_but_lower_ranked": behavior["b_s0_technically_feasible_but_lower_ranked"],
            },
            "C": {
                "selected": expected["c"],
                "gpu_b_dependent_s1_technically_feasible": split_c["technically_feasible"],
                "gpu_b_dependent_s1_classification": split_c["state"],
                "reason": split_c["policy_reasons"],
            },
        },
        "physical_execution": {
            "both_selected_shapes_execute_correctly_through_r3_seam": physical_pass,
            "scenario_a_selected_resources": physical["a"]["selected_resources"],
            "scenario_b_selected_resources": physical["b"]["selected_resources"],
            "scenario_b_startup_reconciliation_passed": physical["b"]["correctness"]["startup_reconciliation_passed"],
            "scenario_b_resident_invariants_passed": physical["b"]["correctness"]["resident_invariants_passed"],
        },
        "decision_digests": {key.upper(): decisions[key]["digest"] for key in "abc"},
        "physical_execution_digests": {key.upper(): physical[key]["digest"] if "digest" in physical[key] else "sha256:" + hashlib.sha256((args.artifact_dir / f"physical-{key}.json").read_bytes()).hexdigest() for key in "ab"},
        "correctness": {
            key.upper(): {
                "workloads": physical[key]["workloads"],
                "exact_generated_sequences": physical[key]["correctness"]["exact_generated_sequences"],
                "selected_logits_within_r2_v2_threshold": physical[key]["correctness"]["selected_logits_within_r2_v2_threshold"],
                "max_absolute_deviation": physical[key]["correctness"]["max_absolute_deviation"],
                "max_relative_deviation": physical[key]["correctness"]["max_relative_deviation"],
                "nan_count": physical[key]["correctness"]["nan_count"],
                "inf_count": physical[key]["correctness"]["inf_count"],
            }
            for key in "ab"
        },
        "gates": gates,
        "test_summary": {
            "passed": tests["total_passed"],
            "failed": tests["total_failed"],
            "deselected": tests["total_deselected"],
            "artifact": "test-summary.json",
        },
        "limitations": [
            "Research-internal schemas and strategy seam are not public APIs.",
            "Planning is setup-time only; there is no dynamic replanning.",
            "Evidence applies only to the exact accepted mappings, topology, backend/runtime, and W4 warm workload context.",
            "RETAIN remains occupied capacity and is not treated as a live-evictable cache.",
        ],
        "scope_not_proven": [
            "production planner quality",
            "dynamic replanning",
            "adaptive demand learning",
            "multi-node execution",
            "R4 network viability",
            "plan epochs or elasticity",
            "stable public planner or strategy APIs",
            "generalized performance prediction",
            "multi-vendor execution",
            "transparent cache eviction or rematerialization",
        ],
        "result": "R3_MINIMUM_AUTOMATIC_PLANNING_PASS" if passed else "R3_MINIMUM_AUTOMATIC_PLANNING_FAIL",
        "passed": passed,
    }
    write_json_with_sha(args.out, payload)
    print(json.dumps({"out": str(args.out), "result": payload["result"], "gates": gates}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
