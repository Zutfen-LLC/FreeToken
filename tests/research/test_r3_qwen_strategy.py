from benchmarks.inferswarm_r3.build_artifacts import (
    evidence_catalog,
    resource_snapshot,
    synthetic_planner_proof,
)
from benchmarks.inferswarm_r3.qwen_strategy import compile_selected, planning_problem
from freetoken.research.r3_planner import FEASIBLE_UNRANKED, POLICY_EXCLUDED, RANKED, freeze, plan


def _policy(excluded=()):
    return freeze({"schema": "test.policy/1", "excluded_compute_unit_ids": list(excluded), "reservations_bytes": {}})


def _objective(metric, direction):
    return freeze({"schema": "test.objective/1", "metric": metric, "direction": direction, "unit": "test", "statistic": "median", "evidence_context": {"workload_geometry": "W4-prompt121-generate32-prefill64-warm"}})


def _item(decision, candidate_id):
    return next(item for item in decision["evaluations"] if item["id"] == candidate_id)


def test_required_a_b_c_selections_and_state_separation():
    problem, snapshot = planning_problem(), resource_snapshot()
    # The catalog builder only hashes provenance; use the checked-in accepted artifact.
    from pathlib import Path
    catalog = evidence_catalog(Path(__file__).resolve().parents[2])
    baseline = "s0.source-backed-single-offload[whole-model-slot=gpu-a]"
    split = "s1.resident-two-slot-split[opaque-slot-a=gpu-a,opaque-slot-b=gpu-b]"

    a = plan(problem, snapshot, _policy(), _objective("warm_decode_tok_s", "MAXIMIZE"), catalog)
    assert a["selected_candidate_id"] == baseline
    assert _item(a, split)["technically_feasible"] and _item(a, split)["state"] == RANKED
    assert a["unused_resources"] == [{"compute_unit_id": "gpu-b", "reason": "not needed by the highest-ranked candidate"}]

    b = plan(problem, snapshot, _policy(), _objective("warm_ttft_ms", "MINIMIZE"), catalog)
    assert b["selected_candidate_id"] == split
    assert _item(b, baseline)["technically_feasible"] and _item(b, baseline)["rank"] == 2

    c = plan(problem, snapshot, _policy(["gpu-b"]), _objective("warm_decode_tok_s", "MAXIMIZE"), catalog)
    assert c["selected_candidate_id"] == baseline
    assert _item(c, split)["technically_feasible"] and _item(c, split)["state"] == POLICY_EXCLUDED


def test_unmeasured_mappings_remain_feasible_unranked():
    from pathlib import Path
    problem, snapshot = planning_problem(), resource_snapshot()
    catalog = evidence_catalog(Path(__file__).resolve().parents[2])
    decision = plan(problem, snapshot, _policy(), _objective("warm_decode_tok_s", "MAXIMIZE"), catalog)
    assert _item(decision, "s0.source-backed-single-offload[whole-model-slot=gpu-b]")["state"] == FEASIBLE_UNRANKED
    assert _item(decision, "s1.resident-two-slot-split[opaque-slot-a=gpu-b,opaque-slot-b=gpu-a]")["state"] == FEASIBLE_UNRANKED


def test_compiler_reuses_existing_paths_and_preserves_lifecycles():
    from pathlib import Path
    problem, snapshot = planning_problem(), resource_snapshot()
    catalog = evidence_catalog(Path(__file__).resolve().parents[2])
    a = plan(problem, snapshot, _policy(), _objective("warm_decode_tok_s", "MAXIMIZE"), catalog)
    b = plan(problem, snapshot, _policy(), _objective("warm_ttft_ms", "MINIMIZE"), catalog)
    compiled_a = compile_selected(a, problem, a["inputs"])
    compiled_b = compile_selected(b, problem, b["inputs"])
    assert compiled_a["realization_path"] == "freetoken-supported-single-resource-offload"
    assert compiled_a["source_lifecycle"] == "RETAIN_REQUIRED_SOURCE_BACKING"
    assert compiled_b["realization_path"] == "inferswarm-r2-frozen-plan-coordinator"
    assert compiled_b["source_lifecycle"] == "RELEASE_AFTER_FINAL_RESIDENCY"
    assert compiled_b["r2_frozen_plan_digest"].startswith("sha256:")


def test_implementation_commit_propagates_through_decision_and_compilation():
    implementation_commit = "a" * 40
    problem = planning_problem(implementation_commit)
    snapshot = resource_snapshot(implementation_commit)
    from pathlib import Path

    catalog = evidence_catalog(Path(__file__).resolve().parents[2], implementation_commit)
    decision = plan(
        problem,
        snapshot,
        _policy(),
        _objective("warm_decode_tok_s", "MAXIMIZE"),
        catalog,
    )
    compiled = compile_selected(decision, problem, decision["inputs"])
    assert decision["implementation_commit"] == implementation_commit
    assert compiled["implementation_commit"] == implementation_commit


def test_retained_synthetic_proof_uses_the_same_generic_planner():
    proof = synthetic_planner_proof("a" * 40)
    assert proof["passed"]
    assert proof["cpu_testable"]
    assert proof["same_generic_planner"] == "freetoken.research.r3_planner.plan"
    assert proof["selected_candidate_id"] == "violet-shape[amber-slot=copper-unit]"


def test_physical_resource_records_preserve_exact_selected_identities():
    from benchmarks.inferswarm_r3.run_selected import _selected_resources

    problem, snapshot = planning_problem(), resource_snapshot()
    from pathlib import Path

    catalog = evidence_catalog(Path(__file__).resolve().parents[2])
    decision = plan(
        problem,
        snapshot,
        _policy(),
        _objective("warm_ttft_ms", "MINIMIZE"),
        catalog,
    )
    resources = _selected_resources(decision, snapshot)
    assert [item["compute_unit_id"] for item in resources] == ["gpu-a", "gpu-b"]
    assert all(item["stable_device_id"].startswith("GPU-") for item in resources)
