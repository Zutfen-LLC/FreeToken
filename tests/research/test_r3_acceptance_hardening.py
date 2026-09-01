import json
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.inferswarm_r3.compose_result import _candidate_behavior
from benchmarks.inferswarm_r3.run_selected import (
    _selected_resources,
    _validate_compiled_decision,
    _validate_r2_plan,
)
from freetoken.research.r2_local_split import freeze_plan


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "a": "s0.source-backed-single-offload[whole-model-slot=gpu-a]",
    "b": "s1.resident-two-slot-split[opaque-slot-a=gpu-a,opaque-slot-b=gpu-b]",
    "c": "s0.source-backed-single-offload[whole-model-slot=gpu-a]",
}


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def test_r2_plan_identity_digest_drift_fails_closed_before_materialization(tmp_path):
    decision = _load("docs/inferswarm_r3/decision-b.json")
    compiled = _load("docs/inferswarm_r3/compiled-plan-b.json")
    snapshot = _load("docs/inferswarm_r3/resource-snapshot.json")
    plan = _load("docs/inferswarm_r2/frozen-plan.json")
    plan["plan_version"] = "otherwise-valid-identity-drift"
    drifted = freeze_plan(plan)
    path = tmp_path / "drifted-r2-plan.json"
    path.write_text(json.dumps(drifted))

    with pytest.raises(RuntimeError, match="identity differs"):
        _validate_r2_plan(path, compiled, _selected_resources(decision, snapshot))


def test_r2_execution_mapping_drift_fails_closed(tmp_path):
    decision = _load("docs/inferswarm_r3/decision-b.json")
    compiled = _load("docs/inferswarm_r3/compiled-plan-b.json")
    snapshot = _load("docs/inferswarm_r3/resource-snapshot.json")
    plan = _load("docs/inferswarm_r2/frozen-plan.json")
    plan["execution"][0]["compute_unit_id"] = "gpu-b"
    plan["execution"][1]["compute_unit_id"] = "gpu-a"
    drifted = freeze_plan(plan)
    compiled["r2_frozen_plan_digest"] = drifted["digest"]
    path = tmp_path / "reverse-placement-r2-plan.json"
    path.write_text(json.dumps(drifted))

    with pytest.raises(RuntimeError, match="slot mapping differs"):
        _validate_r2_plan(path, compiled, _selected_resources(decision, snapshot))


def test_compiled_input_digests_must_exactly_match_decision_inputs():
    decision = _load("docs/inferswarm_r3/decision-b.json")
    compiled = _load("docs/inferswarm_r3/compiled-plan-b.json")
    _validate_compiled_decision(compiled, decision)
    compiled["input_digests"] = deepcopy(compiled["input_digests"])
    compiled["input_digests"]["objective_digest"] = "sha256:drift"
    with pytest.raises(RuntimeError, match="input digests differ"):
        _validate_compiled_decision(compiled, decision)


def test_composer_derives_loser_ranks_and_unused_resource_explanation():
    decisions = {
        key: _load(f"docs/inferswarm_r3/decision-{key}.json") for key in "abc"
    }
    behavior = _candidate_behavior(decisions, EXPECTED)
    assert behavior["a_s1_technically_feasible_but_lower_ranked"]
    assert behavior["a_unused_gpu_b_explained"]
    assert behavior["b_s0_technically_feasible_but_lower_ranked"]

    decisions["a"]["unused_resources"][0]["reason"] = "unexplained"
    decisions["b"]["evaluations"] = deepcopy(decisions["b"]["evaluations"])
    next(
        item for item in decisions["b"]["evaluations"] if item["id"] == EXPECTED["a"]
    )["state"] = "FEASIBLE_UNRANKED"
    behavior = _candidate_behavior(decisions, EXPECTED)
    assert not behavior["a_unused_gpu_b_explained"]
    assert not behavior["b_s0_technically_feasible_but_lower_ranked"]
