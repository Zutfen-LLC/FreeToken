from copy import deepcopy
from pathlib import Path

import pytest

from freetoken.research.r3_planner import (
    EnvironmentDriftError,
    FEASIBLE_UNRANKED,
    POLICY_EXCLUDED,
    RANKED,
    TECHNICALLY_INFEASIBLE,
    enumerate_candidates,
    freeze,
    plan,
    validate_decision_environment,
)


def _snapshot(*, capacity=1_000, host_capacity=2_000):
    return freeze(
        {
            "schema": "synthetic.resources/1",
            "context": "synthetic",
            "nodes": [
                {
                    "id": "node",
                    "compute_units": [
                        {
                            "id": "c0",
                            "stable_device_id": "synthetic-0",
                            "memory_resource_id": "m0",
                            "capabilities": ["opaque-cap", "other-cap"],
                            "integrity_eligible": True,
                        },
                        {
                            "id": "c1",
                            "stable_device_id": "synthetic-1",
                            "memory_resource_id": "m1",
                            "capabilities": ["opaque-cap"],
                            "integrity_eligible": True,
                        },
                    ],
                    "memory_resources": [
                        {"id": "m0", "kind": "accelerator", "capacity_bytes": capacity},
                        {"id": "m1", "kind": "accelerator", "capacity_bytes": capacity},
                        {"id": "ram", "kind": "system-ram", "capacity_bytes": host_capacity},
                    ],
                }
            ],
            "links": [
                {
                    "id": "p0",
                    "source_memory_resource_id": "m0",
                    "target_memory_resource_id": "m1",
                    "capabilities": ["opaque-transport"],
                }
            ],
        }
    )


def _problem(*, two_shapes=True, required=100, transient=50, lifecycle="TRANSIENT_RELEASE_AFTER_FINALIZATION"):
    shapes = [
        {
            "id": "one",
            "slots": [
                {
                    "id": "alpha",
                    "required_capabilities": ["opaque-cap"],
                    "memory": {"persistent_required_bytes": required},
                }
            ],
            "materializations": [
                {"id": "source", "memory_kind": "system-ram", "bytes": transient, "lifecycle": lifecycle}
            ],
            "strategy_payload": {"secret": "planner-must-not-read"},
        }
    ]
    if two_shapes:
        shapes.append(
            {
                "id": "two",
                "slots": [
                    {"id": "left", "required_capabilities": ["opaque-cap"], "memory": {"persistent_required_bytes": required}},
                    {"id": "right", "required_capabilities": ["opaque-cap"], "memory": {"persistent_required_bytes": required}},
                ],
                "distinct_slot_groups": [["left", "right"]],
                "paths": [
                    {"id": "boundary", "from_slot": "left", "to_slot": "right", "required_capabilities": ["opaque-transport"]}
                ],
                "strategy_payload": {"different_secret": [1, 2, 3]},
            }
        )
    return freeze({"schema": "synthetic.problem/1", "evidence_context": {"model_revision": "r1", "runtime": "v1"}, "shapes": shapes})


def _policy(**updates):
    value = {"schema": "synthetic.policy/1", "excluded_compute_unit_ids": [], "reservations_bytes": {}}
    value.update(updates)
    return freeze(value)


def _objective(metric="speed", direction="MAXIMIZE"):
    return freeze({"schema": "synthetic.objective/1", "metric": metric, "direction": direction, "unit": "widgets/s", "statistic": "median"})


def _catalog(problem, snapshot, records=None, *, context=None):
    context = context or {"model_revision": "r1", "runtime": "v1"}
    if records is None:
        mappings = {item["id"]: item for item in enumerate_candidates(problem, snapshot)}
        records = [
            {"id": "one-c0-speed", "shape_id": "one", "mapping": mappings["one[alpha=c0]"]["mapping"], "required_context": context, "freshness": "CURRENT", "class": "MEASURED", "metric": {"name": "speed", "value": 10.0}},
            {"id": "one-c0-latency", "shape_id": "one", "mapping": mappings["one[alpha=c0]"]["mapping"], "required_context": context, "freshness": "CURRENT", "class": "MEASURED", "metric": {"name": "latency", "value": 100.0}},
        ]
        if "two[left=c0,right=c1]" in mappings:
            records.extend(
                [
                    {"id": "two-c0-c1-speed", "shape_id": "two", "mapping": mappings["two[left=c0,right=c1]"]["mapping"], "required_context": context, "freshness": "ACCEPTED_COMPATIBLE", "class": "MEASURED", "metric": {"name": "speed", "value": 8.0}},
                    {"id": "two-c0-c1-latency", "shape_id": "two", "mapping": mappings["two[left=c0,right=c1]"]["mapping"], "required_context": context, "freshness": "CURRENT", "class": "MEASURED", "metric": {"name": "latency", "value": 20.0}},
                ]
            )
    return freeze({"schema": "synthetic.evidence/1", "context": context, "records": records})


def _run(problem=None, snapshot=None, policy=None, objective=None, catalog=None):
    problem = problem or _problem()
    snapshot = snapshot or _snapshot()
    return plan(problem, snapshot, policy or _policy(), objective or _objective(), catalog or _catalog(problem, snapshot))


def _evaluation(decision, candidate_id):
    return next(item for item in decision["evaluations"] if item["id"] == candidate_id)


def test_enumeration_and_digest_are_deterministic_and_one_shape_maps_to_many_units():
    problem, snapshot = _problem(), _snapshot()
    ids = [item["id"] for item in enumerate_candidates(problem, snapshot)]
    assert ids == sorted(ids)
    assert "one[alpha=c0]" in ids and "one[alpha=c1]" in ids
    assert _run(problem, snapshot)["digest"] == _run(problem, snapshot)["digest"]


def test_distinct_constraint_rejects_same_unit_but_preserves_audit_candidate():
    decision = _run()
    item = _evaluation(decision, "two[left=c0,right=c0]")
    assert item["state"] == TECHNICALLY_INFEASIBLE
    assert "distinct" in item["technical_reasons"][0]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda p, s: p["shapes"][0]["slots"][0].update(allowed_compute_unit_ids=["absent"]), "missing"),
        (lambda p, s: p["shapes"][0]["slots"][0].update(required_capabilities=["unsupported"]), "lacks capabilities"),
    ],
)
def test_missing_or_incompatible_compute_is_technical_failure(mutation, reason):
    problem, snapshot = _problem(two_shapes=False), _snapshot()
    mutation(problem, snapshot)
    problem = freeze(problem)
    item = _run(problem, snapshot, catalog=_catalog(problem, snapshot, records=[]))["evaluations"][0]
    assert item["state"] == TECHNICALLY_INFEASIBLE
    assert reason in " ".join(item["technical_reasons"])


def test_persistent_vram_and_required_host_ram_capacity_are_enforced():
    vram = _run(_problem(two_shapes=False, required=1_001), _snapshot())
    assert vram["evaluations"][0]["state"] == TECHNICALLY_INFEASIBLE
    host = _run(_problem(two_shapes=False, transient=2_001, lifecycle="PERSISTENT_REQUIRED"), _snapshot())
    assert host["evaluations"][0]["state"] == TECHNICALLY_INFEASIBLE


def test_release_transient_counts_at_peak_but_not_steady():
    decision = _run(_problem(two_shapes=False, transient=1_901), _snapshot(host_capacity=2_000))
    accounting = decision["evaluations"][0]["memory_accounting"]["ram"]
    assert accounting["realization_peak_occupied_bytes"] == 1_901
    assert accounting["steady_occupied_bytes"] == 0
    assert accounting["released_after_lifecycle_boundary_bytes"] == 1_901


def test_retain_is_occupied_and_never_credited_as_evictable():
    problem = _problem(two_shapes=False, transient=2_001, lifecycle="PERSISTENT_OPTIONAL_RETAIN")
    item = _run(problem, _snapshot(host_capacity=2_000))["evaluations"][0]
    assert item["state"] == TECHNICALLY_INFEASIBLE
    assert item["memory_accounting"]["ram"]["persistent_optional_bytes"] == 2_001


def test_missing_link_is_technical_failure():
    snapshot = _snapshot()
    snapshot["links"] = []
    snapshot = freeze(snapshot)
    item = _evaluation(_run(snapshot=snapshot, catalog=_catalog(_problem(), snapshot, records=[])), "two[left=c0,right=c1]")
    assert item["state"] == TECHNICALLY_INFEASIBLE
    assert "Link/path" in " ".join(item["technical_reasons"])


def test_policy_exclusion_and_quarantine_are_not_technical_failure():
    excluded = _run(policy=_policy(excluded_compute_unit_ids=["c1"]))
    item = _evaluation(excluded, "two[left=c0,right=c1]")
    assert item["technically_feasible"] and item["state"] == POLICY_EXCLUDED
    snapshot = _snapshot()
    snapshot["nodes"][0]["compute_units"][1]["integrity_eligible"] = False
    snapshot = freeze(snapshot)
    quarantined = _run(snapshot=snapshot, catalog=_catalog(_problem(), snapshot))
    item = _evaluation(quarantined, "two[left=c0,right=c1]")
    assert item["technically_feasible"] and item["state"] == POLICY_EXCLUDED


def test_operator_reservation_is_policy_exclusion_not_technical_failure():
    decision = _run(
        problem=_problem(two_shapes=False, required=900),
        policy=_policy(reservations_bytes={"m0": 200, "m1": 200}),
    )
    item = _evaluation(decision, "one[alpha=c0]")
    assert item["technically_feasible"] and item["state"] == POLICY_EXCLUDED
    assert "operator reservation" in item["policy_reasons"][0]


def test_slow_plan_is_feasible_and_extra_resource_can_remain_unused():
    decision = _run()
    split = _evaluation(decision, "two[left=c0,right=c1]")
    assert split["technically_feasible"] and split["state"] == RANKED and split["rank"] == 2
    assert decision["selected_candidate_id"] == "one[alpha=c0]"
    assert decision["unused_resources"] == [{"compute_unit_id": "c1", "reason": "not needed by the highest-ranked candidate"}]


@pytest.mark.parametrize("mismatch", ["model_revision", "runtime"])
def test_context_mismatch_is_feasible_unranked(mismatch):
    problem, snapshot = _problem(two_shapes=False), _snapshot()
    catalog = _catalog(problem, snapshot)
    catalog["records"][0]["required_context"] = {
        **catalog["records"][0]["required_context"],
        mismatch: "wrong",
    }
    catalog = freeze(catalog)
    item = _run(problem, snapshot, catalog=catalog)["evaluations"][0]
    assert item["state"] == FEASIBLE_UNRANKED
    assert "context mismatch" in item["evidence"][0]["reasons"][0]


def test_resource_mapping_mismatch_and_stale_evidence_are_unranked():
    problem, snapshot = _problem(two_shapes=False), _snapshot()
    catalog = _catalog(problem, snapshot)
    catalog["records"][0]["mapping"] = {"alpha": "c1"}
    catalog["records"][0]["freshness"] = "STALE"
    catalog = freeze(catalog)
    item = _evaluation(_run(problem, snapshot, catalog=catalog), "one[alpha=c0]")
    assert item["state"] == FEASIBLE_UNRANKED
    assert {reason for reason in item["evidence"][0]["reasons"]} == {"resource mapping mismatch", "evidence is stale or lacks a compatibility statement"}


def test_no_evidence_returns_explicit_no_selection():
    problem, snapshot = _problem(), _snapshot()
    decision = _run(problem, snapshot, catalog=_catalog(problem, snapshot, records=[]))
    assert decision["selection_status"] == "NO_AUTOMATIC_SELECTION_INSUFFICIENT_EVIDENCE"
    assert decision["selected_candidate_id"] is None
    assert any(item["state"] == FEASIBLE_UNRANKED for item in decision["evaluations"])


def test_objective_changes_selection_without_changing_feasibility():
    speed = _run()
    latency = _run(objective=_objective("latency", "MINIMIZE"))
    assert speed["selected_candidate_id"] == "one[alpha=c0]"
    assert latency["selected_candidate_id"] == "two[left=c0,right=c1]"
    assert all(item["technically_feasible"] for item in (speed["evaluations"][0], latency["evaluations"][0]))


def test_tie_break_is_id_order_not_container_order():
    problem, snapshot = _problem(two_shapes=False), _snapshot()
    records = []
    for candidate in reversed(enumerate_candidates(problem, snapshot)):
        records.append({"id": candidate["id"], "shape_id": "one", "mapping": candidate["mapping"], "required_context": {}, "freshness": "CURRENT", "metric": {"name": "speed", "value": 1}})
    catalog = _catalog(problem, snapshot, records=records, context={})
    assert _run(problem, snapshot, catalog=catalog)["selected_candidate_id"] == "one[alpha=c0]"


def test_decision_is_frozen_and_environment_drift_fails_closed():
    snapshot = _snapshot()
    decision = _run(snapshot=snapshot)
    validate_decision_environment(decision, snapshot)
    changed = deepcopy(snapshot)
    changed["nodes"][0]["memory_resources"][0]["capacity_bytes"] -= 1
    with pytest.raises(EnvironmentDriftError, match="new planning decision"):
        validate_decision_environment(decision, freeze(changed))


def test_mixed_implementation_provenance_fails_closed():
    problem, snapshot = _problem(), _snapshot()
    problem["implementation_commit"] = "a" * 40
    snapshot["implementation_commit"] = "b" * 40
    with pytest.raises(ValueError, match="different implementation commits"):
        _run(freeze(problem), freeze(snapshot))


def test_generic_planner_never_inspects_strategy_payload_or_model_nouns():
    problem, snapshot = _problem(), _snapshot()
    first = _run(problem, snapshot)
    problem["shapes"][0]["strategy_payload"] = {"arbitrary": object().__class__.__name__}
    problem = freeze(problem)
    second = _run(problem, snapshot, catalog=_catalog(problem, snapshot))
    assert first["selected_candidate_id"] == second["selected_candidate_id"]
    source = (Path(__file__).parents[2] / "python/freetoken/research/r3_planner.py").read_text()
    forbidden = [
        "qwen",
        "moe",
        "expert",
        "router",
        "top-k",
        "top_k",
        "nvfp4",
        "triton",
        "cuda graph",
        "transformer layer",
        "[0,19)",
        "[19,40)",
    ]
    assert not any(noun in source.lower() for noun in forbidden)
