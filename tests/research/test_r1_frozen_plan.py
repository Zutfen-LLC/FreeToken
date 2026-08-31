from __future__ import annotations

from copy import deepcopy
from typing import ClassVar

import pytest
from freetoken.research.r1_frozen_plan import (
    FrozenPlanError,
    ReconciliationError,
    freeze_plan,
    realize_frozen_plan,
    reconcile_realization,
    validate_frozen_plan,
)


class FakeAdapter:
    representations: ClassVar[dict[str, set[str]]] = {
        "state.fixed": {"native-device", "native-host"},
        "state.mutable": {"runtime-device"},
    }

    def __init__(self):
        self.released = []

    def supports_representation(self, state_id, representation):
        return representation in self.representations.get(state_id, set())

    def supports_execution(self, execution):
        return execution.get("strategy_unit") == "fixture.unit"

    def begin(self, plan, environment):
        self.plan = plan

    def realize_materialization(self, item):
        return {
            "actual_representation": item["representation"],
            "actual_memory_resource_id": item["memory_resource_id"],
            "observed_bytes": item["expected_bytes"],
            "lifecycle_state": "live",
            "status": "PLANNED_AND_REALIZED",
        }

    def release_materialization(self, item):
        self.released.append(item["id"])
        return {"observed_bytes": 0}

    def activate_execution(self, item):
        return {
            "execution_id": item["id"],
            "compute_unit_id": item["compute_unit_id"],
            "status": "ACTIVE",
        }

    def observe_authorities(self):
        return deepcopy(self.plan["authorities"])


def _unfrozen_plan():
    return {
        "schema": "inferswarm.r1.frozen-plan/1",
        "model": {"repository": "repo/model", "revision": "rev"},
        "resources": {
            "swarm_id": "swarm",
            "nodes": [
                {
                    "id": "node",
                    "compute_units": [{"id": "cu"}],
                    "memory_resources": [
                        {"id": "ram", "capacity_bytes": 1000},
                        {"id": "vram", "capacity_bytes": 1000},
                    ],
                }
            ],
        },
        "logical_state_units": [
            {"id": "state.fixed", "semantic_class": "immutable_source"},
            {"id": "state.mutable", "semantic_class": "mutable_authoritative"},
        ],
        "materializations": [
            {
                "id": "stage",
                "logical_state_id": "state.fixed",
                "representation": "native-host",
                "memory_resource_id": "ram",
                "role": "staging",
                "requirement": "required",
                "persistence": "transient",
                "expected_bytes": 100,
            },
            {
                "id": "resident",
                "logical_state_id": "state.fixed",
                "representation": "native-device",
                "memory_resource_id": "vram",
                "role": "required_residency",
                "requirement": "required",
                "persistence": "persistent",
                "expected_bytes": 100,
            },
            {
                "id": "mutable",
                "logical_state_id": "state.mutable",
                "representation": "runtime-device",
                "memory_resource_id": "vram",
                "role": "mutable_authority",
                "requirement": "required",
                "persistence": "persistent",
                "expected_bytes": 10,
            },
            {
                "id": "optional",
                "logical_state_id": "state.fixed",
                "representation": "native-host",
                "memory_resource_id": "ram",
                "role": "optional_cache",
                "requirement": "optional",
                "persistence": "persistent",
                "expected_bytes": 100,
            },
        ],
        "authorities": [
            {
                "logical_state_id": "state.mutable",
                "materialization_id": "mutable",
                "lineage": "epoch-1",
            }
        ],
        "execution": [
            {
                "id": "exec",
                "strategy_unit": "fixture.unit",
                "compute_unit_id": "cu",
                "required_state": [
                    {
                        "logical_state_id": "state.fixed",
                        "representations": ["native-device"],
                        "memory_resources": ["vram"],
                    },
                    {
                        "logical_state_id": "state.mutable",
                        "representations": ["runtime-device"],
                        "memory_resources": ["vram"],
                    },
                ],
            }
        ],
        "forbidden_persistent_materializations": [
            {"logical_state_id": "state.fixed", "memory_resource_id": "ram"}
        ],
    }


def _plan():
    plan = _unfrozen_plan()
    # The optional host cache is used only by the omission test, not the valid base.
    plan["forbidden_persistent_materializations"] = []
    return freeze_plan(plan)


def _environment():
    return {
        "model_repository": "repo/model",
        "model_revision": "rev",
        "resources": deepcopy(_unfrozen_plan()["resources"]),
    }


def _refreeze(plan):
    plan.pop("digest", None)
    return freeze_plan(plan)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p, e: e["resources"].update(nodes=[]), "Node 'node'"),
        (
            lambda p, e: e["resources"]["nodes"][0].update(compute_units=[]),
            "Compute Unit 'cu'",
        ),
        (
            lambda p, e: e["resources"]["nodes"][0].update(memory_resources=[]),
            "Memory Resource 'ram'",
        ),
        (
            lambda p, e: p["logical_state_units"].pop(0),
            "missing Logical State Unit 'state.fixed'",
        ),
        (
            lambda p, e: p["logical_state_units"].append(
                deepcopy(p["logical_state_units"][0])
            ),
            "duplicate Logical State Unit",
        ),
        (
            lambda p, e: p["materializations"][0].update(memory_resource_id="missing"),
            "missing Memory Resource 'missing'",
        ),
        (
            lambda p, e: p["materializations"][0].update(representation="unsupported"),
            "unsupported representation",
        ),
        (
            lambda p, e: e["resources"]["nodes"][0]["memory_resources"][1].update(
                capacity_bytes=1
            ),
            "capacity 1 is insufficient",
        ),
        (
            lambda p, e: p["authorities"].append(deepcopy(p["authorities"][0])),
            "exactly one authoritative lineage; found 2",
        ),
        (
            lambda p, e: p["execution"][0]["required_state"][0].update(
                representations=["native-host"]
            ),
            "no planned legal materialization/path",
        ),
    ],
)
def test_validation_fails_closed_for_invalid_references_and_constraints(
    mutate, message
):
    plan, environment = _plan(), _environment()
    mutate(plan, environment)
    plan = _refreeze(plan)
    with pytest.raises(FrozenPlanError, match=message):
        validate_frozen_plan(plan, environment, FakeAdapter())


def test_forbidden_persistent_host_mirror_is_rejected():
    plan = _plan()
    plan["forbidden_persistent_materializations"] = [
        {"logical_state_id": "state.fixed", "memory_resource_id": "ram"}
    ]
    plan = _refreeze(plan)
    with pytest.raises(
        FrozenPlanError,
        match="forbidden persistent materialization requested: 'optional'",
    ):
        validate_frozen_plan(plan, _environment(), FakeAdapter())


def test_digest_and_model_revision_mismatches_are_rejected():
    plan = _plan()
    plan["materializations"][0]["expected_bytes"] += 1
    with pytest.raises(FrozenPlanError, match="digest mismatch"):
        validate_frozen_plan(plan, _environment(), FakeAdapter())
    plan = _plan()
    environment = _environment()
    environment["model_revision"] = "wrong"
    with pytest.raises(FrozenPlanError, match="revision does not match"):
        validate_frozen_plan(plan, environment, FakeAdapter())


def test_optional_omission_and_staging_release_are_successful():
    adapter = FakeAdapter()
    realized = realize_frozen_plan(_plan(), _environment(), adapter)
    records = {
        item["materialization_id"]: item for item in realized.observed_materializations
    }
    assert records["optional"]["status"] == "OPTIONAL_NOT_REALIZED"
    assert records["stage"]["status"] == "TRANSIENT_RELEASED_AS_PLANNED"
    assert adapter.released == ["stage"]
    assert realized.reconciliation["passed"]


def _observed(plan):
    return [
        {
            "materialization_id": item["id"],
            "logical_state_id": item["logical_state_id"],
            "actual_representation": item["representation"],
            "actual_memory_resource_id": item["memory_resource_id"],
            "observed_bytes": item["expected_bytes"],
            "status": "PLANNED_AND_REALIZED",
            "lifecycle_state": "live",
            "persistence": item["persistence"],
        }
        for item in plan["materializations"]
        if item["requirement"] == "required"
    ]


@pytest.mark.parametrize(
    "mutate, field",
    [
        (lambda o: o.pop(1), "planned_not_realized"),
        (
            lambda o: o.append(
                {
                    "materialization_id": "hidden",
                    "persistence": "persistent",
                    "lifecycle_state": "live",
                }
            ),
            "unplanned_persistent",
        ),
        (lambda o: o[1].update(actual_memory_resource_id="ram"), "mismatches"),
    ],
)
def test_reconciliation_detects_missing_unplanned_and_wrong_resource(mutate, field):
    plan = _plan()
    observed = _observed(plan)
    mutate(observed)
    result = reconcile_realization(
        plan,
        observed,
        [{"execution_id": "exec", "compute_unit_id": "cu"}],
        plan["authorities"],
    )
    assert result[field]
    assert not result["passed"]


def test_reconciliation_rejects_wrong_execution_and_authority():
    plan = _plan()
    result = reconcile_realization(
        plan,
        _observed(plan),
        [{"execution_id": "exec", "compute_unit_id": "other"}],
        [
            {
                "logical_state_id": "state.mutable",
                "materialization_id": "mutable",
                "lineage": "wrong",
            }
        ],
    )
    assert any("ran on" in item for item in result["mismatches"])
    assert any("authority mismatch" in item for item in result["mismatches"])


def test_realizer_raises_on_observed_mismatch():
    class BadAdapter(FakeAdapter):
        def realize_materialization(self, item):
            record = dict(super().realize_materialization(item))
            if item["id"] == "resident":
                record["actual_memory_resource_id"] = "ram"
            return record

    with pytest.raises(ReconciliationError, match="actual_memory_resource_id"):
        realize_frozen_plan(_plan(), _environment(), BadAdapter())
