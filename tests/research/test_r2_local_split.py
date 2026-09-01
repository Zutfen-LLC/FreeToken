from __future__ import annotations

from copy import deepcopy

import pytest
from freetoken.research.r2_local_split import (
    LocalSplitPlanError,
    LocalSplitReconciliationError,
    classify_movement,
    freeze_plan,
    reconcile_observed,
    validate_boundary_payload,
    validate_frozen_plan,
    validate_participant,
)


def _unfrozen():
    states = [
        {"id": "a.fixed", "semantic_class": "immutable_source"},
        {"id": "a.mutable", "semantic_class": "mutable_authoritative"},
        {"id": "b.fixed", "semantic_class": "immutable_source"},
        {"id": "b.mutable", "semantic_class": "mutable_authoritative"},
        {
            "id": "shared",
            "semantic_class": "immutable_source",
            "sharing": "explicitly_duplicated_immutable",
        },
    ]
    materials = []
    for prefix, execution, memory in (
        ("a", "exec.a", "vram.a"),
        ("b", "exec.b", "vram.b"),
    ):
        materials.extend(
            [
                {
                    "id": f"mat.{prefix}.fixed",
                    "logical_state_id": f"{prefix}.fixed",
                    "execution_id": execution,
                    "memory_resource_id": memory,
                    "role": "required_residency",
                    "requirement": "required",
                    "persistence": "persistent",
                    "expected_bytes": 100,
                },
                {
                    "id": f"mat.{prefix}.mutable",
                    "logical_state_id": f"{prefix}.mutable",
                    "execution_id": execution,
                    "memory_resource_id": memory,
                    "role": "mutable_authority",
                    "requirement": "required",
                    "persistence": "persistent",
                    "expected_bytes": 10,
                },
                {
                    "id": f"mat.{prefix}.shared",
                    "logical_state_id": "shared",
                    "execution_id": execution,
                    "memory_resource_id": memory,
                    "role": "required_residency",
                    "requirement": "required",
                    "persistence": "persistent",
                    "expected_bytes": 5,
                },
            ]
        )
    return {
        "schema": "inferswarm.r2.local-split-plan/1",
        "model": {"repository": "repo/model", "revision": "rev"},
        "resources": {
            "nodes": [
                {
                    "id": "node",
                    "compute_units": [
                        {"id": "cu.a", "stable_device_id": "GPU-a"},
                        {"id": "cu.b", "stable_device_id": "GPU-b"},
                    ],
                    "memory_resources": [
                        {"id": "vram.a"},
                        {"id": "vram.b"},
                        {"id": "ram"},
                    ],
                }
            ],
            "links": [{"id": "link", "kind": "registered-pinned-host-staging"}],
        },
        "logical_state_units": states,
        "materializations": materials,
        "authorities": [
            {"logical_state_id": "a.mutable", "execution_id": "exec.a"},
            {"logical_state_id": "b.mutable", "execution_id": "exec.b"},
        ],
        "execution": [
            {"id": "exec.a", "compute_unit_id": "cu.a"},
            {"id": "exec.b", "compute_unit_id": "cu.b"},
        ],
        "strategy": {
            "number_of_layers": 4,
            "blocks": [
                {
                    "execution_id": "exec.a",
                    "ordinary_layer_ids": [0, 1],
                    "owned_state_ids": ["a.fixed", "shared"],
                    "mutable_state_ids": ["a.mutable"],
                },
                {
                    "execution_id": "exec.b",
                    "ordinary_layer_ids": [2, 3],
                    "owned_state_ids": ["b.fixed", "shared"],
                    "mutable_state_ids": ["b.mutable"],
                },
            ],
        },
        "boundary": {
            "producer_execution_id": "exec.a",
            "consumer_execution_id": "exec.b",
            "transport_path_id": "link",
            "contract": {
                "dtype": "bfloat16",
                "layout": "opaque-pair",
                "planes": 2,
                "row_width": 4,
                "element_bytes": 2,
                "decode_payload_bytes": 16,
            },
        },
        "provenance": {
            "baseline": {
                "model_repository": "repo/model",
                "model_revision": "rev",
                "workload_digest": "x",
            },
            "candidate": {
                "model_repository": "repo/model",
                "model_revision": "rev",
                "workload_digest": "x",
            },
        },
    }


def _plan():
    return freeze_plan(_unfrozen())


def _environment():
    return {
        "model": {"repository": "repo/model", "revision": "rev"},
        "resources": {
            "nodes": [
                {
                    "id": "node",
                    "compute_units": [
                        {"id": "cu.a", "stable_device_id": "GPU-a"},
                        {"id": "cu.b", "stable_device_id": "GPU-b"},
                    ],
                    "memory_resources": [
                        {"id": "vram.a", "capacity_bytes": 1000},
                        {"id": "vram.b", "capacity_bytes": 1000},
                        {"id": "ram", "capacity_bytes": 1000},
                    ],
                }
            ]
        },
    }


def _refreeze(plan):
    plan.pop("digest", None)
    return freeze_plan(plan)


def test_valid_two_resource_plan_and_declared_shared_state():
    result = validate_frozen_plan(_plan(), _environment())
    assert result["passed"] and result["ordinary_layer_intersection"] == []


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda p: p["execution"].pop(), "exactly two"),
        (
            lambda p: p["strategy"]["blocks"][1].update(ordinary_layer_ids=[1, 2, 3]),
            "overlap",
        ),
        (
            lambda p: p["strategy"]["blocks"][1].update(ordinary_layer_ids=[3]),
            "coverage is incomplete",
        ),
        (
            lambda p: p["logical_state_units"][-1].pop("sharing"),
            "illegal execution overlap",
        ),
        (
            lambda p: p["strategy"]["blocks"][1]["mutable_state_ids"].append(
                "a.mutable"
            ),
            "mutable authority overlap",
        ),
        (
            lambda p: p["boundary"].update(producer_execution_id="exec.b"),
            "wrong execution boundary producer",
        ),
        (
            lambda p: p["boundary"].update(consumer_execution_id="exec.a"),
            "wrong execution boundary consumer",
        ),
        (
            lambda p: p["resources"]["links"][0].update(kind="magic"),
            "unsupported transport",
        ),
        (
            lambda p: p["boundary"]["contract"].update(dtype="int8"),
            "unsupported boundary dtype",
        ),
        (
            lambda p: p["boundary"]["contract"].update(decode_payload_bytes=1),
            "byte-count mismatch",
        ),
        (
            lambda p: p["materializations"][0].update(logical_state_id="b.fixed"),
            "unplanned materialization",
        ),
        (
            lambda p: p["provenance"]["candidate"].update(model_revision="other"),
            "provenance mismatch",
        ),
    ],
)
def test_plan_invariants_fail_closed(mutate, message):
    plan = _plan()
    mutate(plan)
    plan = _refreeze(plan)
    with pytest.raises(LocalSplitPlanError, match=message):
        validate_frozen_plan(plan, _environment())


def test_wrong_gpu_assignment_and_participant_digest_rejected():
    plan = _plan()
    environment = _environment()
    environment["resources"]["nodes"][0]["compute_units"][0]["stable_device_id"] = (
        "GPU-wrong"
    )
    with pytest.raises(LocalSplitPlanError, match="stable device assignment"):
        validate_frozen_plan(plan, environment)
    with pytest.raises(LocalSplitPlanError, match="digest mismatch"):
        validate_participant(
            plan,
            execution_id="exec.a",
            plan_digest_value="wrong",
            stable_device_id="GPU-a",
            materialization_ids=[],
        )
    with pytest.raises(LocalSplitPlanError, match="unplanned materializations"):
        validate_participant(
            plan,
            execution_id="exec.a",
            plan_digest_value=plan["digest"],
            stable_device_id="GPU-a",
            materialization_ids=["mat.b.fixed"],
        )


def test_boundary_runtime_contract_rejects_dtype_layout_and_bytes():
    plan = _plan()
    assert (
        validate_boundary_payload(
            plan,
            producer_execution_id="exec.a",
            consumer_execution_id="exec.b",
            dtype="bfloat16",
            layout="opaque-pair",
            token_count=2,
            payload_bytes=32,
        )
        == 32
    )
    with pytest.raises(LocalSplitPlanError, match="dtype/layout"):
        validate_boundary_payload(
            plan,
            producer_execution_id="exec.a",
            consumer_execution_id="exec.b",
            dtype="float16",
            layout="opaque-pair",
            token_count=2,
            payload_bytes=32,
        )
    with pytest.raises(LocalSplitPlanError, match="byte-count"):
        validate_boundary_payload(
            plan,
            producer_execution_id="exec.a",
            consumer_execution_id="exec.b",
            dtype="bfloat16",
            layout="opaque-pair",
            token_count=2,
            payload_bytes=31,
        )


def _observed(plan):
    participants = []
    for execution_id in ("exec.a", "exec.b"):
        participants.append(
            {
                "execution_id": execution_id,
                "unexplained_persistent_host_mirror_bytes": 0,
                "unexpected_checkpoint_keys": [],
                "materializations": [
                    {
                        "materialization_id": item["id"],
                        "observed_bytes": item["expected_bytes"],
                        "memory_resource_id": item["memory_resource_id"],
                        "persistence": item["persistence"],
                    }
                    for item in plan["materializations"]
                    if item["execution_id"] == execution_id
                ],
            }
        )
    return {
        "participants": participants,
        "movement": [
            {"classification": "model_state", "bytes": 0},
            {"classification": "activation", "bytes": 32},
            {"classification": "control", "bytes": 4},
        ],
    }


def test_hidden_persistent_state_and_host_mirror_rejected():
    plan = _plan()
    observed = _observed(plan)
    assert reconcile_observed(plan, observed)["passed"]
    hidden = deepcopy(observed)
    hidden["participants"][0]["materializations"].append(
        {"materialization_id": "hidden", "persistence": "persistent"}
    )
    with pytest.raises(LocalSplitReconciliationError, match="hidden persistent"):
        reconcile_observed(plan, hidden)
    mirror = deepcopy(observed)
    mirror["participants"][1]["unexplained_persistent_host_mirror_bytes"] = 1
    with pytest.raises(LocalSplitReconciliationError, match="host mirror"):
        reconcile_observed(plan, mirror)


def test_transport_accounting_classification_is_exact():
    assert classify_movement(
        [
            {"classification": "activation", "bytes": 10},
            {"classification": "control", "bytes": 2},
        ]
    ) == {"activation": 10, "control": 2, "model_state": 0}
    with pytest.raises(LocalSplitReconciliationError, match="invalid transport"):
        classify_movement([{"classification": "weights", "bytes": 10}])
