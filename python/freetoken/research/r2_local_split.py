"""Internal R2 local split-plan validation and reconciliation.

The structures in this module are research mappings, not a public planner,
strategy, worker, or wire API.  Strategy-specific payload meaning remains in
the experiment adapter.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

PLAN_SCHEMA = "inferswarm.r2.local-split-plan/1"
RESULT_SCHEMA = "inferswarm.r2.local-split-execution/1"
SUPPORTED_TRANSPORT_KINDS = {
    "accelerator-peer",
    "accelerator-ipc",
    "registered-pinned-host-staging",
    "serialized-local-byte-stream",
}
MOVEMENT_CLASSES = {"model_state", "activation", "control"}


class LocalSplitPlanError(ValueError):
    """The frozen local split plan is invalid and must not execute."""


class LocalSplitReconciliationError(RuntimeError):
    """Observed execution did not faithfully realize the plan."""


def canonical_plan_bytes(plan: Mapping[str, Any]) -> bytes:
    payload = dict(plan)
    payload.pop("digest", None)
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def plan_digest(plan: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_plan_bytes(plan)).hexdigest()


def freeze_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    frozen = json.loads(json.dumps(plan))
    frozen["digest"] = f"sha256:{plan_digest(frozen)}"
    return frozen


def _items_by_id(
    items: Sequence[Mapping[str, Any]], kind: str
) -> dict[str, Mapping[str, Any]]:
    ids = [item.get("id") for item in items]
    bad = [
        index
        for index, value in enumerate(ids)
        if not isinstance(value, str) or not value
    ]
    if bad:
        raise LocalSplitPlanError(f"{kind} entries missing stable id at indexes {bad}")
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise LocalSplitPlanError(f"duplicate {kind} id(s): {duplicates}")
    return {str(item["id"]): item for item in items}


def _error(errors: list[str], message: str) -> None:
    errors.append(message)


def validate_frozen_plan(
    plan: Mapping[str, Any], environment: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail-closed validation for one preselected two-block local plan."""
    errors: list[str] = []
    if plan.get("schema") != PLAN_SCHEMA:
        _error(errors, f"schema must be {PLAN_SCHEMA!r}")
    calculated = f"sha256:{plan_digest(plan)}"
    if plan.get("digest") != calculated:
        _error(errors, "frozen plan digest mismatch")
    for key in ("repository", "revision"):
        if plan.get("model", {}).get(key) != environment.get("model", {}).get(key):
            _error(errors, f"plan model {key} does not match environment")

    try:
        nodes = _items_by_id(plan.get("resources", {}).get("nodes", []), "Node")
        links = _items_by_id(plan.get("resources", {}).get("links", []), "Link")
        states = _items_by_id(plan.get("logical_state_units", []), "Logical State Unit")
        materials = _items_by_id(plan.get("materializations", []), "materialization")
        executions = _items_by_id(plan.get("execution", []), "execution unit")
    except LocalSplitPlanError as exc:
        _error(errors, str(exc))
        nodes = links = states = materials = executions = {}

    compute: dict[str, Mapping[str, Any]] = {}
    memory: dict[str, Mapping[str, Any]] = {}
    for node in nodes.values():
        for item in node.get("compute_units", []):
            if item.get("id") in compute:
                _error(errors, f"duplicate Compute Unit id {item.get('id')!r}")
            compute[str(item.get("id"))] = item
        for item in node.get("memory_resources", []):
            if item.get("id") in memory:
                _error(errors, f"duplicate Memory Resource id {item.get('id')!r}")
            memory[str(item.get("id"))] = item

    env_units = {
        item.get("id"): item
        for node in environment.get("resources", {}).get("nodes", [])
        for item in node.get("compute_units", [])
    }
    env_memory = {
        item.get("id"): item
        for node in environment.get("resources", {}).get("nodes", [])
        for item in node.get("memory_resources", [])
    }
    participant_units = [item.get("compute_unit_id") for item in executions.values()]
    if len(executions) != 2 or len(set(participant_units)) != 2:
        _error(errors, "R2 requires exactly two execution units on two Compute Units")
    for unit_id, intended in compute.items():
        actual = env_units.get(unit_id)
        if actual is None:
            _error(errors, f"Compute Unit {unit_id!r} is unavailable")
        elif intended.get("stable_device_id") != actual.get("stable_device_id"):
            _error(
                errors, f"Compute Unit {unit_id!r} stable device assignment mismatch"
            )
    for resource_id in memory:
        if resource_id not in env_memory:
            _error(errors, f"Memory Resource {resource_id!r} is unavailable")

    strategy = plan.get("strategy", {})
    blocks = strategy.get("blocks", [])
    if len(blocks) != 2:
        _error(errors, "strategy must contain exactly two blocks")
    layer_sets: list[set[int]] = []
    all_layers = set(range(int(strategy.get("number_of_layers", 0))))
    owned_state: dict[str, set[str]] = {}
    mutable_state: dict[str, set[str]] = {}
    for block in blocks:
        execution_id = block.get("execution_id")
        if execution_id not in executions:
            _error(errors, f"block references missing execution {execution_id!r}")
        layers = set(block.get("ordinary_layer_ids", []))
        layer_sets.append(layers)
        owned_state[str(execution_id)] = set(block.get("owned_state_ids", []))
        mutable_state[str(execution_id)] = set(block.get("mutable_state_ids", []))
        for state_id in (
            owned_state[str(execution_id)] | mutable_state[str(execution_id)]
        ):
            if state_id not in states:
                _error(
                    errors, f"block references missing Logical State Unit {state_id!r}"
                )
    if len(layer_sets) == 2:
        if layer_sets[0] & layer_sets[1]:
            _error(errors, "ordinary layer ownership overlap is illegal")
        if layer_sets[0] | layer_sets[1] != all_layers:
            _error(errors, "Block A/B ordinary layer coverage is incomplete")
    mutable_intersection = (
        mutable_state.get(str(blocks[0].get("execution_id")), set())
        & mutable_state.get(str(blocks[1].get("execution_id")), set())
        if len(blocks) == 2
        else set()
    )
    if mutable_intersection:
        _error(
            errors,
            f"mutable authority overlap is illegal: {sorted(mutable_intersection)}",
        )

    ordinary_owners: dict[str, list[str]] = {}
    for execution_id, state_ids in owned_state.items():
        for state_id in state_ids:
            ordinary_owners.setdefault(state_id, []).append(execution_id)
    for state_id, owners in ordinary_owners.items():
        if len(owners) <= 1:
            continue
        state = states.get(state_id, {})
        if state.get("sharing") != "explicitly_duplicated_immutable":
            _error(errors, f"ordinary state {state_id!r} has illegal execution overlap")

    authorities: dict[str, list[Mapping[str, Any]]] = {}
    for authority in plan.get("authorities", []):
        authorities.setdefault(str(authority.get("logical_state_id")), []).append(
            authority
        )
    for state_id, state in states.items():
        if state.get("semantic_class") == "mutable_authoritative":
            records = authorities.get(state_id, [])
            if len(records) != 1:
                _error(
                    errors, f"mutable state {state_id!r} requires exactly one authority"
                )
            elif records[0].get("execution_id") not in executions:
                _error(
                    errors,
                    f"mutable state {state_id!r} authority has missing execution",
                )

    required_by_memory: dict[str, int] = {}
    for mid, item in materials.items():
        state_id = item.get("logical_state_id")
        execution_id = item.get("execution_id")
        resource_id = item.get("memory_resource_id")
        if state_id not in states:
            _error(errors, f"materialization {mid!r} references missing state")
        if execution_id not in executions:
            _error(errors, f"materialization {mid!r} references missing execution")
        if resource_id not in memory:
            _error(
                errors, f"materialization {mid!r} references missing Memory Resource"
            )
        if item.get("role") == "staging" and item.get("persistence") == "persistent":
            _error(errors, f"materialization {mid!r} is persistent staging")
        if (
            item.get("requirement") == "required"
            and item.get("persistence") == "persistent"
        ):
            byte_count = item.get("expected_bytes")
            if not isinstance(byte_count, int) or byte_count < 0:
                _error(errors, f"materialization {mid!r} has invalid expected bytes")
            else:
                required_by_memory[str(resource_id)] = (
                    required_by_memory.get(str(resource_id), 0) + byte_count
                )
        if execution_id in owned_state and state_id not in (
            owned_state[execution_id] | mutable_state[execution_id]
        ):
            _error(errors, f"worker assigned unplanned materialization {mid!r}")
    for resource_id, required in required_by_memory.items():
        capacity = env_memory.get(resource_id, {}).get("capacity_bytes")
        if isinstance(capacity, int) and required > capacity:
            _error(errors, f"Memory Resource {resource_id!r} capacity is insufficient")

    boundary = plan.get("boundary", {})
    producer = boundary.get("producer_execution_id")
    consumer = boundary.get("consumer_execution_id")
    expected_order = [block.get("execution_id") for block in blocks]
    if expected_order and producer != expected_order[0]:
        _error(errors, "wrong execution boundary producer")
    if len(expected_order) == 2 and consumer != expected_order[1]:
        _error(errors, "wrong execution boundary consumer")
    transport = links.get(boundary.get("transport_path_id"), {})
    if transport.get("kind") not in SUPPORTED_TRANSPORT_KINDS:
        _error(errors, "unsupported transport path")
    contract = boundary.get("contract", {})
    if contract.get("dtype") not in {"bfloat16", "float16", "float32"}:
        _error(errors, "unsupported boundary dtype")
    if not isinstance(contract.get("layout"), str) or not contract.get("layout"):
        _error(errors, "boundary layout is required")
    planes = contract.get("planes")
    width = contract.get("row_width")
    element_bytes = contract.get("element_bytes")
    decode_bytes = contract.get("decode_payload_bytes")
    if not all(
        isinstance(value, int) and value > 0 for value in (planes, width, element_bytes)
    ):
        _error(errors, "invalid boundary geometry")
    elif decode_bytes != planes * width * element_bytes:
        _error(errors, "boundary decode byte-count mismatch")

    provenance = plan.get("provenance", {})
    baseline = provenance.get("baseline", {})
    candidate = provenance.get("candidate", {})
    for field in ("model_repository", "model_revision", "workload_digest"):
        if baseline.get(field) != candidate.get(field):
            _error(errors, f"baseline/candidate provenance mismatch for {field}")

    if errors:
        raise LocalSplitPlanError("invalid frozen R2 plan:\n- " + "\n- ".join(errors))
    return {
        "passed": True,
        "digest": calculated,
        "required_bytes_by_memory": required_by_memory,
        "ordinary_layer_union": sorted(set().union(*layer_sets)),
        "ordinary_layer_intersection": sorted(layer_sets[0] & layer_sets[1]),
    }


def validate_participant(
    plan: Mapping[str, Any],
    *,
    execution_id: str,
    plan_digest_value: str,
    stable_device_id: str,
    materialization_ids: Sequence[str],
) -> dict[str, Any]:
    if plan_digest_value != plan.get("digest"):
        raise LocalSplitPlanError("participant plan digest mismatch")
    executions = {item["id"]: item for item in plan.get("execution", [])}
    execution = executions.get(execution_id)
    if execution is None:
        raise LocalSplitPlanError(f"unknown participant execution {execution_id!r}")
    units = {
        unit["id"]: unit
        for node in plan.get("resources", {}).get("nodes", [])
        for unit in node.get("compute_units", [])
    }
    expected_device = units[execution["compute_unit_id"]].get("stable_device_id")
    if stable_device_id != expected_device:
        raise LocalSplitPlanError("participant stable GPU/resource assignment mismatch")
    planned = {
        item["id"]
        for item in plan.get("materializations", [])
        if item.get("execution_id") == execution_id
    }
    unplanned = sorted(set(materialization_ids) - planned)
    if unplanned:
        raise LocalSplitPlanError(
            f"participant requested unplanned materializations: {unplanned}"
        )
    return {
        "passed": True,
        "execution_id": execution_id,
        "stable_device_id": stable_device_id,
    }


def validate_boundary_payload(
    plan: Mapping[str, Any],
    *,
    producer_execution_id: str,
    consumer_execution_id: str,
    dtype: str,
    layout: str,
    token_count: int,
    payload_bytes: int,
) -> int:
    boundary = plan["boundary"]
    contract = boundary["contract"]
    if producer_execution_id != boundary["producer_execution_id"]:
        raise LocalSplitPlanError("wrong execution boundary producer")
    if consumer_execution_id != boundary["consumer_execution_id"]:
        raise LocalSplitPlanError("wrong execution boundary consumer")
    if dtype != contract["dtype"] or layout != contract["layout"]:
        raise LocalSplitPlanError("boundary dtype/layout mismatch")
    expected = (
        token_count
        * contract["planes"]
        * contract["row_width"]
        * contract["element_bytes"]
    )
    if payload_bytes != expected:
        raise LocalSplitPlanError(
            f"boundary byte-count mismatch: {payload_bytes} != {expected}"
        )
    return expected


def classify_movement(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals = {kind: 0 for kind in sorted(MOVEMENT_CLASSES)}
    for record in records:
        kind = record.get("classification")
        count = record.get("bytes")
        if kind not in MOVEMENT_CLASSES or not isinstance(count, int) or count < 0:
            raise LocalSplitReconciliationError("invalid transport accounting record")
        totals[str(kind)] += count
    return totals


def reconcile_observed(
    plan: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, Any]:
    """Audit physical ownership, persistence, authority, and movement evidence."""
    errors: list[str] = []
    planned_materials = {item["id"]: item for item in plan.get("materializations", [])}
    seen: set[str] = set()
    unexplained_host = 0
    for execution in observed.get("participants", []):
        execution_id = execution.get("execution_id")
        for material in execution.get("materializations", []):
            mid = material.get("materialization_id")
            intended = planned_materials.get(mid)
            if intended is None:
                if material.get("persistence") == "persistent":
                    errors.append(f"hidden persistent state {mid!r}")
                continue
            seen.add(str(mid))
            if intended.get("execution_id") != execution_id:
                errors.append(f"materialization {mid!r} observed on wrong execution")
            if material.get("observed_bytes") != intended.get("expected_bytes"):
                errors.append(f"materialization {mid!r} byte mismatch")
            if material.get("memory_resource_id") != intended.get("memory_resource_id"):
                errors.append(f"materialization {mid!r} resource mismatch")
        unexplained_host += int(
            execution.get("unexplained_persistent_host_mirror_bytes", 0)
        )
        if execution.get("unexpected_checkpoint_keys"):
            errors.append(
                f"execution {execution_id!r} fetched unplanned checkpoint keys"
            )
    required = {
        mid
        for mid, item in planned_materials.items()
        if item.get("requirement") == "required"
    }
    missing = sorted(required - seen)
    if missing:
        errors.append(f"required materializations not observed: {missing}")
    if unexplained_host:
        errors.append(f"unexplained persistent host mirror bytes: {unexplained_host}")
    movement = classify_movement(observed.get("movement", []))
    if movement["model_state"]:
        errors.append("steady-state model-state movement is nonzero")
    if errors:
        raise LocalSplitReconciliationError(
            "R2 reconciliation failed:\n- " + "\n- ".join(errors)
        )
    return {
        "passed": True,
        "missing_required_materializations": [],
        "unplanned_persistent_materializations": [],
        "unexplained_persistent_host_mirror_bytes": 0,
        "movement_bytes": movement,
    }


__all__ = [
    "MOVEMENT_CLASSES",
    "PLAN_SCHEMA",
    "RESULT_SCHEMA",
    "SUPPORTED_TRANSPORT_KINDS",
    "LocalSplitPlanError",
    "LocalSplitReconciliationError",
    "canonical_plan_bytes",
    "classify_movement",
    "freeze_plan",
    "plan_digest",
    "reconcile_observed",
    "validate_boundary_payload",
    "validate_frozen_plan",
    "validate_participant",
]
