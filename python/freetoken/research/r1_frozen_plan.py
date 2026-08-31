"""Internal R1 frozen-plan realization substrate.

This module deliberately consumes plain mappings.  It proves doctrine semantics
without publishing an ExecutionPlan or model-strategy API.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

PLAN_SCHEMA = "inferswarm.r1.frozen-plan/1"
RESULT_SCHEMA = "inferswarm.r1.frozen-plan-realization/1"


class FrozenPlanError(ValueError):
    """A frozen research plan is invalid and must not be realized."""


class ReconciliationError(RuntimeError):
    """Observed state does not faithfully realize the frozen plan."""


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


def _ids(items: list[dict[str, Any]], kind: str) -> set[str]:
    values = [item.get("id") for item in items]
    missing = [
        i for i, value in enumerate(values) if not isinstance(value, str) or not value
    ]
    if missing:
        raise FrozenPlanError(f"{kind} entries missing stable id at indexes {missing}")
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise FrozenPlanError(f"duplicate {kind} id(s): {duplicates}")
    return set(values)


def validate_frozen_plan(
    plan: Mapping[str, Any],
    environment: Mapping[str, Any],
    adapter: ResearchPlanAdapter,
) -> dict[str, Any]:
    errors: list[str] = []
    if plan.get("schema") != PLAN_SCHEMA:
        errors.append(f"schema must be {PLAN_SCHEMA!r}, got {plan.get('schema')!r}")
    expected_digest = plan.get("digest")
    actual_digest = f"sha256:{plan_digest(plan)}"
    if expected_digest != actual_digest:
        errors.append(
            f"frozen plan digest mismatch: declared {expected_digest!r}, calculated {actual_digest!r}"
        )
    if plan.get("model", {}).get("repository") != environment.get("model_repository"):
        errors.append("plan model repository does not match realization environment")
    if plan.get("model", {}).get("revision") != environment.get("model_revision"):
        errors.append("plan model revision does not match realization environment")

    resources = plan.get("resources", {})
    env_resources = environment.get("resources", {})
    if resources.get("swarm_id") != env_resources.get("swarm_id"):
        errors.append(
            f"Swarm {resources.get('swarm_id')!r} is absent from the environment"
        )
    try:
        node_ids = _ids(resources.get("nodes", []), "Node")
        env_node_ids = _ids(env_resources.get("nodes", []), "environment Node")
        state_ids = _ids(plan.get("logical_state_units", []), "Logical State Unit")
        materialization_ids = _ids(plan.get("materializations", []), "materialization")
    except FrozenPlanError as exc:
        errors.append(str(exc))
        node_ids, env_node_ids, state_ids, materialization_ids = (
            set(),
            set(),
            set(),
            set(),
        )

    compute: dict[str, dict[str, Any]] = {}
    memory: dict[str, dict[str, Any]] = {}
    env_compute: dict[str, dict[str, Any]] = {}
    env_memory: dict[str, dict[str, Any]] = {}
    for node in resources.get("nodes", []):
        if node.get("id") not in node_ids:
            continue
        for unit in node.get("compute_units", []):
            if unit.get("id") in compute:
                errors.append(f"duplicate Compute Unit id {unit.get('id')!r}")
            compute[unit.get("id")] = unit
        for resource in node.get("memory_resources", []):
            if resource.get("id") in memory:
                errors.append(f"duplicate Memory Resource id {resource.get('id')!r}")
            memory[resource.get("id")] = resource
    for node in env_resources.get("nodes", []):
        for unit in node.get("compute_units", []):
            env_compute[unit.get("id")] = unit
        for resource in node.get("memory_resources", []):
            env_memory[resource.get("id")] = resource
    backing_ids = {item.get("id") for item in resources.get("backing_sources", [])}
    link_ids = {item.get("id") for item in resources.get("links", [])}
    for node_id in sorted(node_ids - env_node_ids):
        errors.append(f"referenced Node {node_id!r} does not exist in the environment")
    for unit_id in sorted(set(compute) - set(env_compute)):
        errors.append(
            f"referenced Compute Unit {unit_id!r} does not exist in the environment"
        )
    for resource_id in sorted(set(memory) - set(env_memory)):
        errors.append(
            f"referenced Memory Resource {resource_id!r} does not exist in the environment"
        )

    states = {item.get("id"): item for item in plan.get("logical_state_units", [])}
    for state_id, state in states.items():
        backing_id = state.get("backing_source_id")
        if backing_id is not None and backing_id not in backing_ids:
            errors.append(
                f"Logical State Unit {state_id!r} references missing backing/source {backing_id!r}"
            )
    persistent_by_state: dict[str, list[dict[str, Any]]] = {}
    required_bytes_by_memory: dict[str, int] = {}
    for materialization in plan.get("materializations", []):
        mid = materialization.get("id", "<missing>")
        state_id = materialization.get("logical_state_id")
        resource_id = materialization.get("memory_resource_id")
        if state_id not in state_ids:
            errors.append(
                f"materialization {mid!r} references missing Logical State Unit {state_id!r}"
            )
        if resource_id not in memory:
            errors.append(
                f"materialization {mid!r} references missing Memory Resource {resource_id!r}"
            )
        elif resource_id not in env_memory:
            errors.append(
                f"materialization {mid!r} targets unavailable Memory Resource {resource_id!r}"
            )
        path_id = materialization.get("path_id")
        if path_id is not None and path_id not in link_ids:
            errors.append(
                f"materialization {mid!r} references missing Link/path {path_id!r}"
            )
        representation = materialization.get("representation")
        if not adapter.supports_representation(state_id, representation):
            errors.append(
                f"materialization {mid!r} requests unsupported representation {representation!r} for {state_id!r}"
            )
        role = materialization.get("role")
        persistence = materialization.get("persistence")
        requirement = materialization.get("requirement")
        if role == "staging" and persistence == "persistent":
            errors.append(
                f"materialization {mid!r} has contradictory staging and persistent roles"
            )
        if (
            role == "mutable_authority"
            and states.get(state_id, {}).get("semantic_class")
            != "mutable_authoritative"
        ):
            errors.append(
                f"materialization {mid!r} assigns mutable authority to non-mutable state {state_id!r}"
            )
        if persistence == "persistent":
            persistent_by_state.setdefault(state_id, []).append(materialization)
        expected_bytes = materialization.get("expected_bytes")
        if (
            requirement == "required"
            and persistence == "persistent"
            and isinstance(expected_bytes, int)
        ):
            required_bytes_by_memory[resource_id] = (
                required_bytes_by_memory.get(resource_id, 0) + expected_bytes
            )

    for selector in plan.get("forbidden_persistent_materializations", []):
        for materialization in persistent_by_state.get(
            selector.get("logical_state_id"), []
        ):
            if materialization.get("memory_resource_id") == selector.get(
                "memory_resource_id"
            ):
                errors.append(
                    f"forbidden persistent materialization requested: {materialization.get('id')!r}"
                )

    for resource_id, required_bytes in required_bytes_by_memory.items():
        capacity = env_memory.get(resource_id, {}).get("capacity_bytes")
        if isinstance(capacity, int) and required_bytes > capacity:
            errors.append(
                f"Memory Resource {resource_id!r} capacity {capacity} is insufficient for {required_bytes} required persistent bytes"
            )

    authorities: dict[str, list[dict[str, Any]]] = {}
    for authority in plan.get("authorities", []):
        state_id = authority.get("logical_state_id")
        if state_id not in state_ids:
            errors.append(
                f"authority references missing Logical State Unit {state_id!r}"
            )
        if authority.get("materialization_id") not in materialization_ids:
            errors.append(
                f"authority for {state_id!r} references missing materialization {authority.get('materialization_id')!r}"
            )
        authorities.setdefault(state_id, []).append(authority)
    for state_id, state in states.items():
        if (
            state.get("semantic_class") == "mutable_authoritative"
            and len(authorities.get(state_id, [])) != 1
        ):
            errors.append(
                f"mutable state {state_id!r} requires exactly one authoritative lineage; found {len(authorities.get(state_id, []))}"
            )

    executions = plan.get("execution", [])
    try:
        _ids(executions, "execution unit")
    except FrozenPlanError as exc:
        errors.append(str(exc))
    for execution in executions:
        eid = execution.get("id", "<missing>")
        unit_id = execution.get("compute_unit_id")
        if unit_id not in compute or unit_id not in env_compute:
            errors.append(
                f"execution unit {eid!r} references missing Compute Unit {unit_id!r}"
            )
        for requirement in execution.get("required_state", []):
            state_id = requirement.get("logical_state_id")
            if state_id not in state_ids:
                errors.append(
                    f"execution unit {eid!r} references missing Logical State Unit {state_id!r}"
                )
                continue
            legal = [
                item
                for item in plan.get("materializations", [])
                if item.get("logical_state_id") == state_id
                and item.get("requirement") == "required"
                and item.get("persistence") == "persistent"
                and item.get("representation") in requirement.get("representations", [])
                and item.get("memory_resource_id")
                in requirement.get("memory_resources", [])
            ]
            if not legal:
                errors.append(
                    f"execution unit {eid!r} has no planned legal materialization/path for required state {state_id!r}"
                )
        if not adapter.supports_execution(execution):
            errors.append(
                f"execution unit {eid!r} is not legal for the fixture adapter"
            )

    if errors:
        raise FrozenPlanError("invalid frozen plan:\n- " + "\n- ".join(errors))
    return {
        "passed": True,
        "digest": actual_digest,
        "required_bytes_by_memory": required_bytes_by_memory,
    }


class ResearchPlanAdapter(Protocol):
    def supports_representation(
        self, logical_state_id: str, representation: str
    ) -> bool: ...
    def supports_execution(self, execution: Mapping[str, Any]) -> bool: ...
    def begin(
        self, plan: Mapping[str, Any], environment: Mapping[str, Any]
    ) -> None: ...
    def realize_materialization(
        self, materialization: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...
    def release_materialization(
        self, materialization: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...
    def activate_execution(self, execution: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def observe_authorities(self) -> list[Mapping[str, Any]]: ...


@dataclass
class FrozenPlanRealization:
    plan: Mapping[str, Any]
    validation: Mapping[str, Any]
    observed_materializations: list[dict[str, Any]]
    observed_execution: list[dict[str, Any]]
    observed_authorities: list[dict[str, Any]]
    reconciliation: dict[str, Any]
    adapter: ResearchPlanAdapter


def reconcile_realization(
    plan: Mapping[str, Any],
    observed_materializations: list[Mapping[str, Any]],
    observed_execution: list[Mapping[str, Any]],
    observed_authorities: list[Mapping[str, Any]],
) -> dict[str, Any]:
    planned = {item["id"]: item for item in plan.get("materializations", [])}
    observed = {
        item.get("materialization_id"): item for item in observed_materializations
    }
    planned_not_realized: list[str] = []
    unplanned_persistent: list[str] = []
    mismatches: list[str] = []
    for mid, intended in planned.items():
        actual = observed.get(mid)
        if actual is None:
            if intended.get("requirement") == "required":
                planned_not_realized.append(mid)
            continue
        if (
            intended.get("requirement") == "optional"
            and actual.get("status") == "OPTIONAL_NOT_REALIZED"
        ):
            continue
        if (
            intended.get("persistence") == "transient"
            and actual.get("status") == "TRANSIENT_RELEASED_AS_PLANNED"
        ):
            continue
        if actual.get("status") != "PLANNED_AND_REALIZED":
            mismatches.append(f"{mid}: unexpected status {actual.get('status')!r}")
        for intended_key, actual_key in (
            ("logical_state_id", "logical_state_id"),
            ("representation", "actual_representation"),
            ("memory_resource_id", "actual_memory_resource_id"),
        ):
            if intended.get(intended_key) != actual.get(actual_key):
                mismatches.append(
                    f"{mid}: {actual_key} {actual.get(actual_key)!r} != planned {intended.get(intended_key)!r}"
                )
        expected_bytes = intended.get("expected_bytes")
        if (
            isinstance(expected_bytes, int)
            and actual.get("observed_bytes") != expected_bytes
        ):
            mismatches.append(
                f"{mid}: observed_bytes {actual.get('observed_bytes')!r} != expected {expected_bytes}"
            )
    for item in observed_materializations:
        mid = item.get("materialization_id")
        if (
            mid not in planned
            and item.get("persistence") == "persistent"
            and item.get("lifecycle_state") == "live"
        ):
            unplanned_persistent.append(str(mid))
    expected_exec = {item["id"]: item for item in plan.get("execution", [])}
    actual_exec = {item.get("execution_id"): item for item in observed_execution}
    for eid, intended in expected_exec.items():
        actual = actual_exec.get(eid)
        if actual is None:
            mismatches.append(f"execution {eid!r} was not activated")
        elif actual.get("compute_unit_id") != intended.get("compute_unit_id"):
            mismatches.append(
                f"execution {eid!r} ran on {actual.get('compute_unit_id')!r}, planned {intended.get('compute_unit_id')!r}"
            )
    expected_authority = {
        (item["logical_state_id"], item["materialization_id"], item["lineage"])
        for item in plan.get("authorities", [])
    }
    actual_authority = {
        (
            item.get("logical_state_id"),
            item.get("materialization_id"),
            item.get("lineage"),
        )
        for item in observed_authorities
    }
    if expected_authority != actual_authority:
        mismatches.append(
            f"mutable authority mismatch: observed {sorted(actual_authority)!r}, planned {sorted(expected_authority)!r}"
        )
    result = {
        "planned_not_realized": planned_not_realized,
        "unplanned_persistent": unplanned_persistent,
        "mismatches": mismatches,
        "passed": not planned_not_realized
        and not unplanned_persistent
        and not mismatches,
    }
    return result


def realize_frozen_plan(
    plan: Mapping[str, Any],
    environment: Mapping[str, Any],
    adapter: ResearchPlanAdapter,
) -> FrozenPlanRealization:
    validation = validate_frozen_plan(plan, environment, adapter)
    adapter.begin(plan, environment)
    observed: list[dict[str, Any]] = []
    for materialization in plan.get("materializations", []):
        if materialization.get("requirement") == "optional" and not materialization.get(
            "realize", False
        ):
            observed.append(
                {
                    "materialization_id": materialization["id"],
                    "logical_state_id": materialization["logical_state_id"],
                    "status": "OPTIONAL_NOT_REALIZED",
                    "lifecycle_state": "absent",
                    "persistence": materialization["persistence"],
                }
            )
            continue
        record = dict(adapter.realize_materialization(materialization))
        record.setdefault("materialization_id", materialization["id"])
        record.setdefault("logical_state_id", materialization["logical_state_id"])
        record.setdefault("intended_role", materialization["role"])
        record.setdefault("intended_representation", materialization["representation"])
        record.setdefault(
            "intended_memory_resource_id", materialization["memory_resource_id"]
        )
        record.setdefault("intended_persistence", materialization["persistence"])
        record.setdefault("persistence", materialization["persistence"])
        observed.append(record)

    realized_required_ids = {
        item.get("logical_state_id")
        for item in observed
        if item.get("status") == "PLANNED_AND_REALIZED"
    }
    for materialization, record in zip(
        plan.get("materializations", []), observed, strict=True
    ):
        if (
            materialization.get("role") != "staging"
            or materialization.get("persistence") != "transient"
        ):
            continue
        final_exists = any(
            candidate.get("logical_state_id") == materialization.get("logical_state_id")
            and candidate.get("requirement") == "required"
            and candidate.get("persistence") == "persistent"
            and candidate.get("logical_state_id") in realized_required_ids
            for candidate in plan.get("materializations", [])
        )
        if final_exists:
            released = dict(adapter.release_materialization(materialization))
            record.update(released)
            record["status"] = "TRANSIENT_RELEASED_AS_PLANNED"
            record["lifecycle_state"] = "released"

    execution = [
        dict(adapter.activate_execution(item)) for item in plan.get("execution", [])
    ]
    authorities = [dict(item) for item in adapter.observe_authorities()]
    reconciliation = reconcile_realization(plan, observed, execution, authorities)
    if not reconciliation["passed"]:
        raise ReconciliationError(json.dumps(reconciliation, sort_keys=True))
    return FrozenPlanRealization(
        plan, validation, observed, execution, authorities, reconciliation, adapter
    )
