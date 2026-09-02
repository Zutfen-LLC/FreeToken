"""Minimum automatic planner used by the InferSwarm R3 research proof.

This module deliberately speaks only in resource, state, constraint, and
evidence terms.  Shapes and their opaque payloads belong to a strategy.  The
structures are versioned research records, not a public scheduling API.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

DECISION_SCHEMA = "inferswarm.r3.planner-decision/1"
RANKED = "RANKED"
FEASIBLE_UNRANKED = "FEASIBLE_UNRANKED"
TECHNICALLY_INFEASIBLE = "TECHNICALLY_INFEASIBLE"
INTEGRITY_EXCLUDED = "INTEGRITY_EXCLUDED"
POLICY_EXCLUDED = "POLICY_EXCLUDED"
EVIDENCE_EXCLUDED = "EVIDENCE_EXCLUDED"


class PlanningInputError(ValueError):
    """A frozen R3 planner input is malformed or has a bad digest."""


class EnvironmentDriftError(RuntimeError):
    """The execution environment no longer matches the selected decision."""


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    payload = deepcopy(dict(value))
    payload.pop("digest", None)
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def freeze(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["digest"] = digest(result)
    return result


def require_frozen(value: Mapping[str, Any], label: str) -> None:
    if value.get("digest") != digest(value):
        raise PlanningInputError(f"{label} digest mismatch")


def _by_id(items: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for item in items:
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise PlanningInputError(f"{label} lacks a stable id")
        if item_id in result:
            raise PlanningInputError(f"duplicate {label} id {item_id!r}")
        result[item_id] = item
    return result


def _resource_indexes(snapshot: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    compute: dict[str, Mapping[str, Any]] = {}
    memory: dict[str, Mapping[str, Any]] = {}
    for node in snapshot.get("nodes", []):
        for item in node.get("compute_units", []):
            if item.get("id") in compute:
                raise PlanningInputError(f"duplicate Compute Unit id {item.get('id')!r}")
            record = dict(item)
            record["node_id"] = node.get("id")
            compute[str(item.get("id"))] = record
        for item in node.get("memory_resources", []):
            if item.get("id") in memory:
                raise PlanningInputError(f"duplicate Memory Resource id {item.get('id')!r}")
            record = dict(item)
            record["node_id"] = node.get("id")
            memory[str(item.get("id"))] = record
    return compute, memory, _by_id(snapshot.get("links", []), "Link")


def _available_bytes(resource: Mapping[str, Any]) -> int:
    capacity = resource.get("capacity_bytes")
    if not isinstance(capacity, int) or capacity < 0:
        return -1
    reservations = int(resource.get("reservation_bytes", 0))
    return capacity - reservations


def _compatible(unit: Mapping[str, Any], required: Sequence[str]) -> list[str]:
    if unit.get("availability", "AVAILABLE") != "AVAILABLE":
        return [f"Compute Unit {unit['id']!r} is unavailable"]
    capabilities = set(unit.get("capabilities", []))
    missing = sorted(set(required) - capabilities)
    return ([f"Compute Unit {unit['id']!r} lacks capabilities {missing}"] if missing else [])


def enumerate_candidates(
    problem: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Enumerate strategy-legal mappings in stable order; payload is never inspected."""
    require_frozen(problem, "strategy problem")
    require_frozen(snapshot, "resource snapshot")
    compute, _, _ = _resource_indexes(snapshot)
    candidates: list[dict[str, Any]] = []
    for shape in sorted(problem.get("shapes", []), key=lambda item: item["id"]):
        slots = sorted(shape.get("slots", []), key=lambda item: item["id"])
        pools: list[list[str]] = []
        for slot in slots:
            allowed = slot.get("allowed_compute_unit_ids")
            pool = sorted(set(allowed) if allowed is not None else compute)
            # Keep explicitly requested but absent units so the reason survives audit.
            pools.append(pool)
        for assignment in itertools.product(*pools):
            mapping = dict(zip((slot["id"] for slot in slots), assignment, strict=True))
            legal_reasons: list[str] = []
            for group in shape.get("distinct_slot_groups", []):
                assigned = [mapping[slot_id] for slot_id in group]
                if len(set(assigned)) != len(assigned):
                    legal_reasons.append(f"slots {list(group)!r} require distinct resources")
            for group in shape.get("colocated_slot_groups", []):
                assigned = [mapping[slot_id] for slot_id in group]
                if len(set(assigned)) != 1:
                    legal_reasons.append(f"slots {list(group)!r} require co-location")
            mapping_id = ",".join(f"{key}={mapping[key]}" for key in sorted(mapping))
            candidates.append(
                {
                    "id": f"{shape['id']}[{mapping_id}]",
                    "shape_id": shape["id"],
                    "mapping": mapping,
                    "strategy_legal": not legal_reasons,
                    "strategy_legality_reasons": legal_reasons,
                }
            )
    return sorted(candidates, key=lambda item: item["id"])


def _path_matches(
    link: Mapping[str, Any], source_memory: str, target_memory: str, required: set[str]
) -> bool:
    endpoints = {link.get("source_memory_resource_id"), link.get("target_memory_resource_id")}
    if endpoints != {source_memory, target_memory}:
        return False
    return required <= set(link.get("capabilities", [])) and link.get("available", True)


def _evaluate_technical(
    candidate: Mapping[str, Any],
    shape: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    compute, memory, links = _resource_indexes(snapshot)
    reasons = list(candidate["strategy_legality_reasons"])
    required: defaultdict[str, int] = defaultdict(int)
    optional: defaultdict[str, int] = defaultdict(int)
    transient: defaultdict[str, int] = defaultdict(int)
    slot_memory: dict[str, str] = {}

    slots = _by_id(shape.get("slots", []), "slot")
    for slot_id, compute_id in candidate["mapping"].items():
        slot = slots[slot_id]
        unit = compute.get(compute_id)
        if unit is None:
            reasons.append(f"Compute Unit {compute_id!r} is missing")
            continue
        reasons.extend(_compatible(unit, slot.get("required_capabilities", [])))
        memory_id = unit.get("memory_resource_id")
        if memory_id not in memory:
            reasons.append(f"Compute Unit {compute_id!r} has no available Memory Resource")
            continue
        slot_memory[slot_id] = memory_id
        demand = slot.get("memory", {})
        required[memory_id] += int(demand.get("persistent_required_bytes", 0))
        optional[memory_id] += int(demand.get("persistent_optional_bytes", 0))
        transient[memory_id] += int(demand.get("transient_peak_bytes", 0))

    # Shape-level materializations cover host RAM, checkpoint staging, and similar roles.
    for item in shape.get("materializations", []):
        matches = sorted(
            resource_id
            for resource_id, resource in memory.items()
            if resource.get("kind") == item.get("memory_kind")
        )
        if len(matches) != 1:
            reasons.append(f"required Memory Resource kind {item.get('memory_kind')!r} is missing or ambiguous")
            continue
        memory_id = matches[0]
        byte_count = int(item.get("bytes", 0))
        lifecycle = item.get("lifecycle", "PERSISTENT_REQUIRED")
        if lifecycle == "PERSISTENT_REQUIRED":
            required[memory_id] += byte_count
        elif lifecycle == "PERSISTENT_OPTIONAL_RETAIN":
            # RETAIN is occupied capacity.  It is never credited as live-evictable.
            optional[memory_id] += byte_count
        elif lifecycle == "TRANSIENT_RELEASE_AFTER_FINALIZATION":
            # RELEASE affects steady accounting only; feasibility still includes its peak.
            transient[memory_id] += byte_count
        else:
            reasons.append(f"unsupported materialization lifecycle {lifecycle!r}")

    memory_accounting = {}
    for memory_id in sorted(set(required) | set(optional) | set(transient)):
        resource = memory[memory_id]
        available = _available_bytes(resource)
        peak = required[memory_id] + optional[memory_id] + transient[memory_id]
        steady = required[memory_id] + optional[memory_id]
        memory_accounting[memory_id] = {
            "persistent_required_bytes": required[memory_id],
            "persistent_optional_bytes": optional[memory_id],
            "transient_realization_peak_bytes": transient[memory_id],
            "released_after_lifecycle_boundary_bytes": transient[memory_id],
            "steady_occupied_bytes": steady,
            "realization_peak_occupied_bytes": peak,
            "available_after_reservation_bytes": available,
        }
        if available < 0:
            reasons.append(f"Memory Resource {memory_id!r} has invalid capacity")
        elif peak > available:
            reasons.append(
                f"Memory Resource {memory_id!r} requires peak {peak} bytes but only {available} are available"
            )

    selected_paths = []
    for requirement in shape.get("paths", []):
        source = slot_memory.get(requirement["from_slot"])
        target = slot_memory.get(requirement["to_slot"])
        if source is None or target is None:
            continue
        matches = sorted(
            link_id
            for link_id, link in links.items()
            if _path_matches(link, source, target, set(requirement.get("required_capabilities", [])))
        )
        if not matches:
            reasons.append(
                f"no supported Link/path from {source!r} to {target!r} for {requirement['id']!r}"
            )
        else:
            selected_paths.append({"requirement_id": requirement["id"], "link_id": matches[0]})

    return {
        "technically_feasible": not reasons,
        "technical_reasons": reasons,
        "memory_accounting": memory_accounting,
        "selected_paths": selected_paths,
    }


def _evaluate_integrity(
    candidate: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    compute, _, _ = _resource_indexes(snapshot)
    reasons = []
    for compute_id in sorted(set(candidate["mapping"].values())):
        unit = compute.get(compute_id)
        if unit is not None and not unit.get("integrity_eligible", True):
            reasons.append(f"Compute Unit {compute_id!r} is integrity-ineligible/quarantined")
    return not reasons, reasons


def _evaluate_policy(
    candidate: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    policy: Mapping[str, Any],
    memory_accounting: Mapping[str, Mapping[str, int]],
) -> tuple[bool, list[str]]:
    compute, memory, _ = _resource_indexes(snapshot)
    excluded = set(policy.get("excluded_compute_unit_ids", []))
    approved = policy.get("approved_compute_unit_ids")
    approved_set = set(approved) if approved is not None else None
    reasons = []
    for compute_id in sorted(set(candidate["mapping"].values())):
        if compute_id in excluded:
            reasons.append(f"Compute Unit {compute_id!r} is operator-excluded")
        if approved_set is not None and compute_id not in approved_set:
            reasons.append(f"Compute Unit {compute_id!r} is not operator-approved")
    for memory_id, reserved in sorted(policy.get("reservations_bytes", {}).items()):
        if memory_id not in memory:
            reasons.append(f"operator reservation names missing Memory Resource {memory_id!r}")
            continue
        if not isinstance(reserved, int) or reserved < 0:
            reasons.append(f"operator reservation for {memory_id!r} is invalid")
            continue
        peak = memory_accounting.get(memory_id, {}).get(
            "realization_peak_occupied_bytes", 0
        )
        available = _available_bytes(memory[memory_id]) - reserved
        if peak > available:
            reasons.append(
                f"operator reservation leaves only {available} bytes on {memory_id!r} "
                f"for a {peak}-byte realization peak"
            )
    return not reasons, reasons


def _context_mismatches(
    evidence: Mapping[str, Any], context: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[str]:
    reasons = []
    if evidence.get("shape_id") != candidate["shape_id"]:
        reasons.append("candidate shape mismatch")
    if evidence.get("mapping") != candidate["mapping"]:
        reasons.append("resource mapping mismatch")
    for key, expected in evidence.get("required_context", {}).items():
        if context.get(key) != expected:
            reasons.append(f"evidence context mismatch for {key}")
    if evidence.get("freshness") not in ("CURRENT", "ACCEPTED_COMPATIBLE"):
        reasons.append("evidence is stale or lacks a compatibility statement")
    return reasons


def _applicable_evidence(
    candidate: Mapping[str, Any],
    catalog: Mapping[str, Any],
    objective: Mapping[str, Any],
    planning_context: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit every record and return ranking and admission evidence separately.

    R3 only had objective-valued records, so records for another metric were
    silently skipped.  R5A needs accepted path/capacity evidence to be visible
    even when the declared ranking objective is an end-to-end service metric.
    ``role=ADMISSION_CONSTRAINT`` is deliberately generic: the strategy binds
    the record to a candidate/context and supplies a normalized comparison.
    """
    applicable = []
    admission = []
    considered = []
    records = sorted(
        catalog.get("records", []),
        key=lambda item: (
            item.get("metric", {}).get("name") != objective.get("metric"),
            item["id"],
        ),
    )
    for record in records:
        reasons = _context_mismatches(record, planning_context, candidate)
        metric = record.get("metric", {})
        role = record.get("role", "RANKING_OBJECTIVE")
        if role == "RANKING_OBJECTIVE":
            if metric.get("name") != objective.get("metric"):
                reasons.append(
                    f"objective metric mismatch: evidence {metric.get('name')!r}, "
                    f"objective {objective.get('metric')!r}"
                )
            if metric.get("unit") != objective.get("unit"):
                reasons.append(
                    f"metric unit mismatch: evidence {metric.get('unit')!r}, "
                    f"objective {objective.get('unit')!r}"
                )
            if metric.get("statistic") != objective.get("statistic"):
                reasons.append(
                    f"metric statistic mismatch: evidence {metric.get('statistic')!r}, "
                    f"objective {objective.get('statistic')!r}"
                )
        elif role == "ADMISSION_CONSTRAINT":
            constraint = record.get("constraint", {})
            if constraint.get("comparison") not in ("LTE", "GTE"):
                reasons.append("admission constraint comparison must be LTE or GTE")
            if not isinstance(constraint.get("threshold"), (int, float)):
                reasons.append("admission constraint lacks a numeric threshold")
            if metric.get("unit") != constraint.get("unit"):
                reasons.append("admission metric and threshold units differ")
        else:
            reasons.append(f"unsupported evidence role {role!r}")
        audit = {
            "evidence_id": record["id"],
            "applicable": not reasons,
            "reasons": reasons,
            "evidence_class": record.get("evidence_class", record.get("class")),
            "freshness": record.get("freshness"),
            "confidence": record.get("confidence"),
            "metric": {
                "name": metric.get("name"),
                "value": metric.get("value"),
                "unit": metric.get("unit"),
                "statistic": metric.get("statistic"),
            },
        }
        # Preserve the accepted R3 audit shape byte-for-byte for legacy records;
        # R5A records opt in to the richer generic identity/provenance fields.
        for key in ("producer_identity", "evidence_identity", "measurement_status"):
            if key in record:
                audit[key] = deepcopy(record[key])
        if "role" in record:
            audit["role"] = role
        if "constraint" in record:
            audit["constraint"] = deepcopy(record["constraint"])
        if "provenance" in record:
            audit["provenance"] = deepcopy(record["provenance"])
        if role == "ADMISSION_CONSTRAINT" and not reasons:
            threshold = float(record["constraint"]["threshold"])
            value = float(metric["value"])
            comparison = record["constraint"]["comparison"]
            passed = value <= threshold if comparison == "LTE" else value >= threshold
            audit["constraint_passed"] = passed
            if "role" in record:
                audit["influence"] = (
                    "candidate admitted by applicable constraint"
                    if passed
                    else "candidate excluded by applicable constraint"
                )
            admission.append(dict(record))
        elif role == "RANKING_OBJECTIVE" and not reasons:
            if "role" in record or "provenance" in record:
                audit["influence"] = "included in declared-objective ranking aggregate"
            applicable.append(dict(record))
        else:
            if "role" in record or "provenance" in record:
                audit["influence"] = "rejected; did not affect admission or ranking"
        considered.append(audit)
    return applicable, admission, considered


def plan(
    problem: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    policy: Mapping[str, Any],
    objective: Mapping[str, Any],
    evidence_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate, rank, and freeze one setup-time decision from frozen inputs."""
    for value, label in (
        (problem, "strategy problem"),
        (snapshot, "resource snapshot"),
        (policy, "operator policy"),
        (objective, "objective"),
        (evidence_catalog, "evidence catalog"),
    ):
        require_frozen(value, label)
    implementation_commits = {
        value.get("implementation_commit")
        for value in (problem, snapshot, policy, objective, evidence_catalog)
        if value.get("implementation_commit") is not None
    }
    if len(implementation_commits) > 1:
        raise PlanningInputError("planner inputs name different implementation commits")
    direction = objective.get("direction")
    if direction not in ("MINIMIZE", "MAXIMIZE"):
        raise PlanningInputError("objective direction must be MINIMIZE or MAXIMIZE")
    shapes = _by_id(problem.get("shapes", []), "shape")
    planning_context = {
        **problem.get("evidence_context", {}),
        **snapshot.get("evidence_context", {}),
        **objective.get("evidence_context", {}),
    }
    evaluations = []
    for candidate in enumerate_candidates(problem, snapshot):
        technical = _evaluate_technical(candidate, shapes[candidate["shape_id"]], snapshot)
        integrity_ok, integrity_reasons = _evaluate_integrity(candidate, snapshot)
        policy_ok, policy_reasons = _evaluate_policy(
            candidate, snapshot, policy, technical["memory_accounting"]
        )
        applicable, admission, evidence_audit = _applicable_evidence(
            candidate, evidence_catalog, objective, planning_context
        )
        failed_constraints = []
        for record in admission:
            value = float(record["metric"]["value"])
            threshold = float(record["constraint"]["threshold"])
            comparison = record["constraint"]["comparison"]
            if not (value <= threshold if comparison == "LTE" else value >= threshold):
                failed_constraints.append(record["id"])
        metric = None
        state = TECHNICALLY_INFEASIBLE
        if technical["technically_feasible"]:
            if not integrity_ok:
                state = INTEGRITY_EXCLUDED
            elif not policy_ok:
                state = POLICY_EXCLUDED
            elif failed_constraints:
                state = EVIDENCE_EXCLUDED
            else:
                state = FEASIBLE_UNRANKED
            if integrity_ok and policy_ok and not failed_constraints and applicable:
                # One immutable catalog may contain repetitions; rank its declared aggregate.
                values = [float(record["metric"]["value"]) for record in applicable]
                metric = statistics.median(values)
                state = RANKED
        evaluations.append(
            {
                **candidate,
                **technical,
                "integrity_eligible": integrity_ok,
                "integrity_reasons": integrity_reasons,
                "policy_eligible": policy_ok,
                "policy_reasons": policy_reasons,
                "evidence": evidence_audit,
                "applicable_evidence_ids": [item["id"] for item in applicable],
                "applicable_admission_evidence_ids": [item["id"] for item in admission],
                "failed_admission_evidence_ids": failed_constraints,
                "objective_metric": (
                    {
                        "name": objective["metric"],
                        "value": metric,
                        "unit": objective.get("unit"),
                        "statistic": objective.get("statistic", "median"),
                    }
                    if metric is not None
                    else None
                ),
                "state": state,
            }
        )
    rankable = [item for item in evaluations if item["state"] == RANKED]
    rankable.sort(
        key=lambda item: (
            (-item["objective_metric"]["value"] if direction == "MAXIMIZE" else item["objective_metric"]["value"]),
            item["id"],
        )
    )
    for rank, item in enumerate(rankable, 1):
        item["rank"] = rank
    selected = rankable[0] if rankable else None
    used = set(selected["mapping"].values()) if selected else set()
    all_compute, _, _ = _resource_indexes(snapshot)
    unused = []
    for compute_id in sorted(set(all_compute) - used):
        reason = "not needed by the highest-ranked candidate"
        if compute_id in set(policy.get("excluded_compute_unit_ids", [])):
            reason = "operator-excluded"
        unused.append({"compute_unit_id": compute_id, "reason": reason})
    body = {
        "schema": DECISION_SCHEMA,
        "implementation_commit": problem.get("implementation_commit"),
        "inputs": {
            "resource_snapshot_digest": snapshot["digest"],
            "strategy_problem_digest": problem["digest"],
            "policy_digest": policy["digest"],
            "objective_digest": objective["digest"],
            "evidence_catalog_digest": evidence_catalog["digest"],
        },
        "selection_status": "SELECTED" if selected else "NO_AUTOMATIC_SELECTION_INSUFFICIENT_EVIDENCE",
        "selected_candidate_id": selected["id"] if selected else None,
        "selected_mapping": deepcopy(selected["mapping"]) if selected else None,
        "evaluations": sorted(evaluations, key=lambda item: item["id"]),
        "unused_resources": unused,
        "planning_phase": "SETUP_ONLY",
    }
    return freeze(body)


def validate_decision_environment(
    decision: Mapping[str, Any], current_snapshot: Mapping[str, Any]
) -> None:
    """Fail closed instead of repairing a selected plan after environment drift."""
    require_frozen(decision, "planner decision")
    require_frozen(current_snapshot, "current resource snapshot")
    expected = decision.get("inputs", {}).get("resource_snapshot_digest")
    if current_snapshot.get("digest") != expected:
        raise EnvironmentDriftError(
            "resource snapshot changed after planning; obtain a new planning decision"
        )


def selected_evaluation(decision: Mapping[str, Any]) -> Mapping[str, Any]:
    require_frozen(decision, "planner decision")
    selected_id = decision.get("selected_candidate_id")
    for evaluation in decision.get("evaluations", []):
        if evaluation["id"] == selected_id:
            return evaluation
    raise PlanningInputError("decision has no selected candidate")
