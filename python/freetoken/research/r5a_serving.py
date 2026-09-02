"""Static InferSwarm serving orchestration for the R5A research gate.

The types are deliberately research-internal and model-neutral.  A Model
Execution Strategy supplies legal candidates and compiles one evaluated
candidate into the immutable execution-plan body.  This module owns ordering,
freeze/reconciliation, explicit evidence-collection overrides, and request
isolation; it does not know model, transport, or backend nouns.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from .r3_planner import (
    FEASIBLE_UNRANKED,
    RANKED,
    PlanningInputError,
    freeze,
    plan,
    require_frozen,
    selected_evaluation,
    validate_decision_environment,
)

PLAN_SCHEMA = "inferswarm.r5a.static-execution-plan/1"


class RealizationMismatchError(RuntimeError):
    """Observed heavyweight realization differs from the frozen plan."""


class ServingIsolationError(RuntimeError):
    """A request/session identity was reused or crossed another request."""


def checkpoint_identity_from_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Read the reused R4 gate's nested retained-check schema fail-closed."""
    if gate.get("result") != "ALL_PREFLIGHT_CHECKS_PASSED":
        raise ValueError("R5A environment cannot retain an unsuccessful preflight")
    try:
        return dict(gate["checks"]["checkpoint_identity"])
    except KeyError as exc:
        raise ValueError("preflight gate lacks checkpoint_identity check") from exc


class StaticRuntime(Protocol):
    def generate(
        self,
        *,
        session_id: int,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        on_token: Callable[[int, int, Mapping[str, Any]], None] | None = None,
    ) -> Mapping[str, Any]: ...

    def report(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class RealizedStaticPlan:
    runtime: StaticRuntime
    observation: Mapping[str, Any]


def _selected_or_override(
    decision: Mapping[str, Any], override_candidate_id: str | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return evaluation and an auditable authorization record."""
    if override_candidate_id is None:
        evaluation = dict(selected_evaluation(decision))
        return evaluation, {
            "mode": "AUTOMATIC_PLANNER_SELECTION",
            "candidate_id": evaluation["id"],
            "planner_selected_candidate_id": decision["selected_candidate_id"],
        }
    evaluations = {item["id"]: item for item in decision.get("evaluations", [])}
    if override_candidate_id not in evaluations:
        raise PlanningInputError(
            f"evidence-collection override names unknown candidate {override_candidate_id!r}"
        )
    evaluation = dict(evaluations[override_candidate_id])
    if evaluation.get("state") not in (RANKED, FEASIBLE_UNRANKED):
        raise PlanningInputError(
            "evidence-collection override may run only a technically feasible, "
            "integrity-eligible, policy-eligible candidate with no failed admission constraint"
        )
    return evaluation, {
        "mode": "CONTROLLED_EVIDENCE_COLLECTION_OVERRIDE",
        "candidate_id": evaluation["id"],
        "planner_selected_candidate_id": decision.get("selected_candidate_id"),
        "automatic_selection_preserved": True,
    }


def freeze_execution_plan(
    *,
    decision: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    authorization: Mapping[str, Any],
    compiled_body: Mapping[str, Any],
    objective: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the complete immutable setup record before realization begins."""
    for value, label in (
        (decision, "planner decision"),
        (objective, "objective"),
        (policy, "operator policy"),
    ):
        require_frozen(value, label)
    required = {
        "strategy_identity",
        "model_identity",
        "participants",
        "compute_units",
        "representations",
        "backend_choices",
        "state_placement",
        "state_authority",
        "semantic_boundaries",
        "expected_resource_accounting",
    }
    missing = sorted(required - set(compiled_body))
    if missing:
        raise PlanningInputError(f"strategy compiler omitted execution-plan fields {missing}")
    body = {
        "schema": PLAN_SCHEMA,
        "planner_decision_digest": decision["digest"],
        "candidate_id": evaluation["id"],
        "mapping": deepcopy(evaluation["mapping"]),
        "selection_authorization": deepcopy(dict(authorization)),
        "strategy_identity": deepcopy(compiled_body["strategy_identity"]),
        "model_identity": deepcopy(compiled_body["model_identity"]),
        "participants": deepcopy(compiled_body["participants"]),
        "compute_units": deepcopy(compiled_body["compute_units"]),
        "representations": deepcopy(compiled_body["representations"]),
        "backend_choices": deepcopy(compiled_body["backend_choices"]),
        "state_placement": deepcopy(compiled_body["state_placement"]),
        "state_authority": deepcopy(compiled_body["state_authority"]),
        "semantic_boundaries": deepcopy(compiled_body["semantic_boundaries"]),
        "evidence_used": deepcopy(evaluation.get("applicable_evidence_ids", [])),
        "admission_evidence_used": deepcopy(
            evaluation.get("applicable_admission_evidence_ids", [])
        ),
        "evidence_audit": deepcopy(evaluation.get("evidence", [])),
        "objective": deepcopy(dict(objective)),
        "policy_inputs": deepcopy(dict(policy)),
        "exclusions": [
            deepcopy(item)
            for item in decision.get("evaluations", [])
            if item.get("state") not in (RANKED, FEASIBLE_UNRANKED)
        ],
        "lower_ranked_feasible_candidates": [
            {
                "candidate_id": item["id"],
                "state": item["state"],
                "rank": item.get("rank"),
                "objective_metric": deepcopy(item.get("objective_metric")),
            }
            for item in decision.get("evaluations", [])
            if item["id"] != evaluation["id"]
            and item.get("state") in (RANKED, FEASIBLE_UNRANKED)
        ],
        "unused_resources": deepcopy(decision.get("unused_resources", [])),
        "expected_resource_accounting": deepcopy(
            compiled_body["expected_resource_accounting"]
        ),
        "strategy_realization": deepcopy(compiled_body.get("strategy_realization", {})),
        "frozen_before_heavyweight_realization": True,
    }
    return freeze(body)


def reconcile_realization(
    execution_plan: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any]:
    """Mechanically compare correctness-bearing intended and observed state."""
    require_frozen(execution_plan, "execution plan")
    comparisons = {
        "plan_digest": (execution_plan["digest"], observation.get("plan_digest")),
        "participants": (execution_plan["participants"], observation.get("participants")),
        "compute_units": (execution_plan["compute_units"], observation.get("compute_units")),
        "representations": (
            execution_plan["representations"],
            observation.get("representations"),
        ),
        "backend_choices": (
            execution_plan["backend_choices"],
            observation.get("backend_choices"),
        ),
        "state_placement": (
            execution_plan["state_placement"],
            observation.get("state_placement"),
        ),
        "state_authority": (
            execution_plan["state_authority"],
            observation.get("state_authority"),
        ),
        "semantic_boundaries": (
            execution_plan["semantic_boundaries"],
            observation.get("semantic_boundaries"),
        ),
    }
    mismatches = [
        {"field": key, "intended": intended, "observed": observed}
        for key, (intended, observed) in comparisons.items()
        if intended != observed
    ]
    result = {
        "schema": "inferswarm.r5a.realization-reconciliation/1",
        "plan_digest": execution_plan["digest"],
        "matched": not mismatches,
        "mismatches": mismatches,
    }
    if mismatches:
        raise RealizationMismatchError(
            "observed realization does not match frozen execution plan: "
            + ", ".join(item["field"] for item in mismatches)
        )
    return result


class StaticServingController:
    """Plan once, realize once, and serve isolated requests without substitution."""

    def __init__(
        self,
        *,
        problem: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        policy: Mapping[str, Any],
        objective: Mapping[str, Any],
        evidence_catalog: Mapping[str, Any],
        compiler: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        realizer: Callable[[Mapping[str, Any]], RealizedStaticPlan],
        override_candidate_id: str | None = None,
    ) -> None:
        self.decision = plan(problem, snapshot, policy, objective, evidence_catalog)
        evaluation, authorization = _selected_or_override(
            self.decision, override_candidate_id
        )
        compiled = compiler(evaluation)
        self.execution_plan = freeze_execution_plan(
            decision=self.decision,
            evaluation=evaluation,
            authorization=authorization,
            compiled_body=compiled,
            objective=objective,
            policy=policy,
        )
        # The current resource graph must still be exactly the graph used by the
        # planner immediately before heavyweight realization starts.
        validate_decision_environment(self.decision, snapshot)
        self.plan_frozen_at_ns = time.perf_counter_ns()
        realized = realizer(self.execution_plan)
        self.realization_completed_at_ns = time.perf_counter_ns()
        self.reconciliation = reconcile_realization(
            self.execution_plan, realized.observation
        )
        self.runtime = realized.runtime
        self._execution_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._active: set[int] = set()
        self._completed: set[int] = set()
        self._requests: list[dict[str, Any]] = []
        self._max_outstanding = 0

    def serve_tokens(
        self,
        *,
        session_id: int,
        prompt_token_ids: Sequence[int],
        max_new_tokens: int,
        on_token: Callable[[int, int, Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        admitted_at = time.perf_counter_ns()
        with self._state_lock:
            if session_id in self._active or session_id in self._completed:
                raise ServingIsolationError(f"session id {session_id} was reused")
            self._active.add(session_id)
            self._max_outstanding = max(self._max_outstanding, len(self._active))
        try:
            # The accepted R4 primitive is single-connection/static-plan. Multiple
            # requests may be outstanding at the serving surface, but execute in
            # bounded FIFO admission order through this immutable plan.
            with self._execution_lock:
                execution_started = time.perf_counter_ns()
                result = dict(
                    self.runtime.generate(
                        session_id=session_id,
                        prompt_token_ids=list(prompt_token_ids),
                        max_new_tokens=max_new_tokens,
                        on_token=on_token,
                    )
                )
                execution_ended = time.perf_counter_ns()
            if result.get("session_id") != session_id:
                raise ServingIsolationError(
                    f"runtime returned session {result.get('session_id')!r} for {session_id}"
                )
            if result.get("plan_digest") != self.execution_plan["digest"]:
                raise RealizationMismatchError("runtime silently substituted a plan")
            record = {
                "session_id": session_id,
                "plan_digest": self.execution_plan["digest"],
                "admitted_at_ns": admitted_at,
                "execution_started_ns": execution_started,
                "execution_ended_ns": execution_ended,
                "queue_wait_ns": execution_started - admitted_at,
                "request_wall_ns": execution_ended - admitted_at,
                "prompt_token_count": len(prompt_token_ids),
                "generated_token_count": len(result.get("generated_token_ids", [])),
            }
            result["serving_record"] = record
            with self._state_lock:
                self._requests.append(record)
            return result
        finally:
            with self._state_lock:
                self._active.discard(session_id)
                self._completed.add(session_id)

    def report(self) -> dict[str, Any]:
        with self._state_lock:
            requests = deepcopy(self._requests)
            active = sorted(self._active)
            maximum = self._max_outstanding
        return {
            "schema": "inferswarm.r5a.static-serving-report/1",
            "plan_digest": self.execution_plan["digest"],
            "selection_authorization": deepcopy(
                self.execution_plan["selection_authorization"]
            ),
            "reconciliation": deepcopy(self.reconciliation),
            "plan_frozen_at_ns": self.plan_frozen_at_ns,
            "realization_completed_at_ns": self.realization_completed_at_ns,
            "plan_frozen_before_realization_completed": (
                self.plan_frozen_at_ns < self.realization_completed_at_ns
            ),
            "active_session_ids": active,
            "max_outstanding_requests": maximum,
            "requests": requests,
            "runtime": deepcopy(dict(self.runtime.report())),
        }

    def close(self) -> None:
        self.runtime.close()


__all__ = [
    "PLAN_SCHEMA",
    "RealizationMismatchError",
    "RealizedStaticPlan",
    "ServingIsolationError",
    "StaticServingController",
    "checkpoint_identity_from_gate",
    "freeze_execution_plan",
    "reconcile_realization",
]
