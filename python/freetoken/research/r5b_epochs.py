"""Research-internal execution-plan epochs for the InferSwarm R5B proof.

This module deliberately contains no model, accelerator, transport, or public
protocol nouns.  Strategies own recovery and safe-boundary semantics; the
controller owns immutable plan generations, single authority, commit ordering,
resource-event replanning, activation, retirement, and late-result fencing.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol

from .r3_planner import freeze, plan, require_frozen, selected_evaluation
from .r5a_serving import (
    RealizedStaticPlan,
    RealizationMismatchError,
    ServingIsolationError,
    freeze_execution_plan,
    reconcile_realization,
)


class AuthorityUncertainError(RuntimeError):
    """The controller cannot prove which epoch owns mutable authority."""


class RecoveryUnavailableError(RuntimeError):
    """Required mutable state has no trustworthy recovery source."""


class ActiveEpochLostError(RuntimeError):
    """The active plan is non-executable and no replacement activated."""


class EpochRuntime(Protocol):
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


class TransitionStrategy(Protocol):
    """Model-owned transition contract; opaque to the generic controller."""

    def safe_boundary(
        self, *, session: "SessionLedger", trigger: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def replay_input(self, *, session: "SessionLedger") -> Sequence[int]: ...

    def recovery_contract(
        self, *, session: "SessionLedger", trigger: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def next_token(self, result: Mapping[str, Any]) -> int: ...


@dataclass
class SessionLedger:
    session_id: int
    prompt_token_ids: list[int]
    max_new_tokens: int
    sampling_inputs: Mapping[str, Any]
    committed_token_ids: list[int] = field(default_factory=list)
    committed_epoch_ids: list[str] = field(default_factory=list)
    committed_plan_digests: list[str] = field(default_factory=list)
    committed_at_ns: list[int] = field(default_factory=list)
    failed: bool = False
    failure_reason: str | None = None

    @property
    def committed_position(self) -> int:
        return len(self.committed_token_ids)

    def boundary(self, active_epoch_id: str, plan_digest: str) -> dict[str, Any]:
        return {
            "schema": "inferswarm.r5b.committed-boundary/1",
            "session_id": self.session_id,
            "prompt_token_ids": list(self.prompt_token_ids),
            "committed_generated_token_ids": list(self.committed_token_ids),
            "committed_position": self.committed_position,
            "active_epoch_id": active_epoch_id,
            "plan_digest": plan_digest,
            "sampling_inputs": deepcopy(dict(self.sampling_inputs)),
        }


@dataclass
class Epoch:
    epoch_id: str
    generation: int
    execution_plan: Mapping[str, Any]
    planner_decision: Mapping[str, Any]
    resource_snapshot: Mapping[str, Any]
    runtime: EpochRuntime
    reconciliation: Mapping[str, Any]
    activated_at_ns: int
    state: str = "ACTIVE"
    retired_at_ns: int | None = None
    reclaimed_at_ns: int | None = None
    reclamation: Mapping[str, Any] | None = None


def _now() -> int:
    return time.perf_counter_ns()


def _event_copy(event: Mapping[str, Any]) -> dict[str, Any]:
    required = {"event_id", "kind", "resource_id", "authenticated", "observed_at_ns"}
    missing = sorted(required - set(event))
    if missing:
        raise ValueError(f"resource event lacks {missing}")
    if event["authenticated"] is not True:
        raise ValueError("resource event is not authenticated")
    return deepcopy(dict(event))


class EpochServingController:
    """One-scope epoch state machine above the accepted frozen-plan machinery."""

    def __init__(
        self,
        *,
        problem: Mapping[str, Any],
        initial_snapshot: Mapping[str, Any],
        policy: Mapping[str, Any],
        objective: Mapping[str, Any],
        evidence_catalog: Mapping[str, Any],
        compiler: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        realizer: Callable[[Mapping[str, Any]], RealizedStaticPlan],
        transition_strategy: TransitionStrategy,
        transition_policy: Callable[
            [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any] | None],
            Mapping[str, Any],
        ],
        initial_override_candidate_id: str | None = None,
    ) -> None:
        for value, label in (
            (problem, "strategy problem"),
            (initial_snapshot, "initial resource snapshot"),
            (policy, "operator policy"),
            (objective, "objective"),
            (evidence_catalog, "evidence catalog"),
        ):
            require_frozen(value, label)
        self.problem = problem
        self.policy = policy
        self.objective = objective
        self.evidence_catalog = evidence_catalog
        self.compiler = compiler
        self.realizer = realizer
        self.transition_strategy = transition_strategy
        self.transition_policy = transition_policy
        self._lock = threading.RLock()
        self._execution_lock = threading.Lock()
        self._events: deque[tuple[dict[str, Any], Mapping[str, Any]]] = deque()
        self._epochs: list[Epoch] = []
        self._transitions: list[dict[str, Any]] = []
        self._event_audit: list[dict[str, Any]] = []
        self._late_rejections: list[dict[str, Any]] = []
        self._sessions: dict[int, SessionLedger] = {}
        self._session_records: list[dict[str, Any]] = []
        self._completed_sessions: set[int] = set()
        self._generation = 0
        self._runtime_session_sequence = 0
        self._closed = False
        decision, execution_plan = self._plan(
            initial_snapshot, override_candidate_id=initial_override_candidate_id
        )
        realized = self._realize(execution_plan)
        activated = _now()
        self._active = Epoch(
            epoch_id=self._epoch_id(self._generation, execution_plan["digest"]),
            generation=self._generation,
            execution_plan=execution_plan,
            planner_decision=decision,
            resource_snapshot=initial_snapshot,
            runtime=realized.runtime,
            reconciliation=reconcile_realization(execution_plan, realized.observation),
            activated_at_ns=activated,
        )
        self._epochs.append(self._active)

    @staticmethod
    def _epoch_id(generation: int, plan_digest: str) -> str:
        return f"research-generation-{generation}:{plan_digest.split(':')[-1][:12]}"

    def _plan(
        self,
        snapshot: Mapping[str, Any],
        *,
        override_candidate_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        decision = plan(
            self.problem, snapshot, self.policy, self.objective, self.evidence_catalog
        )
        if override_candidate_id is None:
            evaluation = dict(selected_evaluation(decision))
            authorization = {
                "mode": "AUTOMATIC_PLANNER_SELECTION",
                "candidate_id": evaluation["id"],
                "planner_selected_candidate_id": decision["selected_candidate_id"],
            }
        else:
            evaluations = {item["id"]: item for item in decision["evaluations"]}
            evaluation = dict(evaluations[override_candidate_id])
            if evaluation.get("state") not in ("RANKED", "FEASIBLE_UNRANKED"):
                raise ActiveEpochLostError(
                    f"requested initial candidate is not feasible: {override_candidate_id}"
                )
            authorization = {
                "mode": "CONTROLLED_INITIAL_STATE_OVERRIDE",
                "candidate_id": evaluation["id"],
                "planner_selected_candidate_id": decision.get("selected_candidate_id"),
                "automatic_selection_preserved": True,
            }
        execution_plan = freeze_execution_plan(
            decision=decision,
            evaluation=evaluation,
            authorization=authorization,
            compiled_body=self.compiler(evaluation),
            objective=self.objective,
            policy=self.policy,
        )
        return decision, execution_plan

    def _realize(self, execution_plan: Mapping[str, Any]) -> RealizedStaticPlan:
        realized = self.realizer(execution_plan)
        reconcile_realization(execution_plan, realized.observation)
        return realized

    @property
    def active_epoch(self) -> Epoch:
        with self._lock:
            if self._active.state != "ACTIVE":
                raise AuthorityUncertainError("no unique active mutable authority")
            return self._active

    def submit_resource_event(
        self, event: Mapping[str, Any], snapshot: Mapping[str, Any]
    ) -> None:
        """Queue one validated event/snapshot; no polling scheduler is created."""
        copied = _event_copy(event)
        require_frozen(snapshot, "post-event resource snapshot")
        if copied.get("resource_snapshot_digest") != snapshot.get("digest"):
            raise ValueError("resource event does not bind the supplied snapshot")
        with self._lock:
            self._events.append((copied, snapshot))
            self._event_audit.append({**deepcopy(copied), "accepted": True})
            runtime = self._active.runtime
        if copied.get("active_plan_executable", True) is False:
            fail_resource = getattr(runtime, "fail_resource", None)
            if fail_resource is None:
                raise ActiveEpochLostError(
                    "runtime has no controlled participant-loss seam"
                )
            fail_resource(str(copied["resource_id"]))

    def _runtime_session_id(self, logical_session_id: int) -> int:
        self._runtime_session_sequence += 1
        return logical_session_id * 1_000_000 + self._runtime_session_sequence

    def _result_envelope(
        self,
        *,
        epoch: Epoch,
        session: SessionLedger,
        token_id: int,
        runtime_result: Mapping[str, Any],
        boundary: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "schema": "inferswarm.r5b.result-envelope/1",
            "epoch_id": epoch.epoch_id,
            "generation": epoch.generation,
            "plan_digest": epoch.execution_plan["digest"],
            "session_id": session.session_id,
            "position": session.committed_position,
            "token_id": int(token_id),
            "boundary": deepcopy(dict(boundary or {})),
            "runtime_session_id": runtime_result.get("session_id"),
            "arrived_at_ns": _now(),
        }

    def accept_result(
        self, envelope: Mapping[str, Any], session: SessionLedger
    ) -> bool:
        """Commit only a current-epoch, next-position result; fence all old work."""
        with self._lock:
            reason = None
            if self._active.state != "ACTIVE":
                reason = "NO_ACTIVE_AUTHORITY"
            elif envelope.get("epoch_id") != self._active.epoch_id:
                reason = "RETIRED_OR_SUPERSEDED_EPOCH"
            elif envelope.get("plan_digest") != self._active.execution_plan["digest"]:
                reason = "PLAN_DIGEST_MISMATCH"
            elif envelope.get("session_id") != session.session_id:
                reason = "SESSION_MISMATCH"
            elif envelope.get("position") != session.committed_position:
                reason = "NON_NEXT_COMMIT_POSITION"
            if reason is not None:
                self._late_rejections.append(
                    {
                        "rejected_at_ns": _now(),
                        "reason": reason,
                        "active_epoch_id": getattr(self._active, "epoch_id", None),
                        "envelope": deepcopy(dict(envelope)),
                        "committed_position_unchanged": session.committed_position,
                    }
                )
                return False
            session.committed_token_ids.append(int(envelope["token_id"]))
            session.committed_epoch_ids.append(self._active.epoch_id)
            session.committed_plan_digests.append(self._active.execution_plan["digest"])
            session.committed_at_ns.append(_now())
            return True

    def _retire_and_reclaim(self, epoch: Epoch) -> None:
        epoch.state = "RETIRED"
        epoch.retired_at_ns = _now()
        epoch.runtime.close()
        epoch.reclamation = deepcopy(
            dict(getattr(epoch.runtime, "reclamation_report", {}) or {})
        )
        epoch.state = "RECLAIMED"
        epoch.reclaimed_at_ns = _now()

    def _process_event(self, session: SessionLedger) -> None:
        with self._lock:
            if not self._events:
                return
            event, snapshot = self._events.popleft()
            old = self._active
        started = _now()
        record: dict[str, Any] = {
            "schema": "inferswarm.r5b.authority-transition/1",
            "execution_scope": session.session_id,
            "old_epoch_id": old.epoch_id,
            "old_generation": old.generation,
            "old_plan_digest": old.execution_plan["digest"],
            "transition_trigger": deepcopy(event),
            "resource_snapshot_digest": snapshot["digest"],
            "preparation_started_at_ns": started,
            "mutable_authority_before": old.epoch_id,
            "latest_committed_recovery_boundary": session.boundary(
                old.epoch_id, old.execution_plan["digest"]
            ),
            "status": "PREPARING",
        }
        try:
            decision, replacement_plan = self._plan(snapshot)
        except Exception as exc:
            record.update(
                {
                    "status": "ABORTED_PLANNING",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "mutable_authority_after": old.epoch_id,
                    "ended_at_ns": _now(),
                }
            )
            self._transitions.append(record)
            if event.get("active_plan_executable", True) is False:
                raise ActiveEpochLostError(record["failure_reason"]) from exc
            return
        record["planning_ended_at_ns"] = _now()
        record["replanning_wall_ns"] = record["planning_ended_at_ns"] - started
        policy_decision = dict(self.transition_policy(old.execution_plan, replacement_plan, event))
        record.update(
            {
                "planner_decision_digest": decision["digest"],
                "replacement_plan_digest": replacement_plan["digest"],
                "replacement_candidate_id": replacement_plan["candidate_id"],
                "selection_authorization": deepcopy(
                    replacement_plan["selection_authorization"]
                ),
                "applicable_evidence": deepcopy(replacement_plan["evidence_used"]),
                "rejected_evidence": [
                    item
                    for item in replacement_plan["evidence_audit"]
                    if not item.get("applicable")
                ],
                "transition_policy": deepcopy(policy_decision),
            }
        )
        if replacement_plan["candidate_id"] == old.execution_plan["candidate_id"]:
            record.update(
                {
                    "status": "RETAINED_CURRENT_EPOCH",
                    "failure_reason": "planner retained the current candidate",
                    "mutable_authority_after": old.epoch_id,
                    "ended_at_ns": _now(),
                }
            )
            self._transitions.append(record)
            return
        if not policy_decision.get("authorize", False):
            record.update(
                {
                    "status": "RETAINED_BY_POLICY",
                    "failure_reason": policy_decision.get("reason", "not authorized"),
                    "mutable_authority_after": old.epoch_id,
                    "ended_at_ns": _now(),
                }
            )
            self._transitions.append(record)
            if event.get("active_plan_executable", True) is False:
                raise ActiveEpochLostError(record["failure_reason"])
            return

        safe = dict(self.transition_strategy.safe_boundary(session=session, trigger=event))
        recovery = dict(
            self.transition_strategy.recovery_contract(session=session, trigger=event)
        )
        record["safe_boundary_selected"] = deepcopy(safe)
        record["recovery_contract"] = deepcopy(recovery)
        if not safe.get("safe", False):
            raise AuthorityUncertainError("strategy refused the transition boundary")
        if not recovery.get("continuation_legal", False):
            if event.get("active_plan_executable", True) is False:
                old.state = "NON_EXECUTABLE"
                old.retired_at_ns = _now()
                old.runtime.close()
                old.reclaimed_at_ns = _now()
            record.update(
                {
                    "status": "FAILED_UNRECOVERABLE_STATE",
                    "failure_reason": recovery.get(
                        "reason", "strategy has no trustworthy recovery source"
                    ),
                    "mutable_authority_after": None,
                    "ended_at_ns": _now(),
                }
            )
            self._transitions.append(record)
            session.failed = True
            session.failure_reason = record["failure_reason"]
            raise RecoveryUnavailableError(record["failure_reason"])

        overlap = bool(policy_decision.get("overlap_preparation", False))
        realized: RealizedStaticPlan | None = None
        try:
            if overlap:
                realized = self._realize(replacement_plan)
            record["preparation_ended_at_ns"] = _now()
            record["preparation_mode"] = (
                "MAKE_BEFORE_BREAK" if overlap else "SAFE_INTERRUPTED_COLD_CUTOVER"
            )
        except Exception as exc:
            record.update(
                {
                    "status": "ABORTED_PREPARATION",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "mutable_authority_after": old.epoch_id,
                    "ended_at_ns": _now(),
                    "partial_replacement_activated": False,
                }
            )
            self._transitions.append(record)
            if event.get("active_plan_executable", True) is False:
                raise ActiveEpochLostError(record["failure_reason"]) from exc
            return

        cutover_started = _now()
        if not overlap:
            self._retire_and_reclaim(old)
            try:
                realized = self._realize(replacement_plan)
            except Exception as exc:
                record.update(
                    {
                        "status": "FAILED_COLD_REALIZATION",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "mutable_authority_after": None,
                        "ended_at_ns": _now(),
                    }
                )
                self._transitions.append(record)
                session.failed = True
                session.failure_reason = record["failure_reason"]
                raise ActiveEpochLostError(record["failure_reason"]) from exc
        assert realized is not None
        reconciliation = reconcile_realization(replacement_plan, realized.observation)
        self._generation += 1
        replacement = Epoch(
            epoch_id=self._epoch_id(self._generation, replacement_plan["digest"]),
            generation=self._generation,
            execution_plan=replacement_plan,
            planner_decision=decision,
            resource_snapshot=snapshot,
            runtime=realized.runtime,
            reconciliation=reconciliation,
            activated_at_ns=_now(),
        )
        with self._lock:
            if overlap:
                old.state = "SETTLING"
            self._active = replacement
            self._epochs.append(replacement)
        if overlap:
            self._retire_and_reclaim(old)
        ended = _now()
        replay_range = [0, session.committed_position]
        record.update(
            {
                "replacement_epoch_id": replacement.epoch_id,
                "replacement_generation": replacement.generation,
                "activation_timestamp_ns": replacement.activated_at_ns,
                "activation_order": replacement.generation,
                "retirement_state": old.state,
                "resource_reclamation_state": old.state,
                "mutable_authority_after": replacement.epoch_id,
                "recovery_replay_source": recovery.get("source"),
                "recovery_replay_range": replay_range,
                "replay_token_count": session.committed_position,
                "authority_cutover_started_at_ns": cutover_started,
                "authority_cutover_ended_at_ns": ended,
                "authority_cutover_ns": ended - cutover_started,
                "status": "ACTIVATED",
                "ended_at_ns": ended,
            }
        )
        self._transitions.append(record)
        if event.get("inject_late_result") and session.committed_token_ids:
            self.inject_late_result(
                epoch_id=old.epoch_id,
                plan_digest=old.execution_plan["digest"],
                session=session,
                position=session.committed_position,
                token_id=session.committed_token_ids[-1],
            )

    def serve_tokens(
        self,
        *,
        session_id: int,
        prompt_token_ids: Sequence[int],
        max_new_tokens: int,
        sampling_inputs: Mapping[str, Any],
        on_token: Callable[[int, int, Mapping[str, Any]], None] | None = None,
        after_commit: Callable[[int, "EpochServingController"], None] | None = None,
    ) -> dict[str, Any]:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        with self._lock:
            if session_id in self._sessions or session_id in self._completed_sessions:
                raise ServingIsolationError(f"session id {session_id} was reused")
            session = SessionLedger(
                session_id=session_id,
                prompt_token_ids=list(prompt_token_ids),
                max_new_tokens=max_new_tokens,
                sampling_inputs=deepcopy(dict(sampling_inputs)),
            )
            self._sessions[session_id] = session
        started = _now()
        try:
            while session.committed_position < max_new_tokens:
                self._process_event(session)
                epoch = self.active_epoch
                replay_started = _now()
                replay_input = list(self.transition_strategy.replay_input(session=session))
                boundary_holder: dict[str, Any] = {}
                token_holder: list[int] = []

                def capture(_step: int, token: int, boundary: Mapping[str, Any]) -> None:
                    if _step == 0:
                        token_holder.append(int(token))
                        boundary_holder.update(deepcopy(dict(boundary)))

                runtime_session_id = self._runtime_session_id(session_id)
                with self._execution_lock:
                    try:
                        result = dict(
                            epoch.runtime.generate(
                                session_id=runtime_session_id,
                                prompt_token_ids=replay_input,
                                # Accepted R2/R4 service measurement requires one
                                # decode interval. Commit only step zero; step one is
                                # explicitly speculative and discarded before replay.
                                max_new_tokens=2,
                                on_token=capture,
                            )
                        )
                    except Exception:
                        # A failure event may have arrived during execution.  If so,
                        # the committed ledger remains untouched and recovery starts
                        # from its last boundary.  Otherwise propagate the failure.
                        if not self._events:
                            raise
                        self._process_event(session)
                        continue
                if result.get("plan_digest") != epoch.execution_plan["digest"]:
                    raise RealizationMismatchError("runtime silently substituted a plan")
                token = (
                    token_holder[-1]
                    if token_holder
                    else self.transition_strategy.next_token(result)
                )
                envelope = self._result_envelope(
                    epoch=epoch,
                    session=session,
                    token_id=token,
                    runtime_result=result,
                    boundary=boundary_holder,
                )
                if not self.accept_result(envelope, session):
                    continue
                step = session.committed_position - 1
                commit = {
                    **deepcopy(envelope),
                    "committed_at_ns": session.committed_at_ns[-1],
                    "replay_input_token_count": len(replay_input),
                    "replay_wall_ns": _now() - replay_started,
                    "speculative_uncommitted_tokens_discarded": 1,
                }
                if on_token is not None:
                    on_token(step, token, commit)
                if after_commit is not None:
                    after_commit(step, self)
            completed = {
                "session_id": session_id,
                "generated_token_ids": list(session.committed_token_ids),
                "committed_epoch_ids": list(session.committed_epoch_ids),
                "committed_plan_digests": list(session.committed_plan_digests),
                "latest_committed_boundary": session.boundary(
                    self.active_epoch.epoch_id,
                    self.active_epoch.execution_plan["digest"],
                ),
                "request_wall_ns": _now() - started,
            }
            with self._lock:
                self._session_records.append(deepcopy(completed))
            return completed
        except Exception as exc:
            session.failed = True
            session.failure_reason = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._session_records.append(
                    {
                        "session_id": session_id,
                        "prompt_token_ids": list(session.prompt_token_ids),
                        "generated_token_ids": list(session.committed_token_ids),
                        "committed_epoch_ids": list(session.committed_epoch_ids),
                        "committed_plan_digests": list(session.committed_plan_digests),
                        "failed": True,
                        "failure_reason": session.failure_reason,
                        "latest_committed_position": session.committed_position,
                        "request_wall_ns": _now() - started,
                    }
                )
            raise
        finally:
            with self._lock:
                self._sessions.pop(session_id, None)
                self._completed_sessions.add(session_id)

    def inject_late_result(
        self,
        *,
        epoch_id: str,
        plan_digest: str,
        session: SessionLedger,
        position: int,
        token_id: int,
    ) -> bool:
        """Bounded physical-path fault seam used to prove retired fencing."""
        return self.accept_result(
            {
                "schema": "inferswarm.r5b.result-envelope/1",
                "epoch_id": epoch_id,
                "plan_digest": plan_digest,
                "session_id": session.session_id,
                "position": position,
                "token_id": token_id,
                "injection": "CONTROLLED_LATE_REAL_SERVING_RESULT",
                "arrived_at_ns": _now(),
            },
            session,
        )

    def report(self) -> dict[str, Any]:
        with self._lock:
            epochs = [
                {
                    "epoch_id": item.epoch_id,
                    "generation": item.generation,
                    "plan_digest": item.execution_plan["digest"],
                    "candidate_id": item.execution_plan["candidate_id"],
                    "resource_snapshot_digest": item.resource_snapshot["digest"],
                    "planner_decision_digest": item.planner_decision["digest"],
                    "activated_at_ns": item.activated_at_ns,
                    "state": item.state,
                    "retired_at_ns": item.retired_at_ns,
                    "reclaimed_at_ns": item.reclaimed_at_ns,
                    "reclamation": deepcopy(dict(item.reclamation or {})),
                    "reconciliation": deepcopy(dict(item.reconciliation)),
                    "execution_plan": deepcopy(dict(item.execution_plan)),
                    "planner_decision": deepcopy(dict(item.planner_decision)),
                }
                for item in self._epochs
            ]
            active = self._active
            return {
                "schema": "inferswarm.r5b.epoch-serving-report/1",
                "active_epoch_id": active.epoch_id if active.state == "ACTIVE" else None,
                "active_plan_digest": active.execution_plan["digest"],
                "single_mutable_authority": active.state == "ACTIVE",
                "epochs": epochs,
                "resource_events": deepcopy(self._event_audit),
                "transitions": deepcopy(self._transitions),
                "late_result_rejection_count": len(self._late_rejections),
                "late_result_rejections": deepcopy(self._late_rejections),
                "sessions": deepcopy(self._session_records),
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            active = self._active
        if active.state in ("ACTIVE", "NON_EXECUTABLE", "SETTLING"):
            self._retire_and_reclaim(active)


__all__ = [
    "ActiveEpochLostError",
    "AuthorityUncertainError",
    "EpochServingController",
    "RecoveryUnavailableError",
    "SessionLedger",
    "TransitionStrategy",
]
