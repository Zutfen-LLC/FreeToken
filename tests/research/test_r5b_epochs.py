from __future__ import annotations

from copy import deepcopy

import pytest

from freetoken.research.r3_planner import freeze
from freetoken.research.r5a_serving import RealizedStaticPlan
from freetoken.research.r5b_epochs import (
    EpochServingController,
    RecoveryUnavailableError,
    SessionLedger,
)


def _problem():
    return freeze(
        {
            "schema": "test.r5b.problem/1",
            "implementation_commit": "producer",
            "evidence_context": {"runtime_context": "runtime"},
            "shapes": [
                {
                    "id": "same-node",
                    "slots": [
                        {
                            "id": "left",
                            "allowed_compute_unit_ids": ["a0"],
                            "required_capabilities": ["execute"],
                            "memory": {"persistent_required_bytes": 1},
                        },
                        {
                            "id": "right",
                            "allowed_compute_unit_ids": ["a1"],
                            "required_capabilities": ["execute"],
                            "memory": {"persistent_required_bytes": 1},
                        },
                    ],
                    "distinct_slot_groups": [["left", "right"]],
                },
                {
                    "id": "two-node",
                    "slots": [
                        {
                            "id": "left",
                            "allowed_compute_unit_ids": ["a0"],
                            "required_capabilities": ["execute"],
                            "memory": {"persistent_required_bytes": 1},
                        },
                        {
                            "id": "right",
                            "allowed_compute_unit_ids": ["b0"],
                            "required_capabilities": ["execute"],
                            "memory": {"persistent_required_bytes": 1},
                        },
                    ],
                    "distinct_slot_groups": [["left", "right"]],
                },
            ],
        }
    )


def _snapshot(*, a1: bool, b0: bool, label: str):
    def node(node_id, units):
        return {
            "id": node_id,
            "compute_units": [
                {
                    "id": unit_id,
                    "memory_resource_id": f"{unit_id}.vram",
                    "capabilities": ["execute"],
                    "availability": "AVAILABLE" if available else "UNAVAILABLE",
                    "integrity_eligible": True,
                }
                for unit_id, available in units
            ],
            "memory_resources": [
                {"id": f"{unit_id}.vram", "kind": "vram", "capacity_bytes": 10}
                for unit_id, _ in units
            ],
        }

    return freeze(
        {
            "schema": f"test.r5b.snapshot/{label}",
            "implementation_commit": "producer",
            "evidence_context": {"runtime_context": "runtime"},
            "nodes": [
                node("node-a", [("a0", True), ("a1", a1)]),
                node("node-b", [("b0", b0)]),
            ],
            "links": [],
        }
    )


def _evidence():
    def record(record_id, shape, mapping, value):
        return {
            "id": record_id,
            "role": "RANKING_OBJECTIVE",
            "producer_identity": "producer",
            "evidence_identity": record_id,
            "shape_id": shape,
            "mapping": mapping,
            "required_context": {"runtime_context": "runtime"},
            "freshness": "CURRENT",
            "measurement_status": "MEASURED",
            "evidence_class": "MEASURED_MATCHED_SERVING",
            "confidence": "EXACT_CONTEXT",
            "metric": {
                "name": "ttft_ms",
                "value": value,
                "unit": "ms",
                "statistic": "median",
            },
            "provenance": {"artifact": record_id},
        }

    return freeze(
        {
            "schema": "test.r5b.evidence/1",
            "implementation_commit": "producer",
            "records": [
                record("same", "same-node", {"left": "a0", "right": "a1"}, 1),
                record("network", "two-node", {"left": "a0", "right": "b0"}, 5),
            ],
        }
    )


def _compiled(evaluation):
    units = list(evaluation["mapping"].values())
    return {
        "strategy_identity": {"id": "test-strategy"},
        "model_identity": {"revision": "test"},
        "participants": sorted({f"node-{unit[0]}" for unit in units}),
        "compute_units": units,
        "representations": [{"id": "weights", "kind": "test"}],
        "backend_choices": [{"backend": "test"}],
        "state_placement": [{"state": "mutable", "units": units}],
        "state_authority": [{"state": "mutable", "owner": "active-epoch"}],
        "semantic_boundaries": [{"kind": "opaque"}],
        "expected_resource_accounting": evaluation["memory_accounting"],
    }


class _Runtime:
    def __init__(self, plan, *, fail=False):
        if fail:
            raise RuntimeError("controlled preparation failure")
        self.plan = plan
        self.closed = False

    def generate(self, *, session_id, prompt_token_ids, max_new_tokens, on_token=None):
        assert not self.closed
        assert max_new_tokens == 1
        token = prompt_token_ids[-1] + 1
        if on_token:
            on_token(0, token, {"checksum": f"token-{token}"})
        return {
            "session_id": session_id,
            "plan_digest": self.plan["digest"],
            "generated_token_ids": [token],
        }

    def report(self):
        return {"closed": self.closed}

    def close(self):
        self.closed = True

    def fail_resource(self, resource_id):
        self.failed_resource = resource_id


class _Strategy:
    def safe_boundary(self, *, session, trigger):
        return {
            "safe": True,
            "kind": "test-token-boundary",
            "committed_position": session.committed_position,
        }

    def replay_input(self, *, session):
        return session.prompt_token_ids + session.committed_token_ids

    def recovery_contract(self, *, session, trigger):
        trusted = trigger.get("trusted_recovery", True)
        return {
            "continuation_legal": trusted,
            "source": "retained-prompt-and-committed-token-ids" if trusted else None,
            "reason": None if trusted else "controlled trusted-history loss",
        }

    def next_token(self, result):
        return result["generated_token_ids"][-1]


def _controller(*, fail_candidate=None):
    policy = freeze(
        {
            "schema": "test.r5b.policy/1",
            "implementation_commit": "producer",
            "excluded_compute_unit_ids": [],
        }
    )
    objective = freeze(
        {
            "schema": "test.r5b.objective/1",
            "implementation_commit": "producer",
            "metric": "ttft_ms",
            "direction": "MINIMIZE",
            "unit": "ms",
            "statistic": "median",
        }
    )

    def realizer(execution_plan):
        runtime = _Runtime(
            execution_plan,
            fail=execution_plan["candidate_id"].startswith(fail_candidate or "!"),
        )
        observation = {
            key: deepcopy(execution_plan[key])
            for key in (
                "participants",
                "compute_units",
                "representations",
                "backend_choices",
                "state_placement",
                "state_authority",
                "semantic_boundaries",
            )
        }
        observation["plan_digest"] = execution_plan["digest"]
        return RealizedStaticPlan(runtime=runtime, observation=observation)

    def transition_policy(_old, _new, event):
        return {
            "authorize": True,
            "reason": "beneficial or required replacement",
            "overlap_preparation": event.get("overlap", True),
        }

    return EpochServingController(
        problem=_problem(),
        initial_snapshot=_snapshot(a1=False, b0=True, label="initial"),
        policy=policy,
        objective=objective,
        evidence_catalog=_evidence(),
        compiler=_compiled,
        realizer=realizer,
        transition_strategy=_Strategy(),
        transition_policy=transition_policy,
    )


def _event(snapshot, event_id, kind, resource, **extra):
    return {
        "event_id": event_id,
        "kind": kind,
        "resource_id": resource,
        "authenticated": True,
        "observed_at_ns": 1,
        "resource_snapshot_digest": snapshot["digest"],
        **extra,
    }


def test_scale_up_down_back_up_preserves_exact_output_and_distinct_authority():
    controller = _controller()
    up = _snapshot(a1=True, b0=True, label="up")
    down = _snapshot(a1=False, b0=True, label="down")
    back = _snapshot(a1=True, b0=True, label="back")

    def after_commit(step, active):
        if step == 0:
            active.submit_resource_event(_event(up, "up", "AVAILABLE", "a1"), up)
        elif step == 1:
            active.submit_resource_event(
                _event(
                    down,
                    "loss",
                    "PARTICIPANT_LOST",
                    "a1",
                    active_plan_executable=False,
                ),
                down,
            )
        elif step == 2:
            active.submit_resource_event(
                _event(back, "return", "RETURNED", "a1"), back
            )

    result = controller.serve_tokens(
        session_id=7,
        prompt_token_ids=[10],
        max_new_tokens=5,
        sampling_inputs={"temperature": 0},
        after_commit=after_commit,
    )
    assert result["generated_token_ids"] == [11, 12, 13, 14, 15]
    assert len(set(result["committed_epoch_ids"])) == 4
    report = controller.report()
    assert [item["status"] for item in report["transitions"]] == [
        "ACTIVATED",
        "ACTIVATED",
        "ACTIVATED",
    ]
    assert report["single_mutable_authority"] is True
    assert [item["generation"] for item in report["epochs"]] == [0, 1, 2, 3]
    assert all(item["state"] == "RECLAIMED" for item in report["epochs"][:-1])


def test_late_retired_epoch_result_is_rejected_without_state_or_accounting_change():
    controller = _controller()
    session = SessionLedger(9, [1], 3, {"temperature": 0})
    old = controller.active_epoch
    up = _snapshot(a1=True, b0=True, label="late-up")
    controller.submit_resource_event(_event(up, "up", "AVAILABLE", "a1"), up)
    controller._process_event(session)
    assert controller.inject_late_result(
        epoch_id=old.epoch_id,
        plan_digest=old.execution_plan["digest"],
        session=session,
        position=0,
        token_id=999,
    ) is False
    assert session.committed_token_ids == []
    rejection = controller.report()["late_result_rejections"][0]
    assert rejection["reason"] == "RETIRED_OR_SUPERSEDED_EPOCH"
    assert rejection["committed_position_unchanged"] == 0


def test_failed_overlap_preparation_keeps_old_epoch_authoritative():
    controller = _controller(fail_candidate="same-node")
    old = controller.active_epoch
    up = _snapshot(a1=True, b0=True, label="failed-prep")
    controller.submit_resource_event(
        _event(up, "up", "AVAILABLE", "a1", overlap=True), up
    )
    result = controller.serve_tokens(
        session_id=11,
        prompt_token_ids=[20],
        max_new_tokens=2,
        sampling_inputs={"temperature": 0},
    )
    assert result["generated_token_ids"] == [21, 22]
    assert set(result["committed_epoch_ids"]) == {old.epoch_id}
    transition = controller.report()["transitions"][0]
    assert transition["status"] == "ABORTED_PREPARATION"
    assert transition["mutable_authority_after"] == old.epoch_id
    assert transition["partial_replacement_activated"] is False


def test_unrecoverable_authority_loss_fails_closed():
    controller = _controller()
    up = _snapshot(a1=True, b0=True, label="recover-up")
    controller.submit_resource_event(_event(up, "up", "AVAILABLE", "a1"), up)

    def after_commit(step, active):
        if step == 0:
            lost = _snapshot(a1=False, b0=True, label="unrecoverable")
            active.submit_resource_event(
                _event(
                    lost,
                    "loss",
                    "PARTICIPANT_LOST",
                    "a1",
                    active_plan_executable=False,
                    trusted_recovery=False,
                ),
                lost,
            )

    with pytest.raises(RecoveryUnavailableError, match="trusted-history loss"):
        controller.serve_tokens(
            session_id=12,
            prompt_token_ids=[30],
            max_new_tokens=3,
            sampling_inputs={"temperature": 0},
            after_commit=after_commit,
        )
    report = controller.report()
    assert report["transitions"][-1]["status"] == "FAILED_UNRECOVERABLE_STATE"
    assert report["transitions"][-1]["mutable_authority_after"] is None
    assert report["single_mutable_authority"] is False


def test_unauthenticated_or_unbound_resource_event_is_rejected():
    controller = _controller()
    snapshot = _snapshot(a1=True, b0=True, label="bad-event")
    event = _event(snapshot, "bad", "AVAILABLE", "a1")
    event["authenticated"] = False
    with pytest.raises(ValueError, match="not authenticated"):
        controller.submit_resource_event(event, snapshot)
    event["authenticated"] = True
    event["resource_snapshot_digest"] = "sha256:wrong"
    with pytest.raises(ValueError, match="does not bind"):
        controller.submit_resource_event(event, snapshot)
