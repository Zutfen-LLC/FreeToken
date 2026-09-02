from __future__ import annotations

import threading
from copy import deepcopy

import pytest

from freetoken.research.r3_planner import EVIDENCE_EXCLUDED, freeze
from freetoken.research.r5a_serving import (
    RealizationMismatchError,
    RealizedStaticPlan,
    ServingIsolationError,
    StaticServingController,
)


def _inputs(*, demand=2.947, limit=747.12):
    problem = freeze(
        {
            "schema": "test.strategy/1",
            "implementation_commit": "frozen-sha",
            "evidence_context": {
                "model_revision": "revision",
                "runtime_context": "runtime",
                "network_context": "link",
            },
            "shapes": [
                {
                    "id": "local",
                    "slots": [
                        {
                            "id": "only",
                            "allowed_compute_unit_ids": ["a"],
                            "required_capabilities": ["execute"],
                            "memory": {"persistent_required_bytes": 10},
                        }
                    ],
                },
                {
                    "id": "distributed",
                    "slots": [
                        {
                            "id": "left",
                            "allowed_compute_unit_ids": ["a"],
                            "required_capabilities": ["execute"],
                            "memory": {"persistent_required_bytes": 10},
                        },
                        {
                            "id": "right",
                            "allowed_compute_unit_ids": ["b"],
                            "required_capabilities": ["execute"],
                            "memory": {"persistent_required_bytes": 10},
                        },
                    ],
                    "paths": [
                        {
                            "id": "opaque-boundary",
                            "from_slot": "left",
                            "to_slot": "right",
                            "required_capabilities": ["transport"],
                        }
                    ],
                },
            ],
        }
    )
    snapshot = freeze(
        {
            "schema": "test.resources/1",
            "implementation_commit": "frozen-sha",
            "evidence_context": {
                "runtime_context": "runtime",
                "network_context": "link",
            },
            "nodes": [
                {
                    "id": "node-a",
                    "compute_units": [
                        {
                            "id": "a",
                            "memory_resource_id": "a-vram",
                            "capabilities": ["execute"],
                            "integrity_eligible": True,
                        }
                    ],
                    "memory_resources": [
                        {"id": "a-vram", "kind": "vram", "capacity_bytes": 100}
                    ],
                },
                {
                    "id": "node-b",
                    "compute_units": [
                        {
                            "id": "b",
                            "memory_resource_id": "b-vram",
                            "capabilities": ["execute"],
                            "integrity_eligible": True,
                        }
                    ],
                    "memory_resources": [
                        {"id": "b-vram", "kind": "vram", "capacity_bytes": 100}
                    ],
                },
            ],
            "links": [
                {
                    "id": "path",
                    "source_memory_resource_id": "a-vram",
                    "target_memory_resource_id": "b-vram",
                    "capabilities": ["transport"],
                }
            ],
        }
    )
    policy = freeze(
        {
            "schema": "test.policy/1",
            "implementation_commit": "frozen-sha",
            "excluded_compute_unit_ids": [],
        }
    )
    objective = freeze(
        {
            "schema": "test.objective/1",
            "implementation_commit": "frozen-sha",
            "metric": "request_wall_ms",
            "direction": "MINIMIZE",
            "unit": "ms",
            "statistic": "median",
        }
    )
    context = {
        "model_revision": "revision",
        "runtime_context": "runtime",
        "network_context": "link",
    }

    def ranking(record_id, shape, mapping, value):
        return {
            "id": record_id,
            "role": "RANKING_OBJECTIVE",
            "producer_identity": "producer",
            "evidence_identity": record_id,
            "shape_id": shape,
            "mapping": mapping,
            "required_context": context,
            "freshness": "CURRENT",
            "measurement_status": "MEASURED",
            "evidence_class": "MEASURED_MATCHED_SERVING",
            "confidence": "EXACT_CONTEXT",
            "metric": {
                "name": "request_wall_ms",
                "value": value,
                "unit": "ms",
                "statistic": "median",
            },
            "provenance": {"artifact": "arm.json"},
        }

    evidence = freeze(
        {
            "schema": "test.evidence/1",
            "implementation_commit": "frozen-sha",
            "records": [
                ranking("local-service", "local", {"only": "a"}, 10.0),
                ranking(
                    "distributed-service",
                    "distributed",
                    {"left": "a", "right": "b"},
                    20.0,
                ),
                {
                    "id": "accepted-path-capacity",
                    "role": "ADMISSION_CONSTRAINT",
                    "producer_identity": "r4-producer",
                    "evidence_identity": "r4-capacity-disposition",
                    "shape_id": "distributed",
                    "mapping": {"left": "a", "right": "b"},
                    "required_context": context,
                    "freshness": "ACCEPTED_COMPATIBLE",
                    "measurement_status": "MEASURED",
                    "evidence_class": "MEASURED_ACCEPTED_PATH_CAPACITY",
                    "confidence": "EXACT_CONTEXT",
                    "metric": {
                        "name": "application_network_demand",
                        "value": demand,
                        "unit": "Mb/s",
                        "statistic": "peak",
                    },
                    "constraint": {
                        "comparison": "LTE",
                        "threshold": limit,
                        "unit": "Mb/s",
                    },
                    "provenance": {"artifact": "docs/inferswarm_r4/result.json"},
                },
            ],
        }
    )
    return problem, snapshot, policy, objective, evidence


def _compiled(evaluation):
    units = sorted(set(evaluation["mapping"].values()))
    participants = [f"node-{unit}" for unit in units]
    return {
        "strategy_identity": {"id": "opaque-strategy", "version": 1},
        "model_identity": {"repository": "model", "revision": "revision"},
        "participants": participants,
        "compute_units": units,
        "representations": [{"id": "state", "representation": "opaque"}],
        "backend_choices": [{"participant": node, "backend": "native"} for node in participants],
        "state_placement": [{"state": "state", "compute_unit": units[0]}],
        "state_authority": [{"state": "mutable", "participant": participants[0]}],
        "semantic_boundaries": ([{"id": "boundary", "from": "a", "to": "b"}] if len(units) == 2 else []),
        "expected_resource_accounting": evaluation["memory_accounting"],
    }


class _Runtime:
    def __init__(self, digest, entered=None, release=None):
        self.digest = digest
        self.entered = entered
        self.release = release
        self.sessions = []

    def generate(self, *, session_id, prompt_token_ids, max_new_tokens, on_token=None):
        self.sessions.append(session_id)
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            assert self.release.wait(5)
        tokens = [prompt_token_ids[-1]] * max_new_tokens
        if on_token is not None:
            for step, token in enumerate(tokens):
                on_token(step, token, {})
        return {
            "session_id": session_id,
            "plan_digest": self.digest,
            "generated_token_ids": tokens,
        }

    def report(self):
        return {"sessions": list(self.sessions), "fallbacks": 0}

    def close(self):
        pass


def _controller(*, override=None, observation_mutation=None, entered=None, release=None):
    problem, snapshot, policy, objective, evidence = _inputs()

    def realizer(execution_plan):
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
        if observation_mutation:
            observation_mutation(observation)
        return RealizedStaticPlan(
            runtime=_Runtime(execution_plan["digest"], entered, release),
            observation=observation,
        )

    return StaticServingController(
        problem=problem,
        snapshot=snapshot,
        policy=policy,
        objective=objective,
        evidence_catalog=evidence,
        compiler=_compiled,
        realizer=realizer,
        override_candidate_id=override,
    )


def test_automatic_selection_is_preserved_and_plan_is_frozen_before_realization():
    controller = _controller()
    assert controller.execution_plan["candidate_id"] == "local[only=a]"
    assert controller.execution_plan["selection_authorization"]["mode"] == "AUTOMATIC_PLANNER_SELECTION"
    assert controller.report()["plan_frozen_before_realization_completed"] is True
    assert controller.execution_plan["digest"].startswith("sha256:")
    assert controller.execution_plan["lower_ranked_feasible_candidates"]


def test_controlled_distributed_override_is_explicit_and_keeps_automatic_result():
    candidate = "distributed[left=a,right=b]"
    controller = _controller(override=candidate)
    authorization = controller.execution_plan["selection_authorization"]
    assert controller.execution_plan["candidate_id"] == candidate
    assert authorization == {
        "mode": "CONTROLLED_EVIDENCE_COLLECTION_OVERRIDE",
        "candidate_id": candidate,
        "planner_selected_candidate_id": "local[only=a]",
        "automatic_selection_preserved": True,
    }
    audit = next(
        row
        for row in controller.execution_plan["evidence_audit"]
        if row["evidence_id"] == "accepted-path-capacity"
    )
    assert audit["applicable"] and audit["constraint_passed"]
    assert audit["measurement_status"] == "MEASURED"
    assert audit["provenance"]["artifact"].endswith("result.json")


def test_applicable_capacity_evidence_can_exclude_without_hard_coded_domain_logic():
    problem, snapshot, policy, objective, evidence = _inputs(demand=800.0)
    from freetoken.research.r3_planner import plan

    decision = plan(problem, snapshot, policy, objective, evidence)
    distributed = next(
        row for row in decision["evaluations"] if row["shape_id"] == "distributed"
    )
    assert distributed["state"] == EVIDENCE_EXCLUDED
    assert distributed["failed_admission_evidence_ids"] == ["accepted-path-capacity"]


def test_realization_mismatch_fails_closed_before_serving():
    with pytest.raises(RealizationMismatchError, match="participants"):
        _controller(observation_mutation=lambda value: value.update(participants=[]))


def test_runtime_plan_substitution_and_session_reuse_fail_closed():
    controller = _controller()
    controller.runtime.digest = "sha256:wrong"
    with pytest.raises(RealizationMismatchError, match="substituted"):
        controller.serve_tokens(session_id=1, prompt_token_ids=[1], max_new_tokens=1)
    with pytest.raises(ServingIsolationError, match="reused"):
        controller.serve_tokens(session_id=1, prompt_token_ids=[1], max_new_tokens=1)


def test_two_outstanding_requests_are_isolated_on_one_static_plan():
    entered = threading.Event()
    release = threading.Event()
    controller = _controller(entered=entered, release=release)
    results = []

    def run(session_id):
        results.append(
            controller.serve_tokens(
                session_id=session_id,
                prompt_token_ids=[session_id],
                max_new_tokens=2,
            )
        )

    first = threading.Thread(target=run, args=(11,))
    second = threading.Thread(target=run, args=(12,))
    first.start()
    assert entered.wait(2)
    second.start()
    while controller.report()["max_outstanding_requests"] < 2:
        pass
    release.set()
    first.join(2)
    second.join(2)
    assert {row["session_id"] for row in results} == {11, 12}
    assert all(row["plan_digest"] == controller.execution_plan["digest"] for row in results)
    assert controller.report()["max_outstanding_requests"] == 2
