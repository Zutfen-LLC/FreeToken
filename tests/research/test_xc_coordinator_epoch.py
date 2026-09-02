"""Coordinator-side epoch/remote-realizer integration tests (inferswarm #67).

Drives the real ``EpochServingController`` (planning, frozen Execution Plan,
commit/fencing) against the remote ``xc_wire`` seam with an in-memory fake
Node-agent runtime — no torch, no model, no GPU.  These are the tests that
physically run on the CPU-only Coordinator host to prove control-plane purity.
"""

from __future__ import annotations

import threading

import pytest

from freetoken.research.r5b_epochs import EpochServingController
from freetoken.research.xc_coordinator import make_remote_realizer

import benchmarks.inferswarm_xc.node_agent as node_agent_module
from tests.research.test_xc_seam import (
    SCOPE,
    FakeRuntime,
    _AgentHarness,
    _frozen_plan,
    _matching_observation,
)

EXPECTED_TOKENS = [9764, 393, 45, 283, 220, 24, 22, 853]


class _EpochHarness(_AgentHarness):
    """Fake agent whose runtime replays the known-good reference tokens."""

    def __init__(self):
        super().__init__(FakeRuntime(tokens=EXPECTED_TOKENS))


@pytest.fixture()
def epoch_agent():
    harness = _EpochHarness()
    yield harness
    harness._listener.close()


class _LocalRealizer:
    """Bypass planner-dependent strategy bodies: realize the fake plan."""

    def __init__(self, harness):
        self.harness = harness

    def __call__(self, execution_plan):
        runtime = self.harness._build(execution_plan)
        from freetoken.research.r5a_serving import RealizedStaticPlan

        return RealizedStaticPlan(
            runtime=runtime, observation=dict(runtime.observation)
        )


class _StepStrategy:
    """Minimal transition strategy mirroring Qwen's committed-token boundary."""

    def safe_boundary(self, *, session, trigger):
        return {"safe": True}

    def replay_input(self, *, session):
        return list(session.prompt_token_ids) + list(session.committed_token_ids)

    def recovery_contract(self, *, session, trigger):
        return {
            "continuation_legal": True,
            "restart_replay_legal": True,
            "source": "committed-token replay",
        }

    def next_token(self, result):
        tokens = result.get("generated_token_ids", [])
        assert len(tokens) == 1
        return int(tokens[0])


def _fake_environment(producer_sha: str) -> dict:
    from freetoken.research.r3_planner import freeze

    return freeze(
        {
            "schema": "inferswarm.r5a.frozen-environment/1",
            "implementation_commit": producer_sha,
            "runtime_context": "test-runtime-context",
            "network_context": "test-network-context",
            "node_a": {
                "node_id": "node.inferswarm01",
                "gpus": [
                    {"uuid": "GPU-a0", "pci_bdf": "00000000:02:00.0",
                     "vram_total_bytes": 12 * 1024**3, "availability": "AVAILABLE",
                     "integrity_eligible": True},
                    {"uuid": "GPU-a1", "pci_bdf": "00000000:03:00.0",
                     "vram_total_bytes": 12 * 1024**3, "availability": "UNAVAILABLE",
                     "integrity_eligible": True},
                ],
            },
            "node_b": {
                "node_id": "node.inferswarm03",
                "gpus": [
                    {"uuid": "GPU-b0", "pci_bdf": "00000000:01:00.0",
                     "vram_total_bytes": 12 * 1024**3, "availability": "AVAILABLE",
                     "integrity_eligible": True},
                ],
            },
            "network": {
                "link_id": "link.node-a-to-node-b.tcp",
                "negotiated_mbps": 1000,
                "available": True,
            },
        }
    )


def _snapshot_from(environment: dict, *, gpu_a1_available: bool) -> dict:
    from benchmarks.inferswarm_r5b.strategy import snapshot_with_availability

    return snapshot_with_availability(environment, gpu_a1_available=gpu_a1_available)


TEST_COMMIT = "test" * 10


def _test_problem() -> dict:
    from freetoken.research.r3_planner import freeze

    return freeze(
        {
            "schema": "inferswarm.test.strategy-problem/1",
            "implementation_commit": TEST_COMMIT,
            "strategy": {"id": "test.strategy/1"},
            "model": {"repository": "test/model", "revision": "testrev"},
            "evidence_context": {"model_revision": "testrev"},
            "shapes": [
                {
                    "id": "resident-two-node-two-slot",
                    "slots": [
                        {
                            "id": "slot-a",
                            "allowed_compute_unit_ids": ["gpu.node-a.0"],
                            "required_capabilities": ["freetoken-resident-block-a-v1"],
                            "memory": {"persistent_required_bytes": 1024},
                        },
                        {
                            "id": "slot-b",
                            "allowed_compute_unit_ids": ["gpu.node-b.0"],
                            "required_capabilities": ["freetoken-resident-block-b-v1"],
                            "memory": {"persistent_required_bytes": 1024},
                        },
                    ],
                    "distinct_slot_groups": [["slot-a", "slot-b"]],
                    "paths": [
                        {
                            "id": "strategy-boundary",
                            "from_slot": "slot-a",
                            "to_slot": "slot-b",
                            "required_capabilities": ["freetoken-static-boundary-v1"],
                        }
                    ],
                    "strategy_payload": {"realization": "test"},
                }
            ],
        }
    )


def _test_evidence_catalog() -> dict:
    from freetoken.research.r3_planner import freeze

    return freeze(
        {
            "schema": "inferswarm.test.evidence-catalog/1",
            "records": [
                {
                    "id": "test-ranking-evidence",
                    "role": "RANKING_OBJECTIVE",
                    "producer_identity": TEST_COMMIT,
                    "evidence_identity": "test-evidence",
                    "shape_id": "resident-two-node-two-slot",
                    "mapping": {"slot-a": "gpu.node-a.0", "slot-b": "gpu.node-b.0"},
                    "required_context": {"model_revision": "testrev"},
                    "freshness": "CURRENT",
                    "measurement_status": "MEASURED",
                    "evidence_class": "TEST",
                    "confidence": "EXACT_CONTEXT",
                    "metric": {
                        "name": "ttft_ms",
                        "value": 1.0,
                        "unit": "ms",
                        "statistic": "median",
                    },
                }
            ],
        }
    )


def _test_objective() -> dict:
    from freetoken.research.r3_planner import freeze

    return freeze(
        {
            "schema": "inferswarm.test.objective/1",
            "id": "test",
            "metric": "ttft_ms",
            "direction": "MINIMIZE",
            "unit": "ms",
            "statistic": "median",
        }
    )


def _controller(harness) -> EpochServingController:
    from freetoken.research.r3_planner import freeze

    environment = _fake_environment(TEST_COMMIT)
    problem = _test_problem()
    snapshot = _snapshot_from(environment, gpu_a1_available=False)
    policy = freeze(
        {"schema": "inferswarm.test.operator-policy/1", "excluded_compute_unit_ids": []}
    )
    objective = _test_objective()
    evidence = _test_evidence_catalog()

    def compiler(evaluation):
        return _frozen_plan()

    return EpochServingController(
        problem=problem,
        initial_snapshot=snapshot,
        policy=policy,
        objective=objective,
        evidence_catalog=evidence,
        compiler=compiler,
        realizer=_LocalRealizer(harness),
        transition_strategy=_StepStrategy(),
        transition_policy=lambda old, new, event: {"authorize": False},
    )


class TestExternalCoordinatorEpochFlow:
    def test_serve_tokens_commits_reference_exactly(self, epoch_agent):
        controller = _controller(epoch_agent)
        completed = controller.serve_tokens(
            session_id=1,
            prompt_token_ids=[9764, 393, 45],
            max_new_tokens=len(EXPECTED_TOKENS),
            sampling_inputs={"temperature": 0.0},
        )
        assert completed["generated_token_ids"] == EXPECTED_TOKENS
        assert len(completed["committed_epoch_ids"]) == len(EXPECTED_TOKENS)
        assert len(set(completed["committed_epoch_ids"])) == 1
        controller.close()

    def test_controller_report_retains_attribution(self, epoch_agent):
        controller = _controller(epoch_agent)
        controller.serve_tokens(
            session_id=1,
            prompt_token_ids=[1, 2, 3],
            max_new_tokens=4,
            sampling_inputs={"temperature": 0.0},
        )
        report = controller.report()
        assert report["single_mutable_authority"] is True
        # The top-level active realization reflects the initial generation-0
        # activation, not only a later replacement.
        assert report["active_realization_id"] == (
            report["epochs"][0]["realization_authorization"]["realization_id"]
        )
        session = report["sessions"][0]
        assert session["generated_token_ids"] == EXPECTED_TOKENS[:4]
        assert all(session["committed_epoch_ids"])
        controller.close()

    def test_stale_duplicate_result_fenced(self, epoch_agent):
        # Controlled negative arm mirroring the accepted R5B seam: after a
        # token commits, inject a duplicate/stale result (already-committed
        # position, old epoch id) through the same acceptance path and prove
        # mechanical rejection without mutating the commit ledger.
        controller = _controller(epoch_agent)
        injections: list[bool] = []

        def after_commit(step, ctrl):
            if step == 0:
                ledger = ctrl._sessions[1]
                injections.append(
                    ctrl.inject_late_result(
                        epoch_id="research-generation-0:deadbeefcafe",
                        plan_digest=ctrl.active_epoch.execution_plan["digest"],
                        session=ledger,
                        position=0,  # duplicate of the just-committed position
                        token_id=999,
                    )
                )
                injections.append(
                    ctrl.inject_late_result(
                        epoch_id=ctrl.active_epoch.epoch_id,
                        plan_digest=ctrl.active_epoch.execution_plan["digest"],
                        session=ledger,
                        position=0,  # duplicate position, current epoch
                        token_id=999,
                    )
                )

        completed = controller.serve_tokens(
            session_id=1,
            prompt_token_ids=[1, 2, 3],
            max_new_tokens=4,
            sampling_inputs={"temperature": 0.0},
            after_commit=after_commit,
        )
        assert injections == [False, False], "stale/duplicate results were accepted"
        assert completed["generated_token_ids"] == EXPECTED_TOKENS[:4]
        report = controller.report()
        assert len(report["late_result_rejections"]) == 2
        reasons = {item["reason"] for item in report["late_result_rejections"]}
        assert "RETIRED_OR_SUPERSEDED_EPOCH" in reasons
        assert "NON_NEXT_COMMIT_POSITION" in reasons
        assert report["sessions"][0]["generated_token_ids"] == EXPECTED_TOKENS[:4]
        controller.close()

    def test_remote_realizer_used_when_provided(self, epoch_agent):
        """The same controller accepts the remote realizer without change."""
        from freetoken.research.r3_planner import freeze

        realizer = make_remote_realizer(
            host="127.0.0.1", port=epoch_agent.port, scope_id=SCOPE
        )
        controller = EpochServingController(
            problem=_test_problem(),
            initial_snapshot=_snapshot_from(
                _fake_environment(TEST_COMMIT), gpu_a1_available=False
            ),
            policy=freeze({"schema": "t/2"}),
            objective=_test_objective(),
            evidence_catalog=_test_evidence_catalog(),
            compiler=lambda evaluation: _frozen_plan(),
            realizer=realizer,
            transition_strategy=_StepStrategy(),
            transition_policy=lambda old, new, event: {"authorize": False},
        )
        completed = controller.serve_tokens(
            session_id=1,
            prompt_token_ids=[5, 6],
            max_new_tokens=2,
            sampling_inputs={"temperature": 0.0},
        )
        assert completed["generated_token_ids"] == EXPECTED_TOKENS[:2]
        controller.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
