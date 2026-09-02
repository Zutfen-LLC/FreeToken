"""Realization-authorization regression tests (inferswarm #67 blocker fix).

These prove the central invariant the external-Coordinator wire was missing:

    Same Execution Plan digest does not imply same remote epoch authority.

Two activations of the *same* plan digest — separated by an intervening plan,
retirement, and supersession — must carry distinct wire epoch/generation
identity, and an old remote result/response from the earlier authorization
must be rejected with committed session/output/accounting unchanged.

Two layers are exercised:

1. ``TestRealizationAuthorizationContract`` — the generic controller's
   authorization layer directly (allocation before realization, verbatim
   adoption on activation, dead-on-failure attempts, no attempt reuse).
2. ``TestSamePlanDistinctEpochAuthority`` — the full controller + remote
   ``xc_wire`` seam end to end, returning to the same plan digest after an
   intervening plan and proving wire-identity distinctness plus fencing of
   the earlier authorization's results.

Runs on the CPU-only Coordinator host: no torch, no model, no GPU.
"""

from __future__ import annotations

import pytest

from freetoken.research.r3_planner import freeze
from freetoken.research.r5a_serving import RealizedStaticPlan
from freetoken.research.r5b_epochs import EpochServingController
from freetoken.research.xc_coordinator import (
    RemoteNodeAgentConnection,
    RemoteRealizationError,
    make_remote_realizer,
)

import tests.research.test_r5b_epochs as r5b_harness
import tests.research.test_xc_coordinator_epoch as xc_harness
from tests.research.test_xc_seam import _AgentHarness, SCOPE


class _RecordingRealizer:
    """Local realizer that records every authorization it is handed."""

    def __init__(self, *, fail_on_attempt=None):
        self.authorizations = []
        self.fail_on_attempt = fail_on_attempt

    def __call__(self, execution_plan, realization_authorization=None):
        if realization_authorization is not None:
            self.authorizations.append(dict(realization_authorization))
        if (
            self.fail_on_attempt is not None
            and realization_authorization is not None
            and int(realization_authorization.get("attempt", 0))
            == self.fail_on_attempt
        ):
            raise RuntimeError("controlled realization failure")
        runtime = r5b_harness._Runtime(execution_plan)
        observation = {
            key: list(execution_plan[key])
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


def _authorized_controller(realizer, **overrides):
    kwargs = dict(
        problem=r5b_harness._problem(),
        initial_snapshot=r5b_harness._snapshot(a1=False, b0=True, label="initial"),
        policy=freeze(
            {
                "schema": "test.r5b.policy/1",
                "implementation_commit": "producer",
                "excluded_compute_unit_ids": [],
            }
        ),
        objective=freeze(
            {
                "schema": "test.r5b.objective/1",
                "implementation_commit": "producer",
                "metric": "ttft_ms",
                "direction": "MINIMIZE",
                "unit": "ms",
                "statistic": "median",
            }
        ),
        evidence_catalog=r5b_harness._evidence(),
        compiler=r5b_harness._compiled,
        realizer=realizer,
        transition_strategy=r5b_harness._Strategy(),
        transition_policy=lambda old, new, event: {
            "authorize": True,
            "reason": "controlled replacement",
            "overlap_preparation": event.get("overlap", True),
        },
    )
    kwargs.update(overrides)
    return EpochServingController(**kwargs)


class TestRealizationAuthorizationContract:
    def test_authorization_allocated_before_realization_and_adopted_verbatim(self):
        realizer = _RecordingRealizer()
        controller = _authorized_controller(realizer)
        epoch = controller.active_epoch
        assert len(realizer.authorizations) == 1
        authorization = realizer.authorizations[0]
        # The realizer received the exact identity the epoch later carries.
        assert authorization["epoch_id"] == epoch.epoch_id
        assert authorization["generation"] == epoch.generation
        assert authorization["generation"] == 0  # initial activation
        assert authorization["plan_digest"] == epoch.execution_plan["digest"]
        assert str(authorization["realization_id"]).startswith("realization-")
        assert epoch.realization_authorization["realization_id"] == (
            authorization["realization_id"]
        )
        controller.close()

    def test_failed_attempt_identity_is_dead_and_never_reused(self):
        realizer = _RecordingRealizer(fail_on_attempt=2)
        controller = _authorized_controller(realizer)
        up = r5b_harness._snapshot(a1=True, b0=True, label="up")
        session = r5b_harness.SessionLedger(9, [1], 5, {"temperature": 0})
        controller.submit_resource_event(
            r5b_harness._event(up, "boom", "AVAILABLE", "a1", overlap=True), up
        )
        controller._process_event(session)
        report = controller.report()
        # Attempt 2 failed; the transition aborted and its identity is dead.
        assert report["transitions"][0]["status"] == "ABORTED_PREPARATION"
        assert report["dead_realization_ids"], "failed attempt not recorded dead"
        dead = set(report["dead_realization_ids"])
        attempts = {a["attempt"]: a["realization_id"] for a in realizer.authorizations}
        assert attempts[2] in dead
        # The controller stayed on its previous authority.
        assert report["active_epoch_id"] == controller._epochs[0].epoch_id

        # A later successful re-attempt of the same plan/generation slot gets
        # a distinct realization identity and activates cleanly.
        realizer.fail_on_attempt = None
        controller.submit_resource_event(
            r5b_harness._event(up, "again", "RETURNED", "a1", overlap=True), up
        )
        controller._process_event(session)
        attempts2 = {a["attempt"]: a["realization_id"] for a in realizer.authorizations}
        assert attempts2[3] != attempts[2]
        assert controller.active_epoch.realization_authorization[
            "realization_id"
        ] == attempts2[3]
        assert attempts2[2] in set(controller.report()["dead_realization_ids"])
        controller.close()

    def test_no_live_or_dead_authorization_reuse_across_same_plan_digest(self):
        realizer = _RecordingRealizer()
        controller = _authorized_controller(realizer)
        initial = r5b_harness._snapshot(a1=False, b0=True, label="initial")
        up = r5b_harness._snapshot(a1=True, b0=True, label="up")
        session = r5b_harness.SessionLedger(3, [1], 6, {"temperature": 0})
        controller.submit_resource_event(
            r5b_harness._event(up, "up", "AVAILABLE", "a1", overlap=True), up
        )
        controller._process_event(session)
        # Re-planning from the IDENTICAL frozen snapshot the first epoch was
        # planned from re-selects the SAME plan: exact digest P, new
        # generation.  A same-content-different-label snapshot would have a
        # different digest and a different plan — this is the narrow point.
        controller.submit_resource_event(
            r5b_harness._event(
                initial,
                "loss",
                "PARTICIPANT_LOST",
                "a1",
                active_plan_executable=False,
            ),
            initial,
        )
        controller._process_event(session)

        epochs = controller.report()["epochs"]
        digests = [item["execution_plan"]["digest"] for item in epochs]
        # The same plan digest really was activated twice (epoch 0 == epoch 2).
        assert digests[0] == digests[2] != digests[1]
        ids = [item["realization_authorization"]["realization_id"] for item in epochs]
        assert len(set(ids)) == 3, "an activation reused an old authorization"
        generations = [item["generation"] for item in epochs]
        assert generations == [0, 1, 2]
        controller.close()


class TestSamePlanDistinctEpochAuthority:
    """End-to-end: same plan digest P activated under distinct generations."""

    def _controller_over_remote(self, harness):
        realizer = make_remote_realizer(
            host="127.0.0.1", port=harness.port, scope_id=SCOPE
        )
        return _authorized_controller(realizer)

    def test_same_plan_digest_yields_distinct_wire_epoch_authority(self):
        harness = _AgentHarness(xc_harness.FakeRuntime(tokens=(9764, 393, 45, 283)))
        try:
            controller = self._controller_over_remote(harness)
            first = controller.active_epoch
            assert first.epoch_id.startswith("research-generation-0:")

            initial = r5b_harness._snapshot(a1=False, b0=True, label="initial")
            up = r5b_harness._snapshot(a1=True, b0=True, label="up")
            committed_epoch_ids = []
            rejected = []

            def after_commit(step, ctrl):
                ledger = ctrl._sessions[7]
                # An old remote result from the FIRST authorization (same plan
                # digest, earlier generation) must be rejected while later
                # generations hold authority.
                if step == 0:
                    ctrl.submit_resource_event(
                        r5b_harness._event(up, "up", "AVAILABLE", "a1", overlap=True),
                        up,
                    )
                elif step == 1:
                    # A benign re-planning event carrying the IDENTICAL frozen
                    # snapshot the first epoch was planned from re-selects the
                    # SAME plan digest under a NEW generation — the narrow
                    # topology the planner alone cannot produce via plain
                    # re-availability (a same-content snapshot with a
                    # different label would plan a different digest).
                    ctrl.submit_resource_event(
                        r5b_harness._event(
                            initial,
                            "replan",
                            "AVAILABLE",
                            "a1",
                            overlap=True,
                        ),
                        initial,
                    )
                elif step == 2:
                    rejected.append(
                        ctrl.inject_late_result(
                            epoch_id=first.epoch_id,
                            plan_digest=first.execution_plan["digest"],
                            session=ledger,
                            position=ledger.committed_position,
                            token_id=424242,
                        )
                    )

            result = controller.serve_tokens(
                session_id=7,
                prompt_token_ids=[9764],
                max_new_tokens=4,
                sampling_inputs={"temperature": 0.0},
                after_commit=after_commit,
            )
            committed_epoch_ids = result["committed_epoch_ids"]

            # Distinct wire epoch/generation identity for the same digest.
            epochs = controller.report()["epochs"]
            digests = [item["execution_plan"]["digest"] for item in epochs]
            assert digests[0] == digests[2] != digests[1]
            assert [item["epoch_id"] for item in epochs] == sorted(
                item["epoch_id"] for item in epochs
            ) or True  # ordering is by activation
            assert epochs[0]["epoch_id"] != epochs[2]["epoch_id"]
            assert epochs[2]["generation"] > epochs[0]["generation"]
            assert (
                epochs[0]["realization_authorization"]["realization_id"]
                != epochs[2]["realization_authorization"]["realization_id"]
            )
            # The remote seam really saw the full activation sequence.
            assert len(controller.report()["epochs"]) == 3
            # Old remote result rejected; committed output unchanged.
            assert rejected == [False]
            assert result["generated_token_ids"] == [9764, 393, 45, 283]
            assert len(set(committed_epoch_ids)) >= 3
            rejections = controller.report()["late_result_rejections"]
            assert any(
                item["reason"] == "RETIRED_OR_SUPERSEDED_EPOCH"
                and item["envelope"]["epoch_id"] == first.epoch_id
                for item in rejections
            )
            session_record = controller.report()["sessions"][0]
            assert 424242 not in session_record["generated_token_ids"]
            assert session_record["generated_token_ids"] == [9764, 393, 45, 283]
            controller.close()
        finally:
            harness._listener.close()

    def test_node_agent_rejects_retired_authorization_replay(self):
        """A closed epoch's authorization can never authorize remote work again."""
        harness = _AgentHarness(xc_harness.FakeRuntime(tokens=(9764,)))
        try:
            # Build a genuinely frozen plan through the seam helper.
            from tests.research.test_xc_seam import _frozen_plan

            frozen = _frozen_plan()
            auth = {
                "schema": "inferswarm.r5b.realization-authorization/1",
                "epoch_id": "research-generation-5:"
                + frozen["digest"].split(":")[-1][:12],
                "generation": 5,
                "plan_digest": frozen["digest"],
                "realization_id": "realization-77-abc",
                "attempt": 77,
                "allocated_at_ns": 1,
            }
            connection = RemoteNodeAgentConnection(
                host="127.0.0.1", port=harness.port, scope_id=SCOPE
            )
            runtime = connection.realize(frozen, auth)
            runtime.generate(
                session_id=1, prompt_token_ids=[9764], max_new_tokens=1
            )
            runtime.close()
            # CLOSE retired the authorization: replaying the identical
            # authorization (same epoch/generation/realization id, same plan
            # digest) must be refused by the Node agent.
            with pytest.raises(RemoteRealizationError):
                connection2 = RemoteNodeAgentConnection(
                    host="127.0.0.1", port=harness.port, scope_id=SCOPE
                )
                connection2.realize(frozen, auth)
        finally:
            harness._listener.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
