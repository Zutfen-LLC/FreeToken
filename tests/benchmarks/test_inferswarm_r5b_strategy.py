from __future__ import annotations

from copy import deepcopy

from inferswarm_r5b.requalify_evidence import requalify
from inferswarm_r5b.runtime import EpochIsolatedR4Runtime
from inferswarm_r5b.strategy import (
    QwenTokenBoundaryStrategy,
    policy_evaluator,
    transition_policy,
)
from freetoken.research.r3_planner import freeze
from freetoken.research.r5b_epochs import SessionLedger


def test_qwen_contract_replays_exact_committed_history_and_fails_without_it():
    strategy = QwenTokenBoundaryStrategy()
    session = SessionLedger(1, [10, 20], 3, {"temperature": 0})
    session.committed_token_ids.extend([30, 40])
    assert strategy.replay_input(session=session) == [10, 20, 30, 40]
    safe = strategy.safe_boundary(session=session, trigger={})
    assert safe["safe"] and safe["committed_position"] == 2
    assert strategy.recovery_contract(session=session, trigger={})[
        "continuation_legal"
    ]
    assert not strategy.recovery_contract(
        session=session, trigger={"trusted_history_available": False}
    )["continuation_legal"]


def test_frozen_policy_uses_economics_for_optimization_and_feasibility_for_loss():
    policy = transition_policy("a" * 40)
    evaluate = policy_evaluator(policy)
    old = {"candidate_id": "resident-two-node-two-slot[a=b]"}
    new = {"candidate_id": "resident-same-node-two-slot[a=b]"}
    optimization = evaluate(old, new, {"expected_remaining_requests": 64})
    assert optimization["authorize"]
    assert optimization["overlap_preparation"] is False
    recovery = evaluate(
        new,
        old,
        {"active_plan_executable": False, "expected_remaining_requests": 0},
    )
    assert recovery["authorize"] and recovery["failure_recovery"]


def test_r5a_numeric_evidence_is_requalified_without_rewriting_measurement():
    source = freeze(
        {
            "schema": "source/1",
            "implementation_commit": "old",
            "records": [
                {
                    "id": "r5a-value",
                    "producer_identity": "old",
                    "required_context": {
                        "runtime_context": "old-runtime",
                        "network_context": "old-link",
                    },
                    "evidence_class": "MEASURED_R5A_MATCHED_HTTP_SERVING",
                    "freshness": "CURRENT",
                    "confidence": "EXACT_CONTEXT",
                    "metric": {"value": 373.6170495},
                    "provenance": {"artifact": "accepted"},
                }
            ],
        }
    )
    environment = freeze(
        {
            "schema": "environment/1",
            "implementation_commit": "new",
            "runtime_context": "new-runtime",
            "network_context": "new-link",
        }
    )
    result = requalify(
        deepcopy(source),
        environment,
        producer_sha="new",
        regression_evidence="test-summary.json",
    )
    record = result["records"][0]
    assert record["metric"]["value"] == 373.6170495
    assert record["provenance"]["numeric_value_remeasured"] is False
    assert record["required_context"]["runtime_context"] == "new-runtime"


def test_isolated_r4_facade_preserves_generated_boundary_callbacks():
    runtime = object.__new__(EpochIsolatedR4Runtime)
    runtime._closed = False
    runtime._rpc = lambda operation, **payload: {
        "result": {
            "generated_token_ids": [11, 12],
            "boundaries": [
                {"generated_step": 0, "checksum": "a"},
                {"generated_step": 1, "checksum": "b"},
            ],
        }
    }
    seen = []
    result = runtime.generate(
        session_id=1,
        prompt_token_ids=[7],
        max_new_tokens=2,
        on_token=lambda step, token, boundary: seen.append(
            (step, token, boundary["checksum"])
        ),
    )
    assert result["generated_token_ids"] == [11, 12]
    assert seen == [(0, 11, "a"), (1, 12, "b")]
