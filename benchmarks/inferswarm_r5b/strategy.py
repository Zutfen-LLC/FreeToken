"""Pinned Qwen transition semantics and bounded R5B research policy."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

try:
    from benchmarks.inferswarm_r5a.strategy import (
        GPU_A_SECONDARY,
        LOCAL_SPLIT_SHAPE,
        NETWORK_SHAPE,
        compile_candidate as compile_r5a_candidate,
        operator_policy as r5a_operator_policy,
        planning_problem as r5a_planning_problem,
        resource_snapshot,
    )
except ModuleNotFoundError:
    from inferswarm_r5a.strategy import (
        GPU_A_SECONDARY,
        LOCAL_SPLIT_SHAPE,
        NETWORK_SHAPE,
        compile_candidate as compile_r5a_candidate,
        operator_policy as r5a_operator_policy,
        planning_problem as r5a_planning_problem,
        resource_snapshot,
    )
from freetoken.research.r3_planner import freeze

STRATEGY_ID = "freetoken.qwen36-epoch-coarse-serving/1"


class QwenTokenBoundaryStrategy:
    """The proving model's explicit recovery contract; not a universal rule."""

    def safe_boundary(self, *, session, trigger):
        return {
            "safe": True,
            "kind": "qwen-generated-token-commit-boundary",
            "strategy_id": STRATEGY_ID,
            "committed_position": session.committed_position,
            "old_epoch_mutations_settled": True,
            "scope": "this pinned deterministic Qwen strategy only",
        }

    def replay_input(self, *, session):
        # The accepted R2/R4 runtimes reconstruct their block-local KV/recurrent
        # state by prefilling the exact prompt plus already committed output.
        return list(session.prompt_token_ids) + list(session.committed_token_ids)

    def recovery_contract(self, *, session, trigger):
        trusted = bool(trigger.get("trusted_history_available", True))
        deterministic = session.sampling_inputs.get("temperature", 0) == 0
        legal = trusted and deterministic
        return {
            "schema": "inferswarm.r5b.qwen-recovery-contract/1",
            "strategy_id": STRATEGY_ID,
            "safe_boundary": "committed generated-token boundary",
            "mutable_state": [
                "block-local KV cache",
                "block-local linear-recurrent state",
                "host committed-output authority ledger",
            ],
            "placement_change_survival": ["host committed-output authority ledger"],
            "reconstructible_state": [
                "block-local KV cache",
                "block-local linear-recurrent state",
            ],
            "trusted_inputs": [
                "exact original prompt token IDs",
                "exact committed generated token IDs",
                "committed position",
                "deterministic sampling inputs",
                "model revision and plan identity",
            ],
            "source": (
                "retained exact prompt plus committed generated-token history"
                if legal
                else None
            ),
            "replay_output_emitted": False,
            "continuation_legal": legal,
            "restart_replay_legal": legal,
            "reason": (
                "trusted deterministic replay through latest committed boundary"
                if legal
                else "trusted deterministic reconstruction inputs unavailable"
            ),
        }

    def next_token(self, result):
        tokens = result.get("generated_token_ids", [])
        if len(tokens) != 1:
            raise RuntimeError("Qwen replay step did not produce exactly one token")
        return int(tokens[0])


def planning_problem(implementation_commit: str) -> dict[str, Any]:
    value = deepcopy(r5a_planning_problem(implementation_commit))
    value.pop("digest", None)
    value["schema"] = "inferswarm.r5b.qwen-strategy-problem/1"
    value["strategy"] = {"id": STRATEGY_ID}
    return freeze(value)


def operator_policy(implementation_commit: str) -> dict[str, Any]:
    value = deepcopy(r5a_operator_policy(implementation_commit))
    value.pop("digest", None)
    value["schema"] = "inferswarm.r5b.operator-policy/1"
    value["static_plan_only"] = False
    value["epoch_reconfiguration"] = True
    return freeze(value)


def compile_candidate(evaluation, *, r4_plan, local_plan):
    compiled = compile_r5a_candidate(
        dict(evaluation), r4_plan=r4_plan, local_plan=local_plan
    )
    compiled["strategy_identity"] = {
        "id": STRATEGY_ID,
        "model_specific": True,
        "public_api": False,
    }
    compiled.setdefault("strategy_realization", {})["transition_contract"] = {
        "safe_boundary": "qwen-generated-token-commit-boundary",
        "mutable_runtime_state": ["block-local-kv", "block-local-linear-recurrent"],
        "recovery": "replay-exact-prompt-plus-committed-output",
        "replay_emission": "suppressed",
        "fail_without_trusted_history": True,
    }
    return compiled


def snapshot_with_availability(
    environment: dict[str, Any], *, gpu_a1_available: bool, gpu_b0_available: bool = True
):
    current = deepcopy(environment)
    current["node_a"]["gpus"][1]["availability"] = (
        "AVAILABLE" if gpu_a1_available else "UNAVAILABLE"
    )
    current["node_b"]["gpus"][0]["availability"] = (
        "AVAILABLE" if gpu_b0_available else "UNAVAILABLE"
    )
    snapshot = deepcopy(resource_snapshot(current))
    snapshot.pop("digest", None)
    snapshot["schema"] = "inferswarm.r5b.resource-evidence-snapshot/1"
    return freeze(snapshot)


def transition_policy(implementation_commit: str) -> dict[str, Any]:
    """Predeclared bounded economics; values are never tuned from R5B results."""
    return freeze(
        {
            "schema": "inferswarm.r5b.transition-policy/1",
            "implementation_commit": implementation_commit,
            "correctness_and_feasibility_first": True,
            "operator_policy": "automatic within this bounded physical campaign",
            "objective": "minimize applicable median TTFT",
            "candidate_median_ttft_ms": {
                LOCAL_SPLIT_SHAPE: 373.6170495,
                NETWORK_SHAPE: 1877.4567285,
                "source-backed-single-resource": 2630.5871575,
            },
            "preparation_cost_estimate_seconds": {
                LOCAL_SPLIT_SHAPE: 68.131071624,
                NETWORK_SHAPE: 47.826231041,
            },
            "transition_interruption_budget_seconds": 90.0,
            "minimum_expected_remaining_requests": 64,
            "minimum_resource_stability_confidence": 0.9,
            "required_resource_stability_confidence": 1.0,
            "full_overlap_physically_feasible": False,
            "reason": (
                "R2 and R4 both require GPU A0; retain immutable preparation and use "
                "a strategy-safe interrupted/cold cutover rather than claim overlap"
            ),
        }
    )


def policy_evaluator(policy: dict[str, Any]):
    def shape(plan):
        return str(plan["candidate_id"]).split("[", 1)[0]

    def evaluate(old_plan, replacement_plan, event):
        old_shape = shape(old_plan)
        new_shape = shape(replacement_plan)
        failure = event.get("active_plan_executable", True) is False
        values = policy["candidate_median_ttft_ms"]
        expected_gain_ms = values[old_shape] - values[new_shape]
        requests = int(event.get("expected_remaining_requests", 64))
        aggregate_gain_seconds = expected_gain_ms * requests / 1000
        preparation_seconds = policy["preparation_cost_estimate_seconds"][new_shape]
        stability = float(event.get("resource_stability_confidence", 1.0))
        economically_positive = aggregate_gain_seconds > preparation_seconds
        authorize = failure or (
            expected_gain_ms > 0
            and requests >= policy["minimum_expected_remaining_requests"]
            and stability >= policy["minimum_resource_stability_confidence"]
            and economically_positive
        )
        return {
            "authorize": authorize,
            "reason": (
                "required-resource loss: choose best correct feasible survivor"
                if failure
                else (
                    "expected aggregate objective improvement exceeds preparation cost"
                    if authorize
                    else "bounded economics/stability threshold not satisfied"
                )
            ),
            "failure_recovery": failure,
            "expected_ttft_improvement_ms_per_request": expected_gain_ms,
            "expected_remaining_requests": requests,
            "expected_aggregate_improvement_seconds": aggregate_gain_seconds,
            "preparation_cost_estimate_seconds": preparation_seconds,
            "resource_stability_confidence": stability,
            "overlap_preparation": policy["full_overlap_physically_feasible"],
            "policy_digest": policy["digest"],
        }

    return evaluate


def validate_gpu_a1_event(event: dict[str, Any], environment: dict[str, Any]) -> None:
    expected = environment["node_a"]["gpus"][1]
    if event.get("resource_id") != GPU_A_SECONDARY:
        raise ValueError("R5B control seam event names an unexpected resource")
    observed = event.get("observed_identity", {})
    for key in ("uuid", "pci_bdf", "vram_total_bytes"):
        if observed.get(key) != expected.get(key):
            raise ValueError(f"R5B resource event {key} does not match frozen GPU A1")
    if observed.get("integrity_eligible") is not True:
        raise ValueError("R5B resource event is not integrity eligible")
    if observed.get("representation_backend_compatible") is not True:
        raise ValueError("R5B resource event lacks representation/backend compatibility")


__all__ = [
    "QwenTokenBoundaryStrategy",
    "STRATEGY_ID",
    "compile_candidate",
    "operator_policy",
    "planning_problem",
    "policy_evaluator",
    "snapshot_with_availability",
    "transition_policy",
    "validate_gpu_a1_event",
]
