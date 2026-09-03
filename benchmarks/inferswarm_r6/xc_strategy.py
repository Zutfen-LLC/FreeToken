"""R6 Gemma strategy adapter for the external-Coordinator serving path.

Mirrors the accepted R5B/#67 strategy contract (planning problem, operator
policy, transition strategy, candidate compilation) with every Qwen/expert
concept replaced by the dense-Gemma census-driven candidates.  The generic
planner, epoch controller, and xc wire consume this unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

try:
    from benchmarks.inferswarm_r6.strategy import (
        MODEL_REPOSITORY,
        MODEL_REVISION,
        STRATEGY_ID,
    )
except ModuleNotFoundError:
    from inferswarm_r6.strategy import (  # type: ignore
        MODEL_REPOSITORY,
        MODEL_REVISION,
        STRATEGY_ID,
    )

# Census-derived frozen stage weights (measured on the checkpoint, producer
# 2c8e381+; see docs/inferswarm_r6/METHODOLOGY.md).  The CPU-only
# Coordinator must not need checkpoint bytes to plan.
STAGE_WEIGHT_BYTES = [9256814624, 7278939168, 7278946848 + 2013731840]
THREE_STAGE_SHAPE = "resident-two-node-three-slot"

from freetoken.research.r3_planner import freeze

MODEL_PATH_DEFAULT = "/srv/models/gemma-r6"


class GemmaTokenBoundaryStrategy:
    """Dense Gemma recovery contract: replay-prefill reconstructs all
    block-local KV state; the host committed-output ledger is the only
    state that survives placement changes."""

    def safe_boundary(self, *, session, trigger):
        return {
            "safe": True,
            "kind": "gemma-generated-token-commit-boundary",
            "strategy_id": STRATEGY_ID,
            "committed_position": session.committed_position,
            "old_epoch_mutations_settled": True,
            "scope": "this pinned deterministic dense Gemma strategy only",
        }

    def replay_input(self, *, session):
        return list(session.prompt_token_ids) + list(session.committed_token_ids)

    def recovery_contract(self, *, session, trigger):
        trusted = bool(trigger.get("trusted_history_available", True))
        deterministic = session.sampling_inputs.get("temperature", 0) == 0
        legal = trusted and deterministic
        return {
            "schema": "inferswarm.r6.gemma-recovery-contract/1",
            "strategy_id": STRATEGY_ID,
            "safe_boundary": "committed generated-token boundary",
            "mutable_state": [
                "stage-local full-attention KV cache",
                "stage-local sliding-window KV cache",
                "host committed-output authority ledger",
            ],
            "placement_change_survival": ["host committed-output authority ledger"],
            "reconstructible_state": [
                "stage-local full-attention KV cache",
                "stage-local sliding-window KV cache",
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
        if len(tokens) < 1:
            raise RuntimeError("Gemma replay step produced no token")
        # The committed token is the step-0 (prefill-final) token; the
        # accepted controller discards the speculative step-1 output.
        return int(tokens[0])


def planning_problem(implementation_commit: str) -> dict[str, Any]:
    """Legal shapes from frozen census constants (no checkpoint access)."""
    return freeze(
        {
            "schema": "inferswarm.r6.gemma-strategy-problem/1",
            "implementation_commit": implementation_commit,
            "strategy": {"id": STRATEGY_ID},
            "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
            "evidence_context": {"model_revision": MODEL_REVISION},
            "shapes": [
                {
                    "id": THREE_STAGE_SHAPE,
                    "slots": [
                        {
                            "id": "slot-stage-1",
                            "allowed_compute_unit_ids": ["gpu.node-a.0"],
                            "required_capabilities": [
                                "freetoken-resident-stage-first-v1"
                            ],
                            "memory": {
                                "persistent_required_bytes": STAGE_WEIGHT_BYTES[0]
                            },
                        },
                        {
                            "id": "slot-stage-2",
                            "allowed_compute_unit_ids": ["gpu.node-a.1"],
                            "required_capabilities": [
                                "freetoken-resident-stage-middle-v1"
                            ],
                            "memory": {
                                "persistent_required_bytes": STAGE_WEIGHT_BYTES[1]
                            },
                        },
                        {
                            "id": "slot-stage-3",
                            "allowed_compute_unit_ids": ["gpu.node-b.0"],
                            "required_capabilities": [
                                "freetoken-resident-stage-last-v1"
                            ],
                            "memory": {
                                "persistent_required_bytes": STAGE_WEIGHT_BYTES[2]
                            },
                        },
                    ],
                    "distinct_slot_groups": [
                        ["slot-stage-1", "slot-stage-2", "slot-stage-3"]
                    ],
                    "paths": [
                        {
                            "id": "strategy-boundary-1",
                            "from_slot": "slot-stage-1",
                            "to_slot": "slot-stage-2",
                            "required_capabilities": [
                                "freetoken-static-boundary-v1"
                            ],
                        },
                        {
                            "id": "strategy-boundary-2",
                            "from_slot": "slot-stage-2",
                            "to_slot": "slot-stage-3",
                            "required_capabilities": [
                                "freetoken-static-boundary-v1"
                            ],
                        },
                    ],
                    "strategy_payload": {
                        "realization": "r6-dense-three-stage-chain",
                    },
                }
            ],
        }
    )


def operator_policy(implementation_commit: str) -> dict[str, Any]:
    return freeze(
        {
            "schema": "inferswarm.r6.operator-policy/1",
            "implementation_commit": implementation_commit,
            "excluded_compute_unit_ids": [],
            "reservations_bytes": {},
            "integrity_policy": "quarantined-resources-cannot-participate",
            "static_plan_only": False,
            "epoch_reconfiguration": True,
        }
    )


_NODE_BY_UNIT = {
    "gpu.node-a.0": "node.inferswarm01",
    "gpu.node-a.1": "node.inferswarm01",
    "gpu.node-b.0": "node.inferswarm03",
}


def compile_candidate(evaluation, *, chain_plan: dict) -> dict[str, Any]:
    """Compile the evaluated candidate into the complete dense plan body.

    ``chain_plan`` is the producer-bound 3-stage participant plan (the
    same artifact the stage services verify); the compiled body carries
    its digest so realization cannot silently substitute it.
    """
    mapping = evaluation["mapping"]
    units = list(dict.fromkeys(mapping.values()))
    participants = list(dict.fromkeys(_NODE_BY_UNIT[u] for u in units))
    state_placement = []
    authority = []
    representations = []
    stage_specs = [
        ("stage-1", "slot-stage-1", "embeddings+layers[0,16)"),
        ("stage-2", "slot-stage-2", "layers[16,32)"),
        ("stage-3", "slot-stage-3", "layers[32,48)+norm+tied-head"),
    ]
    for stage_id, slot, description in stage_specs:
        unit = mapping[slot]
        state_placement.extend(
            [
                {
                    "logical_state_id": f"state.{stage_id}.immutable",
                    "compute_unit_id": unit,
                    "role": "required_residency",
                },
                {
                    "logical_state_id": f"state.{stage_id}.mutable-kv",
                    "compute_unit_id": unit,
                    "role": "mutable_authority",
                },
            ]
        )
        authority.append(
            {
                "logical_state_id": f"state.{stage_id}.mutable-kv",
                "participant": _NODE_BY_UNIT[unit],
                "lineage": "r6-dense-replay-prefill",
            }
        )
        representations.append(
            {
                "logical_state_id": f"state.{stage_id}.immutable",
                "representation": "freetoken-dense-selective-bf16",
            }
        )
    geometry = chain_plan.get("boundary_geometry", {})
    boundaries = []
    for boundary_id, (from_slot, to_slot) in enumerate(
        [("slot-stage-1", "slot-stage-2"), ("slot-stage-2", "slot-stage-3")]
    ):
        boundaries.append(
            {
                "id": f"boundary.{boundary_id}",
                "from_compute_unit_id": mapping[from_slot],
                "to_compute_unit_id": mapping[to_slot],
                "semantic_contract": {
                    "dtype": geometry.get("dtype", "bfloat16"),
                    "layout": "plane-major-contiguous",
                    "planes": geometry.get("planes", 1),
                    "row_width": geometry.get("row_width", 3840),
                    "decode_bytes": geometry.get("decode_bytes"),
                    "prefill_chunk_rows": geometry.get("prefill_chunk_rows"),
                    "prefill_bytes": geometry.get("prefill_bytes"),
                },
            }
        )
    accounting = deepcopy(evaluation.get("memory_accounting", {}))
    accounting["strategy_host_lifecycle"] = {
        "dense_stream_staging": "no persistent host mirror; per-tensor "
        "streamed device-ward loads released after each stage realizes",
        "declared_shared_state_bytes": (
            chain_plan.get("declared_shared_state", {}).get("bytes")
        ),
        "release_after_final_residency": True,
    }
    return {
        "strategy_identity": {"id": STRATEGY_ID},
        "model_identity": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "participants": participants,
        "compute_units": units,
        "representations": representations,
        "backend_choices": [
            {"participant": p, "backend": "freetoken-dense-stage-native"}
            for p in participants
        ],
        "state_placement": state_placement,
        "state_authority": authority,
        "semantic_boundaries": boundaries,
        "expected_resource_accounting": accounting,
        "strategy_realization": {
            "realization": "r6-dense-three-stage-chain",
            "participant_plan_digest": chain_plan["digest"],
        },
    }


__all__ = [
    "GemmaTokenBoundaryStrategy",
    "MODEL_REPOSITORY",
    "MODEL_REVISION",
    "STRATEGY_ID",
    "compile_candidate",
    "operator_policy",
    "planning_problem",
]
