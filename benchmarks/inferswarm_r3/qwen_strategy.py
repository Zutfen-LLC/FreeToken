"""Pinned Qwen R3 strategy and compiler; all model semantics stay here."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from freetoken.research.r3_planner import freeze, require_frozen, selected_evaluation

MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
REVISION = "491c2f1ea524c639598bf8fa787a93fed5a6fbce"
GPU_A = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"
GPU_B = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
R2_PLAN_DIGEST = "sha256:6128dd6705d6d692df3d5fc11cc130dba5c010cfff40c0e4c5ec7c19e1b78ff0"


def planning_problem(implementation_commit: str | None = None) -> dict[str, Any]:
    """Expose two legal shapes without choosing a resource or objective winner."""
    return freeze(
        {
            "schema": "inferswarm.r3.qwen-planning-problem/1",
            "implementation_commit": implementation_commit,
            "model": {"repository": MODEL, "revision": REVISION},
            "evidence_context": {"model_revision": REVISION},
            "shapes": [
                {
                    "id": "s0.source-backed-single-offload",
                    "slots": [
                        {
                            "id": "whole-model-slot",
                            "required_capabilities": ["freetoken-offload-v1"],
                            "memory": {
                                "persistent_required_bytes": 10_733_223_936,
                                "persistent_optional_bytes": 0,
                                "transient_peak_bytes": 0,
                            },
                        }
                    ],
                    "materializations": [
                        {
                            "id": "ordinary-offload-source",
                            "memory_kind": "system-ram",
                            "bytes": 18_182_307_840,
                            "lifecycle": "PERSISTENT_REQUIRED",
                        }
                    ],
                    "strategy_payload": {
                        "runtime": "ordinary-source-backed-offload",
                        "baseline_config": "docs/inferswarm_r2/baseline-config.json",
                    },
                },
                {
                    "id": "s1.resident-two-slot-split",
                    "slots": [
                        {
                            "id": "opaque-slot-a",
                            "required_capabilities": ["freetoken-resident-block-v1"],
                            "memory": {
                                "persistent_required_bytes": 10_861_202_432,
                                "persistent_optional_bytes": 0,
                                "transient_peak_bytes": 0,
                            },
                        },
                        {
                            "id": "opaque-slot-b",
                            "required_capabilities": ["freetoken-resident-block-v1"],
                            "memory": {
                                "persistent_required_bytes": 11_170_278_912,
                                "persistent_optional_bytes": 0,
                                "transient_peak_bytes": 0,
                            },
                        },
                    ],
                    "distinct_slot_groups": [["opaque-slot-a", "opaque-slot-b"]],
                    "paths": [
                        {
                            "id": "activation-boundary",
                            "from_slot": "opaque-slot-a",
                            "to_slot": "opaque-slot-b",
                            "required_capabilities": ["registered-host-staging"],
                        }
                    ],
                    "materializations": [
                        {
                            "id": "slot-a-source-staging",
                            "memory_kind": "system-ram",
                            "bytes": 8_636_596_224,
                            "lifecycle": "TRANSIENT_RELEASE_AFTER_FINALIZATION",
                        },
                        {
                            "id": "slot-b-source-staging",
                            "memory_kind": "system-ram",
                            "bytes": 9_545_711_616,
                            "lifecycle": "TRANSIENT_RELEASE_AFTER_FINALIZATION",
                        },
                        {
                            "id": "registered-activation-buffer",
                            "memory_kind": "system-ram",
                            "bytes": 524_288,
                            "lifecycle": "PERSISTENT_REQUIRED",
                        },
                    ],
                    "strategy_payload": {
                        "opaque_unit_geometry": [[0, 19], [19, 40]],
                        "boundary": {
                            "dtype": "bfloat16",
                            "planes": 2,
                            "decode_bytes": 8192,
                            "prefill_chunk_rows": 64,
                            "prefill_bytes": 524288,
                        },
                        "host_staging_policy": "release_after_final_residency",
                        "accepted_r2_plan_digest": R2_PLAN_DIGEST,
                    },
                },
            ],
        }
    )


def compile_selected(
    decision: dict[str, Any], problem: dict[str, Any], input_digests: dict[str, str]
) -> dict[str, Any]:
    """Compile a generic selection to an immutable existing-runtime invocation."""
    require_frozen(problem, "strategy problem")
    evaluation = selected_evaluation(decision)
    shapes = {shape["id"]: shape for shape in problem["shapes"]}
    shape = shapes[evaluation["shape_id"]]
    common = {
        "schema": "inferswarm.r3.compiled-selected-plan/1",
        "implementation_commit": problem.get("implementation_commit"),
        "planner_decision_digest": decision["digest"],
        "input_digests": deepcopy(input_digests),
        "candidate_id": evaluation["id"],
        "mapping": deepcopy(evaluation["mapping"]),
    }
    if shape["id"] == "s0.source-backed-single-offload":
        common.update(
            {
                "realization_path": "freetoken-supported-single-resource-offload",
                "source_lifecycle": "RETAIN_REQUIRED_SOURCE_BACKING",
                "runtime_config_ref": "docs/inferswarm_r2/baseline-config.json",
                "compute_unit_id": evaluation["mapping"]["whole-model-slot"],
            }
        )
    elif shape["id"] == "s1.resident-two-slot-split":
        common.update(
            {
                "realization_path": "inferswarm-r2-frozen-plan-coordinator",
                "source_lifecycle": "RELEASE_AFTER_FINAL_RESIDENCY",
                "r2_frozen_plan_ref": "docs/inferswarm_r2/frozen-plan.json",
                "r2_frozen_plan_digest": R2_PLAN_DIGEST,
                "adapter_data": deepcopy(shape["strategy_payload"]),
            }
        )
    else:
        raise ValueError(f"strategy cannot compile unknown shape {shape['id']!r}")
    return freeze(common)
