"""Pinned Qwen strategy adapter for R5A; generic planning stays model-neutral."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from freetoken.research.r3_planner import freeze

MODEL_REPOSITORY = "nvidia/Qwen3.6-35B-A3B-NVFP4"
MODEL_REVISION = "491c2f1ea524c639598bf8fa787a93fed5a6fbce"
STRATEGY_ID = "freetoken.qwen36-static-coarse-serving/1"
LOCAL_SHAPE = "source-backed-single-resource"
LOCAL_SPLIT_SHAPE = "resident-same-node-two-slot"
NETWORK_SHAPE = "resident-two-node-two-slot"
GPU_A = "gpu.node-a.0"
GPU_A_SECONDARY = "gpu.node-a.1"
GPU_B = "gpu.node-b.0"

PERSISTENT_A = 10_861_202_432
PERSISTENT_B = 11_170_278_912
PERSISTENT_LOCAL = 10_733_223_936
STAGING_A = 8_636_596_224
STAGING_B = 9_545_711_616
ACTIVATION_BUFFER = 524_288


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def planning_problem(implementation_commit: str) -> dict[str, Any]:
    """Expose legal shapes; the generic planner chooses among mappings."""
    common_boundary = {
        "id": "strategy-boundary",
        "from_slot": "slot-a",
        "to_slot": "slot-b",
        "required_capabilities": ["freetoken-static-boundary-v1"],
    }
    return freeze(
        {
            "schema": "inferswarm.r5a.qwen-strategy-problem/1",
            "implementation_commit": implementation_commit,
            "strategy": {"id": STRATEGY_ID},
            "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
            "evidence_context": {"model_revision": MODEL_REVISION},
            "shapes": [
                {
                    "id": LOCAL_SHAPE,
                    "slots": [
                        {
                            "id": "whole-model",
                            "allowed_compute_unit_ids": [GPU_A],
                            "required_capabilities": ["freetoken-source-offload-v1"],
                            "memory": {"persistent_required_bytes": PERSISTENT_LOCAL},
                        }
                    ],
                    "strategy_payload": {"realization": "ordinary-freetoken-offload"},
                },
                {
                    "id": LOCAL_SPLIT_SHAPE,
                    "slots": [
                        {
                            "id": "slot-a",
                            "allowed_compute_unit_ids": [GPU_A],
                            "required_capabilities": ["freetoken-resident-block-a-v1"],
                            "memory": {"persistent_required_bytes": PERSISTENT_A},
                        },
                        {
                            "id": "slot-b",
                            "allowed_compute_unit_ids": [GPU_A_SECONDARY],
                            "required_capabilities": ["freetoken-resident-block-b-v1"],
                            "memory": {"persistent_required_bytes": PERSISTENT_B},
                        },
                    ],
                    "distinct_slot_groups": [["slot-a", "slot-b"]],
                    "paths": [common_boundary],
                    "strategy_payload": {"realization": "accepted-r2-local-split"},
                },
                {
                    "id": NETWORK_SHAPE,
                    "slots": [
                        {
                            "id": "slot-a",
                            "allowed_compute_unit_ids": [GPU_A],
                            "required_capabilities": ["freetoken-resident-block-a-v1"],
                            "memory": {"persistent_required_bytes": PERSISTENT_A},
                        },
                        {
                            "id": "slot-b",
                            "allowed_compute_unit_ids": [GPU_B],
                            "required_capabilities": ["freetoken-resident-block-b-v1"],
                            "memory": {"persistent_required_bytes": PERSISTENT_B},
                        },
                    ],
                    "distinct_slot_groups": [["slot-a", "slot-b"]],
                    "paths": [common_boundary],
                    "strategy_payload": {
                        "realization": "accepted-r4-backend-native-boundary",
                        "semantic_geometry": {
                            "opaque_units": [[0, 19], [19, 40]],
                            "dtype": "bfloat16",
                            "planes": 2,
                            "decode_bytes": 8192,
                            "prefill_chunk_rows": 64,
                            "prefill_bytes": 524288,
                        },
                    },
                },
            ],
        }
    )


def resource_snapshot(environment: dict[str, Any]) -> dict[str, Any]:
    """Normalize a freshly frozen physical environment into planner resources."""
    required = (
        "implementation_commit",
        "runtime_context",
        "network_context",
        "node_a",
        "node_b",
    )
    missing = [key for key in required if key not in environment]
    if missing:
        raise ValueError(f"frozen environment lacks {missing}")

    def unit(unit_id, record, capabilities):
        return {
            "id": unit_id,
            "stable_device_id": record["uuid"],
            "pci_bdf": record["pci_bdf"],
            "memory_resource_id": f"{unit_id}.vram",
            "availability": record.get("availability", "AVAILABLE"),
            "integrity_eligible": record.get("integrity_eligible", True),
            "capabilities": capabilities,
        }

    a = environment["node_a"]
    b = environment["node_b"]
    a0, a1 = a["gpus"][0], a["gpus"][1]
    b0 = b["gpus"][0]
    return freeze(
        {
            "schema": "inferswarm.r5a.resource-evidence-snapshot/1",
            "implementation_commit": environment["implementation_commit"],
            "environment_digest": environment.get("digest"),
            "evidence_context": {
                "runtime_context": environment["runtime_context"],
                "network_context": environment["network_context"],
            },
            "nodes": [
                {
                    "id": a["node_id"],
                    "compute_units": [
                        unit(
                            GPU_A,
                            a0,
                            [
                                "freetoken-source-offload-v1",
                                "freetoken-resident-block-a-v1",
                            ],
                        ),
                        unit(
                            GPU_A_SECONDARY,
                            a1,
                            ["freetoken-resident-block-b-v1"],
                        ),
                    ],
                    "memory_resources": [
                        {
                            "id": f"{GPU_A}.vram",
                            "kind": "accelerator-vram-a0",
                            "capacity_bytes": a0["vram_total_bytes"],
                            "reservation_bytes": a0.get("reservation_bytes", 0),
                        },
                        {
                            "id": f"{GPU_A_SECONDARY}.vram",
                            "kind": "accelerator-vram-a1",
                            "capacity_bytes": a1["vram_total_bytes"],
                            "reservation_bytes": a1.get("reservation_bytes", 0),
                        },
                    ],
                },
                {
                    "id": b["node_id"],
                    "compute_units": [
                        unit(
                            GPU_B,
                            b0,
                            ["freetoken-resident-block-b-v1"],
                        )
                    ],
                    "memory_resources": [
                        {
                            "id": f"{GPU_B}.vram",
                            "kind": "accelerator-vram-b0",
                            "capacity_bytes": b0["vram_total_bytes"],
                            "reservation_bytes": b0.get("reservation_bytes", 0),
                        }
                    ],
                },
            ],
            "links": [
                {
                    "id": "path.node-a.local-staging",
                    "source_memory_resource_id": f"{GPU_A}.vram",
                    "target_memory_resource_id": f"{GPU_A_SECONDARY}.vram",
                    "available": True,
                    "capabilities": ["freetoken-static-boundary-v1"],
                },
                {
                    "id": environment["network"]["link_id"],
                    "source_memory_resource_id": f"{GPU_A}.vram",
                    "target_memory_resource_id": f"{GPU_B}.vram",
                    "available": environment["network"].get("available", True),
                    "capabilities": ["freetoken-static-boundary-v1"],
                    "negotiated_mbps": environment["network"]["negotiated_mbps"],
                },
            ],
        }
    )


def operator_policy(implementation_commit: str) -> dict[str, Any]:
    return freeze(
        {
            "schema": "inferswarm.r5a.operator-policy/1",
            "implementation_commit": implementation_commit,
            "excluded_compute_unit_ids": [],
            "reservations_bytes": {},
            "integrity_policy": "quarantined-resources-cannot-participate",
            "static_plan_only": True,
        }
    )


def objective(implementation_commit: str, metric: str = "ttft_ms") -> dict[str, Any]:
    if metric not in ("ttft_ms", "complete_request_wall_ms", "decode_tok_s"):
        raise ValueError(f"unsupported frozen R5A objective {metric!r}")
    maximize = metric == "decode_tok_s"
    return freeze(
        {
            "schema": "inferswarm.r5a.objective/1",
            "implementation_commit": implementation_commit,
            "id": f"matched-serving-{metric}",
            "metric": metric,
            "direction": "MAXIMIZE" if maximize else "MINIMIZE",
            "unit": "tok/s" if maximize else "ms",
            "statistic": "median",
            "evidence_context": {"workload_geometry": "W2-W4-generate32-static"},
        }
    )


def _mapping(shape: str) -> dict[str, str]:
    if shape == LOCAL_SHAPE:
        return {"whole-model": GPU_A}
    if shape == LOCAL_SPLIT_SHAPE:
        return {"slot-a": GPU_A, "slot-b": GPU_A_SECONDARY}
    if shape == NETWORK_SHAPE:
        return {"slot-a": GPU_A, "slot-b": GPU_B}
    raise ValueError(shape)


def accepted_r4_evidence(repository_root: Path) -> list[dict[str, Any]]:
    """Normalize accepted R4 demand/capacity without claiming new applicability."""
    source = repository_root / "docs/inferswarm_r4/result.json"
    value = json.loads(source.read_text())
    disposition = value["one_gbe_disposition"]
    demand = disposition["actual_clean_arm_workload_wire_demand"]["peak_a_to_b_mbps"]
    limit = disposition["criterion"]["applicable_demand_limit_mbps"]
    required_context = {
        "model_revision": MODEL_REVISION,
        "runtime_context": f"accepted-r4-producer:{value['implementation_producer_sha']}",
        "network_context": "1GbE-full-duplex-MTU1500-eno1-enp5s0",
        "workload_geometry": "W2-W4-generate32-static",
    }
    return [
        {
            "id": "accepted-r4-application-demand-vs-capacity",
            "role": "ADMISSION_CONSTRAINT",
            "producer_identity": value["implementation_producer_sha"],
            "evidence_identity": value["r4_plan_digest"],
            "shape_id": NETWORK_SHAPE,
            "mapping": _mapping(NETWORK_SHAPE),
            "required_context": required_context,
            "freshness": "ACCEPTED_COMPATIBLE",
            "measurement_status": "MEASURED",
            "evidence_class": "MEASURED_ACCEPTED_R4_APPLICATION_DEMAND",
            "confidence": "EXACT_CONTEXT_ONLY",
            "metric": {
                "name": "application_network_demand",
                "value": demand,
                "unit": "Mb/s",
                "statistic": "peak",
            },
            "constraint": {"comparison": "LTE", "threshold": limit, "unit": "Mb/s"},
            "provenance": {
                "source_artifact": "docs/inferswarm_r4/result.json",
                "source_artifact_sha256": file_sha256(source),
                "source_plan_digest": value["r4_plan_digest"],
                "disposition": disposition["disposition"],
            },
        }
    ]


def serving_evidence_records(
    arm_summary: dict[str, Any], *, shape_id: str, context: dict[str, Any]
) -> list[dict[str, Any]]:
    """Normalize a matched R5A HTTP serving arm into objective evidence."""
    records = []
    mapping = _mapping(shape_id)
    producer = arm_summary["producer_freetoken_sha"]
    for metric, unit in (
        ("ttft_ms", "ms"),
        ("complete_request_wall_ms", "ms"),
        ("decode_tok_s", "tok/s"),
    ):
        if metric not in arm_summary["summary"]:
            continue
        records.append(
            {
                "id": f"r5a-{arm_summary['arm_id']}-{metric}",
                "role": "RANKING_OBJECTIVE",
                "producer_identity": producer,
                "evidence_identity": arm_summary["artifact_sha256"],
                "shape_id": shape_id,
                "mapping": mapping,
                "required_context": deepcopy(context),
                "freshness": "CURRENT",
                "measurement_status": "MEASURED",
                "evidence_class": "MEASURED_R5A_MATCHED_HTTP_SERVING",
                "confidence": "EXACT_CONTEXT",
                "metric": {
                    "name": metric,
                    "value": arm_summary["summary"][metric]["median"],
                    "unit": unit,
                    "statistic": "median",
                },
                "provenance": {
                    "arm_id": arm_summary["arm_id"],
                    "artifact_sha256": arm_summary["artifact_sha256"],
                },
            }
        )
    return records


def evidence_catalog(
    implementation_commit: str,
    repository_root: Path,
    serving_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return freeze(
        {
            "schema": "inferswarm.r5a.evidence-catalog/1",
            "implementation_commit": implementation_commit,
            "records": accepted_r4_evidence(repository_root)
            + list(serving_records or []),
        }
    )


def compile_candidate(
    evaluation: dict[str, Any], *, r4_plan: dict[str, Any], local_plan: dict[str, Any]
) -> dict[str, Any]:
    """Compile generic selection into complete strategy-owned semantics."""
    shape = evaluation["shape_id"]
    mapping = evaluation["mapping"]
    if mapping != _mapping(shape):
        raise ValueError("Qwen strategy refuses an unrecognized mapping")
    node_by_unit = {
        GPU_A: "node.inferswarm01",
        GPU_A_SECONDARY: "node.inferswarm01",
        GPU_B: "node.inferswarm03",
    }
    units = list(dict.fromkeys(mapping.values()))
    participants = list(dict.fromkeys(node_by_unit[unit] for unit in units))
    distributed = shape in (LOCAL_SPLIT_SHAPE, NETWORK_SHAPE)
    state_placement = []
    authority = []
    representations = []
    if distributed:
        for role, unit in (("a", mapping["slot-a"]), ("b", mapping["slot-b"])):
            state_placement.extend(
                [
                    {"logical_state_id": f"state.block-{role}.routed", "compute_unit_id": unit, "role": "required_residency"},
                    {"logical_state_id": f"state.block-{role}.mutable-runtime", "compute_unit_id": unit, "role": "mutable_authority"},
                ]
            )
            authority.append(
                {"logical_state_id": f"state.block-{role}.mutable-runtime", "participant": node_by_unit[unit], "lineage": "r5a-static"}
            )
            representations.append(
                {"logical_state_id": f"state.block-{role}.routed", "representation": "freetoken-nvfp4-slot-banks"}
            )
    else:
        state_placement.append(
            {"logical_state_id": "state.whole-model", "compute_unit_id": GPU_A, "role": "source-backed-execution"}
        )
        authority.append(
            {"logical_state_id": "state.session", "participant": node_by_unit[GPU_A], "lineage": "r5a-static"}
        )
        representations.append(
            {"logical_state_id": "state.whole-model", "representation": "freetoken-source-backed-native"}
        )
    boundary = (
        [
            {
                "id": "boundary.block-a-to-block-b",
                "from_compute_unit_id": mapping["slot-a"],
                "to_compute_unit_id": mapping["slot-b"],
                "semantic_contract": {
                    "dtype": "bfloat16",
                    "layout": "plane-major-contiguous",
                    "planes": 2,
                    "decode_bytes": 8192,
                    "prefill_chunk_rows": 64,
                    "prefill_bytes": 524288,
                },
            }
        ]
        if distributed
        else []
    )
    accounting = deepcopy(evaluation["memory_accounting"])
    accounting["strategy_host_lifecycle"] = {
        "slot_a_staging_peak_bytes": STAGING_A if distributed else 18_182_307_840,
        "slot_b_staging_peak_bytes": STAGING_B if distributed else 0,
        "registered_activation_buffer_bytes": ACTIVATION_BUFFER if distributed else 0,
        "release_after_final_residency": distributed,
    }
    return {
        "strategy_identity": {"id": STRATEGY_ID},
        "model_identity": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "participants": participants,
        "compute_units": units,
        "representations": representations,
        "backend_choices": [
            {"participant": participant, "backend": "freetoken-backend-native-resident"}
            for participant in participants
        ],
        "state_placement": state_placement,
        "state_authority": authority,
        "semantic_boundaries": boundary,
        "expected_resource_accounting": accounting,
        "strategy_realization": {
            "path": (
                "r4-persistent-boundary" if shape == NETWORK_SHAPE else
                "r2-local-split" if shape == LOCAL_SPLIT_SHAPE else
                "ordinary-freetoken-source-offload"
            ),
            "participant_plan_digest": (
                r4_plan.get("digest") if shape == NETWORK_SHAPE else
                local_plan.get("digest") if shape == LOCAL_SPLIT_SHAPE else None
            ),
        },
    }


__all__ = [
    "GPU_A",
    "GPU_A_SECONDARY",
    "GPU_B",
    "LOCAL_SHAPE",
    "LOCAL_SPLIT_SHAPE",
    "MODEL_REPOSITORY",
    "MODEL_REVISION",
    "NETWORK_SHAPE",
    "accepted_r4_evidence",
    "compile_candidate",
    "evidence_catalog",
    "objective",
    "operator_policy",
    "planning_problem",
    "resource_snapshot",
    "serving_evidence_records",
]
