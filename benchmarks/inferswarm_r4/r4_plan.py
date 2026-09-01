"""R4 frozen two-Node split plan and planner seam.

Extends the accepted R2 local split plan with exactly one new architectural
variable: the two blocks execute on two physical Nodes joined by a persistent
ordinary-TCP path over negotiated 1 GbE.  Strategy geometry, state ownership,
materializations, and the boundary contract are carried over unchanged from
the accepted R2 frozen plan.

Also exposes the R3-generic planner seam: the network candidate shape and a
resource snapshot with two nodes, so the generic planner can classify it
FEASIBLE_UNRANKED (no pre-existing performance evidence).  No model nouns
enter the generic planner.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from freetoken.research.r2_local_split import (
    LocalSplitPlanError,
    freeze_plan,
    plan_digest,
)
from freetoken.research.r3_planner import freeze as freeze_generic

R4_PLAN_SCHEMA = "inferswarm.r4.two-node-split-plan/1"
NETWORK_TRANSPORT_KIND = "persistent-tcp-ethernet"
NODE_A_ID = "node.inferswarm01"
NODE_B_ID = "node.inferswarm03"
GPU_A_UUID = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"
GPU_B_UUID = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
MODEL_REPOSITORY = "nvidia/Qwen3.6-35B-A3B-NVFP4"
MODEL_REVISION = "491c2f1ea524c639598bf8fa787a93fed5a6fbce"
ACCEPTED_R2_PLAN_DIGEST = (
    "sha256:6128dd6705d6d692df3d5fc11cc130dba5c010cfff40c0e4c5ec7c19e1b78ff0"
)
# Node-side transport endpoint registered host activation buffer.
REGISTERED_BUFFER_BYTES = 524_288
# Canonical network arm facts (must be re-proven by preflight before runs).
CANONICAL_LINK = {
    "kind": NETWORK_TRANSPORT_KIND,
    "speed_mbps": 1000,
    "duplex": "full",
    "mtu": 1500,
    "node_a_lan_ipv4": "10.0.0.141",
    "node_b_lan_ipv4": "10.0.0.219",
    "node_a_interface": "eno1",
    "node_b_interface": "enp5s0",
    "tcp_nodelay": True,
    "persistent": True,
}


def build_r4_plan(
    r2_plan: dict[str, Any],
    *,
    node_a_hardware: dict[str, Any],
    node_b_hardware: dict[str, Any],
    link_freeze: dict[str, Any],
) -> dict[str, Any]:
    """Derive the R4 plan from the accepted R2 frozen plan.

    The R2 plan's nodes/links are replaced with two physical Nodes and a
    persistent-TCP link; compute-unit stable device ids, memory resources,
    strategy blocks, materializations, boundary contract, and provenance are
    carried over unchanged.
    """

    if r2_plan.get("digest") != ACCEPTED_R2_PLAN_DIGEST:
        raise LocalSplitPlanError("base plan is not the accepted R2 frozen plan")
    plan = deepcopy(r2_plan)
    plan["schema"] = R4_PLAN_SCHEMA
    gpu_a = _selected_gpu(node_a_hardware, GPU_A_UUID)
    gpu_b = _selected_gpu(node_b_hardware, GPU_B_UUID)
    resources = plan["resources"]
    resources["nodes"] = [
        {
            "id": NODE_A_ID,
            "role": "node-a-block-a",
            "compute_units": [_unit_for(gpu_a, "gpu-a")],
            "memory_resources": [
                _vram_resource(gpu_a, "gpu-a.vram"),
                _host_resource(node_a_hardware, "node-a.ram"),
            ],
        },
        {
            "id": NODE_B_ID,
            "role": "node-b-block-b",
            "compute_units": [_unit_for(gpu_b, "gpu-b")],
            "memory_resources": [
                _vram_resource(gpu_b, "gpu-b.vram"),
                _host_resource(node_b_hardware, "node-b.ram"),
            ],
        },
    ]
    resources["links"] = [
        {
            "id": "link.node-a-to-node-b.tcp",
            "kind": NETWORK_TRANSPORT_KIND,
            "from_node": NODE_A_ID,
            "to_node": NODE_B_ID,
            **deepcopy(link_freeze),
        }
    ]
    plan["boundary"]["transport_path_id"] = "link.node-a-to-node-b.tcp"
    plan["provenance"] = deepcopy(plan.get("provenance", {}))
    plan["provenance"]["r4"] = {
        "derived_from_r2_plan_digest": ACCEPTED_R2_PLAN_DIGEST,
        "node_a_hardware_digest": node_a_hardware.get("digest"),
        "node_b_hardware_digest": node_b_hardware.get("digest"),
        "link_freeze_digest": freeze_generic(link_freeze)["digest"],
        "variable_changed": "node-network-locality-only",
    }
    frozen = freeze_plan(plan)
    return frozen


def _selected_gpu(hardware: dict[str, Any], uuid: str) -> dict[str, Any]:
    for gpu in hardware.get("gpus", []):
        if gpu.get("uuid") == uuid:
            return gpu
    raise LocalSplitPlanError(
        f"frozen GPU {uuid} absent from node hardware profile"
    )


def _unit_for(gpu: dict[str, Any], unit_id: str) -> dict[str, Any]:
    return {
        "id": unit_id,
        "kind": "cuda-gpu",
        "stable_device_id": gpu["uuid"],
        "model": gpu.get("name"),
        "pci_bdf": gpu.get("pci_bus_id"),
    }


def _vram_resource(gpu: dict[str, Any], resource_id: str) -> dict[str, Any]:
    return {
        "id": resource_id,
        "kind": "gpu-vram",
        "capacity_bytes": int(gpu["memory_total_bytes"]),
    }


def _host_resource(hardware: dict[str, Any], resource_id: str) -> dict[str, Any]:
    return {
        "id": resource_id,
        "kind": "host-memory",
        "capacity_bytes": int(hardware["memory"]["mem_total_kib"]) * 1024,
    }


def validate_network_candidate(
    r4_plan: dict[str, Any], decision: dict[str, Any]
) -> dict[str, Any]:
    """Confirm the generic planner classified the network candidate
    FEASIBLE_UNRANKED (evidence-collection authorization, not preference)."""

    evaluation = None
    for item in decision.get("evaluations", []):
        if str(item.get("id", "")).startswith("s1r4.two-node-network-split"):
            evaluation = item
    if evaluation is None:
        raise LocalSplitPlanError(
            "planner decision does not contain the two-node network candidate"
        )
    if evaluation.get("state") != "FEASIBLE_UNRANKED":
        raise LocalSplitPlanError(
            f"network candidate planner state is {evaluation.get('state')!r}, "
            "expected FEASIBLE_UNRANKED (no pre-existing performance evidence)"
        )
    return {
        "candidate_id": evaluation["id"],
        "state": evaluation["state"],
        "integrity_eligible": evaluation.get("integrity_eligible"),
        "policy_eligible": evaluation.get("policy_eligible"),
        "applicable_evidence_ids": evaluation.get("applicable_evidence_ids", []),
        "r4_plan_digest": r4_plan["digest"],
        "authorization": "evidence-collection-only",
    }


def r4_resource_snapshot(
    node_a_hardware: dict[str, Any],
    node_b_hardware: dict[str, Any],
    *,
    implementation_commit: str | None = None,
) -> dict[str, Any]:
    """Two-node resource snapshot for the generic R3 planner."""

    nodes = []
    for hardware, node_id, unit_id in (
        (node_a_hardware, NODE_A_ID, "gpu-a"),
        (node_b_hardware, NODE_B_ID, "gpu-b"),
    ):
        gpu = _selected_gpu(hardware, GPU_A_UUID if node_id == NODE_A_ID else GPU_B_UUID)
        nodes.append(
            {
                "id": node_id,
                "compute_units": [
                    {
                        "id": unit_id,
                        "stable_device_id": gpu["uuid"],
                        "memory_resource_id": f"{unit_id}.vram",
                        "availability": "AVAILABLE",
                        "integrity_eligible": True,
                        "capabilities": [
                            "freetoken-offload-v1",
                            "freetoken-resident-block-v1",
                        ],
                    }
                ],
                "memory_resources": [
                    {
                        "id": f"{unit_id}.vram",
                        "kind": "gpu-vram",
                        "capacity_bytes": int(gpu["memory_total_bytes"]),
                        "reservation_bytes": 0,
                    },
                    {
                        "id": f"{node_id}.ram",
                        "kind": (
                            "node-a-system-ram"
                            if node_id == NODE_A_ID
                            else "node-b-system-ram"
                        ),
                        "capacity_bytes": int(hardware["memory"]["mem_total_kib"]) * 1024,
                        "reservation_bytes": 0,
                    },
                ],
            }
        )
    return {
        "schema": "inferswarm.r4.resource-snapshot/1",
        "implementation_commit": implementation_commit,
        "nodes": nodes,
        "links": [
            {
                "id": "link.node-a-to-node-b.tcp",
                "source_memory_resource_id": "gpu-a.vram",
                "target_memory_resource_id": "gpu-b.vram",
                "available": True,
                "capabilities": ["persistent-tcp-ethernet"],
                "decode_payload_bytes": 8192,
                "prefill_chunk_payload_bytes": 524288,
            }
        ],
    }


def r4_network_problem(implementation_commit: str | None = None) -> dict[str, Any]:
    """Generic planning problem exposing the network candidate shape.

    Mirrors the accepted R3 s1 shape but with node-distinct slots and the
    network path; deliberately carries no Qwen noun.  Byte figures come from
    the accepted R2/R3 strategy payload lineage.
    """

    return freeze_generic(
        {
            "schema": "inferswarm.r4.network-planning-problem/1",
            "implementation_commit": implementation_commit,
            "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
            "evidence_context": {"model_revision": MODEL_REVISION},
            "shapes": [
                {
                    "id": "s1r4.two-node-network-split",
                    "slots": [
                        {
                            "id": "opaque-slot-a",
                            "allowed_compute_unit_ids": ["gpu-a"],
                            "required_capabilities": ["freetoken-resident-block-v1"],
                            "memory": {
                                "persistent_required_bytes": 10_861_202_432,
                                "persistent_optional_bytes": 0,
                                "transient_peak_bytes": 0,
                            },
                        },
                        {
                            "id": "opaque-slot-b",
                            "allowed_compute_unit_ids": ["gpu-b"],
                            "required_capabilities": ["freetoken-resident-block-v1"],
                            "memory": {
                                "persistent_required_bytes": 11_170_278_912,
                                "persistent_optional_bytes": 0,
                                "transient_peak_bytes": 0,
                            },
                        },
                    ],
                    "distinct_slot_groups": [["opaque-slot-a", "opaque-slot-b"]],
                    "node_distinct_slot_groups": [["opaque-slot-a", "opaque-slot-b"]],
                    "paths": [
                        {
                            "id": "activation-boundary",
                            "from_slot": "opaque-slot-a",
                            "to_slot": "opaque-slot-b",
                            "required_capabilities": ["persistent-tcp-ethernet"],
                        }
                    ],
                    "materializations": [
                        {
                            "id": "slot-a-source-staging",
                            "memory_kind": "node-a-system-ram",
                            "bytes": 8_636_596_224,
                            "lifecycle": "TRANSIENT_RELEASE_AFTER_FINALIZATION",
                        },
                        {
                            "id": "slot-b-source-staging",
                            "memory_kind": "node-b-system-ram",
                            "bytes": 9_545_711_616,
                            "lifecycle": "TRANSIENT_RELEASE_AFTER_FINALIZATION",
                        },
                        {
                            "id": "registered-activation-buffer-a",
                            "memory_kind": "node-a-system-ram",
                            "bytes": REGISTERED_BUFFER_BYTES,
                            "lifecycle": "PERSISTENT_REQUIRED",
                        },
                        {
                            "id": "registered-activation-buffer-b",
                            "memory_kind": "node-b-system-ram",
                            "bytes": REGISTERED_BUFFER_BYTES,
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
                        "accepted_r2_plan_digest": ACCEPTED_R2_PLAN_DIGEST,
                    },
                }
            ],
        }
    )


def write_r4_plan(plan: dict[str, Any], out_dir: Path) -> None:
    from freetoken.research.n0_model_block import write_json_with_sha

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_with_sha(out_dir / "r4-frozen-plan.json", plan)


def load_r4_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text())
    if plan.get("schema") != R4_PLAN_SCHEMA:
        raise LocalSplitPlanError(f"not an R4 plan: {plan.get('schema')!r}")
    if plan.get("digest") != f"sha256:{plan_digest(plan)}":
        raise LocalSplitPlanError("R4 plan digest mismatch")
    return plan


__all__ = [
    "ACCEPTED_R2_PLAN_DIGEST",
    "CANONICAL_LINK",
    "GPU_A_UUID",
    "GPU_B_UUID",
    "MODEL_REPOSITORY",
    "MODEL_REVISION",
    "NETWORK_TRANSPORT_KIND",
    "NODE_A_ID",
    "NODE_B_ID",
    "R4_PLAN_SCHEMA",
    "REGISTERED_BUFFER_BYTES",
    "build_r4_plan",
    "load_r4_plan",
    "r4_network_problem",
    "r4_resource_snapshot",
    "validate_network_candidate",
    "write_r4_plan",
]
