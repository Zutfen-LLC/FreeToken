"""Freeze the one preflight-selected R2 plan and matched baseline config."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha
from freetoken.research.r1_frozen_plan import freeze_plan as freeze_r1
from freetoken.research.r2_local_split import freeze_plan as freeze_r2

MODEL_REPOSITORY = "nvidia/Qwen3.6-35B-A3B-NVFP4"
MODEL_REVISION = "491c2f1ea524c639598bf8fa787a93fed5a6fbce"
R1_BASE = "6a242a34083c3080aa6d8f92625a6be4a0d124db"
GPU_A_UUID = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"
GPU_B_UUID = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
RUNTIME_CAPACITY = 17_152
PREFILL_CHUNK = 64
GENERATED_TOKENS = 32
SAFETY_BYTES = 512 * 1024 * 1024

COMPONENTS = {
    "a": {
        "non_routed": 1_689_347_736,
        "routed": 8_636_596_224,
        "mutable": 172_703_744,
        "graph_backend": 362_554_728,
    },
    "b": {
        "non_routed": 1_019_085_480,
        "routed": 9_545_711_616,
        "mutable": 242_958_336,
        "graph_backend": 362_523_480,
    },
}


def _gpu_records() -> dict[str, dict]:
    fields = (
        "index,uuid,name,memory.total,pci.bus_id,pcie.link.gen.current,"
        "pcie.link.width.current,compute_cap,driver_version"
    )
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
    )
    records = {}
    for line in output.splitlines():
        item = dict(
            zip(
                fields.split(","),
                (value.strip() for value in line.split(",")),
                strict=True,
            )
        )
        item["capacity_bytes"] = int(item["memory.total"]) * 1024 * 1024
        records[item["uuid"]] = item
    return records


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _r1_participant(
    *,
    role: str,
    gpu_uuid: str,
    vram_id: str,
    compute_id: str,
    block_plan: dict,
    components: dict,
) -> dict:
    prefix = f"state.block-{role}"
    mats = [
        {
            "id": f"mat.block-{role}.routed-staging",
            "logical_state_id": f"{prefix}.routed",
            "representation": "checkpoint-native-host",
            "memory_resource_id": "node-a.ram",
            "role": "staging",
            "requirement": "required",
            "persistence": "transient",
            "expected_bytes": components["routed"],
        },
        {
            "id": f"mat.block-{role}.non-routed-vram",
            "logical_state_id": f"{prefix}.non-routed",
            "representation": "freetoken-native-device",
            "memory_resource_id": vram_id,
            "role": "required_residency",
            "requirement": "required",
            "persistence": "persistent",
            "expected_bytes": components["non_routed"],
        },
        {
            "id": f"mat.block-{role}.routed-vram",
            "logical_state_id": f"{prefix}.routed",
            "representation": "freetoken-nvfp4-slot-banks",
            "memory_resource_id": vram_id,
            "role": "required_residency",
            "requirement": "required",
            "persistence": "persistent",
            "expected_bytes": components["routed"],
        },
        {
            "id": f"mat.block-{role}.mutable-runtime-vram",
            "logical_state_id": f"{prefix}.mutable-runtime",
            "representation": "freetoken-block-runtime-device",
            "memory_resource_id": vram_id,
            "role": "mutable_authority",
            "requirement": "required",
            "persistence": "persistent",
            "expected_bytes": components["mutable"],
        },
        {
            "id": f"mat.block-{role}.graph-backend-vram",
            "logical_state_id": f"{prefix}.graph-backend",
            "representation": "freetoken-captured-backend-device",
            "memory_resource_id": vram_id,
            "role": "required_residency",
            "requirement": "required",
            "persistence": "persistent",
            "expected_bytes": components["graph_backend"],
        },
    ]
    states = [
        {
            "id": f"{prefix}.non-routed",
            "semantic_class": "immutable_source",
            "backing_source_id": "checkpoint",
        },
        {
            "id": f"{prefix}.routed",
            "semantic_class": "immutable_source",
            "backing_source_id": "checkpoint",
        },
        {"id": f"{prefix}.mutable-runtime", "semantic_class": "mutable_authoritative"},
        {"id": f"{prefix}.graph-backend", "semantic_class": "derived_reconstructible"},
    ]
    return freeze_r1(
        {
            "schema": "inferswarm.r1.frozen-plan/1",
            "plan_version": f"r2-block-{role}-participant-v1",
            "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
            "resources": {
                "swarm_id": "r2-local-proof",
                "nodes": [
                    {
                        "id": "node-a",
                        "compute_units": [
                            {"id": compute_id, "stable_device_id": gpu_uuid}
                        ],
                        "memory_resources": [
                            {"id": "node-a.ram", "kind": "system-ram"},
                            {"id": vram_id, "kind": "accelerator-vram"},
                        ],
                    }
                ],
                "backing_sources": [{"id": "checkpoint"}],
                "links": [
                    {
                        "id": f"checkpoint-to-block-{role}",
                        "kind": "bounded-selective-staging",
                    }
                ],
            },
            "logical_state_units": states,
            "materializations": mats,
            "authorities": [
                {
                    "logical_state_id": f"{prefix}.mutable-runtime",
                    "materialization_id": f"mat.block-{role}.mutable-runtime-vram",
                    "lineage": "r2-epoch-1",
                }
            ],
            "execution": [
                {
                    "id": f"exec.block-{role}",
                    "strategy_unit": "opaque-contiguous-block-v1",
                    "compute_unit_id": compute_id,
                    "required_state": [
                        {
                            "logical_state_id": state["id"],
                            "representations": [mat["representation"]],
                            "memory_resources": [vram_id],
                        }
                        for state, mat in zip(states, mats[1:], strict=True)
                    ],
                }
            ],
            "forbidden_persistent_materializations": [
                {
                    "logical_state_id": f"{prefix}.routed",
                    "memory_resource_id": "node-a.ram",
                }
            ],
            "adapter_data": {
                "role": role,
                "spec": block_plan["spec"],
                "allowed_tensor_keys": block_plan["allowed_tensor_keys"],
                "runtime_capacity_tokens": RUNTIME_CAPACITY,
                "prefill_chunk_tokens": PREFILL_CHUNK,
            },
        }
    )


def build(
    n0: dict, manifest_sha: str, transport_preflight_sha: str
) -> tuple[dict, dict, dict]:
    gpu = _gpu_records()
    for uuid in (GPU_A_UUID, GPU_B_UUID):
        if uuid not in gpu:
            raise RuntimeError(f"frozen GPU {uuid} is absent")
    if n0["split_boundary"] != 19:
        raise RuntimeError("N0 candidate split changed; refuse to freeze")
    participant_a = _r1_participant(
        role="a",
        gpu_uuid=GPU_A_UUID,
        vram_id="gpu-a.vram",
        compute_id="gpu-a",
        block_plan=n0["block_a"],
        components=COMPONENTS["a"],
    )
    participant_b = _r1_participant(
        role="b",
        gpu_uuid=GPU_B_UUID,
        vram_id="gpu-b.vram",
        compute_id="gpu-b",
        block_plan=n0["block_b"],
        components=COMPONENTS["b"],
    )
    resources = {
        "nodes": [
            {
                "id": "node-a",
                "compute_units": [
                    {
                        "id": "gpu-a",
                        "stable_device_id": GPU_A_UUID,
                        "model": gpu[GPU_A_UUID]["name"],
                    },
                    {
                        "id": "gpu-b",
                        "stable_device_id": GPU_B_UUID,
                        "model": gpu[GPU_B_UUID]["name"],
                    },
                ],
                "memory_resources": [
                    {
                        "id": "gpu-a.vram",
                        "kind": "accelerator-vram",
                        "capacity_bytes": gpu[GPU_A_UUID]["capacity_bytes"],
                    },
                    {
                        "id": "gpu-b.vram",
                        "kind": "accelerator-vram",
                        "capacity_bytes": gpu[GPU_B_UUID]["capacity_bytes"],
                    },
                    {
                        "id": "node-a.ram",
                        "kind": "system-ram",
                        "capacity_bytes": 134_984_794_112,
                    },
                ],
            }
        ],
        "links": [
            {
                "id": "link.gpu-a-host-gpu-b",
                "kind": "registered-pinned-host-staging",
                "source_memory_resource": "gpu-a.vram",
                "staging_memory_resource": "node-a.ram",
                "target_memory_resource": "gpu-b.vram",
                "preflight_sha256": transport_preflight_sha,
            }
        ],
        "backing_sources": [{"id": "checkpoint", "kind": "local-filesystem"}],
    }
    states = []
    materials = []
    authorities = []
    blocks = []
    for role, execution, memory, layers, components in (
        ("a", "exec.block-a", "gpu-a.vram", list(range(19)), COMPONENTS["a"]),
        ("b", "exec.block-b", "gpu-b.vram", list(range(19, 40)), COMPONENTS["b"]),
    ):
        prefix = f"state.block-{role}"
        role_states = [
            f"{prefix}.non-routed",
            f"{prefix}.routed",
            f"{prefix}.mutable-runtime",
            f"{prefix}.graph-backend",
        ]
        states.extend(
            [
                {"id": role_states[0], "semantic_class": "immutable_source"},
                {"id": role_states[1], "semantic_class": "immutable_source"},
                {"id": role_states[2], "semantic_class": "mutable_authoritative"},
                {"id": role_states[3], "semantic_class": "derived_reconstructible"},
            ]
        )
        for name, key in (
            ("non-routed", "non_routed"),
            ("routed", "routed"),
            ("mutable-runtime", "mutable"),
            ("graph-backend", "graph_backend"),
        ):
            materials.append(
                {
                    "id": f"mat.block-{role}.{name}-vram",
                    "logical_state_id": f"{prefix}.{name}",
                    "execution_id": execution,
                    "memory_resource_id": memory,
                    "role": "mutable_authority"
                    if name == "mutable-runtime"
                    else "required_residency",
                    "requirement": "required",
                    "persistence": "persistent",
                    "expected_bytes": components[key],
                }
            )
        materials.append(
            {
                "id": f"mat.block-{role}.routed-staging",
                "logical_state_id": f"{prefix}.routed",
                "execution_id": execution,
                "memory_resource_id": "node-a.ram",
                "role": "staging",
                "requirement": "required",
                "persistence": "transient",
                "expected_bytes": components["routed"],
            }
        )
        authorities.append(
            {
                "logical_state_id": role_states[2],
                "execution_id": execution,
                "lineage": "r2-epoch-1",
            }
        )
        blocks.append(
            {
                "execution_id": execution,
                "ordinary_layer_ids": layers,
                "owned_state_ids": [role_states[0], role_states[1], role_states[3]],
                "mutable_state_ids": [role_states[2]],
            }
        )
    provenance = {
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "workload_digest": manifest_sha,
    }
    plan = freeze_r2(
        {
            "schema": "inferswarm.r2.local-split-plan/1",
            "plan_version": "r2-n0-split-v1",
            "created_from_r1_base": R1_BASE,
            "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
            "resources": resources,
            "logical_state_units": states,
            "materializations": materials,
            "authorities": authorities,
            "execution": [
                {"id": "exec.block-a", "compute_unit_id": "gpu-a"},
                {"id": "exec.block-b", "compute_unit_id": "gpu-b"},
            ],
            "strategy": {
                "strategy_id": "pinned-qwen-contiguous-n0-v1",
                "number_of_layers": 40,
                "blocks": blocks,
                "selected_before_performance": True,
            },
            "boundary": {
                "id": "boundary.block-a-to-b",
                "producer_execution_id": "exec.block-a",
                "consumer_execution_id": "exec.block-b",
                "source_memory_resource": "gpu-a.vram",
                "target_memory_resource": "gpu-b.vram",
                "transport_path_id": "link.gpu-a-host-gpu-b",
                "representation": "opaque-strategy-boundary-v1",
                "semantic_contract": "adapter-owned",
                "cadence": "per prefill chunk / per decode step",
                "contract": {
                    "dtype": "bfloat16",
                    "layout": "plane-major-contiguous",
                    "planes": 2,
                    "row_width": 2048,
                    "element_bytes": 2,
                    "decode_payload_bytes": 8192,
                    "prefill_chunk_payload_bytes": 524288,
                },
            },
            "runtime_capacity": {
                "max_prompt_tokens": 16819,
                "generated_tokens": GENERATED_TOKENS,
                "max_sequence_tokens": RUNTIME_CAPACITY,
                "prefill_chunk_tokens": PREFILL_CHUNK,
                "concurrency": 1,
                "explicit_margin_tokens": RUNTIME_CAPACITY - 16819 - GENERATED_TOKENS,
            },
            "memory": {
                "safety_headroom_bytes_per_vram": SAFETY_BYTES,
                "required_by_resource": {
                    "gpu-a.vram": sum(COMPONENTS["a"].values()),
                    "gpu-b.vram": sum(COMPONENTS["b"].values()),
                    "node-a.ram": 524288,
                },
            },
            "provenance": {"baseline": dict(provenance), "candidate": dict(provenance)},
            "participant_r1_plans": {
                "exec.block-a": participant_a,
                "exec.block-b": participant_b,
            },
            "adapter_data": {
                "n0_split": 19,
                "block_a": n0["block_a"],
                "block_b": n0["block_b"],
            },
        }
    )
    baseline = {
        "schema": "inferswarm.r2.baseline-config/1",
        "freetoken_base": R1_BASE,
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "compute_unit": {"id": "baseline-gpu", "stable_device_id": GPU_A_UUID},
        "execution": "current supported single-GPU FreeToken offload path",
        "runtime_capacity": plan["runtime_capacity"],
        "workload_manifest_sha256": manifest_sha,
        "workloads": ["W1", "W2", "W3", "W4"],
        "sampling": "deterministic-greedy",
        "backend": {
            "attention": "fi",
            "moe": "offload",
            "nvfp4": "triton",
            "decode": "captured-bs1",
            "moe_cache_slots": 3774,
            "moe_cpu_layers": 0,
            "prefill_overlap": True,
        },
        "server": {
            "max_running_requests": 1,
            "num_tokens": RUNTIME_CAPACITY,
            "kv_reserve_tokens": RUNTIME_CAPACITY,
            "max_prefill_length": PREFILL_CHUNK,
            "memory_ratio": 0.85,
            "sampling_defaults": "none",
        },
    }
    capacity = {
        "schema": "inferswarm.r2.capacity-preflight/1",
        "status": "N0_SPLIT_FITS",
        "candidate_boundary": 19,
        "selection_rule": n0["split_rule"],
        "selected_gpu_records": [gpu[GPU_A_UUID], gpu[GPU_B_UUID]],
        "components": COMPONENTS,
        "projected_required_bytes": {
            "gpu-a.vram": sum(COMPONENTS["a"].values()),
            "gpu-b.vram": sum(COMPONENTS["b"].values()),
        },
        "safety_headroom_bytes": SAFETY_BYTES,
        "remaining_after_required_bytes": {
            "gpu-a.vram": gpu[GPU_A_UUID]["capacity_bytes"]
            - sum(COMPONENTS["a"].values()),
            "gpu-b.vram": gpu[GPU_B_UUID]["capacity_bytes"]
            - sum(COMPONENTS["b"].values()),
        },
        "component_provenance": {
            "non_routed_and_routed": "fresh selective component inspection from accepted R1 lineage",
            "mutable": "capacity-only projection from historical N1 component measurements; retained R2 observes actual bytes",
            "graph_backend": "fresh pre-retained two-process realization observation from accepted R1 lineage",
            "checkpoint_census_sha256": hashlib.sha256(
                json.dumps(n0, sort_keys=True).encode()
            ).hexdigest(),
        },
        "passed_with_explicit_safety": all(
            gpu[uuid]["capacity_bytes"] - sum(COMPONENTS[role].values()) >= SAFETY_BYTES
            for uuid, role in ((GPU_A_UUID, "a"), (GPU_B_UUID, "b"))
        ),
    }
    if not capacity["passed_with_explicit_safety"]:
        raise RuntimeError("N0 split fails capacity preflight with safety headroom")
    return plan, baseline, capacity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n0-plan", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--transport-preflight", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    plan, baseline, capacity = build(
        json.loads(args.n0_plan.read_text()),
        _sha(args.workload_manifest),
        _sha(args.transport_preflight),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("frozen-plan.json", plan),
        ("baseline-config.json", baseline),
        ("capacity-preflight.json", capacity),
    ):
        write_json_with_sha(args.out_dir / name, payload)
    print(
        json.dumps(
            {
                "plan_digest": plan["digest"],
                "split": 19,
                "transport": plan["resources"]["links"][0]["kind"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
