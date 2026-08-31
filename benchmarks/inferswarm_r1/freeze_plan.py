"""Write the predeclared R1 research plan and integrity record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from freetoken.research.r1_frozen_plan import PLAN_SCHEMA, freeze_plan

MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
REVISION = "491c2f1ea524c639598bf8fa787a93fed5a6fbce"


def build_plan(n0_plan: dict) -> dict:
    block = n0_plan["block_a"]
    return freeze_plan(
        {
            "schema": PLAN_SCHEMA,
            "plan_version": "r1-block-a-v1",
            "model": {"repository": MODEL, "revision": REVISION},
            "resources": {
                "swarm_id": "r1-test",
                "nodes": [
                    {
                        "id": "node-a",
                        "compute_units": [
                            {
                                "id": "gpu0",
                                "kind": "cuda-gpu",
                                "device_uuid": "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099",
                                "memory_resources": ["gpu0.vram"],
                            }
                        ],
                        "memory_resources": [
                            {
                                "id": "gpu0.vram",
                                "kind": "accelerator-vram",
                                "capacity_bytes": 12485525504,
                            },
                            {
                                "id": "node-a.ram",
                                "kind": "system-ram",
                                "capacity_bytes": 134984794112,
                            },
                        ],
                    }
                ],
                "backing_sources": [
                    {
                        "id": "qwen-checkpoint",
                        "kind": "checkpoint",
                        "repository": MODEL,
                        "revision": REVISION,
                    }
                ],
                "links": [
                    {
                        "id": "checkpoint-to-ram",
                        "source": "qwen-checkpoint",
                        "target": "node-a.ram",
                        "kind": "bounded-read",
                    },
                    {
                        "id": "ram-to-vram",
                        "source": "node-a.ram",
                        "target": "gpu0.vram",
                        "kind": "host-to-device",
                    },
                ],
            },
            "logical_state_units": [
                {
                    "id": "state.block-a.non-routed",
                    "semantic_class": "immutable_source",
                    "backing_source_id": "qwen-checkpoint",
                },
                {
                    "id": "state.block-a.routed",
                    "semantic_class": "immutable_source",
                    "backing_source_id": "qwen-checkpoint",
                },
                {
                    "id": "state.block-a.mutable-runtime",
                    "semantic_class": "mutable_authoritative",
                },
                {
                    "id": "state.block-a.replay-baseline",
                    "semantic_class": "derived_reconstructible",
                    "reconstruction_inputs": ["state.block-a.mutable-runtime"],
                },
            ],
            "materializations": [
                {
                    "id": "mat.routed.ram-stage",
                    "logical_state_id": "state.block-a.routed",
                    "representation": "checkpoint-native-host",
                    "memory_resource_id": "node-a.ram",
                    "role": "staging",
                    "requirement": "required",
                    "persistence": "transient",
                    "expected_bytes": 8636596224,
                    "path_id": "checkpoint-to-ram",
                },
                {
                    "id": "mat.non-routed.vram",
                    "logical_state_id": "state.block-a.non-routed",
                    "representation": "freetoken-native-device",
                    "memory_resource_id": "gpu0.vram",
                    "role": "required_residency",
                    "requirement": "required",
                    "persistence": "persistent",
                    "expected_bytes": 1689347736,
                    "path_id": "ram-to-vram",
                },
                {
                    "id": "mat.routed.vram",
                    "logical_state_id": "state.block-a.routed",
                    "representation": "freetoken-nvfp4-slot-banks",
                    "memory_resource_id": "gpu0.vram",
                    "role": "required_residency",
                    "requirement": "required",
                    "persistence": "persistent",
                    "expected_bytes": 8636596224,
                    "path_id": "ram-to-vram",
                },
                {
                    "id": "mat.runtime.vram",
                    "logical_state_id": "state.block-a.mutable-runtime",
                    "representation": "freetoken-block-runtime-device",
                    "memory_resource_id": "gpu0.vram",
                    "role": "mutable_authority",
                    "requirement": "required",
                    "persistence": "persistent",
                    "expected_bytes": 32718848,
                },
                {
                    "id": "mat.replay.vram",
                    "logical_state_id": "state.block-a.replay-baseline",
                    "representation": "freetoken-runtime-snapshot-device",
                    "memory_resource_id": "gpu0.vram",
                    "role": "required_test_state",
                    "requirement": "required",
                    "persistence": "persistent",
                    "expected_bytes": 32718848,
                },
            ],
            "authorities": [
                {
                    "logical_state_id": "state.block-a.mutable-runtime",
                    "materialization_id": "mat.runtime.vram",
                    "lineage": "r1-block-a-epoch-1",
                }
            ],
            "execution": [
                {
                    "id": "exec.block-a",
                    "strategy_unit": "qwen-block-a.same-backend-resident-decode-v1",
                    "compute_unit_id": "gpu0",
                    "required_state": [
                        {
                            "logical_state_id": "state.block-a.non-routed",
                            "representations": ["freetoken-native-device"],
                            "memory_resources": ["gpu0.vram"],
                        },
                        {
                            "logical_state_id": "state.block-a.routed",
                            "representations": ["freetoken-nvfp4-slot-banks"],
                            "memory_resources": ["gpu0.vram"],
                        },
                        {
                            "logical_state_id": "state.block-a.mutable-runtime",
                            "representations": ["freetoken-block-runtime-device"],
                            "memory_resources": ["gpu0.vram"],
                        },
                    ],
                }
            ],
            "forbidden_persistent_materializations": [
                {
                    "logical_state_id": "state.block-a.routed",
                    "memory_resource_id": "node-a.ram",
                }
            ],
            "memory_expectations": {
                "persistent_optional_bytes": 0,
                "unplanned_persistent_bytes": 0,
                "unexplained_persistent_host_mirror_bytes": 0,
            },
            "adapter_data": {
                "spec": block["spec"],
                "allowed_tensor_keys": block["allowed_tensor_keys"],
            },
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n0-plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan = build_plan(json.loads(Path(args.n0_plan).read_text()))
    out = Path(args.out)
    payload = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    out.with_suffix(out.suffix + ".sha256").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  {out.name}\n"
    )
    print(plan["digest"])


if __name__ == "__main__":
    main()
