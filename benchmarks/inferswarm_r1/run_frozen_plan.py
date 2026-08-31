"""Realize and execute the pinned R1 frozen research plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
from freetoken.research.n0_model_block import write_json_with_sha
from freetoken.research.r1_frozen_plan import RESULT_SCHEMA, realize_frozen_plan

from benchmarks.inferswarm_r1.qwen_adapter import QwenBlockAResearchAdapter


def _physical_environment(plan):
    properties = torch.cuda.get_device_properties(0)
    uuid = f"GPU-{torch.cuda.get_device_properties(0).uuid}"
    planned_node = plan["resources"]["nodes"][0]
    planned_compute = planned_node["compute_units"][0]
    if uuid != planned_compute["device_uuid"]:
        raise RuntimeError(
            f"physical device UUID {uuid!r} does not match frozen resource {planned_compute['device_uuid']!r}"
        )
    ram_bytes = (
        int(Path("/proc/meminfo").read_text().split("MemTotal:", 1)[1].split()[0])
        * 1024
    )
    resources = json.loads(json.dumps(plan["resources"]))
    resources["nodes"][0]["memory_resources"] = [
        {
            "id": "gpu0.vram",
            "kind": "accelerator-vram",
            "capacity_bytes": properties.total_memory,
        },
        {"id": "node-a.ram", "kind": "system-ram", "capacity_bytes": ram_bytes},
    ]
    return {
        "model_repository": plan["model"]["repository"],
        "model_revision": plan["model"]["revision"],
        "device": "cuda:0",
        "resources": resources,
        "physical": {
            "hostname": platform.node(),
            "gpu_name": properties.name,
            "gpu_uuid": uuid,
            "gpu_total_memory_bytes": properties.total_memory,
            "gpu_compute_capability": [properties.major, properties.minor],
            "system_ram_bytes": ram_bytes,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())
    file_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    environment = _physical_environment(plan)
    adapter = QwenBlockAResearchAdapter(
        model_path=args.model, fixture_path=args.fixture, repetitions=args.repetitions
    )
    realization = realize_frozen_plan(plan, environment, adapter)
    correctness = adapter.run_repeated_decode_and_audit()
    memory = adapter.memory_report()
    passed = (
        realization.validation["passed"]
        and realization.reconciliation["passed"]
        and correctness["passed"]
        and memory["unplanned_persistent_bytes"] == 0
        and memory["unexplained_persistent_host_mirror_bytes"] == 0
    )
    payload = {
        "schema": RESULT_SCHEMA,
        "status": "R1_FROZEN_PLAN_REALIZATION_PASS"
        if passed
        else "R1_FROZEN_PLAN_REALIZATION_FAIL",
        "passed": passed,
        "plan": {
            "digest": plan["digest"],
            "file_sha256": file_sha,
            "schema": plan["schema"],
            "version": plan["plan_version"],
            "model": plan["model"]["repository"],
            "revision": plan["model"]["revision"],
        },
        "resources": environment,
        "validation": realization.validation,
        "intended": {
            "logical_state_units": plan["logical_state_units"],
            "materializations": plan["materializations"],
            "execution": plan["execution"],
            "authorities": plan["authorities"],
        },
        "observed": {
            "materializations": realization.observed_materializations,
            "execution": realization.observed_execution,
            "authorities": realization.observed_authorities,
            "process_checkpoints": adapter.checkpoints,
        },
        "reconciliation": realization.reconciliation,
        "memory": memory,
        "correctness": correctness,
        "backend_execution": {
            "setup_time_plan_interpretation_only": True,
            "resident_device_mapping": True,
            "hot_path_plan_lookups": 0,
            "cuda_graph": "structurally preserved and regression-tested; not physically captured by this evidence runner",
        },
        "provenance": {
            "freetoken_base_sha": "2d435c5b8addc4032fd1be3198d796e632648ac8",
            "inferswarm_issue": "https://github.com/Zutfen-LLC/inferswarm/issues/50",
        },
        "scope": {
            "proves": "one predeclared doctrine-shaped Block A plan caused exact validated resource/state/materialization realization and correct backend-native repeated decode",
            "does_not_prove": "automatic planning, candidate comparison, R2 split execution, multi-GPU or multi-node execution, networking, elasticity, public planner/strategy APIs, discovery, or other model/vendor generality",
        },
    }
    write_json_with_sha(args.out, payload)
    print(json.dumps(payload, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
