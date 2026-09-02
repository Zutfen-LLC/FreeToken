"""R5B physical freeze layered on the accepted R5A fail-closed preflight."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

try:
    from benchmarks.inferswarm_r5a.preflight import freeze_physical_environment
    from benchmarks.inferswarm_r5b.strategy import transition_policy
except ModuleNotFoundError:
    from inferswarm_r5a.preflight import freeze_physical_environment
    from inferswarm_r5b.strategy import transition_policy
from freetoken.research.n0_model_block import write_json_with_sha
from freetoken.research.r3_planner import freeze

ACCEPTED_BASE = "d9f45a9ef7b5f89800f96c54397202a7d43beb52"


def freeze_r5b(args):
    merge_base = subprocess.check_output(
        ["git", "-C", args.node_a_repo, "merge-base", args.producer_sha, ACCEPTED_BASE],
        text=True,
    ).strip()
    if merge_base != ACCEPTED_BASE:
        raise RuntimeError("R5B producer is not descended from the exact accepted R5A head")
    environment, participant, gate = freeze_physical_environment(args)
    initial = json.loads(json.dumps(environment))
    initial.pop("digest", None)
    initial["schema"] = "inferswarm.r5b.frozen-environment/1"
    initial["accepted_r5a_base_sha"] = ACCEPTED_BASE
    initial["initial_resource_availability"] = {
        "gpu.node-a.0": "AVAILABLE",
        "gpu.node-a.1": "UNAVAILABLE_PARTICIPANT_NOT_STARTED",
        "gpu.node-b.0": "AVAILABLE",
    }
    initial["transition_policy"] = transition_policy(args.producer_sha)
    frozen = freeze(initial)
    write_json_with_sha(args.out_dir / "frozen-environment.json", frozen)
    write_json_with_sha(
        args.out_dir / "initial-resource-graph.json",
        {
            "schema": "inferswarm.r5b.initial-resource-graph/1",
            "producer_sha": args.producer_sha,
            "environment_digest": frozen["digest"],
            "availability": frozen["initial_resource_availability"],
            "gpu_a1_execution_participant_started": False,
            "hidden_gpu_a1_runtime_use": False,
        },
    )
    return frozen, participant, gate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-sha", required=True)
    parser.add_argument("--node-a-repo", required=True)
    parser.add_argument("--node-b-repo", required=True)
    parser.add_argument("--node-a-model", required=True)
    parser.add_argument("--node-b-model", required=True)
    parser.add_argument("--node-b-python", required=True)
    parser.add_argument("--ssh-node-b", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    environment, participant, gate = freeze_r5b(args)
    print(
        json.dumps(
            {
                "environment_digest": environment["digest"],
                "participant_plan_digest": participant["digest"],
                "gate": gate["result"],
                "accepted_base": ACCEPTED_BASE,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
