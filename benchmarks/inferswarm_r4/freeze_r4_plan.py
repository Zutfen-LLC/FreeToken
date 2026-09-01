"""Freeze the R4 two-node plan from accepted R2 plan + node hardware freezes.

Runs on Node A after both node profiles are captured and the canonical link
is proven.  Also runs the generic R3 planner over the two-node candidate to
retain the FEASIBLE_UNRANKED evidence-collection authorization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha
from freetoken.research.r3_planner import plan as generic_plan

from benchmarks.inferswarm_r4.r4_plan import (
    CANONICAL_LINK,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    build_r4_plan,
    r4_network_problem,
    r4_resource_snapshot,
    validate_network_candidate,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2-plan", type=Path, required=True)
    parser.add_argument("--node-a-profile", type=Path, required=True)
    parser.add_argument("--node-b-profile", type=Path, required=True)
    parser.add_argument("--implementation-commit", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    r2_plan = json.loads(args.r2_plan.read_text())
    node_a = json.loads(args.node_a_profile.read_text())
    node_b = json.loads(args.node_b_profile.read_text())
    plan = build_r4_plan(
        r2_plan,
        node_a_hardware=node_a,
        node_b_hardware=node_b,
        link_freeze=CANONICAL_LINK,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_with_sha(args.out_dir / "r4-frozen-plan.json", plan)

    problem = r4_network_problem(args.implementation_commit)
    snapshot = r4_resource_snapshot(node_a, node_b)
    objective = {
        "schema": "inferswarm.r3.objective/1",
        "metric": "warm_decode_tok_s",
        "direction": "MAXIMIZE",
        "unit": "tok/s",
        "statistic": "median",
    }
    policy = {
        "schema": "inferswarm.r3.operator-policy/1",
        "rules": [],
    }
    from freetoken.research.r3_planner import freeze as freeze_generic

    decision = generic_plan(
        problem,
        freeze_generic(snapshot),
        freeze_generic(policy),
        freeze_generic(objective),
        evidence_catalog=freeze_generic({"records": []}),
    )
    authorization = validate_network_candidate(plan, decision)
    write_json_with_sha(args.out_dir / "planner-authorization.json", authorization)
    print(
        json.dumps(
            {
                "r4_plan_digest": plan["digest"],
                "candidate_state": authorization["state"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
