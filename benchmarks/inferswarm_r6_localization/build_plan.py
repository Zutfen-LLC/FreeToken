"""#71 localization: build the #71 participant chain plan (producer-bound).

Re-freezes the accepted 3-stage block plan from this branch's producer SHA
(same accepted freeze_dense_block_plan code, same census, same specs). The
accepted R6 physical producer 44d6c94 remains the historical R6 plan's
producer; the #71 plan is a NEW producer-bound plan for the localization
arms, recorded as such in provenance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from freetoken.research.r6_dense_census import (
        DenseBlockSpec,
        checkpoint_census,
        freeze_dense_block_plan,
    )

    repo = Path(__file__).resolve().parents[2]
    producer = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), "status", "--porcelain"],
        text=True,
    )
    if status:
        raise RuntimeError("plan freeze refuses a dirty tree")

    model = Path(args.model).resolve()
    census = checkpoint_census(model, text_prefix="model.language_model")
    shared = {
        "id": "tied-embedding-lm-head",
        "kind": "tied-weight-shared-state",
        "tensor_keys": ["model.language_model.embed_tokens.weight"],
        "bytes": census["bytes_by_owner_category"]["embedding/input"],
        "materialization_policy": "duplicated-on-first-and-last-stage",
        "reason": "tied lm_head reuses the embedding table (accepted R6 "
                  "declared shared logical state)",
    }
    plan = freeze_dense_block_plan(
        census,
        [
            DenseBlockSpec(0, 16, True, False),
            DenseBlockSpec(16, 32, False, False),
            DenseBlockSpec(32, 48, False, True),
        ],
        declared_shared_state=shared,
    )
    plan["provenance"] = {
        "r6_localization_71": {
            "producer_sha": producer,
            "supersedes_for_localization_only": (
                "docs/inferswarm_r6/chain-plan.json (R6 producer 44d6c94)"
            ),
            "note": "same accepted plan code/specs; re-frozen bound to the "
                    "#71 localization producer SHA",
        }
    }
    plan["runtime_capacity_tokens"] = 256
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": args.out, "producer": producer,
                      "blocks": [b["spec"] for b in plan["blocks"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
