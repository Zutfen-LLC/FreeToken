from __future__ import annotations

import argparse
import json
from pathlib import Path

from freetoken.research.n0_model_block import (
    checkpoint_census,
    freeze_two_block_plan,
    write_json_with_sha,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--census-out", required=True)
    parser.add_argument("--plan-out", required=True)
    args = parser.parse_args()
    config = json.loads((Path(args.model) / "config.json").read_text())
    text = config.get("text_config", config)
    census = checkpoint_census(args.model)
    plan = freeze_two_block_plan(census, list(text["layer_types"]))
    write_json_with_sha(args.census_out, census)
    write_json_with_sha(args.plan_out, plan)
    print(json.dumps({
        "tensor_count": census["tensor_count"],
        "total_checkpoint_bytes": census["total_checkpoint_bytes"],
        "required_text_model_bytes": census["required_text_model_bytes"],
        "split_boundary": plan["split_boundary"],
        "block_a_bytes": plan["block_a"]["owned_checkpoint_bytes"],
        "block_b_bytes": plan["block_b"]["owned_checkpoint_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
