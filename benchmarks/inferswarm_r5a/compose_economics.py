"""Compare matched resident same-node and two-node R5A serving arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha


def compose(local: dict, network: dict) -> dict:
    if local["producer_freetoken_sha"] != network["producer_freetoken_sha"]:
        raise ValueError("matched placement arms have different producers")
    if local["environment_digest"] != network["environment_digest"]:
        raise ValueError("matched placement arms have different frozen environments")
    metrics = {}
    for metric in ("ttft_ms", "complete_request_wall_ms", "decode_tok_s"):
        local_value = float(local["summary"][metric]["median"])
        network_value = float(network["summary"][metric]["median"])
        metrics[metric] = {
            "resident_same_node": local_value,
            "resident_two_node": network_value,
            "two_node_minus_same_node": network_value - local_value,
            "two_node_over_same_node": network_value / local_value,
            "evidence_label": "MEASURED_MATCHED_PLACEMENT_COMPARISON",
        }
    return {
        "schema": "inferswarm.r5a.network-placement-economics/1",
        "producer_freetoken_sha": local["producer_freetoken_sha"],
        "environment_digest": local["environment_digest"],
        "methodology": {
            "same_model_workloads": True,
            "same_warmups_repetitions": True,
            "same_generation_settings": True,
            "same_ordinary_serving_entry": True,
        },
        "metrics": metrics,
        "residual_interpretation": (
            "step wall minus Block A compute minus Block B compute remains a combined "
            "staging/transport/protocol residual in both arms; it is not pure network time"
        ),
        "local_arm_id": local["arm_id"],
        "network_arm_id": network["arm_id"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compose(
        json.loads(args.local.read_text()), json.loads(args.network.read_text())
    )
    write_json_with_sha(args.out, result)
    print(json.dumps({"out": str(args.out), "metrics": result["metrics"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
