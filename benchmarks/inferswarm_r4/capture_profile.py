"""Capture one node's R4 hardware/network freeze profile (CLI).

Usage (on the node itself):
    python -m benchmarks.inferswarm_r4.capture_profile --node node-a|node-b --out profile.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha

from benchmarks.inferswarm_r4.node_preflight import (
    capture_node_profile,
    require_canonical_link,
)

NODES = {
    "node-a": {
        "node_id": "node.inferswarm01",
        "peer_ipv4": "10.0.0.219",
        "interface": "eno1",
        "model_path": "/srv/models/nvidia/Qwen3.6-35B-A3B-NVFP4/491c2f1ea524c639598bf8fa787a93fed5a6fbce",
    },
    "node-b": {
        "node_id": "node.inferswarm03",
        "peer_ipv4": "10.0.0.141",
        "interface": "enp5s0",
        "model_path": "/srv/models/nvidia/Qwen3.6-35B-A3B-NVFP4/491c2f1ea524c639598bf8fa787a93fed5a6fbce",
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", choices=sorted(NODES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    config = NODES[args.node]
    profile = capture_node_profile(
        node_id=config["node_id"],
        peer_ipv4=config["peer_ipv4"],
        interface=config["interface"],
        model_path=config["model_path"],
    )
    link = require_canonical_link(profile)
    write_json_with_sha(args.out, profile)
    print(
        json.dumps(
            {
                "node": args.node,
                "hostname": profile["hostname"],
                "link": link,
                "mem_available_gib": round(
                    profile["memory"]["mem_available_kib"] / 1024 / 1024, 3
                ),
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
