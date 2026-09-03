"""Freeze the R6 physical environment snapshot (CPU-only, no torch).

Captures producer SHA, GPU identities, and network context for the frozen
3-stage topology; writes environment.json (+sha sidecar).  Run on the
coordinator repository root; GPU fields filled from a producer-supplied
nvidia-smi capture passed as JSON files (keeps this host torch-free and
the capture physically measured on the compute nodes).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from freetoken.research.r6_dense_census import write_json_with_sha

MODEL_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
MODEL_SHA = "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d"


def gpu_rows(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        index, uuid, name, total_mib, bdf = (x.strip() for x in line.split(","))
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "vram_total_bytes": int(total_mib) * 1024 * 1024,
                "pci_bdf": bdf,
                "availability": "AVAILABLE",
            }
        )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-a-gpus", required=True,
                        help="nvidia-smi csv from inferswarm01 (2 rows)")
    parser.add_argument("--node-b-gpus", required=True,
                        help="nvidia-smi csv from inferswarm03 (row 0 = stage 3)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    producer = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
    )
    if status:
        raise RuntimeError("environment freeze refuses a dirty source tree")

    a = gpu_rows(args.node_a_gpus)
    b = gpu_rows(args.node_b_gpus)
    if len(a) < 2 or len(b) < 1:
        raise ValueError("expected 2 GPUs on node A and >=1 on node B")

    environment = {
        "schema": "inferswarm.r6.environment-freeze/1",
        "implementation_commit": producer,
        "model": {
            "repository": "google/gemma-4-12B-it",
            "revision": MODEL_REVISION,
            "checkpoint_sha256": MODEL_SHA,
            "serving_scope": "text-only",
            "representation": "native-bf16-safetensors",
        },
        "runtime_context": f"r6-dense-producer:{producer}",
        "network_context": "1GbE-LAN-MTU1500-node-a-to-node-b",
        "node_a": {"node_id": "node.inferswarm01", "gpus": a[:2]},
        "node_b": {"node_id": "node.inferswarm03", "gpus": b[:1]},
        "network": {
            "link_id": "path.node-a-to-node-b.1gbe",
            "negotiated_mbps": 1000,
            "available": True,
        },
    }
    write_json_with_sha(args.out, environment)
    print("wrote", args.out, "producer", producer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
