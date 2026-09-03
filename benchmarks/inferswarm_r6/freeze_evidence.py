"""Freeze the R6 controlled evidence-collection override (issue #65).

The generic planner finds the single legal dense shape feasible but
FEASIBLE_UNRANKED (no applicable ranking evidence exists for a dense
Gemma chain anywhere in the accepted record).  Issue #65 explicitly
allows benchmarking a legal candidate via a controlled evidence-collection
override that remains visibly distinct from automatic selection.  This
script freezes that override record BEFORE the canonical run, bound to
the producer and the validation-run measurement.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from freetoken.research.r6_dense_census import write_json_with_sha

SHAPE_ID = "resident-two-node-three-slot"
MAPPING = {
    "slot-stage-1": "gpu.node-a.0",
    "slot-stage-2": "gpu.node-a.1",
    "slot-stage-3": "gpu.node-b.0",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-run", required=True,
                        help="r6_twonode_validation.json from the chain run")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    producer = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    run = json.loads(Path(args.validation_run).read_text())
    session = run["session"]
    wall_ms = session["complete_request_wall_ns"] / 1e6
    ttft_ms = session["ttft_ns"] / 1e6

    records = {
        "schema": "inferswarm.r6.serving-evidence/1",
        "controlled_evidence_collection_override": {
            "declared": True,
            "distinct_from_automatic_selection": True,
            "reason": "no ranking evidence exists for any dense multi-stage "
            "shape; issue #65 permits a controlled evidence-collection "
            "override for exactly this case",
            "basis": "two-node chain validation run (pre-canonical)",
        },
        "records": [
            {
                "id": "r6-chain-validation-ttft",
                "role": "RANKING_OBJECTIVE",
                "producer_identity": producer,
                "evidence_identity": "r6-two-node-chain-validation",
                "shape_id": SHAPE_ID,
                "mapping": MAPPING,
                "required_context": {
                    "model_revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
                    "runtime_context": f"r6-dense-producer:{producer}",
                    "network_context": "1GbE-LAN-MTU1500-node-a-to-node-b",
                    "workload_geometry": "r6-dense-3stage-chain",
                },
                "freshness": "CURRENT",
                "measurement_status": "MEASURED",
                "evidence_class": "MEASURED_R6_CONTROLLED_CHAIN_VALIDATION",
                "confidence": "EXACT_CONTEXT",
                "metric": {
                    "name": "ttft_ms",
                    "value": ttft_ms,
                    "unit": "ms",
                    "statistic": "single-run",
                },
                "provenance": {
                    "source": args.validation_run,
                    "generated_token_ids": session["generated_token_ids"],
                    "complete_request_wall_ms": wall_ms,
                },
            }
        ],
    }
    write_json_with_sha(args.out, records)
    print("wrote", args.out, "ttft_ms", round(ttft_ms, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
