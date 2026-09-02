"""Create an auditable R5A evidence derivative for an unchanged R2/R4 runtime."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha
from freetoken.research.r3_planner import freeze, require_frozen


def requalify(
    source: dict,
    environment: dict,
    *,
    producer_sha: str,
    regression_evidence: str,
) -> dict:
    require_frozen(source, "accepted R5A serving evidence")
    require_frozen(environment, "R5B frozen environment")
    if environment["implementation_commit"] != producer_sha:
        raise ValueError("R5B environment names another producer")
    records = []
    for original in source["records"]:
        item = deepcopy(original)
        item["id"] = "r5b-requalified-" + original["id"]
        item["producer_identity"] = producer_sha
        item["required_context"]["runtime_context"] = environment["runtime_context"]
        item["required_context"]["network_context"] = environment["network_context"]
        item["freshness"] = "ACCEPTED_COMPATIBLE"
        item["confidence"] = "DEPENDENCY_SCOPED_REQUALIFICATION"
        item["evidence_class"] = "MEASURED_R5A_MATCHED_HTTP_SERVING_REQUALIFIED"
        item["provenance"] = {
            "accepted_measurement": deepcopy(original["provenance"]),
            "accepted_r5a_record_id": original["id"],
            "accepted_r5a_producer_identity": original["producer_identity"],
            "requalification_producer_sha": producer_sha,
            "basis": (
                "R5B adds orchestration above the unchanged accepted R2/R4 "
                "realizers and reruns their focused regressions"
            ),
            "regression_evidence": regression_evidence,
            "numeric_value_remeasured": False,
        }
        records.append(item)
    return freeze(
        {
            "schema": "inferswarm.r5b.serving-evidence-derivative/1",
            "implementation_commit": producer_sha,
            "source_digest": source["digest"],
            "environment_digest": environment["digest"],
            "records": records,
            "claim_scope": (
                "accepted context-specific R5A ranking evidence, dependency-scoped "
                "to the unchanged physical R2/R4 runtime; not new measurements"
            ),
        }
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--producer-sha", required=True)
    parser.add_argument("--regression-evidence", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = requalify(
        json.loads(args.source.read_text()),
        json.loads(args.environment.read_text()),
        producer_sha=args.producer_sha,
        regression_evidence=args.regression_evidence,
    )
    write_json_with_sha(args.out, result)
    print(json.dumps({"out": str(args.out), "records": len(result["records"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
