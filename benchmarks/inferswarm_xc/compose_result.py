"""Validate and retain the canonical external-Coordinator evidence (#67)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED = [9764, 393, 45, 283, 220, 24, 22, 853]


def _load(path: Path):
    return json.loads(path.read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--producer-sha", required=True)
    args = parser.parse_args(argv)
    root = args.dir

    environment = _load(root / "r5b-env/frozen-environment.json")
    assert environment["implementation_commit"] == args.producer_sha
    serving = _load(root / "lifecycle/serving-report.json")
    request = serving["coordinator_scope"]["requests"][0]

    checks = {
        "exact_token_equality": request["generated_token_ids"] == EXPECTED,
        "single_epoch_attribution": len(set(request["committed_epoch_ids"])) == 1,
        "single_plan_attribution": len(set(request["committed_plan_digests"])) == 1,
        "reconciliation_matched": serving["epochs"][0]["reconciliation"]["matched"],
        "session_not_failed": not serving["sessions"][0].get("failed", False),
        "fencing_arm_rejected_all": all(
            not injection["accepted"]
            for injection in request["fencing_arm_injections"]
        ),
        "fencing_rejections_retained": serving["late_result_rejection_count"] == 2,
        "rejection_reasons": sorted(
            item["reason"] for item in serving["late_result_rejections"]
        ),
        "epoch_reclaimed_after_shutdown": serving["epochs"][0]["state"] == "RECLAIMED",
        "reclamation_recorded": bool(serving["epochs"][0]["reclamation"]),
        "final_runtime_report_retained": bool(
            serving["epochs"][0]["final_runtime_report"]
        ),
        "usage_counts": {
            "runtime_generate_calls": len(serving["epochs"][0]["runtime_sessions"]),
        },
    }
    result = {
        "schema": "inferswarm.xc.result/1",
        "gate": "EXTERNAL_COORDINATOR_SEPARATION",
        "implementation_producer_sha": args.producer_sha,
        "environment_digest": environment["digest"],
        "committed_token_ids": request["generated_token_ids"],
        "expected_token_ids": EXPECTED,
        "checks": checks,
        "passed": all(
            value if isinstance(value, bool) else True
            for value in checks.values()
            if not isinstance(value, (list, dict))
        )
        and checks["exact_token_equality"],
        "non_claims": [
            "no Coordinator HA, election, consensus, or term fencing",
            "no public Node-agent API or production wire protocol",
            "no production daemon",
            "no Gemma 4 support",
            "no R6 architecture claim",
            "no new Qwen placement/performance claims",
        ],
    }
    print(json.dumps({"passed": result["passed"], "failed": [
        k for k, v in checks.items() if v is False
    ]}))
    (root / "result-check.json").write_text(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
