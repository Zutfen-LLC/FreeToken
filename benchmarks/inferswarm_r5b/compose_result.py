"""Validate and retain the canonical R5B physical and regression evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import statistics
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha

EXPECTED = [9764, 393, 45, 283, 220, 24, 22, 853]
EXPECTED_CANDIDATE_KINDS = [
    "resident-two-node-two-slot",
    "resident-same-node-two-slot",
    "resident-two-node-two-slot",
    "resident-same-node-two-slot",
]


def _load(path: Path):
    return json.loads(path.read_text())


def _runtime_safety(report):
    records = []
    for epoch in report["epochs"]:
        final = epoch.get("final_runtime_report", {})
        runtimes = []
        if final.get("runtime"):
            runtimes.append(final["runtime"])
        runtimes.extend(
            item["runtime"]
            for item in final.get("participants", {}).values()
            if item.get("runtime")
        )
        if not runtimes:
            records.append(
                {
                    "generation": epoch["generation"],
                    "available": False,
                    "reason": epoch.get("runtime_report_failure"),
                    "controlled_failed_epoch": epoch["generation"] == 1,
                }
            )
            continue
        records.append(
            {
                "generation": epoch["generation"],
                "available": True,
                "participant_count": len(runtimes),
                "resident_only": all(item.get("resident_only") for item in runtimes),
                "host_expert_fetches": sum(item.get("host_expert_fetches", 0) for item in runtimes),
                "resident_source_accesses": sum(item.get("resident_source_accesses", 0) for item in runtimes),
                "fallbacks": sum(item.get("fallbacks", 0) for item in runtimes),
                "steady_model_state_movement_bytes": sum(item.get("steady_model_state_movement_bytes", 0) for item in runtimes),
                "unexplained_persistent_host_mirror_bytes": sum(item.get("unexplained_persistent_host_mirror_bytes", 0) for item in runtimes),
                "vmstat_delta_by_participant": [item.get("vmstat_delta") for item in runtimes],
                "host_materialization_accounting_by_participant": [item.get("host_materialization_accounting") for item in runtimes],
            }
        )
    return records


def compose(staging: Path, destination: Path, physical_producer: str):
    environment = _load(staging / "preflight/frozen-environment.json")
    gate = _load(staging / "preflight/preflight-gate.json")
    active = _load(staging / "lifecycle/serving-report-after-request.json")
    shutdown = _load(staging / "lifecycle/serving-report.json")
    http = _load(staging / "lifecycle/http-lifecycle.json")
    postflight = _load(staging / "lifecycle/post-shutdown-resource-state.json")
    session = active["sessions"][0]
    epochs = active["epochs"]
    transitions = active["transitions"]
    candidate_kinds = [item["candidate_id"].split("[")[0] for item in epochs]
    regression_expected = {"research": 199, "benchmarks": 563, "server": 581}
    regressions = {}
    for name, count in regression_expected.items():
        text = (staging / f"regressions/{name}.txt").read_text()
        match = re.search(rf"{count} passed", text)
        regressions[name] = {
            "expected_passed": count,
            "passed": bool(match),
            "warning_count": 1 if name == "server" and "1 warning" in text else 0,
        }
    negative_text = (staging / "regressions/r5b-negative-arms.txt").read_text()
    regressions["r5b_negative_arms"] = {
        "expected_passed": 4,
        "passed": "4 passed" in negative_text,
        "cases": [
            "failed preparation retains old authority",
            "unrecoverable mutable state fails closed",
            "retired result cannot change state or accounting",
            "participant loss publication is atomic",
        ],
        "warning_count": 0,
    }

    logits = [
        boundary["logits"]
        for epoch in epochs
        for runtime_session in epoch["runtime_sessions"]
        for boundary in runtime_session.get("boundaries", [])
        if boundary.get("logits")
    ]
    physical_loss = epochs[1]["reclamation"]
    checks = {
        "exact_physical_producer": environment["implementation_commit"] == physical_producer,
        "preflight": gate["result"] == "ALL_PREFLIGHT_CHECKS_PASSED",
        "exact_output": session["generated_token_ids"] == EXPECTED,
        "http_accounting": http["usage"] == {"prompt_tokens": 54, "completion_tokens": 8, "total_tokens": 62},
        "three_activated_transitions": len(transitions) == 3 and all(item["status"] == "ACTIVATED" for item in transitions),
        "planner_selected_expected_truthful_sequence": candidate_kinds == EXPECTED_CANDIDATE_KINDS,
        "distinct_epoch_generations": [item["generation"] for item in epochs] == [0, 1, 2, 3] and len({item["epoch_id"] for item in epochs}) == 4,
        "all_realizations_reconciled": all(item["reconciliation"]["matched"] for item in epochs),
        "single_authority_after_request": active["single_mutable_authority"] and active["active_epoch_id"] == epochs[3]["epoch_id"],
        "all_reclaimed_after_shutdown": not shutdown["single_mutable_authority"] and all(item["state"] == "RECLAIMED" for item in shutdown["epochs"]),
        "post_shutdown_gpu_processes_absent": postflight[
            "all_gpu_execution_processes_absent"
        ],
        "late_result_fenced": active["late_result_rejection_count"] == 1 and active["late_result_rejections"][0]["reason"] == "RETIRED_OR_SUPERSEDED_EPOCH",
        "physical_gpu1_worker_loss": physical_loss["participant_exit_codes"]["b"] == -15 and physical_loss["all_participants_stopped"],
        "logits_finite": bool(logits) and all(item["nan_count"] == 0 and item["inf_count"] == 0 for item in logits),
        "regressions": all(item["passed"] for item in regressions.values()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"R5B pass gate failed: {checks}")

    gaps = http["inter_output_gaps_ns"]
    transitions_out = []
    for index, item in enumerate(transitions):
        replacement = epochs[item["replacement_generation"]]
        first = replacement["runtime_sessions"][0]
        transitions_out.append(
            {
                "trigger": item["transition_trigger"]["event_id"],
                "old_epoch_id": item["old_epoch_id"],
                "replacement_epoch_id": item["replacement_epoch_id"],
                "replacement_candidate_id": item["replacement_candidate_id"],
                "planner_decision_digest": item["planner_decision_digest"],
                "resource_snapshot_digest": item["resource_snapshot_digest"],
                "replanning_wall_ns": item["replanning_wall_ns"],
                "immutable_preparation_wall_ns": item["preparation_ended_at_ns"] - item["preparation_started_at_ns"],
                "old_settle_and_reclaim_wall_ns": item["replacement_realization_started_at_ns"] - item["authority_cutover_started_at_ns"],
                "replacement_realization_wall_ns": item["replacement_realization_ended_at_ns"] - item["replacement_realization_started_at_ns"],
                "total_authority_cutover_wall_ns": item["authority_cutover_ns"],
                "client_visible_inter_token_gap_ns": gaps[[1, 2, 5][index]],
                "replay_source": item["recovery_replay_source"],
                "replay_range": item["recovery_replay_range"],
                "replay_token_count": item["replay_token_count"],
                "replay_plus_first_candidate_prefill_wall_ns": first["prefill_wall_ns"],
                "first_post_transition_runtime_request_wall_ns": first["total_request_wall_ns"],
            }
        )

    epoch_service = []
    for item in epochs:
        sessions = item["runtime_sessions"]
        epoch_service.append(
            {
                "generation": item["generation"],
                "epoch_id": item["epoch_id"],
                "plan_digest": item["plan_digest"],
                "candidate_id": item["candidate_id"],
                "runtime_requests": len(sessions),
                "median_runtime_ttft_ns": statistics.median(s["ttft_ns"] for s in sessions),
                "median_decode_tokens_per_second": statistics.median(s["decode_tokens_per_second"] for s in sessions),
                "reclamation": shutdown["epochs"][item["generation"]]["reclamation"],
            }
        )

    result = {
        "schema": "inferswarm.r5b.result/1",
        "result": "R5B_PLAN_EPOCH_RECOVERY_PASS",
        "physical_producer_sha": physical_producer,
        "accepted_r5a_base_sha": environment["accepted_r5a_base_sha"],
        "environment_digest": environment["digest"],
        "methodology_sha256": hashlib.sha256((destination / "METHODOLOGY.md").read_bytes()).hexdigest(),
        "checks": checks,
        "correctness": {
            "prompt_token_count": 54,
            "expected_generated_token_ids": EXPECTED,
            "observed_generated_token_ids": session["generated_token_ids"],
            "committed_epoch_ids": session["committed_epoch_ids"],
            "committed_plan_digests": session["committed_plan_digests"],
            "selected_logits_records": len(logits),
            "nan_count": sum(item["nan_count"] for item in logits),
            "inf_count": sum(item["inf_count"] for item in logits),
        },
        "transitions": transitions_out,
        "epoch_service": epoch_service,
        "runtime_safety": _runtime_safety(shutdown),
        "http": {
            "request_wall_ns": http["request_wall_ns"],
            "maximum_client_visible_inter_token_gap_ns": http["maximum_client_visible_inter_token_gap_ns"],
            "usage": http["usage"],
        },
        "regressions": regressions,
        "measurement_note": "Nanosecond wall times are monotonic observed values. Cutover totals combine explicitly listed settle/reclaim and realization phases and are not labeled network latency.",
        "non_claims": [
            "no zero-downtime claim",
            "no production scheduler, daemon, protocol, or public epoch API",
            "no make-before-break runtime materialization claim",
            "no materially different model or R6 claim",
        ],
    }

    for name in ("preflight", "planning", "configs", "lifecycle", "regressions"):
        shutil.copytree(staging / name, destination / name, dirs_exist_ok=True)
    write_json_with_sha(destination / "test-summary.json", {"schema": "inferswarm.r5b.test-summary/1", "suites": regressions})
    write_json_with_sha(destination / "result.json", result)
    artifacts = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.name in {"MANIFEST.json", "MANIFEST.json.sha256"}:
            continue
        artifacts.append(
            {
                "path": path.relative_to(destination).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    write_json_with_sha(
        destination / "MANIFEST.json",
        {
            "schema": "inferswarm.r5b.artifact-manifest/1",
            "physical_producer_sha": physical_producer,
            "artifacts": artifacts,
        },
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--physical-producer", required=True)
    args = parser.parse_args(argv)
    result = compose(args.staging, args.destination, args.physical_producer)
    print(json.dumps({"result": result["result"], "transitions": len(result["transitions"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
