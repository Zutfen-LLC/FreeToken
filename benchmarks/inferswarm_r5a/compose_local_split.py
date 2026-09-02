"""Compose same-node resident split serving evidence through accepted R2."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from benchmarks.inferswarm_r4.run_experiment import compare_selected_logits, load_workloads
from benchmarks.inferswarm_r5a.compose import _percentile, _session_class
from benchmarks.inferswarm_r5a.strategy import LOCAL_SPLIT_SHAPE, serving_evidence_records
from freetoken.research.n0_model_block import write_json_with_sha
from freetoken.research.r3_planner import freeze, require_frozen


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose_local_split_arm(
    *,
    environment: dict[str, Any],
    http_arm: dict[str, Any],
    serving_report: dict[str, Any],
    reference: dict[str, dict[str, Any]],
    arm_id: str,
    arm_mode: str,
) -> dict[str, Any]:
    require_frozen(environment, "R5A frozen environment")
    execution_plan = serving_report["execution_plan"]
    require_frozen(execution_plan, "R5A local execution plan")
    if execution_plan["strategy_realization"]["path"] != "r2-local-split":
        raise ValueError("serving report is not a same-node R2 realization")
    runtime = serving_report["runtime"]
    if execution_plan["strategy_realization"]["participant_plan_digest"] != runtime[
        "transport_accounting"
    ]["participant_plan_digest"]:
        raise ValueError("local runtime belongs to another participant plan")
    http_by_ordinal = {row["request_ordinal"]: row for row in http_arm["requests"]}
    rows = []
    for session in runtime["sessions"]:
        class_id = _session_class(session, reference)
        request = http_by_ordinal.get(session["session_id"])
        if request is None:
            raise ValueError(f"no HTTP record for local session {session['session_id']}")
        expected = reference[class_id]
        logits = {
            str(item["generated_step"]): item["logits"]
            for item in session["boundaries"]
            if "logits" in item
        }
        selected_logits = compare_selected_logits(logits, expected) if logits else None
        checksums = [
            item.get("producer_sha256") == item.get("consumer_sha256")
            for item in session["boundaries"]
            if item.get("producer_sha256") is not None
        ]
        residual_ns = sum(
            max(
                0,
                int(item["step_wall_ns"])
                - int(item["block_a_compute_ns"])
                - int(item["block_b_compute_ns"]),
            )
            for item in session["boundaries"]
        )
        inter = [int(value) for value in session["inter_token_latency_ns"]]
        rows.append(
            {
                "class_id": class_id,
                "session_id": session["session_id"],
                "sample_kind": request.get("sample_kind", "measured"),
                "repetition": request.get("repetition", 0),
                "prompt_token_ids_exact": session["prompt_token_ids"]
                == expected["prompt_token_ids"],
                "generated_token_ids_exact": session["generated_token_ids"]
                == expected["generated_token_ids"],
                "generated_token_ids": session["generated_token_ids"],
                "selected_logits": selected_logits,
                "boundary_checksums_all_match": all(checksums) if checksums else None,
                "nan_count": selected_logits["nan_count"] if selected_logits else None,
                "inf_count": selected_logits["inf_count"] if selected_logits else None,
                "service": {
                    "ttft_ms": session["ttft_ns"] / 1e6,
                    "prefill_wall_ms": session["prefill_wall_ns"] / 1e6,
                    "prefill_tok_s": session["prompt_token_count"]
                    / (session["prefill_wall_ns"] / 1e9),
                    "decode_tok_s": session["decode_tokens_per_second"],
                    "inter_token_latency_ns": inter,
                    "inter_token_p50_ns": statistics.median(inter),
                    "inter_token_p95_ns": _percentile(inter, 0.95),
                    "complete_request_wall_ms": session["total_request_wall_ns"] / 1e6,
                    "boundary_semantic_bytes": session["boundary_bytes"],
                    "measured_staging_transport_protocol_residual_ms": residual_ns / 1e6,
                    "measured_residual_fraction_of_request": residual_ns
                    / session["total_request_wall_ns"],
                },
                "http_service": {
                    "ttft_ms": request["ttft_ns"] / 1e6,
                    "complete_request_wall_ms": request["complete_request_wall_ns"] / 1e6,
                    "decode_tok_s": request["decode_tok_s"],
                    "output_event_interval_p50_ns": request[
                        "output_event_interval_p50_ns"
                    ],
                    "usage": request["usage"],
                },
            }
        )
    measured = [row for row in rows if row["sample_kind"] == "measured"]
    summary = {}
    for metric in ("ttft_ms", "complete_request_wall_ms", "decode_tok_s"):
        values = [float(row["service"][metric]) for row in measured]
        summary[metric] = {
            "evidence_label": "MEASURED",
            "values": values,
            "median": statistics.median(values),
        }
    reports = runtime["participants"]
    invariant_keys = (
        "fallbacks",
        "host_expert_fetches",
        "resident_source_accesses",
        "unexplained_persistent_host_mirror_bytes",
        "steady_model_state_movement_bytes",
    )
    invariants = {}
    for role in ("a", "b"):
        report = reports[role]["runtime"]
        values = {key: report.get(key) for key in invariant_keys}
        values["graph_recaptures"] = report.get("decode_graph", {}).get("recaptures")
        invariants[f"block_{role}"] = values
    invariants["all_required_zero"] = all(
        value == 0
        for role in ("block_a", "block_b")
        for value in invariants[role].values()
    )
    return {
        "schema": "inferswarm.r5a.local-resident-serving-summary/1",
        "arm_id": arm_id,
        "arm_mode": arm_mode,
        "producer_freetoken_sha": environment["implementation_commit"],
        "environment_digest": environment["digest"],
        "plan_digest": execution_plan["digest"],
        "participant_plan_digest": execution_plan["strategy_realization"][
            "participant_plan_digest"
        ],
        "selection_authorization": execution_plan["selection_authorization"],
        "request_entry_contract": http_arm["request_entry_contract"],
        "bounded_concurrency": http_arm["bounded_concurrency"],
        "max_outstanding_requests": serving_report["max_outstanding_requests"],
        "sessions": sorted(rows, key=lambda row: (row["class_id"], row["repetition"])),
        "summary": summary,
        "correctness": {
            "all_prompt_tokens_exact": all(row["prompt_token_ids_exact"] for row in rows),
            "all_generated_tokens_exact": all(
                row["generated_token_ids_exact"] for row in rows
            ),
            "all_selected_logits_within_threshold": (
                all(
                    row["selected_logits"] is not None
                    and row["selected_logits"]["all_exact"]
                    for row in rows
                )
                if arm_mode == "diagnostic"
                else None
            ),
            "all_boundary_checksums_match": (
                all(row["boundary_checksums_all_match"] is True for row in rows)
                if arm_mode == "diagnostic"
                else None
            ),
            "nan_count": sum(int(row["nan_count"] or 0) for row in rows),
            "inf_count": sum(int(row["inf_count"] or 0) for row in rows),
        },
        "transport_accounting": runtime["transport_accounting"],
        "realization": {
            role: runtime["ready"][role]["realization"] for role in ("a", "b")
        },
        "participant_reports": reports,
        "runtime_invariants": invariants,
        "paging_swap": {
            role: {
                "process_current": reports[role]["runtime"].get("process_current"),
                "vmstat_delta": reports[role]["runtime"].get("vmstat_delta"),
            }
            for role in ("a", "b")
        },
        "http_artifact_sha256": http_arm.get("artifact_sha256"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--http-arm", type=Path, required=True)
    parser.add_argument("--serving-report", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument(
        "--arm-mode", choices=("diagnostic", "clean", "concurrency"), required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence-out", type=Path)
    args = parser.parse_args(argv)
    environment = json.loads(args.environment.read_text())
    http = json.loads(args.http_arm.read_text())
    http["artifact_sha256"] = _sha(args.http_arm)
    result = compose_local_split_arm(
        environment=environment,
        http_arm=http,
        serving_report=json.loads(args.serving_report.read_text()),
        reference=load_workloads(args.reference),
        arm_id=args.arm_id,
        arm_mode=args.arm_mode,
    )
    write_json_with_sha(args.out, result)
    if args.evidence_out:
        source = dict(result)
        source["artifact_sha256"] = _sha(args.out)
        context = {
            "model_revision": environment["model"]["revision"],
            "runtime_context": environment["runtime_context"],
            "network_context": environment["network_context"],
            "workload_geometry": "W2-W4-generate32-static",
        }
        records = freeze(
            {
                "schema": "inferswarm.r5a.serving-evidence-derivative/1",
                "implementation_commit": environment["implementation_commit"],
                "records": serving_evidence_records(
                    source, shape_id=LOCAL_SPLIT_SHAPE, context=context
                ),
            }
        )
        write_json_with_sha(args.evidence_out, records)
    print(json.dumps({"out": str(args.out), "correctness": result["correctness"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
