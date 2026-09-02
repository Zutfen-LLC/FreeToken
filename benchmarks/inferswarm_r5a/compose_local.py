"""Compose a matched ordinary-FreeToken local serving control for R5A ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from benchmarks.inferswarm_r5a.strategy import LOCAL_SHAPE, serving_evidence_records
from freetoken.research.n0_model_block import write_json_with_sha
from freetoken.research.r3_planner import freeze, require_frozen


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose_local(environment: dict, http: dict, arm_id: str) -> dict:
    require_frozen(environment, "R5A frozen environment")
    rows = []
    prefills = list(http.get("new_prefill_records", []))
    for index, request in enumerate(http["requests"]):
        prefill = prefills[index] if index < len(prefills) else None
        rows.append(
            {
                "class_id": request["class_id"],
                "sample_kind": request.get("sample_kind", "measured"),
                "repetition": request.get("repetition", 0),
                "http_service": {
                    "ttft_ms": request["ttft_ns"] / 1e6,
                    "complete_request_wall_ms": request["complete_request_wall_ns"] / 1e6,
                    "decode_tok_s": request["decode_tok_s"],
                    "output_event_intervals_ns": request["output_event_intervals_ns"],
                    "output_event_interval_p50_ns": request["output_event_interval_p50_ns"],
                    "usage": request["usage"],
                },
                "prefill": prefill,
                "response_text_sha256": hashlib.sha256(
                    (request["reasoning_content"] + request["content"]).encode()
                ).hexdigest(),
            }
        )
    summary = {}
    measured_rows = [row for row in rows if row["sample_kind"] == "measured"]
    for metric in ("ttft_ms", "complete_request_wall_ms", "decode_tok_s"):
        values = [float(row["http_service"][metric]) for row in measured_rows]
        summary[metric] = {
            "evidence_label": "MEASURED",
            "values": values,
            "median": statistics.median(values),
        }
    return {
        "schema": "inferswarm.r5a.local-http-control/1",
        "arm_id": arm_id,
        "producer_freetoken_sha": environment["implementation_commit"],
        "environment_digest": environment["digest"],
        "shape_id": LOCAL_SHAPE,
        "request_entry_contract": http["request_entry_contract"],
        "bounded_concurrency": http["bounded_concurrency"],
        "sessions": rows,
        "summary": summary,
        "instrumentation": http["instrumentation_after"],
        "correctness_scope": (
            "HTTP response retained; exact token/logit correctness is supplied by the "
            "separate frozen R2/R4 comparator requalification, not inferred from text."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--http-arm", type=Path, required=True)
    parser.add_argument("--arm-id", default="local-single-resource-http")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence-out", type=Path, required=True)
    args = parser.parse_args(argv)
    environment = json.loads(args.environment.read_text())
    result = compose_local(
        environment, json.loads(args.http_arm.read_text()), args.arm_id
    )
    write_json_with_sha(args.out, result)
    evidence_source = dict(result)
    evidence_source["artifact_sha256"] = _sha(args.out)
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
                evidence_source, shape_id=LOCAL_SHAPE, context=context
            ),
        }
    )
    write_json_with_sha(args.evidence_out, records)
    print(json.dumps({"out": str(args.out), "evidence": str(args.evidence_out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
