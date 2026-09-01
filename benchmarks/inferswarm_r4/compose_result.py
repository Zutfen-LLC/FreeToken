"""Compose the final R4 result.json from retained artifacts.

Runs on Node A after all arms complete.  Reads arm results, Node B
ready/report records, network characterization, microbenchmark, and
hardware freezes; emits the machine-readable final result plus the 1-GbE
disposition classification.  Fails closed if canonical artifacts are
missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CAPACITY_MARGIN = 0.80


def _load(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"canonical artifact missing: {path}")
    return json.loads(path.read_text())


def classify_one_gbe(
    clean_arm: dict[str, Any], microbench: dict[str, Any], iperf: dict[str, Any]
) -> dict[str, Any]:
    """Classify the 1-GbE arm using the frozen 80% capacity margin.

    Peak application wire demand per boundary direction vs the lower
    measured sustainable TCP throughput for that direction.
    """

    lower_throughput_mbps = min(
        iperf["iperf_a_to_b_mbps"], iperf["iperf_b_to_a_mbps"]
    )
    # Peak sustained application wire demand: the transport-only
    # microbenchmark drives the exact largest framed boundary payload
    # (524,288 B) request/response cycle with zero compute, which bounds
    # the model workload's offered load from above (model boundaries add
    # compute between requests).  Socket send() duration is NOT used:
    # it returns when the kernel buffer accepts the bytes, not when the
    # wire transmits them, so it is not a physical wire-rate measure.
    prefill = microbench["sizes"]["524288"]
    decode = microbench["sizes"]["8192"]
    sustained_demand_mbps = prefill["effective_mbps_p50"]
    limit = CAPACITY_MARGIN * lower_throughput_mbps
    retransmits = iperf.get("retransmits_total", 0)
    if sustained_demand_mbps <= limit and retransmits == 0:
        disposition = "R4_1GBE_PRIMITIVE_CAPACITY_VIABLE"
    elif sustained_demand_mbps > limit:
        disposition = "R4_1GBE_PRIMITIVE_CAPACITY_NEGATIVE"
    else:
        disposition = "R4_1GBE_PRIMITIVE_CAPACITY_INCONCLUSIVE"
    return {
        "disposition": disposition,
        "criterion": {
            "capacity_margin": CAPACITY_MARGIN,
            "lower_measured_tcp_throughput_mbps": lower_throughput_mbps,
            "applicable_demand_limit_mbps": round(limit, 2),
        },
        "sustained_application_wire_demand": {
            "mbps_prefill_524288": sustained_demand_mbps,
            "mbps_decode_8192": decode["effective_mbps_p50"],
            "method": "transport-only framed request/response sustained rate",
        },
        "transport_only": {
            "decode_8192_rtt_p50_ns": microbench["sizes"]["8192"]["rtt_ns_p50"],
            "prefill_524288_rtt_p50_ns": microbench["sizes"]["524288"][
                "rtt_ns_p50"
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--iperf-a-to-b", type=float, required=True)
    parser.add_argument("--iperf-b-to-a", type=float, required=True)
    parser.add_argument("--retransmits-total", type=int, default=0)
    parser.add_argument("--producer-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    evidence = args.evidence_dir
    diagnostic = _load(evidence / "arm-diagnostic.json")
    clean = _load(evidence / "arm-clean.json")
    microbench = _load(evidence / "transport-microbenchmark.json")
    node_a = _load(evidence / "node-a-hardware.json")
    node_b = _load(evidence / "node-b-hardware.json")
    plan = _load(evidence / "r4-frozen-plan.json")
    authorization = _load(evidence / "planner-authorization.json")

    def gpu_uuid(profile: dict, uuid: str) -> str:
        for gpu in profile["gpus"]:
            if gpu["uuid"] == uuid:
                return gpu["pci.bus_id"]
        raise KeyError(uuid)

    from benchmarks.inferswarm_r4.r4_plan import GPU_A_UUID, GPU_B_UUID

    runtime_invariants = {}
    for label, arm in (("diagnostic", diagnostic), ("clean", clean)):
        report = arm.get("runtime_report_a_final") or arm["runtime_report_a"]
        runtime_invariants[label] = {
            "node_a_fallbacks": report["fallbacks"],
            "node_a_recaptures": report["decode_graph"]["recaptures"],
            "node_a_host_expert_fetches": report["host_expert_fetches"],
            "node_a_resident_source_accesses": report["resident_source_accesses"],
            "node_a_unexplained_persistent_host_mirror_bytes": report[
                "unexplained_persistent_host_mirror_bytes"
            ],
            "node_a_steady_model_state_movement_bytes": report[
                "steady_model_state_movement_bytes"
            ],
            "node_a_captures": report["decode_graph"]["captures"],
            "node_a_replays": report["decode_graph"]["replays"],
        }
    result = {
        "schema": "inferswarm.r4.result/1",
        "r4_plan_digest": plan["digest"],
        "implementation_producer_sha": args.producer_sha,
        "base_r3_merge": "2ac72d547b2a24a3672d1b83268865db5490084d",
        "planner_authorization": authorization,
        "nodes": {
            "node_a": {
                "hostname": node_a["hostname"],
                "gpu_uuid": GPU_A_UUID,
                "gpu_bdf": gpu_uuid(node_a, GPU_A_UUID),
            },
            "node_b": {
                "hostname": node_b["hostname"],
                "gpu_uuid": GPU_B_UUID,
                "gpu_bdf": gpu_uuid(node_b, GPU_B_UUID),
            },
        },
        "network": {
            "canonical": "1GbE full-duplex MTU1500 direct LAN",
            "node_a_interface": "eno1",
            "node_b_interface": "enp5s0",
            "iperf_a_to_b_mbps": args.iperf_a_to_b,
            "iperf_b_to_a_mbps": args.iperf_b_to_a,
            "retransmits_total": args.retransmits_total,
        },
        "correctness": {
            "all_generated_exact": diagnostic.get("all_generated_exact"),
            "all_selected_logits_within_threshold": diagnostic.get(
                "all_selected_logits_within_threshold"
            ),
            "all_boundary_checksums_match": diagnostic.get(
                "all_boundary_checksums_match"
            ),
        },
        "runtime_invariants": runtime_invariants,
        "clean_measurement_summary": {
            row["class_id"]: {
                "decode_tok_s": row["session"]["decode_tokens_per_second"],
                "inter_token_p50_ns": row["session"]["inter_token_p50_ns"],
                "boundary_semantic_bytes": row["session"][
                    "boundary_semantic_bytes"
                ],
            }
            for row in clean["sessions"]
        },
        "wire_accounting_clean": clean.get("wire_accounting"),
    }
    result["one_gbe_disposition"] = classify_one_gbe(
        clean, microbench, result["network"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from freetoken.research.n0_model_block import write_json_with_sha

    write_json_with_sha(args.out, result)
    print(json.dumps({"one_gbe_disposition": result["one_gbe_disposition"]["disposition"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
