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


def workload_wire_demand(clean_arm: dict[str, Any]) -> dict[str, Any]:
    """Actual clean-arm application wire demand, derived from the retained
    clean-arm workload evidence (real application bytes and real request
    cadence/wall time).

    Gate-review correction: the transport-only microbenchmark measures
    transport service capability, NOT the model workload's offered demand
    (a faster transport would produce a larger number under that
    calculation).  Demand is therefore computed from the frozen workload's
    actual framed bytes per direction over the measured service intervals:

    - A -> B: semantic payload bytes + framing/control bytes per boundary;
    - B -> A: actual result/control frame bytes per boundary.

    Decode steady demand uses the measured decode boundary cadence
    (inter-token p50 wall time); prefill demand uses the measured prefill
    wall time and its real transmitted bytes.  socket.send() durations are
    never used as wire-rate evidence.
    """

    accounting = clean_arm["wire_accounting"]
    frames_tx = accounting["frames_tx"]
    frames_rx = accounting["frames_rx"]
    avg_tx_overhead = accounting["framing_control_bytes_tx"] / frames_tx
    avg_rx_frame = accounting["wire_bytes_rx"] / frames_rx
    per_class = {}
    peak = {"a_to_b_mbps": 0.0, "b_to_a_mbps": 0.0}
    for row in clean_arm["sessions"]:
        session = row["session"]
        class_id = row["class_id"]
        boundaries = session["boundaries"]
        prefill = [b for b in boundaries if b["operation"] == "prefill"]
        decode = [b for b in boundaries if b["operation"] == "decode"]
        decode_step_ab = 8192 + avg_tx_overhead
        decode_step_ba = avg_rx_frame
        decode_p50_ns = session["inter_token_p50_ns"]
        decode_ab = decode_step_ab * 8 / (decode_p50_ns / 1e9) / 1e6
        decode_ba = decode_step_ba * 8 / (decode_p50_ns / 1e9) / 1e6
        prefill_semantic = sum(b["payload_bytes"] for b in prefill)
        # session control frames (hello/open request + acks) fall inside the
        # prefill window; include them so prefill demand is not understated
        prefill_frames_ab = len(prefill) + 2
        prefill_frames_ba = len(prefill) + 2
        prefill_ab = (
            (prefill_semantic + prefill_frames_ab * avg_tx_overhead)
            * 8
            / (session["prefill_wall_ns"] / 1e9)
            / 1e6
        )
        prefill_ba = (
            (prefill_frames_ba * avg_rx_frame)
            * 8
            / (session["prefill_wall_ns"] / 1e9)
            / 1e6
        )
        per_class[class_id] = {
            "decode": {
                "a_to_b_mbps": round(decode_ab, 4),
                "b_to_a_mbps": round(decode_ba, 4),
                "boundary_bytes_a_to_b": round(decode_step_ab, 1),
                "boundary_bytes_b_to_a": round(decode_step_ba, 1),
                "cadence_ns_p50": decode_p50_ns,
                "decode_boundary_count": len(decode),
            },
            "prefill": {
                "a_to_b_mbps": round(prefill_ab, 4),
                "b_to_a_mbps": round(prefill_ba, 4),
                "semantic_bytes": prefill_semantic,
                "wall_ns": session["prefill_wall_ns"],
                "prefill_boundary_count": len(prefill),
            },
        }
        peak["a_to_b_mbps"] = max(
            peak["a_to_b_mbps"], decode_ab, prefill_ab
        )
        peak["b_to_a_mbps"] = max(
            peak["b_to_a_mbps"], decode_ba, prefill_ba
        )
    return {
        "method": "actual clean-arm application bytes / measured service interval "
        "(decode: inter-token p50 cadence; prefill: measured prefill wall time); "
        "framing/control bytes included per direction",
        "wire_accounting_source": "arm-clean.json wire_accounting",
        "per_class": per_class,
        "peak_a_to_b_mbps": round(peak["a_to_b_mbps"], 4),
        "peak_b_to_a_mbps": round(peak["b_to_a_mbps"], 4),
    }


def classify_one_gbe(
    clean_arm: dict[str, Any], microbench: dict[str, Any], iperf: dict[str, Any]
) -> dict[str, Any]:
    """Classify the 1-GbE arm using the frozen 80% capacity margin.

    Gate-review corrected semantics: the ACTUAL clean-arm workload wire
    demand (per direction) is compared against 80% of the lower measured
    sustainable applicable TCP throughput (iperf lower direction).  The
    link is full duplex, so each direction is evaluated independently.
    The transport-only microbenchmark is retained as separate transport
    service-capability evidence and is NOT used as workload demand.
    """

    lower_throughput_mbps = min(
        iperf["iperf_a_to_b_mbps"], iperf["iperf_b_to_a_mbps"]
    )
    limit = CAPACITY_MARGIN * lower_throughput_mbps
    demand = workload_wire_demand(clean_arm)
    peak_ab = demand["peak_a_to_b_mbps"]
    peak_ba = demand["peak_b_to_a_mbps"]
    retransmits = iperf.get("retransmits_total", 0)
    if peak_ab <= limit and peak_ba <= limit and retransmits == 0:
        disposition = "R4_1GBE_PRIMITIVE_CAPACITY_VIABLE"
    elif peak_ab > limit or peak_ba > limit:
        disposition = "R4_1GBE_PRIMITIVE_CAPACITY_NEGATIVE"
    else:
        disposition = "R4_1GBE_PRIMITIVE_CAPACITY_INCONCLUSIVE"
    return {
        "disposition": disposition,
        "criterion": {
            "capacity_margin": CAPACITY_MARGIN,
            "lower_measured_tcp_throughput_mbps": lower_throughput_mbps,
            "applicable_demand_limit_mbps": round(limit, 2),
            "rule": "actual clean-arm workload wire demand (per direction) <= "
            "80% of lower measured sustainable TCP throughput, zero retransmits",
        },
        "actual_clean_arm_workload_wire_demand": demand,
        "measured_sustainable_tcp_capacity": {
            "iperf_a_to_b_mbps": iperf["iperf_a_to_b_mbps"],
            "iperf_b_to_a_mbps": iperf["iperf_b_to_a_mbps"],
            "retransmits_total": retransmits,
        },
        "transport_only_service_measurements": {
            "note": "transport service capability evidence; NOT workload demand",
            "decode_8192_rtt_p50_ns": microbench["sizes"]["8192"]["rtt_ns_p50"],
            "decode_8192_effective_mbps_p50": microbench["sizes"]["8192"][
                "effective_mbps_p50"
            ],
            "prefill_524288_rtt_p50_ns": microbench["sizes"]["524288"]["rtt_ns_p50"],
            "prefill_524288_effective_mbps_p50": microbench["sizes"]["524288"][
                "effective_mbps_p50"
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
    gate = _load(evidence / "preflight-gate.json")
    if gate.get("result") != "ALL_PREFLIGHT_CHECKS_PASSED":
        raise RuntimeError(
            "canonical preflight gate record missing/failed; refusing to "
            "compose a final result from an ungated campaign"
        )

    def gpu_uuid(profile: dict, uuid: str) -> str:
        for gpu in profile["gpus"]:
            if gpu["uuid"] == uuid:
                return gpu.get("pci_bus_id") or gpu["pci.bus_id"]
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
        "preflight_gate": {
            "result": gate["result"],
            "frozen_producer_sha": gate["frozen_producer_sha"],
            "checks": sorted(gate["checks"].keys()),
        },
        "hardware": {
            "node_a": {
                "hostname": node_a["hostname"],
                "cpu": node_a["cpu"],
            },
            "node_b": {
                "hostname": node_b["hostname"],
                "cpu": node_b["cpu"],
                "physical_installed_ram_bytes": node_b["memory"].get(
                    "physical_installed_ram_bytes"
                ),
                "linux_memtotal_bytes": node_b["memory"].get("mem_total_bytes"),
                "memavailable_bytes_at_preflight": node_b["memory"].get(
                    "mem_available_bytes"
                ),
            },
        },
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
