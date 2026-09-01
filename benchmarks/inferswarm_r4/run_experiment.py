"""R4 experiment runner (Node A side): diagnostic and clean measurement arms.

Loads the R4 frozen plan, realizes Block A in-process via the accepted R2
adapter, connects to the Node B service, and runs W2/W4 in the requested
arm.  Emits one machine-readable result per arm plus wire accounting and
the participant runtime report.

Reference comparison uses the frozen R2-v2 selected-logit reference
(docs/inferswarm_r2/reference-v2-session-a.json) with the frozen
comparators (atol/rtol 0.002, selected steps [0, 1, 15, 31]).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

W2_PROMPT_TOKEN_COUNT = 54
W4_PROMPT_TOKEN_COUNT = 121
SELECTED_STEPS = [0, 1, 15, 31]
PREFILL_CHUNK = 64
GENERATED_TOKENS = 32
ATOL = 0.002
RTOL = 0.002


def load_workloads(reference_path: Path) -> dict[str, dict[str, Any]]:
    reference = json.loads(reference_path.read_text())
    workloads = {}
    for row in reference["workloads"]:
        workloads[row["class_id"]] = row
    return workloads


def compare_selected_logits(
    observed: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    """Frozen R2-v2 comparator over the retained boundary logit records."""

    from benchmarks.inferswarm_r2.correctness_support import compare_tensor_records

    comparisons = []
    for step in SELECTED_STEPS:
        got = observed.get(str(step))
        if got is None:
            raise ValueError(f"selected logit evidence missing at step {step}")
        want = expected["selected_logit_steps"][str(step)]
        record = compare_tensor_records(got, want)
        comparisons.append(
            {
                "generated_step": step,
                "exact": record.get("exact"),
                "max_absolute_deviation": record.get("max_absolute_deviation"),
                "max_relative_deviation": record.get("max_relative_deviation"),
                "within_threshold": (
                    record.get("max_absolute_deviation", 1.0) <= ATOL
                    and record.get("max_relative_deviation", 1.0) <= RTOL
                ),
            }
        )
    return {
        "comparisons": comparisons,
        "all_exact": all(item["exact"] for item in comparisons),
        "nan_count": sum(
            int(item.get("nan_count", 0)) for item in observed.values()
            if isinstance(item, dict)
        ),
        "inf_count": sum(
            int(item.get("inf_count", 0)) for item in observed.values()
            if isinstance(item, dict)
        ),
    }


def run_arm(
    *,
    arm: str,
    class_ids: list[str],
    plan: dict[str, Any],
    model_path: str,
    peer_host: str,
    peer_port: int,
    reference: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    import torch

    from benchmarks.inferswarm_r2.preflight_transport import _register, _unregister
    from benchmarks.inferswarm_r2.qwen_split_adapter import (
        HIDDEN_SIZE,
        QwenSplitResearchAdapter,
    )
    from benchmarks.inferswarm_r4.node_a_coordinator import (
        NodeACoordinator,
        run_session,
    )
    from freetoken.research.r1_frozen_plan import realize_frozen_plan
    from freetoken.research.r2_local_split import validate_participant

    diagnostic = arm == "diagnostic"
    execution_id = "exec.block-a"
    r1_plan = plan["participant_r1_plans"][execution_id]
    from benchmarks.inferswarm_r4.r4_plan import GPU_A_UUID, MODEL_REVISION

    validate_participant(
        plan,
        execution_id=execution_id,
        plan_digest_value=plan["digest"],
        stable_device_id=GPU_A_UUID,
        materialization_ids=[item["id"] for item in r1_plan["materializations"]],
    )
    environment = {
        "model_repository": plan["model"]["repository"],
        "model_revision": plan["model"]["revision"],
        "resources": r1_plan["resources"],
    }
    adapter = QwenSplitResearchAdapter(
        role="a",
        model_path=model_path,
        host_staging_policy="release_after_final_residency",
    )
    realized = realize_frozen_plan(r1_plan, environment, adapter)
    runtime = adapter.runtime
    if runtime is None:
        raise RuntimeError("Node A realizer did not construct its runtime")
    buffer_bytes = 2 * 64 * HIDDEN_SIZE * 2
    host_u8 = torch.empty(buffer_bytes, dtype=torch.uint8)
    _register(host_u8, buffer_bytes)
    arm_result: dict[str, Any] = {
        "schema": "inferswarm.r4.arm-result/1",
        "arm": arm,
        "diagnostic_transfer": diagnostic,
        "class_ids": class_ids,
        "producer_freetoken_sha": plan.get("provenance", {})
        .get("r4", {})
        .get("producer_sha"),
        "sessions": [],
        "realization": {
            "validation": realized.validation,
            "reconciliation": realized.reconciliation,
            "materializations": realized.observed_materializations,
            "execution": realized.observed_execution,
            "authorities": realized.observed_authorities,
        },
        "runtime_report_a": runtime.report("P4_ready_for_resident_execution"),
    }
    try:
        coordinator = NodeACoordinator(
            plan=plan,
            model_path=model_path,
            peer_host=peer_host,
            peer_port=peer_port,
            diagnostic=diagnostic,
            runtime=runtime,
            host_buffer=host_u8,
        )
        try:
            for index, class_id in enumerate(class_ids):
                expected = reference[class_id]
                session = run_session(
                    coordinator,
                    session_id=1000 + index,
                    prompt_ids=expected["prompt_token_ids"],
                    max_new_tokens=GENERATED_TOKENS,
                    prefill_chunk=PREFILL_CHUNK,
                    capture_steps=set(SELECTED_STEPS) if diagnostic else set(),
                )
                row = {
                    "class_id": class_id,
                    "session": session,
                    "prompt_token_ids_exact": (
                        session["prompt_token_ids"] == expected["prompt_token_ids"]
                    ),
                    "generated_token_ids_exact": (
                        session["generated_token_ids"]
                        == expected["generated_token_ids"]
                    ),
                }
                if diagnostic:
                    observed_logits = {
                        str(item["generated_step"]): item["logits"]
                        for item in session["boundaries"]
                        if "logits" in item
                    }
                    row["selected_logits"] = compare_selected_logits(
                        observed_logits, expected
                    )
                    row["boundary_checksums_all_match"] = all(
                        item["producer_sha256"] == item["consumer_sha256"]
                        for item in session["boundaries"]
                        if item["producer_sha256"] is not None
                    )
                arm_result["sessions"].append(row)
            arm_result["wire_accounting"] = coordinator.report()
        finally:
            coordinator.close()
        arm_result["runtime_report_a_final"] = runtime.report()
    finally:
        _unregister(host_u8)
    if diagnostic:
        arm_result["all_generated_exact"] = all(
            row["generated_token_ids_exact"] for row in arm_result["sessions"]
        )
        arm_result["all_selected_logits_within_threshold"] = all(
            row["selected_logits"]["all_exact"]
            or all(
                c["within_threshold"]
                for c in row["selected_logits"]["comparisons"]
            )
            for row in arm_result["sessions"]
        )
        arm_result["all_boundary_checksums_match"] = all(
            row["boundary_checksums_all_match"] for row in arm_result["sessions"]
        )
    return arm_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("diagnostic", "clean"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--peer-host", default="10.0.0.219")
    parser.add_argument("--peer-port", type=int, default=18485)
    parser.add_argument("--classes", default="W2,W4")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from benchmarks.inferswarm_r4.r4_plan import load_r4_plan

    plan = load_r4_plan(args.plan)
    reference = load_workloads(args.reference)
    result = run_arm(
        arm=args.arm,
        class_ids=args.classes.split(","),
        plan=plan,
        model_path=args.model,
        peer_host=args.peer_host,
        peer_port=args.peer_port,
        reference=reference,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from freetoken.research.n0_model_block import write_json_with_sha

    write_json_with_sha(args.out, result)
    print(json.dumps({"arm": args.arm, "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
