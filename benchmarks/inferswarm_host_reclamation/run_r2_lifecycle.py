"""Run the frozen R2 W2/W4 path under one explicit host-source lifecycle policy."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha

from benchmarks.inferswarm_r2.coordinator import LocalSplitCoordinator
from benchmarks.inferswarm_r2.run_correctness import _compare_logits, _prompt_ids
from benchmarks.inferswarm_r2.v2_support import (
    CANDIDATE_GPU_UUIDS,
    FROZEN_PLAN_DIGEST,
    sha256_file,
    validate_reference_artifact,
)


def _physical_result(reports: dict, policy: str) -> dict:
    participants = {}
    before_available = []
    after_available = []
    for role in ("a", "b"):
        runtime = reports[role]["runtime"]
        snapshots = runtime["host_lifecycle_snapshots"]
        before = snapshots["P2_release_barrier"]
        after = snapshots["P4_coordinated_release_complete"]
        before_rss = before["process_status_bytes"]["VmRSS"]
        after_rss = after["process_status_bytes"]["VmRSS"]
        staging = runtime["host_staging_before_release_bytes"]
        process_reclaimed = max(0, before_rss - after_rss)
        allocations = runtime["host_materialization_allocations"]
        participants[role] = {
            "staging_bytes": staging,
            "process_rss_before_bytes": before_rss,
            "process_rss_after_bytes": after_rss,
            "process_rss_shmem_before_bytes": before["process_status_bytes"][
                "RssShmem"
            ],
            "process_rss_shmem_after_bytes": after["process_status_bytes"][
                "RssShmem"
            ],
            "reclaimed_bytes": process_reclaimed,
            "reclaimed_fraction_of_staging": process_reclaimed / staging,
            "source_tensor_bytes": runtime["host_staging_current_bytes"],
            "source_tensor_objects_alive": sum(
                item["tensor_object_count_alive"] for item in allocations
            ),
            "source_storage_owners_alive": sum(
                item["storage_present"] for item in allocations
            ),
            "worker_alive": after["worker_alive"],
        }
        before_available.append(before["system_meminfo_bytes"]["MemAvailable"])
        after_available.append(after["system_meminfo_bytes"]["MemAvailable"])
    staging_total = sum(item["staging_bytes"] for item in participants.values())
    process_total = sum(item["reclaimed_bytes"] for item in participants.values())
    system_before = int(statistics.median(before_available))
    system_after = int(statistics.median(after_available))
    system_delta = max(0, system_after - system_before)
    reclaimed = min(process_total, system_delta)
    fraction = reclaimed / staging_total
    release_selected = policy == "release_after_final_residency"
    passed = bool(
        release_selected
        and fraction >= 0.80
        and all(
            item["reclaimed_fraction_of_staging"] >= 0.80
            and item["source_tensor_bytes"] == 0
            and item["source_tensor_objects_alive"] == 0
            and item["source_storage_owners_alive"] == 0
            and item["worker_alive"]
            for item in participants.values()
        )
    )
    return {
        "policy": policy,
        "participants": participants,
        "combined": {
            "staging_bytes": staging_total,
            "summed_process_rss_reclaimed_bytes": process_total,
            "system_memavailable_before_bytes": system_before,
            "system_memavailable_after_bytes": system_after,
            "system_memavailable_increase_bytes": system_delta,
            "reclaimed_bytes": reclaimed,
            "reclaimed_fraction_of_staging": fraction,
            "measurement_basis": (
                "min(sum participant VmRSS reductions, coordinated median "
                "system MemAvailable increase)"
            ),
        },
        "tensor_counters_alone_can_pass": False,
        "minimum_fraction": 0.80,
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("retain_reusable_source", "release_after_final_residency"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    from benchmarks.inferswarm_phase0.manifest import load_manifest

    plan = json.loads(args.plan.read_text())
    if plan.get("digest") != FROZEN_PLAN_DIGEST:
        raise ValueError("pre-R3 runner requires the accepted frozen R2 plan")
    reference = json.loads(args.reference.read_text())
    validate_reference_artifact(reference)
    reference_by_class = {row["class_id"]: row for row in reference["workloads"]}
    manifest = load_manifest(args.manifest, canonical=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    classes = ("W2", "W4")
    prompts = {
        class_id: _prompt_ids(tokenizer, manifest.by_class()[class_id])
        for class_id in classes
    }
    rows = []
    coordinator = LocalSplitCoordinator(
        plan_path=str(args.plan),
        model_path=args.model,
        diagnostic=True,
        host_staging_policy=args.policy,
    )
    try:
        startup = coordinator.ready
        for index, class_id in enumerate(classes):
            expected = reference_by_class[class_id]
            if prompts[class_id] != expected["prompt_token_ids"]:
                raise RuntimeError(f"{class_id} prompt IDs differ from R2 v2 reference")
            row = coordinator.run_session(
                session_id=3000 + index,
                prompt_ids=prompts[class_id],
                max_new_tokens=32,
                prefill_chunk=plan["runtime_capacity"]["prefill_chunk_tokens"],
                capture_steps={0, 1, 15, 31},
            )
            actual_logits = {
                item["generated_step"]: item["logits"]
                for item in row["boundaries"]
                if "logits" in item
            }
            comparisons = [
                {
                    "generated_step": step,
                    **_compare_logits(
                        actual_logits[step], expected["selected_logit_steps"][str(step)]
                    ),
                }
                for step in sorted(actual_logits)
            ]
            for boundary in row["boundaries"]:
                boundary.pop("logits", None)
            row.update(
                class_id=class_id,
                exact_generated_sequence=(
                    row["generated_token_ids"] == expected["generated_token_ids"]
                ),
                logit_checkpoints=comparisons,
            )
            rows.append(row)
        reports = coordinator.reports()
        physical = _physical_result(reports, args.policy)
    finally:
        coordinator.shutdown()

    token_pass = all(row["exact_generated_sequence"] for row in rows)
    logits_pass = all(
        len(row["logit_checkpoints"]) == 4
        and all(item["within_canonical_threshold"] for item in row["logit_checkpoints"])
        for row in rows
    )
    byte_exact_logits = all(
        all(item["exact"] for item in row["logit_checkpoints"]) for row in rows
    )
    numerical_pass = all(
        item["nan_count"] == 0 and item["inf_count"] == 0
        for row in rows
        for item in row["logit_checkpoints"]
    )
    boundary_pass = all(
        item["producer_sha256"] == item["consumer_sha256"]
        for row in rows
        for item in row["boundaries"]
    )
    runtime_pass = all(
        reports[role]["runtime"]["decode_graph"]["captures"] == 1
        and reports[role]["runtime"]["decode_graph"]["recaptures"] == 0
        and reports[role]["runtime"]["host_expert_fetches"] == 0
        and reports[role]["runtime"]["resident_source_accesses"] == 0
        and reports[role]["runtime"]["fallbacks"] == 0
        and reports[role]["runtime"]["steady_model_state_movement_bytes"] == 0
        and reports[role]["runtime"]["populate_count"] == 1
        for role in ("a", "b")
    )
    resource_pass = all(
        startup[role]["gpu"]["uuid"] == CANDIDATE_GPU_UUIDS[role]
        for role in ("a", "b")
    )
    correctness_pass = all(
        (token_pass, logits_pass, numerical_pass, boundary_pass, runtime_pass, resource_pass)
    )
    release_mode = args.policy == "release_after_final_residency"
    passed = correctness_pass and (physical["passed"] if release_mode else True)
    payload = {
        "schema": "inferswarm.pre-r3.host-lifecycle/1",
        "producer_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "r2_merge_base": "8627f441c880398389042ce8c0a604f6c4321dfa",
        "policy": args.policy,
        "frozen_split": {"block_a": [0, 19], "block_b": [19, 40]},
        "reference": {
            "path": args.reference.name,
            "sha256": sha256_file(args.reference),
        },
        "workloads": rows,
        "correctness": {
            "exact_generated_tokens": token_pass,
            "selected_logits_within_r2_threshold": logits_pass,
            "selected_logits_byte_exact": byte_exact_logits,
            "nan_inf_zero": numerical_pass,
            "boundary_hashes_exact": boundary_pass,
            "resident_runtime_counters_passed": runtime_pass,
            "resource_identity_passed": resource_pass,
            "passed": correctness_pass,
        },
        "physical_reclamation": physical,
        "participants": reports,
        "shutdown": coordinator.shutdown_records,
        "workers_remained_alive_through_resident_decode": all(
            reports[role]["runtime"]["host_lifecycle_snapshots"][
                "P5_repeated_resident_decode"
            ]["worker_alive"]
            for role in ("a", "b")
        ),
        "passed": passed,
    }
    write_json_with_sha(args.out, payload)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
