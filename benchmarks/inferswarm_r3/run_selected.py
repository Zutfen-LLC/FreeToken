"""Physically execute an already-selected R3 plan (planning is never in the hot path)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from freetoken.research.n0_model_block import write_json_with_sha
from freetoken.research.r3_planner import require_frozen, validate_decision_environment

from benchmarks.inferswarm_r2.qwen_split_adapter import tensor_sha256
from benchmarks.inferswarm_r2.run_correctness import _compare_logits, _prompt_ids
from benchmarks.inferswarm_r2.run_matched_local_control import MatchedLocalRuntime

CLASSES = ("W2", "W4")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _validate_live_resources(snapshot: dict) -> None:
    """Mechanically reject GPU identity/capacity/driver drift before materialization."""
    fields = "uuid,memory.total,driver_version"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
    )
    observed = {}
    for line in output.splitlines():
        uuid, mib, driver = (value.strip() for value in line.split(","))
        observed[uuid] = {"capacity_bytes": int(mib) * 1024 * 1024, "driver": driver}
    expected_driver = snapshot.get("provenance", {}).get("driver")
    for node in snapshot["nodes"]:
        memories = {item["id"]: item for item in node["memory_resources"]}
        for unit in node["compute_units"]:
            uuid = unit["stable_device_id"]
            if uuid not in observed:
                raise RuntimeError(f"planned Compute Unit {uuid} is no longer present")
            memory = memories[unit["memory_resource_id"]]
            if observed[uuid]["capacity_bytes"] != memory["capacity_bytes"]:
                raise RuntimeError(f"planned Memory Resource capacity drifted for {uuid}")
            if expected_driver and observed[uuid]["driver"] != expected_driver:
                raise RuntimeError(f"planned driver context drifted for {uuid}")
    mem_total = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemTotal:")
    )
    expected_ram = next(
        item["capacity_bytes"]
        for node in snapshot["nodes"]
        for item in node["memory_resources"]
        if item["kind"] == "system-ram"
    )
    if mem_total < expected_ram:
        raise RuntimeError("planned system RAM capacity is no longer present")


def _selected_resources(decision: dict, snapshot: dict) -> list[dict]:
    units = {
        unit["id"]: unit
        for node in snapshot["nodes"]
        for unit in node["compute_units"]
    }
    return [
        {
            "slot_id": slot_id,
            "compute_unit_id": compute_id,
            "stable_device_id": units[compute_id]["stable_device_id"],
            "memory_resource_id": units[compute_id]["memory_resource_id"],
        }
        for slot_id, compute_id in sorted(decision["selected_mapping"].items())
    ]


def _validate_r2_resource_identities(plan_path: Path, selected_resources: list[dict]) -> None:
    r2_plan = _load(plan_path)
    r2_units = {
        unit["id"]: unit["stable_device_id"]
        for node in r2_plan["resources"]["nodes"]
        for unit in node["compute_units"]
    }
    for selected in selected_resources:
        compute_id = selected["compute_unit_id"]
        if r2_units.get(compute_id) != selected["stable_device_id"]:
            raise RuntimeError(
                f"R2 realization resource identity drifted for {compute_id!r}"
            )


def _logit_record(logits: torch.Tensor) -> dict:
    value = logits.detach().float()
    return {
        "shape": list(value.shape),
        "float32_sha256": tensor_sha256(value),
        "full_logits": value.cpu().tolist(),
    }


def _summarize(rows: list[dict]) -> dict:
    comparisons = [item for row in rows for item in row["logit_checkpoints"]]
    return {
        "workloads": rows,
        "exact_generated_sequences": all(row["exact_generated_sequence"] for row in rows),
        "selected_logits_within_r2_v2_threshold": all(item["within_canonical_threshold"] for item in comparisons),
        "max_absolute_deviation": max(item["max_absolute_deviation"] for item in comparisons),
        "max_relative_deviation": max(item["max_relative_deviation"] for item in comparisons),
        "nan_count": sum(item["nan_count"] for item in comparisons),
        "inf_count": sum(item["inf_count"] for item in comparisons),
    }


def _run_offload(model: str, manifest, tokenizer, reference: dict) -> dict:
    expected = {row["class_id"]: row for row in reference["workloads"]}
    started = time.perf_counter()
    runtime = MatchedLocalRuntime(model, capacity=17_152, cache_slots=3_774)
    startup_seconds = time.perf_counter() - started
    rows = []
    for class_id in CLASSES:
        runtime.reset_session()
        runtime.cache.reset_stats()
        prompt_ids = _prompt_ids(tokenizer, manifest.by_class()[class_id])
        generated: list[int] = []
        logits_by_step = {}
        for start in range(0, len(prompt_ids), 64):
            ids = prompt_ids[start : start + 64]
            token, logits, _ = runtime.prefill_reference(ids, start, capture_seam=False)
            if start + len(ids) == len(prompt_ids):
                generated.append(token)
                logits_by_step[0] = _logit_record(logits)
        while len(generated) < 32:
            step = len(generated)
            token, logits = runtime.decode(generated[-1], len(prompt_ids) + step - 1)
            generated.append(token)
            if step in {1, 15, 31}:
                logits_by_step[step] = _logit_record(logits)
        comparisons = [
            {"generated_step": step, **_compare_logits(actual, expected[class_id]["selected_logit_steps"][str(step)])}
            for step, actual in sorted(logits_by_step.items())
        ]
        rows.append(
            {
                "class_id": class_id,
                "generated_token_ids": generated,
                "exact_generated_sequence": generated == expected[class_id]["generated_token_ids"],
                "logit_checkpoints": comparisons,
                "ordinary_offload_source_access": runtime.cache.decode_miss_stats(),
            }
        )
    result = _summarize(rows)
    result.update(
        {
            "startup_materialization_seconds": startup_seconds,
            "source_lifecycle": "RETAIN_REQUIRED_SOURCE_BACKING",
            "source_fetch_zero_not_required": True,
            "graph": {"captures": 1, "replays": runtime.graph.replays, "recaptures": 0},
        }
    )
    return result


def _run_split(plan_path: Path, model: str, manifest, tokenizer, reference: dict) -> dict:
    from benchmarks.inferswarm_r2.coordinator import LocalSplitCoordinator

    expected = {row["class_id"]: row for row in reference["workloads"]}
    plan_record = _load(plan_path)
    rows = []
    started = time.perf_counter()
    with LocalSplitCoordinator(plan_path=str(plan_path), model_path=model, diagnostic=True) as coordinator:
        startup_seconds = time.perf_counter() - started
        startup = coordinator.ready
        for index, class_id in enumerate(CLASSES):
            prompt_ids = _prompt_ids(tokenizer, manifest.by_class()[class_id])
            row = coordinator.run_session(
                session_id=3000 + index,
                prompt_ids=prompt_ids,
                max_new_tokens=32,
                prefill_chunk=plan_record["runtime_capacity"]["prefill_chunk_tokens"],
                capture_steps={0, 1, 15, 31},
            )
            logits = {item["generated_step"]: item.pop("logits") for item in row["boundaries"] if "logits" in item}
            comparisons = [
                {"generated_step": step, **_compare_logits(actual, expected[class_id]["selected_logit_steps"][str(step)])}
                for step, actual in sorted(logits.items())
            ]
            rows.append(
                {
                    "class_id": class_id,
                    "generated_token_ids": row["generated_token_ids"],
                    "exact_generated_sequence": row["generated_token_ids"] == expected[class_id]["generated_token_ids"],
                    "logit_checkpoints": comparisons,
                    "boundary_checksums_matched": all(item.get("producer_sha256") == item.get("consumer_sha256") for item in row["boundaries"]),
                }
            )
        reports = coordinator.reports()
    result = _summarize(rows)
    result.update(
        {
            "startup_materialization_seconds": startup_seconds,
            "source_lifecycle": "RELEASE_AFTER_FINAL_RESIDENCY",
            "startup_reconciliation_passed": all(startup[role]["realization"]["reconciliation"]["passed"] for role in ("a", "b")),
            "participants": {role: reports[role]["runtime"] for role in ("a", "b")},
            "resident_invariants_passed": all(
                reports[role]["runtime"]["unexplained_persistent_host_mirror_bytes"] == 0
                and reports[role]["runtime"]["steady_model_state_movement_bytes"] == 0
                and reports[role]["runtime"]["host_expert_fetches"] == 0
                and reports[role]["runtime"]["resident_source_accesses"] == 0
                and reports[role]["runtime"]["fallbacks"] == 0
                and reports[role]["runtime"]["decode_graph"]["recaptures"] == 0
                for role in ("a", "b")
            ),
        }
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("a", "b"), required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("docs/inferswarm_r3"))
    parser.add_argument("--r2-plan", type=Path, default=Path("docs/inferswarm_r2/frozen-plan.json"))
    parser.add_argument("--reference", type=Path, default=Path("docs/inferswarm_r2/reference-v2-session-a.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    # Every frozen decision and environment check happens before model construction.
    decision = _load(args.artifact_dir / f"decision-{args.scenario}.json")
    compiled = _load(args.artifact_dir / f"compiled-plan-{args.scenario}.json")
    snapshot = _load(args.artifact_dir / "resource-snapshot.json")
    frozen_inputs = {
        "resource_snapshot_digest": snapshot,
        "strategy_problem_digest": _load(args.artifact_dir / "strategy-problem.json"),
        "policy_digest": _load(args.artifact_dir / f"policy-{args.scenario}.json"),
        "objective_digest": _load(args.artifact_dir / f"objective-{args.scenario}.json"),
        "evidence_catalog_digest": _load(args.artifact_dir / "evidence-catalog.json"),
    }
    for label, artifact in frozen_inputs.items():
        require_frozen(artifact, label)
        if decision["inputs"].get(label) != artifact["digest"]:
            raise RuntimeError(f"{label} drifted after the frozen decision")
    require_frozen(compiled, "compiled selected plan")
    validate_decision_environment(decision, snapshot)
    if compiled["planner_decision_digest"] != decision["digest"]:
        raise RuntimeError("compiled plan is not linked to the frozen decision")
    implementation_commit = snapshot.get("implementation_commit")
    if not implementation_commit or any(
        artifact.get("implementation_commit") != implementation_commit
        for artifact in (compiled, *frozen_inputs.values())
    ):
        raise RuntimeError("retained artifacts do not share an implementation commit")
    repository_root = Path(__file__).resolve().parents[2]
    live_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    if live_head != implementation_commit:
        raise RuntimeError(
            "implementation commit drifted after planning; regenerate the campaign"
        )
    _validate_live_resources(snapshot)
    selected_resources = _selected_resources(decision, snapshot)

    from inferswarm_phase0.manifest import load_manifest
    from transformers import AutoTokenizer

    manifest = load_manifest(args.manifest, canonical=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    reference = _load(args.reference)
    if args.scenario == "a":
        expected_uuid = next(unit["stable_device_id"] for unit in snapshot["nodes"][0]["compute_units"] if unit["id"] == compiled["compute_unit_id"])
        if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
            raise RuntimeError(f"Scenario A requires CUDA_VISIBLE_DEVICES={expected_uuid}")
        correctness = _run_offload(args.model, manifest, tokenizer, reference)
    else:
        _validate_r2_resource_identities(args.r2_plan, selected_resources)
        correctness = _run_split(args.r2_plan, args.model, manifest, tokenizer, reference)
    passed = (
        correctness["exact_generated_sequences"]
        and correctness["selected_logits_within_r2_v2_threshold"]
        and correctness["nan_count"] == 0
        and correctness["inf_count"] == 0
        and (args.scenario == "a" or correctness["resident_invariants_passed"])
    )
    payload = {
        "schema": "inferswarm.r3.physical-selected-execution/1",
        "implementation_commit": implementation_commit,
        "scenario": args.scenario.upper(),
        "decision_digest": decision["digest"],
        "compiled_plan_digest": compiled["digest"],
        "selected_candidate_id": decision["selected_candidate_id"],
        "selected_mapping": decision["selected_mapping"],
        "selected_resources": selected_resources,
        "decision_frozen_and_environment_validated_before_materialization": True,
        "workloads": list(CLASSES),
        "correctness": correctness,
        "passed": passed,
    }
    write_json_with_sha(args.out, payload)
    print(json.dumps({"out": str(args.out), "scenario": args.scenario, "passed": passed}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
