"""Build frozen R3 inputs, A/B/C decisions, and selected-plan artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from freetoken.research.r3_planner import freeze, plan

from .qwen_strategy import GPU_A, GPU_B, MODEL, REVISION, compile_selected, planning_problem

SCHEMA_CONTEXT = {
    "model_revision": REVISION,
    "runtime_context": "accepted-r2-runtime-compatible-with-2fc64ae",
    "topology_context": "gpu-a--registered-host-staging--gpu-b",
    "workload_geometry": "W4-prompt121-generate32-prefill64-warm",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    path.with_suffix(path.suffix + ".sha256").write_text(hashlib.sha256(payload).hexdigest() + "\n")


def _head(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()


def _require_clean_implementation(repository_root: Path) -> str:
    implementation_commit = _head(repository_root)
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repository_root, text=True
    )
    if status:
        raise RuntimeError(
            "canonical R3 input generation requires a clean implementation commit"
        )
    return implementation_commit


def resource_snapshot(implementation_commit: str | None = None) -> dict:
    return freeze(
        {
            "schema": "inferswarm.r3.resource-snapshot/1",
            "implementation_commit": implementation_commit,
            "evidence_context": {
                "runtime_context": SCHEMA_CONTEXT["runtime_context"],
                "topology_context": SCHEMA_CONTEXT["topology_context"],
            },
            "captured_from": "accepted R2 physical resource graph",
            "provenance": {
                "freetoken_base": "2fc64ae7c79bdc494a52468da329ddafd0adb8ba",
                "driver": "610.57.04",
                "transport_preflight_sha256": "bd310f04066c872fdd02050ef64ce5bd98cbe618f07ce36fdff3909d372986ef",
            },
            "nodes": [
                {
                    "id": "node-a",
                    "compute_units": [
                        {"id": "gpu-a", "stable_device_id": GPU_A, "memory_resource_id": "gpu-a.vram", "availability": "AVAILABLE", "integrity_eligible": True, "capabilities": ["freetoken-offload-v1", "freetoken-resident-block-v1"]},
                        {"id": "gpu-b", "stable_device_id": GPU_B, "memory_resource_id": "gpu-b.vram", "availability": "AVAILABLE", "integrity_eligible": True, "capabilities": ["freetoken-offload-v1", "freetoken-resident-block-v1"]},
                    ],
                    "memory_resources": [
                        {"id": "gpu-a.vram", "kind": "accelerator-vram", "capacity_bytes": 12_884_901_888, "reservation_bytes": 536_870_912},
                        {"id": "gpu-b.vram", "kind": "accelerator-vram", "capacity_bytes": 12_884_901_888, "reservation_bytes": 536_870_912},
                        {"id": "node-a.ram", "kind": "system-ram", "capacity_bytes": 134_984_794_112, "reservation_bytes": 8_589_934_592},
                    ],
                }
            ],
            "links": [
                {"id": "link.gpu-a-host-gpu-b", "source_memory_resource_id": "gpu-a.vram", "target_memory_resource_id": "gpu-b.vram", "available": True, "capabilities": ["registered-host-staging"], "decode_payload_bytes": 8192, "prefill_chunk_payload_bytes": 524288}
            ],
        }
    )


def evidence_catalog(
    repository_root: Path, implementation_commit: str | None = None
) -> dict:
    source = repository_root / "docs/inferswarm_r2/benchmark.json"
    provenance = {
        "source_freetoken_commit": "8627f441c880398389042ce8c0a604f6c4321dfa",
        "source_artifact": str(source.relative_to(repository_root)),
        "source_artifact_sha256": _sha(source),
        "compatibility": "The 2fc64ae descendant changes only post-final-residency host-source ownership/reclamation. It preserves the accepted R2 model, mappings, topology, runtime/backend configuration, and measured warm execution paths; R2 steady-path metrics therefore remain applicable.",
    }
    def record(record_id, shape_id, mapping, metric, value, unit):
        return {
            "id": record_id,
            "shape_id": shape_id,
            "mapping": mapping,
            "required_context": SCHEMA_CONTEXT,
            "freshness": "ACCEPTED_COMPATIBLE",
            "evidence_class": "MEASURED_ACCEPTED_R2_MATCHED_AB",
            "confidence": "EXACT_CONTEXT",
            "metric": {"name": metric, "value": value, "unit": unit, "statistic": "median"},
            "provenance": provenance,
        }
    return freeze(
        {
            "schema": "inferswarm.r3.evidence-catalog/1",
            "implementation_commit": implementation_commit,
            "context": SCHEMA_CONTEXT,
            "records": [
                record("r2-w4-baseline-decode", "s0.source-backed-single-offload", {"whole-model-slot": "gpu-a"}, "warm_decode_tok_s", 73.61368231952689, "tok/s"),
                record("r2-w4-split-decode", "s1.resident-two-slot-split", {"opaque-slot-a": "gpu-a", "opaque-slot-b": "gpu-b"}, "warm_decode_tok_s", 67.01495086043339, "tok/s"),
                record("r2-w4-baseline-ttft", "s0.source-backed-single-offload", {"whole-model-slot": "gpu-a"}, "warm_ttft_ms", 3504.4774749985663, "ms"),
                record("r2-w4-split-ttft", "s1.resident-two-slot-split", {"opaque-slot-a": "gpu-a", "opaque-slot-b": "gpu-b"}, "warm_ttft_ms", 466.800324, "ms"),
            ],
        }
    )


def baseline_memory_census(implementation_commit: str | None = None) -> dict:
    return freeze(
        {
            "schema": "inferswarm.r3.baseline-memory-census/1",
            "implementation_commit": implementation_commit,
            "candidate_shape": "s0.source-backed-single-offload",
            "mechanical_sources": ["docs/inferswarm_r2/benchmark.json", "docs/inferswarm_r2/baseline-config.json"],
            "required_persistent_vram_bytes": 10_733_223_936,
            "required_persistent_host_source_bytes": 18_182_307_840,
            "runtime_capacity_tokens": 17_152,
            "runtime_backend_allocations_included_in_vram_measurement": True,
            "vram_safety_headroom_bytes": 536_870_912,
            "source_lifecycle": "ordinary offload; retained and required while serving",
            "process_rss_is_not_the_feasibility_measure": True,
        }
    )


def synthetic_planner_proof(implementation_commit: str) -> dict:
    """Exercise the production generic planner with arbitrary non-Qwen names."""
    problem = freeze(
        {
            "schema": "inferswarm.r3.synthetic-strategy/1",
            "implementation_commit": implementation_commit,
            "evidence_context": {"synthetic_revision": "violet-1"},
            "shapes": [
                {
                    "id": "violet-shape",
                    "slots": [
                        {
                            "id": "amber-slot",
                            "required_capabilities": ["widget-transform-v1"],
                            "memory": {"persistent_required_bytes": 64},
                        }
                    ],
                    "strategy_payload": {"opaque_domain": "non-model-widget"},
                }
            ],
        }
    )
    snapshot = freeze(
        {
            "schema": "inferswarm.r3.synthetic-resources/1",
            "implementation_commit": implementation_commit,
            "nodes": [
                {
                    "id": "workbench",
                    "compute_units": [
                        {
                            "id": "copper-unit",
                            "memory_resource_id": "copper-memory",
                            "capabilities": ["widget-transform-v1"],
                        }
                    ],
                    "memory_resources": [
                        {
                            "id": "copper-memory",
                            "kind": "synthetic-memory",
                            "capacity_bytes": 1024,
                        }
                    ],
                }
            ],
            "links": [],
        }
    )
    policy = freeze(
        {
            "schema": "inferswarm.r3.synthetic-policy/1",
            "implementation_commit": implementation_commit,
            "excluded_compute_unit_ids": [],
        }
    )
    objective = freeze(
        {
            "schema": "inferswarm.r3.synthetic-objective/1",
            "implementation_commit": implementation_commit,
            "metric": "widgets_per_second",
            "direction": "MAXIMIZE",
            "unit": "widgets/s",
        }
    )
    catalog = freeze(
        {
            "schema": "inferswarm.r3.synthetic-evidence/1",
            "implementation_commit": implementation_commit,
            "records": [
                {
                    "id": "violet-copper-measurement",
                    "shape_id": "violet-shape",
                    "mapping": {"amber-slot": "copper-unit"},
                    "required_context": {"synthetic_revision": "violet-1"},
                    "freshness": "CURRENT",
                    "evidence_class": "SYNTHETIC_MEASURED",
                    "metric": {
                        "name": "widgets_per_second",
                        "value": 12.5,
                        "unit": "widgets/s",
                    },
                }
            ],
        }
    )
    decision = plan(problem, snapshot, policy, objective, catalog)
    expected = "violet-shape[amber-slot=copper-unit]"
    return {
        "schema": "inferswarm.r3.synthetic-planner-proof/1",
        "implementation_commit": implementation_commit,
        "same_generic_planner": "freetoken.research.r3_planner.plan",
        "cpu_testable": True,
        "input_digests": decision["inputs"],
        "decision_digest": decision["digest"],
        "selected_candidate_id": decision["selected_candidate_id"],
        "expected_candidate_id": expected,
        "passed": decision["selected_candidate_id"] == expected,
    }


def build(output: Path, repository_root: Path, implementation_commit: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    problem = planning_problem(implementation_commit)
    snapshot = resource_snapshot(implementation_commit)
    evidence = evidence_catalog(repository_root, implementation_commit)
    policies = {
        "a": freeze({"schema": "inferswarm.r3.operator-policy/1", "implementation_commit": implementation_commit, "scenario": "A", "excluded_compute_unit_ids": [], "reservations_bytes": {}}),
        "b": freeze({"schema": "inferswarm.r3.operator-policy/1", "implementation_commit": implementation_commit, "scenario": "B", "excluded_compute_unit_ids": [], "reservations_bytes": {}}),
        "c": freeze({"schema": "inferswarm.r3.operator-policy/1", "implementation_commit": implementation_commit, "scenario": "C", "excluded_compute_unit_ids": ["gpu-b"], "reservations_bytes": {}}),
    }
    objectives = {
        "a": freeze({"schema": "inferswarm.r3.objective/1", "implementation_commit": implementation_commit, "id": "warm-decode-throughput", "metric": "warm_decode_tok_s", "direction": "MAXIMIZE", "unit": "tok/s", "statistic": "median", "startup_cost_included": False, "evidence_context": {"workload_geometry": SCHEMA_CONTEXT["workload_geometry"]}}),
        "b": freeze({"schema": "inferswarm.r3.objective/1", "implementation_commit": implementation_commit, "id": "warm-request-ttft-w4", "metric": "warm_ttft_ms", "direction": "MINIMIZE", "unit": "ms", "statistic": "median", "startup_cost_included": False, "evidence_context": {"workload_geometry": SCHEMA_CONTEXT["workload_geometry"]}}),
        "c": freeze({"schema": "inferswarm.r3.objective/1", "implementation_commit": implementation_commit, "id": "warm-decode-throughput", "metric": "warm_decode_tok_s", "direction": "MAXIMIZE", "unit": "tok/s", "statistic": "median", "startup_cost_included": False, "evidence_context": {"workload_geometry": SCHEMA_CONTEXT["workload_geometry"]}}),
    }
    _write(output / "resource-snapshot.json", snapshot)
    _write(output / "strategy-problem.json", problem)
    _write(output / "baseline-memory-census.json", baseline_memory_census(implementation_commit))
    _write(output / "evidence-catalog.json", evidence)
    _write(output / "synthetic-planner-proof.json", synthetic_planner_proof(implementation_commit))
    for scenario in "abc":
        _write(output / f"policy-{scenario}.json", policies[scenario])
        _write(output / f"objective-{scenario}.json", objectives[scenario])
        decision = plan(problem, snapshot, policies[scenario], objectives[scenario], evidence)
        _write(output / f"decision-{scenario}.json", decision)
        if scenario in "ab":
            compiled = compile_selected(decision, problem, decision["inputs"])
            _write(output / f"compiled-plan-{scenario}.json", compiled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/inferswarm_r3"))
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    build(args.output, repository_root, _require_clean_implementation(repository_root))


if __name__ == "__main__":
    main()
