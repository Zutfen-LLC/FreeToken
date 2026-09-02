from __future__ import annotations

from pathlib import Path

from inferswarm_r5a.strategy import (
    LOCAL_SPLIT_SHAPE,
    NETWORK_SHAPE,
    accepted_r4_evidence,
    compile_candidate,
    evidence_catalog,
    objective,
    operator_policy,
    planning_problem,
    resource_snapshot,
)
from freetoken.research.r3_planner import FEASIBLE_UNRANKED, freeze, plan


SHA = "584c2ae77ff37b932f4da6cd2b1652b0696066a9"


def _environment(runtime="pre-r5-integrated-runtime"):
    return freeze(
        {
            "schema": "inferswarm.r5a.frozen-environment/1",
            "implementation_commit": SHA,
            "runtime_context": runtime,
            "network_context": "1GbE-full-duplex-MTU1500-eno1-enp5s0",
            "network": {
                "link_id": "link.node-a-to-node-b.tcp",
                "negotiated_mbps": 1000,
            },
            "node_a": {
                "node_id": "node.inferswarm01",
                "gpus": [
                    {
                        "uuid": "GPU-a",
                        "pci_bdf": "0000:02:00.0",
                        "vram_total_bytes": 12_884_901_888,
                    },
                    {
                        "uuid": "GPU-local-b",
                        "pci_bdf": "0000:03:00.0",
                        "vram_total_bytes": 12_884_901_888,
                    },
                ],
            },
            "node_b": {
                "node_id": "node.inferswarm03",
                "gpus": [
                    {
                        "uuid": "GPU-b",
                        "pci_bdf": "0000:01:00.0",
                        "vram_total_bytes": 12_884_901_888,
                    }
                ],
            },
        }
    )


def test_accepted_r4_evidence_is_normalized_with_full_provenance():
    row = accepted_r4_evidence(Path("."))[0]
    assert row["measurement_status"] == "MEASURED"
    assert row["metric"]["value"] == 2.947
    assert row["constraint"]["threshold"] == 747.12
    assert row["provenance"]["source_artifact"] == "docs/inferswarm_r4/result.json"
    assert len(row["provenance"]["source_artifact_sha256"]) == 64


def test_integrated_runtime_rejects_r4_context_instead_of_copying_the_number():
    problem = planning_problem(SHA)
    snapshot = resource_snapshot(_environment())
    decision = plan(
        problem,
        snapshot,
        operator_policy(SHA),
        objective(SHA),
        evidence_catalog(SHA, Path(".")),
    )
    network = next(row for row in decision["evaluations"] if row["shape_id"] == NETWORK_SHAPE)
    assert network["state"] == FEASIBLE_UNRANKED
    audit = next(
        row for row in network["evidence"] if row["evidence_id"].startswith("accepted-r4")
    )
    assert not audit["applicable"]
    assert "runtime_context" in " ".join(audit["reasons"])
    assert audit["influence"] == "rejected; did not affect admission or ranking"


def test_network_compiler_retains_complete_frozen_plan_semantics():
    problem = planning_problem(SHA)
    snapshot = resource_snapshot(_environment())
    # Supply one current exact-context TTFT measurement so this test can compile
    # the automatically selected network candidate without changing R4 evidence.
    current = {
        "id": "current-network-ttft",
        "role": "RANKING_OBJECTIVE",
        "producer_identity": SHA,
        "evidence_identity": "arm-sha",
        "shape_id": NETWORK_SHAPE,
        "mapping": {"slot-a": "gpu.node-a.0", "slot-b": "gpu.node-b.0"},
        "required_context": {
            "model_revision": "491c2f1ea524c639598bf8fa787a93fed5a6fbce",
            "runtime_context": "pre-r5-integrated-runtime",
            "network_context": "1GbE-full-duplex-MTU1500-eno1-enp5s0",
            "workload_geometry": "W2-W4-generate32-static",
        },
        "freshness": "CURRENT",
        "measurement_status": "MEASURED",
        "evidence_class": "MEASURED_R5A_MATCHED_HTTP_SERVING",
        "confidence": "EXACT_CONTEXT",
        "metric": {"name": "ttft_ms", "value": 500.0, "unit": "ms", "statistic": "median"},
    }
    catalog = evidence_catalog(SHA, Path("."), [current])
    decision = plan(problem, snapshot, operator_policy(SHA), objective(SHA), catalog)
    selected = next(row for row in decision["evaluations"] if row["id"] == decision["selected_candidate_id"])
    compiled = compile_candidate(
        selected,
        r4_plan={"digest": "sha256:r4"},
        local_plan={"digest": "sha256:r2"},
    )
    assert compiled["participants"] == ["node.inferswarm01", "node.inferswarm03"]
    assert compiled["strategy_realization"]["participant_plan_digest"] == "sha256:r4"
    assert compiled["semantic_boundaries"][0]["semantic_contract"]["decode_bytes"] == 8192
    assert compiled["expected_resource_accounting"]["strategy_host_lifecycle"]["release_after_final_residency"] is True


def test_local_split_compiler_binds_accepted_r2_plan_without_network_nouns():
    problem = planning_problem(SHA)
    snapshot = resource_snapshot(_environment())
    current = {
        "id": "current-local-split-ttft",
        "role": "RANKING_OBJECTIVE",
        "producer_identity": SHA,
        "evidence_identity": "arm-sha",
        "shape_id": LOCAL_SPLIT_SHAPE,
        "mapping": {"slot-a": "gpu.node-a.0", "slot-b": "gpu.node-a.1"},
        "required_context": {
            "model_revision": "491c2f1ea524c639598bf8fa787a93fed5a6fbce",
            "runtime_context": "pre-r5-integrated-runtime",
            "network_context": "1GbE-full-duplex-MTU1500-eno1-enp5s0",
            "workload_geometry": "W2-W4-generate32-static",
        },
        "freshness": "CURRENT",
        "measurement_status": "MEASURED",
        "evidence_class": "MEASURED_R5A_MATCHED_HTTP_SERVING",
        "confidence": "EXACT_CONTEXT",
        "metric": {"name": "ttft_ms", "value": 400.0, "unit": "ms", "statistic": "median"},
    }
    decision = plan(
        problem,
        snapshot,
        operator_policy(SHA),
        objective(SHA),
        evidence_catalog(SHA, Path("."), [current]),
    )
    selected = next(
        row for row in decision["evaluations"]
        if row["id"] == decision["selected_candidate_id"]
    )
    compiled = compile_candidate(
        selected,
        r4_plan={"digest": "sha256:r4"},
        local_plan={"digest": "sha256:r2"},
    )
    assert compiled["participants"] == ["node.inferswarm01"]
    assert compiled["strategy_realization"] == {
        "path": "r2-local-split",
        "participant_plan_digest": "sha256:r2",
    }
