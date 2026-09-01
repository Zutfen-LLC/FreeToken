from __future__ import annotations

import copy
import json

import pytest

from benchmarks.inferswarm_r2.compose_result_v2 import compose
from benchmarks.inferswarm_r2.v2_support import (
    FROZEN_PLAN_DIGEST,
    GENERATION_SETTINGS,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    REFERENCE_GPU_UUID,
    REFERENCE_RUNTIME_CONFIGURATION,
    SELECTED_STEPS,
    WORKLOAD_MANIFEST_SHA256,
    WORKLOAD_ORDER,
    methodology_record,
    select_reference_pair,
    validate_candidate_pass,
    validate_reference_artifact,
    validate_v2_output_path,
)


def _reference(session: str = "A") -> dict:
    return {
        "schema": "inferswarm.r2.reference-v2/1",
        "evidence_label": "CANONICAL_REFERENCE_CANDIDATE",
        "session": session,
        "methodology": methodology_record(),
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "producer": {"freetoken_commit": "a" * 40, "gpu_uuid": REFERENCE_GPU_UUID},
        "runtime_configuration": copy.deepcopy(REFERENCE_RUNTIME_CONFIGURATION),
        "workload_manifest_sha256": WORKLOAD_MANIFEST_SHA256,
        "selected_steps": SELECTED_STEPS,
        "generation_settings": GENERATION_SETTINGS,
        "workloads": [
            {
                "class_id": class_id,
                "prompt_token_ids": [index, 2],
                "generated_token_ids": list(range(32)),
                "selected_logit_steps": {
                    str(step): {
                        "float32_sha256": "f" * 64,
                        "nan_count": 0,
                        "inf_count": 0,
                        "full_logits": [[float(index), float(step)]],
                    }
                    for step in SELECTED_STEPS
                },
                "layer_18_seam_checkpoints": (
                    [{"pair": {"raw_byte_sha256": "e" * 64}}]
                    if class_id in {"W2", "W4"}
                    else []
                ),
            }
            for index, class_id in enumerate(WORKLOAD_ORDER)
        ],
    }


def _paths(tmp_path, left: dict, right: dict):
    a = tmp_path / "reference-v2-session-a.json"
    b = tmp_path / "reference-v2-session-b.json"
    a.write_text(json.dumps(left))
    b.write_text(json.dumps(right))
    return a, b


def _candidate() -> dict:
    gates = {
        name: True
        for name in (
            "resource_identity",
            "plan",
            "tokens",
            "selected_logits",
            "numerical_health",
            "session_isolation",
            "boundary",
            "backend_native",
            "host_mirror",
            "ownership",
            "model_state_movement",
        )
    }
    return {
        "schema": "inferswarm.r2.correctness-v2/1",
        "methodology": methodology_record(),
        "reference": {"canonical_artifact_sha256": "a" * 64},
        "plan": {"digest": FROZEN_PLAN_DIGEST},
        "acceptance_gates": gates,
        "passed": True,
    }


def test_reference_missing_methodology_provenance_rejected():
    value = _reference()
    del value["methodology"]
    with pytest.raises(ValueError, match="methodology"):
        validate_reference_artifact(value)


def test_wrong_methodology_merge_sha_rejected():
    value = _reference()
    value["methodology"]["merge_commit"] = "b" * 40
    with pytest.raises(ValueError, match="methodology"):
        validate_reference_artifact(value)


def test_wrong_canonical_gpu_uuid_rejected():
    value = _reference()
    value["producer"]["gpu_uuid"] = "GPU-wrong"
    with pytest.raises(ValueError, match="canonical GPU UUID"):
        validate_reference_artifact(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("moe_cache_slots", 3773),
        ("prefill_overlap", True),
        ("prefill_chunk_tokens", 128),
        ("runtime_capacity_tokens", 17151),
        ("graph_policy", "eager"),
        ("session_state_protocol", "warm"),
    ],
)
def test_wrong_frozen_runtime_geometry_rejected(field, value):
    reference = _reference()
    reference["runtime_configuration"][field] = value
    with pytest.raises(ValueError, match="resolved runtime configuration"):
        validate_reference_artifact(reference)


def test_reference_pair_token_mismatch_fails_selection(tmp_path):
    left, right = _reference("A"), _reference("B")
    right["workloads"][0]["generated_token_ids"][3] = 999
    a, b = _paths(tmp_path, left, right)
    result = select_reference_pair(left, right, session_a_path=a, session_b_path=b)
    assert not result["self_consistency_passed"]
    assert not result["candidate_evaluation_authorized"]


def test_reference_pair_logit_threshold_failure_blocks_selection(tmp_path):
    left, right = _reference("A"), _reference("B")
    right["workloads"][2]["selected_logit_steps"]["15"]["full_logits"][0][0] += 1
    a, b = _paths(tmp_path, left, right)
    result = select_reference_pair(left, right, session_a_path=a, session_b_path=b)
    assert not result["logit_consistency_passed"]
    assert result["canonical_reference"] is None


def test_session_b_can_never_become_canonical(tmp_path):
    left, right = _reference("A"), _reference("B")
    a, b = _paths(tmp_path, left, right)
    result = select_reference_pair(left, right, session_a_path=a, session_b_path=b)
    assert result["canonical_reference"] == a.name
    assert result["corroborating_reference"] == b.name
    assert result["selection_rule"] == "session-a-predeclared"


def test_failed_self_consistency_forbids_candidate_evaluation(tmp_path):
    left, right = _reference("A"), _reference("B")
    right["workloads"][1]["generated_token_ids"][0] = -1
    a, b = _paths(tmp_path, left, right)
    result = select_reference_pair(left, right, session_a_path=a, session_b_path=b)
    assert result["candidate_evaluation_authorized"] is False


def test_canonical_session_a_hash_is_required_in_candidate_result():
    value = _candidate()
    del value["reference"]["canonical_artifact_sha256"]
    with pytest.raises(ValueError, match="session A hash"):
        validate_candidate_pass(value)


@pytest.mark.parametrize(
    "name", ["frozen-plan.json", "correctness.json", "benchmark.json", "result.json"]
)
def test_historical_artifact_paths_cannot_be_overwritten(name, tmp_path):
    with pytest.raises(ValueError, match="historical artifact"):
        validate_v2_output_path(tmp_path / name)


def test_v2_result_cannot_declare_pass_unless_all_architectural_gates_pass(tmp_path):
    value = _candidate()
    value["acceptance_gates"]["boundary"] = False
    value["passed"] = True
    (tmp_path / "correctness-v2.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="boundary"):
        compose(tmp_path)


def test_healthy_candidate_with_canonical_hash_passes_gate_validation():
    validate_candidate_pass(_candidate())
