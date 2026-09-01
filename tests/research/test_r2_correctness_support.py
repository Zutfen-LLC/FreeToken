from __future__ import annotations

import hashlib
import json

import pytest

from benchmarks.inferswarm_r2.correctness_support import (
    DIAGNOSTIC_OVERRIDE,
    NONCANONICAL_LABEL,
    compare_tensor_records,
    diagnostic_shared_bytes,
    first_divergence,
    reference_provenance,
    validate_diagnostic_output,
)


def _reference(tmp_path):
    reference = {
        "reference_metadata": {
            "schema": "reference/1",
            "model": "repo/model",
            "revision": "a" * 40,
            "producer_commit": "b" * 40,
            "runtime_configuration": {"attention": "fi"},
            "prefill_chunk_tokens": 64,
            "runtime_capacity_tokens": 17152,
            "session_state_protocol": "fresh-zeroed-state-per-workload",
            "graph_policy": "captured-bs1",
            "selected_steps": [0, 1, 15, 31],
            "workload_order": ["W2", "W4"],
        },
        "workloads": [
            {"class_id": "W2", "prompt_token_ids": [1, 2]},
            {"class_id": "W4", "prompt_token_ids": [3, 4]},
        ],
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    return path, reference


def test_reference_artifact_sha_and_provenance_are_required(tmp_path):
    path, reference = _reference(tmp_path)
    record = reference_provenance(
        reference,
        path,
        required_model="repo/model",
        required_revision="a" * 40,
        required_classes=["W2", "W4"],
        required_prompt_ids={"W2": [1, 2], "W4": [3, 4]},
    )
    assert record["artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["provenance_status"] == "COMPLETE"
    del reference["reference_metadata"]["graph_policy"]
    with pytest.raises(ValueError, match="missing required fields"):
        reference_provenance(
            reference,
            path,
            required_model="repo/model",
            required_revision="a" * 40,
            required_classes=["W2"],
            required_prompt_ids={"W2": [1, 2]},
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("model", "wrong", "model mismatch"), ("revision", "wrong", "revision mismatch")],
)
def test_reference_model_and_revision_mismatch_rejected(
    tmp_path, field, value, message
):
    path, reference = _reference(tmp_path)
    reference["reference_metadata"][field] = value
    with pytest.raises(ValueError, match=message):
        reference_provenance(
            reference,
            path,
            required_model="repo/model",
            required_revision="a" * 40,
            required_classes=["W2"],
            required_prompt_ids={"W2": [1, 2]},
        )


def test_reference_workload_identity_mismatch_rejected(tmp_path):
    path, reference = _reference(tmp_path)
    with pytest.raises(ValueError, match="workload identity mismatch"):
        reference_provenance(
            reference,
            path,
            required_model="repo/model",
            required_revision="a" * 40,
            required_classes=["W2"],
            required_prompt_ids={"W2": [9]},
        )


def test_diagnostic_override_cannot_write_canonical_artifacts(tmp_path):
    for name in ("correctness.json", "result.json"):
        with pytest.raises(ValueError, match="refuses canonical"):
            validate_diagnostic_output(tmp_path / name, diagnostic_override=True)
    validate_diagnostic_output(
        tmp_path / "chunk-controls.json", diagnostic_override=True
    )
    assert DIAGNOSTIC_OVERRIDE.startswith("NONCANONICAL")
    assert NONCANONICAL_LABEL == "NONCANONICAL_DIAGNOSTIC_EVIDENCE"


def test_diagnostic_shared_buffer_can_exceed_frozen_chunk_without_plan_mutation():
    plan = {
        "boundary": {
            "contract": {
                "prefill_chunk_payload_bytes": 64 * 2 * 2048 * 2,
                "planes": 2,
                "row_width": 2048,
                "element_bytes": 2,
            }
        }
    }
    assert diagnostic_shared_bytes(plan, None) == 524288
    assert diagnostic_shared_bytes(plan, 32) == 524288
    assert diagnostic_shared_bytes(plan, 128) == 1048576
    assert plan["boundary"]["contract"]["prefill_chunk_payload_bytes"] == 524288


def test_layer_19_capture_shape_dtype_and_first_divergence():
    left = {"shape": [64, 2048], "dtype": "bfloat16", "raw_byte_sha256": "a" * 64}
    right = dict(left)
    assert compare_tensor_records(left, right)["exact"]
    wrong = {**right, "shape": [63, 2048]}
    with pytest.raises(ValueError, match="shape"):
        compare_tensor_records(left, wrong)
    checkpoints = [
        {"checkpoint": "chunk-1", "comparisons": {"hidden": {"exact": True}}},
        {"checkpoint": "chunk-2", "comparisons": {"hidden": {"exact": False}}},
    ]
    assert first_divergence(checkpoints) == {
        "checkpoint": "chunk-2",
        "first_differing_components": ["hidden"],
    }


def test_state_hash_comparison_detects_difference():
    base = {"shape": [1, 2], "dtype": "bfloat16", "raw_byte_sha256": "a" * 64}
    changed = {**base, "raw_byte_sha256": "b" * 64}
    assert compare_tensor_records(base, changed)["exact"] is False
