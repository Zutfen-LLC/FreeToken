"""Fail-closed gates for the frozen R2 correctness methodology v2."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

METHODOLOGY_REPOSITORY = "Zutfen-LLC/inferswarm"
METHODOLOGY_MERGE_COMMIT = "b05be56317449fe59c1a9adaa8dc81ae14142737"
METHODOLOGY_DOCUMENT = "docs/implementation/r2-correctness-reference-methodology-v2.md"
MODEL_REPOSITORY = "nvidia/Qwen3.6-35B-A3B-NVFP4"
MODEL_REVISION = "491c2f1ea524c639598bf8fa787a93fed5a6fbce"
REFERENCE_GPU_UUID = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"
CANDIDATE_GPU_UUIDS = {
    "a": REFERENCE_GPU_UUID,
    "b": "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55",
}
WORKLOAD_MANIFEST_SHA256 = (
    "10f81e5418a71a68f387632de422c3337cc7ba0518111a8746ad856d0210b24a"
)
FROZEN_PLAN_DIGEST = (
    "sha256:6128dd6705d6d692df3d5fc11cc130dba5c010cfff40c0e4c5ec7c19e1b78ff0"
)
WORKLOAD_ORDER = ["W1", "W2", "W3", "W4"]
SELECTED_STEPS = [0, 1, 15, 31]
HISTORICAL_ARTIFACT_NAMES = frozenset(
    {"frozen-plan.json", "correctness.json", "benchmark.json", "result.json"}
)
REFERENCE_RUNTIME_CONFIGURATION = {
    "attention_backend": "fi",
    "nvfp4_backend": "triton",
    "moe_backend": "offload",
    "moe_cache_slots": 3774,
    "prefill_overlap": False,
    "runtime_capacity_tokens": 17152,
    "prefill_chunk_tokens": 64,
    "page_size": 1,
    "logical_page_mapping": "identity",
    "concurrency": 1,
    "linear_state_slots": 1,
    "graph_policy": "one-full-model-bs1-decode-capture",
    "decode_graph_captures": 1,
    "decode_graph_recaptures": 0,
    "cross_workload_prefix_reuse": "none",
    "session_state_protocol": "fresh-zeroed-state-per-workload",
}
GENERATION_SETTINGS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "ignore_eos": True,
    "generated_tokens": 32,
}


def methodology_record() -> dict:
    return {
        "repository": METHODOLOGY_REPOSITORY,
        "merge_commit": METHODOLOGY_MERGE_COMMIT,
        "document": METHODOLOGY_DOCUMENT,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_v2_output_path(path: Path) -> None:
    if path.name in HISTORICAL_ARTIFACT_NAMES:
        raise ValueError(f"v2 workflow refuses historical artifact path {path.name}")


def _require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{label} mismatch: expected {expected!r}, observed {actual!r}"
        )


def validate_reference_artifact(reference: dict) -> None:
    """Validate all frozen reference identity and resolved-configuration fields."""

    _require_equal(reference.get("methodology"), methodology_record(), "methodology")
    _require_equal(
        reference.get("model"),
        {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "model",
    )
    producer = reference.get("producer", {})
    _require_equal(producer.get("gpu_uuid"), REFERENCE_GPU_UUID, "canonical GPU UUID")
    _require_equal(
        reference.get("runtime_configuration"),
        REFERENCE_RUNTIME_CONFIGURATION,
        "resolved runtime configuration",
    )
    _require_equal(
        reference.get("workload_manifest_sha256"),
        WORKLOAD_MANIFEST_SHA256,
        "workload manifest SHA-256",
    )
    _require_equal(reference.get("selected_steps"), SELECTED_STEPS, "selected steps")
    _require_equal(
        reference.get("generation_settings"), GENERATION_SETTINGS, "generation settings"
    )
    rows = reference.get("workloads", [])
    _require_equal(
        [row.get("class_id") for row in rows], WORKLOAD_ORDER, "workload order"
    )
    for row in rows:
        if not isinstance(row.get("prompt_token_ids"), list):
            raise TypeError(f"{row.get('class_id')} prompt token IDs missing")
        _require_equal(
            len(row.get("generated_token_ids", [])), 32, "generated token count"
        )
        checkpoints = row.get("selected_logit_steps", {})
        _require_equal(
            sorted(map(int, checkpoints)), SELECTED_STEPS, "logit checkpoints"
        )
        for step, record in checkpoints.items():
            if record.get("nan_count") != 0 or record.get("inf_count") != 0:
                raise ValueError(
                    f"numerical health failed at {row['class_id']} step {step}"
                )
            if "full_logits" not in record or "float32_sha256" not in record:
                raise ValueError(
                    f"full logit evidence missing at {row['class_id']} step {step}"
                )
    seam_classes = {
        row["class_id"] for row in rows if row.get("layer_18_seam_checkpoints")
    }
    if not {"W2", "W4"}.issubset(seam_classes):
        raise ValueError("layer-18 seam evidence is required for W2 and W4")


def _compare_logit_records(left: dict, right: dict) -> dict:
    actual = torch.tensor(left["full_logits"], dtype=torch.float32).reshape(-1)
    expected = torch.tensor(right["full_logits"], dtype=torch.float32).reshape(-1)
    if actual.shape != expected.shape:
        raise ValueError("selected logit checkpoint shape mismatch")
    difference = (actual - expected).abs()
    relative = torch.where(
        expected.abs() > 0,
        difference / expected.abs(),
        torch.where(difference == 0, 0.0, float("inf")),
    )
    return {
        "exact": bool(torch.equal(actual, expected)),
        "within_threshold": bool(
            torch.allclose(actual, expected, rtol=0.002, atol=0.002)
        ),
        "rtol": 0.002,
        "atol": 0.002,
        "max_absolute_deviation": float(difference.max().item()),
        "max_relative_deviation": float(relative.max().item()),
        "left_float32_sha256": left["float32_sha256"],
        "right_float32_sha256": right["float32_sha256"],
    }


def select_reference_pair(
    session_a: dict,
    session_b: dict,
    *,
    session_a_path: Path,
    session_b_path: Path,
) -> dict:
    """Apply the predeclared A/B gate. Session B is never selectable."""

    validate_reference_artifact(session_a)
    validate_reference_artifact(session_b)
    identity_fields = (
        "methodology",
        "model",
        "runtime_configuration",
        "workload_manifest_sha256",
        "selected_steps",
        "generation_settings",
    )
    identity_exact = all(
        session_a[field] == session_b[field] for field in identity_fields
    )
    rows_a = {row["class_id"]: row for row in session_a["workloads"]}
    rows_b = {row["class_id"]: row for row in session_b["workloads"]}
    comparisons = []
    tokens_exact = True
    seams_exact = True
    for class_id in WORKLOAD_ORDER:
        left, right = rows_a[class_id], rows_b[class_id]
        prompt_exact = left["prompt_token_ids"] == right["prompt_token_ids"]
        token_exact = left["generated_token_ids"] == right["generated_token_ids"]
        tokens_exact &= prompt_exact and token_exact
        logit_rows = []
        for step in SELECTED_STEPS:
            compared = _compare_logit_records(
                left["selected_logit_steps"][str(step)],
                right["selected_logit_steps"][str(step)],
            )
            logit_rows.append({"generated_step": step, **compared})
        seam = None
        if class_id in {"W2", "W4"}:
            seam = (
                left["layer_18_seam_checkpoints"] == right["layer_18_seam_checkpoints"]
            )
            seams_exact &= seam
        comparisons.append(
            {
                "class_id": class_id,
                "prompt_token_ids_exact": prompt_exact,
                "generated_token_ids_exact": token_exact,
                "selected_logits": logit_rows,
                "layer_18_seam_exact": seam,
            }
        )
    logits_passed = all(
        item["within_threshold"]
        for row in comparisons
        for item in row["selected_logits"]
    )
    passed = identity_exact and tokens_exact and logits_passed and seams_exact
    return {
        "schema": "inferswarm.r2.reference-v2-selection/1",
        "methodology": methodology_record(),
        "session_a": {
            "artifact": session_a_path.name,
            "sha256": sha256_file(session_a_path),
        },
        "session_b": {
            "artifact": session_b_path.name,
            "sha256": sha256_file(session_b_path),
        },
        "identity_exact": identity_exact,
        "token_consistency_passed": tokens_exact,
        "logit_consistency_passed": logits_passed,
        "seam_consistency_passed": seams_exact,
        "comparisons": comparisons,
        "self_consistency_passed": passed,
        "canonical_reference": session_a_path.name if passed else None,
        "corroborating_reference": session_b_path.name if passed else None,
        "selection_rule": "session-a-predeclared",
        "candidate_evaluation_authorized": passed,
    }


def validate_candidate_pass(payload: dict) -> None:
    """Reject a v2 pass unless every frozen architectural gate is evidenced."""

    _require_equal(payload.get("schema"), "inferswarm.r2.correctness-v2/1", "schema")
    _require_equal(payload.get("methodology"), methodology_record(), "methodology")
    _require_equal(
        payload.get("plan", {}).get("digest"), FROZEN_PLAN_DIGEST, "plan digest"
    )
    reference = payload.get("reference", {})
    if not reference.get("canonical_artifact_sha256"):
        raise ValueError("canonical session A hash is required")
    gates = payload.get("acceptance_gates", {})
    required = {
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
    }
    missing = sorted(required - gates.keys())
    if missing:
        raise ValueError(f"candidate acceptance gates missing: {missing}")
    failed = sorted(name for name in required if gates[name] is not True)
    if failed:
        raise ValueError(f"candidate acceptance gates failed: {failed}")
    if payload.get("passed") is not True:
        raise ValueError("all gates pass but correctness-v2 is not marked passed")


__all__ = [
    "CANDIDATE_GPU_UUIDS",
    "FROZEN_PLAN_DIGEST",
    "GENERATION_SETTINGS",
    "METHODOLOGY_MERGE_COMMIT",
    "MODEL_REPOSITORY",
    "MODEL_REVISION",
    "REFERENCE_GPU_UUID",
    "REFERENCE_RUNTIME_CONFIGURATION",
    "SELECTED_STEPS",
    "WORKLOAD_MANIFEST_SHA256",
    "WORKLOAD_ORDER",
    "methodology_record",
    "select_reference_pair",
    "sha256_file",
    "validate_candidate_pass",
    "validate_reference_artifact",
    "validate_v2_output_path",
]
