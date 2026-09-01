"""CPU-testable correctness provenance and diagnostic safety gates for R2."""

from __future__ import annotations

import hashlib
from pathlib import Path

NONCANONICAL_LABEL = "NONCANONICAL_DIAGNOSTIC_EVIDENCE"
DIAGNOSTIC_OVERRIDE = "NONCANONICAL_DIAGNOSTIC_OVERRIDE"
CANONICAL_ARTIFACT_NAMES = frozenset({"correctness.json", "result.json"})
REQUIRED_REFERENCE_FIELDS = frozenset(
    {
        "schema",
        "model",
        "revision",
        "producer_commit",
        "runtime_configuration",
        "prefill_chunk_tokens",
        "runtime_capacity_tokens",
        "session_state_protocol",
        "graph_policy",
        "selected_steps",
        "workload_order",
    }
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_diagnostic_output(path: Path, *, diagnostic_override: bool) -> None:
    if diagnostic_override and path.name in CANONICAL_ARTIFACT_NAMES:
        raise ValueError(
            "diagnostic override refuses canonical correctness.json/result.json output"
        )


def diagnostic_shared_bytes(plan: dict, diagnostic_prefill_chunk: int | None) -> int:
    """Size transient transport staging without changing the frozen plan."""

    canonical = plan["boundary"]["contract"]["prefill_chunk_payload_bytes"]
    if diagnostic_prefill_chunk is None:
        return canonical
    contract = plan["boundary"]["contract"]
    requested = (
        diagnostic_prefill_chunk
        * contract["planes"]
        * contract["row_width"]
        * contract["element_bytes"]
    )
    return max(canonical, requested)


def reference_provenance(
    reference: dict,
    reference_path: Path,
    *,
    required_model: str,
    required_revision: str,
    required_classes: list[str],
    required_prompt_ids: dict[str, list[int]],
    allow_legacy_diagnostic: bool = False,
) -> dict:
    """Validate a self-describing reference and return retained provenance.

    Legacy N1 references may be inspected only by an explicitly noncanonical
    diagnostic. Missing facts stay explicit; they are never guessed into canonical
    provenance.
    """

    metadata = reference.get("reference_metadata")
    if metadata is None:
        if not allow_legacy_diagnostic:
            raise ValueError("reference_metadata is required")
        metadata = {
            "schema": reference.get("schema"),
            "model": reference.get("model"),
            "revision": reference.get("revision"),
            "producer_commit": reference.get("free_token_sha"),
            "runtime_configuration": None,
            "prefill_chunk_tokens": None,
            "runtime_capacity_tokens": None,
            "session_state_protocol": None,
            "graph_policy": None,
            "selected_steps": [0, 1, 15, 31],
            "workload_order": [
                row.get("class_id") for row in reference.get("workloads", [])
            ],
        }
        missing = sorted(
            key for key in REQUIRED_REFERENCE_FIELDS if metadata.get(key) is None
        )
        provenance_status = "LEGACY_REFERENCE_PROVENANCE_INCOMPLETE"
    else:
        missing = sorted(
            key for key in REQUIRED_REFERENCE_FIELDS if metadata.get(key) is None
        )
        if missing:
            raise ValueError(f"reference metadata missing required fields: {missing}")
        provenance_status = "COMPLETE"

    if metadata.get("model") != required_model:
        raise ValueError("reference model mismatch")
    if metadata.get("revision") != required_revision:
        raise ValueError("reference revision mismatch")

    rows = reference.get("workloads", [])
    by_class = {row.get("class_id"): row for row in rows}
    if set(by_class) != set(metadata.get("workload_order", [])):
        raise ValueError("reference workload identity disagrees with metadata")
    for class_id in required_classes:
        if class_id not in by_class:
            raise ValueError(f"reference workload identity missing {class_id}")
        if by_class[class_id].get("prompt_token_ids") != required_prompt_ids[class_id]:
            raise ValueError(f"reference workload identity mismatch for {class_id}")

    return {
        "artifact_sha256": sha256_file(reference_path),
        **metadata,
        "provenance_status": provenance_status,
        "missing_required_metadata": missing,
    }


def tensor_record(tensor) -> dict:
    """Return a stable raw-byte tensor record without retaining its values."""

    import torch

    value = tensor.detach().contiguous()
    raw = value.view(torch.uint8).cpu().numpy().tobytes()
    floating = value.float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "raw_byte_sha256": hashlib.sha256(raw).hexdigest(),
        "min": float(floating.min().item()) if floating.numel() else None,
        "max": float(floating.max().item()) if floating.numel() else None,
    }


def compare_tensor_records(actual: dict, expected: dict) -> dict:
    """Validate geometry and compare hash-only diagnostic records."""

    if actual["shape"] != expected["shape"]:
        raise ValueError("diagnostic tensor shape mismatch")
    if actual["dtype"] != expected["dtype"]:
        raise ValueError("diagnostic tensor dtype mismatch")
    return {
        "shape": actual["shape"],
        "dtype": actual["dtype"],
        "exact": actual["raw_byte_sha256"] == expected["raw_byte_sha256"],
        "actual_raw_byte_sha256": actual["raw_byte_sha256"],
        "reference_raw_byte_sha256": expected["raw_byte_sha256"],
    }


def first_divergence(checkpoints: list[dict]) -> dict | None:
    """Return the first ordered checkpoint whose compared components differ."""

    for checkpoint in checkpoints:
        differing = [
            name
            for name, comparison in checkpoint["comparisons"].items()
            if not comparison["exact"]
        ]
        if differing:
            return {
                "checkpoint": checkpoint["checkpoint"],
                "first_differing_components": differing,
            }
    return None


__all__ = [
    "DIAGNOSTIC_OVERRIDE",
    "NONCANONICAL_LABEL",
    "compare_tensor_records",
    "diagnostic_shared_bytes",
    "first_divergence",
    "reference_provenance",
    "sha256_file",
    "tensor_record",
    "validate_diagnostic_output",
]
