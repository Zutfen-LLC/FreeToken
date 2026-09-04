"""#76 case reducer: 15 frozen envelopes from paired reference/candidate runs.

Pure host-float64 reduction exactly per frozen REDUCER.md
(inferswarm@f394dc9): full-domain absolute differences, fsum-of-squares RMS,
nearest-rank/higher p99, per-case per-family maximum across all declared
checkpoints and replay positions.

Inputs are the per-case capture bundles (.pt, RowPruningSink format) from the
single arm (reference) and the chain arm (candidate). The chain bundles live
across stage subdirs; this module resolves the frozen checkpoint IDs from the
union of stage records using the stage-role mappings.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from benchmarks.inferswarm_76 import (
    CAPTURE_POSITIONS,
    CHECKPOINT_FAMILY_MAP,
    ENVELOPE_CHECKPOINT_IDS,
    SEAM_TO_CHECKPOINT,
    conservative_case_family,
    envelopes_from_case_metrics,
    tensor_metrics,
)

# Chain-stage seam -> frozen checkpoint id (envelope-relevant only).
CHAIN_SEAM_MAP = {
    "first": {
        "embedding_output": "embedding-output",
        "layer0_o_proj_input": "layer-0-o-proj-input",
        "layer0_o_proj_output": "layer-0-o-proj-output",
        "layer15_attn_o_proj_output": "global-layer-15-attention-o-proj-output",
        "after_layer_15": "post-global-layer-15-residual",
    },
    "middle": {
        "after_layer_31": "post-global-layer-31-residual",
    },
    "last": {
        "after_layer_47": "post-global-layer-47-residual",
        "final_norm": "final-normalized-hidden-state",
        "full_final_row_bf16_logits": "full-final-row-bf16-logits",
        "final_row_fp32": "full-final-row-fp32-consumer-logits",
    },
}
SINGLE_SEAM_MAP = SEAM_TO_CHECKPOINT["single"]


def _load_bundle(path: Path) -> list[dict[str, Any]]:
    import torch

    bundle = torch.load(path, map_location="cpu", weights_only=False)
    return [
        {"meta": meta, "tensor": tensor}
        for meta, tensor in zip(bundle["records"], bundle["tensors"])
    ]


def _grouped(
    records: list[dict[str, Any]], seam_map: dict[str, str]
) -> dict[tuple[int, str], Any]:
    out: dict[tuple[int, str], Any] = {}
    for record in records:
        meta = record["meta"]
        name = meta.get("checkpoint")
        if name not in seam_map:
            continue  # boundary probes: exact-layer evidence, not envelopes
        checkpoint_id = seam_map[name]
        position = int(meta["step"])
        key = (position, checkpoint_id)
        if key in out:
            raise ValueError(f"duplicate capture {key}")
        out[key] = record
    return out


def _require_dtype(record: dict[str, Any], checkpoint_id: str) -> None:
    expected = CHECKPOINT_FAMILY_MAP[checkpoint_id][1]
    observed = record["meta"]["dtype"]
    if observed != expected:
        raise ValueError(
            f"{checkpoint_id}: semantic dtype {observed} != frozen {expected}"
        )


def reduce_case(
    *,
    case_id: str,
    reference_bundle: Path,
    chain_bundles: dict[str, Path],
) -> dict[str, Any]:
    """Compare one case; returns envelope hex strings + integrity detail."""
    ref_records = _load_bundle(reference_bundle)
    ref = _grouped(ref_records, SINGLE_SEAM_MAP)

    cand: dict[tuple[int, str], Any] = {}
    stage_of: dict[tuple[int, str], str] = {}
    for stage_role, bundle_path in chain_bundles.items():
        records = _load_bundle(bundle_path)
        grouped = _grouped(records, CHAIN_SEAM_MAP[stage_role])
        for key, record in grouped.items():
            if key in cand:
                raise ValueError(f"duplicate chain capture {key} ({stage_role})")
            cand[key] = record
            stage_of[key] = stage_role

    per_checkpoint: dict[str, dict[str, float]] = {}
    integrity = {"missing_reference": [], "missing_candidate": [],
                 "dtype_mismatches": [], "nan_inf": []}
    for checkpoint_id in sorted(ENVELOPE_CHECKPOINT_IDS):
        for position in CAPTURE_POSITIONS:
            key = (position, checkpoint_id)
            if key not in ref:
                integrity["missing_reference"].append(list(key))
                continue
            if key not in cand:
                integrity["missing_candidate"].append(list(key))
                continue
            _require_dtype(ref[key], checkpoint_id)
            _require_dtype(cand[key], checkpoint_id)
            for side, record in (("ref", ref[key]), ("cand", cand[key])):
                meta = record["meta"]
                if meta["nan_count"] or meta["inf_count"]:
                    integrity["nan_inf"].append(
                        {"side": side, "key": list(key),
                         "nan": meta["nan_count"], "inf": meta["inf_count"]})
            r = ref[key]["tensor"]
            metrics = tensor_metrics(
                [float(v) for v in r.flatten().tolist()],
                [float(v) for v in cand[key]["tensor"].flatten().tolist()],
            )
            per_checkpoint[f"{checkpoint_id}@{position}"] = metrics
    if any(integrity[k] for k in
           ("missing_reference", "missing_candidate", "dtype_mismatches")):
        raise ValueError(f"{case_id}: incomplete capture set: {integrity}")
    if integrity["nan_inf"]:
        raise ValueError(f"{case_id}: NaN/Inf at declared finite checkpoint: "
                         f"{integrity['nan_inf'][:3]}")

    envelopes = envelopes_from_case_metrics(per_checkpoint)
    return {
        "case_id": case_id,
        "envelopes": envelopes,
        "integrity": integrity,
        "checkpoint_count": len(per_checkpoint),
    }
