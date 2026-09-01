"""Compose compact, checksummed NONCANONICAL R2 diagnostic summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha

from .correctness_support import NONCANONICAL_LABEL, sha256_file


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _workload(document: dict, class_id: str) -> dict:
    return next(
        row
        for row in document["correctness"]["workloads"]
        if row["class_id"] == class_id
    )


def _logit_summary(row: dict) -> dict:
    checkpoints = row["logit_checkpoints"]
    return {
        "all_exact": all(item["exact"] for item in checkpoints),
        "within_threshold": all(
            item["within_canonical_threshold"] for item in checkpoints
        ),
        "max_absolute_deviation": max(
            item["max_absolute_deviation"] for item in checkpoints
        ),
        "max_relative_deviation": max(
            item["max_relative_deviation"] for item in checkpoints
        ),
        "nan_count": sum(item["nan_count"] for item in checkpoints),
        "inf_count": sum(item["inf_count"] for item in checkpoints),
        "selected_float32_sha256": {
            str(item["generated_step"]): item["actual_float32_sha256"]
            for item in checkpoints
        },
    }


def _state_exact(split_state: dict, local_state: dict) -> dict:
    kv = {
        layer: split_state["kv_by_global_layer"][layer]["raw_byte_sha256"]
        == record["raw_byte_sha256"]
        for layer, record in local_state["kv_by_global_layer"].items()
        if layer in split_state["kv_by_global_layer"]
    }
    linear = {
        layer: {
            name: split_state["linear_by_global_layer"][layer][name]["raw_byte_sha256"]
            == record[name]["raw_byte_sha256"]
            for name in ("conv", "recurrent")
        }
        for layer, record in local_state["linear_by_global_layer"].items()
        if layer in split_state["linear_by_global_layer"]
    }
    return {
        "kv_by_global_layer": kv,
        "linear_by_global_layer": linear,
        "all_exact": all(kv.values())
        and all(all(parts.values()) for parts in linear.values()),
    }


def _numeric_difference(left, right) -> dict:
    import torch

    left_tensor = torch.tensor(left, dtype=torch.float32)
    right_tensor = torch.tensor(right, dtype=torch.float32)
    difference = (left_tensor - right_tensor).abs()
    relative = torch.where(
        right_tensor.abs() > 0,
        difference / right_tensor.abs(),
        torch.where(difference == 0, 0.0, float("inf")),
    )
    return {
        "exact": bool(torch.equal(left_tensor, right_tensor)),
        "max_absolute_deviation": float(difference.max().item()),
        "max_relative_deviation": float(relative.max().item()),
        "nan_count": int(torch.isnan(left_tensor).sum().item())
        + int(torch.isnan(right_tensor).sum().item()),
        "inf_count": int(torch.isinf(left_tensor).sum().item())
        + int(torch.isinf(right_tensor).sum().item()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--w2-chunk32", type=Path, required=True)
    parser.add_argument("--w4-chunk128", type=Path, required=True)
    parser.add_argument("--w4-split-state", type=Path, required=True)
    parser.add_argument("--w4-local64", type=Path, required=True)
    parser.add_argument("--w4-local128", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    retained = _load(args.retained)
    w2_32_doc = _load(args.w2_chunk32)
    w4_128_doc = _load(args.w4_chunk128)
    split_doc = _load(args.w4_split_state)
    local64_doc = _load(args.w4_local64)
    local128_doc = _load(args.w4_local128)
    w2_64 = _workload(retained, "W2")
    w4_64 = _workload(retained, "W4")
    w2_32 = _workload(w2_32_doc, "W2")
    w4_128 = _workload(w4_128_doc, "W4")
    local64 = local64_doc["workload"]
    local128 = local128_doc["workload"]

    inputs = {
        path.name: sha256_file(path)
        for path in (
            args.retained,
            args.w2_chunk32,
            args.w4_chunk128,
            args.w4_split_state,
            args.w4_local64,
            args.w4_local128,
        )
    }
    common = {
        "evidence_label": NONCANONICAL_LABEL,
        "canonical_threshold": {"rtol": 0.002, "atol": 0.002},
        "input_artifact_sha256": inputs,
    }
    chunk_controls = {
        "schema": "inferswarm.r2.chunk-controls/1",
        **common,
        "cases": [
            {
                "case": "W2 current",
                "prompt_tokens": 54,
                "chunk_tokens": 64,
                "prefill_chunks": 1,
                "tokens_exact": w2_64["exact_generated_sequence"],
                "logits": _logit_summary(w2_64),
            },
            {
                "case": "W2 diagnostic",
                "prompt_tokens": 54,
                "chunk_tokens": 32,
                "prefill_chunks": 2,
                "tokens_exact": w2_32["exact_generated_sequence"],
                "logits": _logit_summary(w2_32),
            },
            {
                "case": "W4 current",
                "prompt_tokens": 121,
                "chunk_tokens": 64,
                "prefill_chunks": 2,
                "tokens_exact": w4_64["exact_generated_sequence"],
                "logits": _logit_summary(w4_64),
            },
            {
                "case": "W4 diagnostic",
                "prompt_tokens": 121,
                "chunk_tokens": 128,
                "prefill_chunks": 1,
                "tokens_exact": w4_128["exact_generated_sequence"],
                "logits": _logit_summary(w4_128),
            },
            {
                "case": "W4 matched local",
                "prompt_tokens": 121,
                "chunk_tokens": 64,
                "prefill_chunks": 2,
                "tokens_exact": local64["exact_generated_sequence"],
                "logits": _logit_summary(local64),
            },
        ],
        "reciprocal_observation": (
            "W2 becomes non-exact when changed from one to two prefill chunks; "
            "W4 step-0/step-1 logits become byte-exact to the legacy reference when "
            "changed from two chunks to one. Later W4 decode divergence shows that the "
            "legacy reference also leaves graph/state protocol unmatched."
        ),
    }
    write_json_with_sha(args.out_dir / "chunk-controls.json", chunk_controls)

    split_chunks = [
        row
        for row in split_doc["correctness"]["workloads"][0]["boundaries"]
        if row["operation"] == "prefill"
    ]
    local_chunks = local64["prefill_checkpoints"]
    checkpoint_rows = []
    for split, local in zip(split_chunks, local_chunks, strict=True):
        checkpoint_rows.append(
            {
                "start": split["position"],
                "token_count": split["token_count"],
                "layer_18_hidden_exact": split["producer_tensors"]["hidden"][
                    "raw_byte_sha256"
                ]
                == local["layer_18_output"]["hidden"]["raw_byte_sha256"],
                "layer_18_residual_exact": split["producer_tensors"]["residual"][
                    "raw_byte_sha256"
                ]
                == local["layer_18_output"]["residual"]["raw_byte_sha256"],
                "local_pair_sha256": local["layer_18_output"]["pair"][
                    "raw_byte_sha256"
                ],
                "producer_sha256": split["producer_sha256"],
                "consumer_sha256": split["consumer_sha256"],
                "transport_exact": split["producer_sha256"] == split["consumer_sha256"],
                "block_a_state": _state_exact(
                    split["producer_state"], local["mutable_state"]
                ),
                "block_b_state": _state_exact(
                    split["consumer_state"], local["mutable_state"]
                ),
                "block_b_output_hidden_exact": split["execution_diagnostic"][
                    "block_output_hidden"
                ]["raw_byte_sha256"]
                == local["block_b_output_hidden"]["raw_byte_sha256"],
                "block_b_output_residual_exact": split["execution_diagnostic"][
                    "block_output_residual"
                ]["raw_byte_sha256"]
                == local["block_b_output_residual"]["raw_byte_sha256"],
                "final_norm_exact": split["execution_diagnostic"]["final_norm"][
                    "raw_byte_sha256"
                ]
                == local["final_norm"]["raw_byte_sha256"],
                "logits_exact": split["execution_diagnostic"]["logits"][
                    "raw_byte_sha256"
                ]
                == local["logits"]["raw_byte_sha256"],
            }
        )
    matched = {
        "schema": "inferswarm.r2.matched-local-comparison/1",
        **common,
        "local_runtime_configuration": local64_doc["runtime_configuration"],
        "local_device": local64_doc["device"],
        "generated_tokens_exact_between_split_and_local": (
            split_doc["correctness"]["workloads"][0]["generated_token_ids"]
            == local64["generated_token_ids"]
        ),
        "selected_logits_byte_identical_between_split_and_local": (
            _logit_summary(split_doc["correctness"]["workloads"][0])[
                "selected_float32_sha256"
            ]
            == _logit_summary(local64)["selected_float32_sha256"]
        ),
        "prefill_checkpoints": checkpoint_rows,
        "all_captured_split_and_local_state_exact": all(
            row["layer_18_hidden_exact"]
            and row["layer_18_residual_exact"]
            and row["transport_exact"]
            and row["block_a_state"]["all_exact"]
            and row["block_b_state"]["all_exact"]
            and row["block_b_output_hidden_exact"]
            and row["block_b_output_residual_exact"]
            and row["final_norm_exact"]
            and row["logits_exact"]
            for row in checkpoint_rows
        ),
    }
    write_json_with_sha(args.out_dir / "matched-local-control.json", matched)

    state64 = local64["prefill_checkpoints"][-1]["mutable_state"]
    state128 = local128["prefill_checkpoints"][-1]["mutable_state"]
    linear_differences = []
    for layer in sorted(state64["linear_by_global_layer"], key=int):
        left = state64["linear_by_global_layer"][layer]
        right = state128["linear_by_global_layer"][layer]
        for component in ("conv", "recurrent"):
            if (
                left[component]["raw_byte_sha256"]
                != right[component]["raw_byte_sha256"]
            ):
                linear_differences.append(
                    {
                        "global_layer": int(layer),
                        "state": component,
                        "chunk64": left[component],
                        "chunk128": right[component],
                    }
                )
    kv_differences = []
    for layer in sorted(state64["kv_by_global_layer"], key=int):
        left = state64["kv_by_global_layer"][layer]
        right = state128["kv_by_global_layer"][layer]
        if left["raw_byte_sha256"] != right["raw_byte_sha256"]:
            kv_differences.append(
                {"global_layer": int(layer), "chunk64": left, "chunk128": right}
            )
    divergence = {
        "schema": "inferswarm.r2.first-divergence/1",
        **common,
        "classification": "REFERENCE_GEOMETRY_MISMATCH",
        "split_specific_first_divergence": None,
        "split_specific_result": (
            "No divergence: split and matched local are byte-identical through both "
            "W4 prefill chunks, all logical mutable state, finalization, selected logits, "
            "and all 32 generated tokens."
        ),
        "first_observed_legacy_reference_divergence": {
            "workload": "W4",
            "checkpoint": "after-second-prefill-chunk",
            "observable": "step-0 logits",
            "global_layer": None,
            "reason_layer_unavailable": "legacy reference retained no layer/state captures",
        },
        "first_chunk_geometry_state_difference": (
            {
                **linear_differences[0],
                "numerical_comparison": _numeric_difference(
                    local64["targeted_first_divergence_values"]["values"],
                    local128["targeted_first_divergence_values"]["values"],
                ),
            }
            if linear_differences
            else None
        ),
        "first_chunk_geometry_kv_difference": (
            kv_differences[0] if kv_differences else None
        ),
        "inference": (
            "The earliest retained state difference between matched local chunk64 and "
            "chunk128 is global layer 0 recurrent state; its conv state remains exact. "
            "This locates the geometry-dependent change to the layer-0 GatedDeltaNet "
            "recurrent update, before any R2 layer-19 boundary or transport."
        ),
    }
    write_json_with_sha(args.out_dir / "first-divergence.json", divergence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
