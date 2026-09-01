"""Capture one fresh one-GPU reference for frozen R2 methodology v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
from freetoken.research.n0_model_block import write_json_with_sha

from .qwen_split_adapter import tensor_sha256
from .run_correctness import _prompt_ids
from .run_matched_local_control import MatchedLocalRuntime, _device_record
from .v2_support import (
    GENERATION_SETTINGS,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    REFERENCE_GPU_UUID,
    REFERENCE_RUNTIME_CONFIGURATION,
    SELECTED_STEPS,
    WORKLOAD_MANIFEST_SHA256,
    WORKLOAD_ORDER,
    methodology_record,
    validate_reference_artifact,
    validate_v2_output_path,
)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _validate_preflight(args, runtime: MatchedLocalRuntime, manifest_sha: str) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != REFERENCE_GPU_UUID:
        raise RuntimeError(
            "canonical reference requires exact CUDA_VISIBLE_DEVICES GPU UUID"
        )
    if args.revision != MODEL_REVISION:
        raise RuntimeError("model revision differs from frozen methodology")
    if Path(args.model).name != MODEL_REVISION:
        raise RuntimeError("resolved model path does not end in the frozen revision")
    if manifest_sha != WORKLOAD_MANIFEST_SHA256:
        raise RuntimeError("workload manifest SHA-256 differs from frozen methodology")
    device = _device_record()
    if device["uuid"] != REFERENCE_GPU_UUID:
        raise RuntimeError("resolved GPU UUID differs from canonical reference GPU")
    if runtime.config.moe_backend != "offload":
        raise RuntimeError("resolved MoE backend differs from ordinary offload")
    if runtime.config.nvfp4_backend != "triton":
        raise RuntimeError("resolved NVFP4 backend differs from Triton")
    if runtime.banks.quant_format != "nvfp4":
        raise RuntimeError("resolved expert representation differs from NVFP4")
    if runtime.cache.cache_size != 3774:
        raise RuntimeError("resolved MoE cache slots differ from 3774")
    if runtime.cache.prefill_overlap is not False:
        raise RuntimeError("resolved prefill overlap must be false")
    if runtime.capacity != 17152:
        raise RuntimeError("resolved runtime capacity differs from 17152")
    expected_pages = torch.arange(17152, dtype=torch.int32, device=runtime.device)
    if runtime.ctx.page_table.shape != (1, 17152) or not torch.equal(
        runtime.ctx.page_table[0], expected_pages
    ):
        raise RuntimeError("logical page mapping is not identity")
    if runtime.config.page_size != 1:
        raise RuntimeError("resolved KV page size differs from 1")
    if runtime.ctx.linear_state_pool.num_slots != 1:
        raise RuntimeError("resolved linear/recurrent state slots differ from one")
    return device


def _logit_record(logits: torch.Tensor) -> dict:
    value = logits.detach().float()
    return {
        "shape": list(value.shape),
        "float32_sha256": tensor_sha256(value),
        "argmax": int(value.argmax(dim=-1).item()),
        "nan_count": int(torch.isnan(value).sum().item()),
        "inf_count": int(torch.isinf(value).sum().item()),
        "full_logits": value.cpu().tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--session", choices=("A", "B"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    validate_v2_output_path(args.out)
    expected_name = f"reference-v2-session-{args.session.lower()}.json"
    if args.out.name != expected_name:
        raise ValueError(f"reference session {args.session} must write {expected_name}")

    from inferswarm_phase0.manifest import load_manifest
    from transformers import AutoTokenizer

    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    manifest = load_manifest(args.manifest, canonical=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    runtime = MatchedLocalRuntime(args.model, capacity=17152, cache_slots=3774)
    device = _validate_preflight(args, runtime, manifest_sha)
    prompt_ids_by_class = {
        class_id: _prompt_ids(tokenizer, manifest.by_class()[class_id])
        for class_id in WORKLOAD_ORDER
    }

    rows = []
    reset_count = 0
    for class_id in WORKLOAD_ORDER:
        runtime.reset_session()
        runtime.cache.reset_stats()
        reset_count += 1
        prompt_ids = prompt_ids_by_class[class_id]
        generated: list[int] = []
        selected_logits: dict[str, dict] = {}
        seam_checkpoints = []
        for start in range(0, len(prompt_ids), 64):
            ids = prompt_ids[start : start + 64]
            token, logits, seam = runtime.prefill_reference(
                ids, start, capture_seam=class_id in {"W2", "W4"}
            )
            if seam is not None:
                seam_checkpoints.append(
                    {"start": start, "token_count": len(ids), **seam}
                )
            if start + len(ids) == len(prompt_ids):
                generated.append(token)
                selected_logits["0"] = _logit_record(logits)
        while len(generated) < 32:
            step = len(generated)
            token, logits = runtime.decode(generated[-1], len(prompt_ids) + step - 1)
            generated.append(token)
            if step in SELECTED_STEPS:
                selected_logits[str(step)] = _logit_record(logits)
        rows.append(
            {
                "class_id": class_id,
                "prompt_token_ids": prompt_ids,
                "prompt_token_count": len(prompt_ids),
                "generated_token_ids": generated,
                "generated_token_count": len(generated),
                "selected_logit_steps": selected_logits,
                "layer_18_seam_checkpoints": seam_checkpoints,
                "nan_count": sum(x["nan_count"] for x in selected_logits.values()),
                "inf_count": sum(x["inf_count"] for x in selected_logits.values()),
                "state_reset": {
                    "direct_kv_zero": True,
                    "direct_linear_recurrent_zero": True,
                    "cross_workload_prefix_reuse": False,
                },
                "ordinary_offload_expert_movement": runtime.cache.decode_miss_stats(),
            }
        )

    payload = {
        "schema": "inferswarm.r2.reference-v2/1",
        "evidence_label": "CANONICAL_REFERENCE_CANDIDATE",
        "session": args.session,
        "methodology": methodology_record(),
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "producer": {
            "freetoken_commit": _git_head(),
            "gpu_uuid": device["uuid"],
            "gpu": device,
        },
        "runtime_configuration": dict(REFERENCE_RUNTIME_CONFIGURATION),
        "runtime_configuration_sha256": hashlib.sha256(
            json.dumps(
                REFERENCE_RUNTIME_CONFIGURATION, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "workload_manifest_sha256": manifest_sha,
        "workload_order": WORKLOAD_ORDER,
        "selected_steps": SELECTED_STEPS,
        "generation_settings": GENERATION_SETTINGS,
        "graph_counters": {
            "captures": 1,
            "replays": runtime.graph.replays,
            "recaptures": 0,
        },
        "state_reset_count": reset_count,
        "workloads": rows,
    }
    validate_reference_artifact(payload)
    write_json_with_sha(args.out, payload)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                "session": args.session,
                "graph_counters": payload["graph_counters"],
                "nan_count": sum(row["nan_count"] for row in rows),
                "inf_count": sum(row["inf_count"] for row in rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
