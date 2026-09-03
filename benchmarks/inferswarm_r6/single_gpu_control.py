"""Bounded-host-memory single-GPU FreeToken control for R6.

This post-failure diagnostic loads the complete BF16 text tower through
per-tensor host mappings, keeps every immutable model tensor on one CUDA
device, and runs the canonical replay-prefill comparator.  It does not alter
or compose the historical R6 verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

EXPECTED_TEXT_BYTES = 23_814_700_640
EXPECTED_CHECKPOINT_SHA256 = (
    "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d"
)
MODEL_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
RUNTIME_CAPACITY_TOKENS = 64
CAPTURE_STEPS = (0, 1, 7)
FROZEN_THRESHOLD = 0.25


def _command(args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(args, text=True, capture_output=True, check=False)
    return {
        "argv": args,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity(repo: Path) -> dict[str, Any]:
    sha = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    )
    return {"sha": sha, "dirty": bool(status), "status": status.splitlines()}


def _machine_census(gpu: str) -> dict[str, Any]:
    query = (
        "name,uuid,pci.bus_id,memory.total,memory.free,memory.used,"
        "driver_version,pstate,pcie.link.gen.current,pcie.link.width.current"
    )
    return {
        "captured_at_unix_ns": time.time_ns(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version,
        "os_release": Path("/etc/os-release").read_text(),
        "proc_meminfo": Path("/proc/meminfo").read_text(),
        "proc_swaps": Path("/proc/swaps").read_text(),
        "lscpu": _command(["lscpu"]),
        "gpu": _command(
            [
                "nvidia-smi",
                "-i",
                gpu,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ]
        ),
        "compute_apps_before_process": _command(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        ),
        "nvidia_smi_full": _command(["nvidia-smi", "-i", gpu, "-q"]),
    }


def _reference_diffs(reference: dict, rows: dict[int, Any]) -> dict[str, float]:
    result = {}
    for step in CAPTURE_STEPS:
        top = reference["step_top32_logits"][str(step)]
        row = rows[step]
        values = row[top["top_indices"]]
        ref_values = values.new_tensor(top["top_values"])
        result[str(step)] = float((values - ref_values).abs().max().item())
    return result


def _distributed_diffs(binary_path: Path, rows: dict[int, Any]) -> dict[str, float]:
    import torch

    vocab = rows[CAPTURE_STEPS[0]].numel()
    raw = torch.from_file(
        str(binary_path),
        shared=False,
        size=len(CAPTURE_STEPS) * vocab,
        dtype=torch.float32,
    ).reshape(len(CAPTURE_STEPS), vocab)
    return {
        str(step): float((rows[step] - raw[index]).abs().max().item())
        for index, step in enumerate(CAPTURE_STEPS)
    }


def run_control(args) -> dict[str, Any]:
    # GPU selection must happen before torch is imported by the runtime.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    repo = Path(__file__).resolve().parents[2]
    model = Path(args.model).resolve()
    reference = json.loads(Path(args.reference).read_text())
    prompt = list(reference["prompt_token_ids"])
    maximum = len(reference["generated_token_ids"])
    if len(prompt) + maximum > RUNTIME_CAPACITY_TOKENS:
        raise RuntimeError("canonical replay exceeds frozen 64-token capacity")

    machine = _machine_census(args.gpu)
    producer = _git_identity(repo)
    report: dict[str, Any] = {
        "schema": "inferswarm.r6.single-gpu-control/1",
        "status": "RUNNING",
        "historical_r6_result_unchanged": True,
        "methodology": "docs/inferswarm_r6/SINGLE_GPU_CONTROL_METHODOLOGY.md",
        "model_revision": MODEL_REVISION,
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "expected_required_text_model_bytes": EXPECTED_TEXT_BYTES,
        "runtime_capacity_tokens": RUNTIME_CAPACITY_TOKENS,
        "attention_backend": "triton",
        "dtype": "bfloat16",
        "topology": {
            "role": "single",
            "layers": [0, 48],
            "semantic_boundaries": [],
            "cpu_weight_offload": False,
        },
        "machine_before_torch": machine,
        "producer": producer,
        "prompt_token_ids": prompt,
        "max_new_tokens": maximum,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    preflight = {
        "canonical_hostname": machine["hostname"] == "inferswarm04",
        "gpu_query_succeeded": machine["gpu"]["returncode"] == 0,
        "no_compute_applications": (
            machine["compute_apps_before_process"]["returncode"] == 0
            and not machine["compute_apps_before_process"]["stdout"]
        ),
        "clean_exact_source": not producer["dirty"],
    }
    report["preflight"] = preflight
    if not all(preflight.values()):
        report["status"] = "SINGLE_GPU_CONTROL_PREFLIGHT_BLOCKED"
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report

    import torch
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.rotary import set_rope_device
    from freetoken.models.loader import drop_page_cache
    from freetoken.research.r6_dense_census import (
        DenseBlockSpec,
        checkpoint_census,
        freeze_dense_block_plan,
    )

    from benchmarks.inferswarm_r6.stage_runtime import GemmaDenseStage

    report["software"] = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": __import__("triton").__version__,
    }
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    set_rope_device(torch.device("cuda:0"))

    census = checkpoint_census(model, text_prefix="model.language_model")
    if census["required_text_model_bytes"] != EXPECTED_TEXT_BYTES:
        raise RuntimeError(
            f"text checkpoint bytes changed: {census['required_text_model_bytes']}"
        )
    if len(census["per_layer"]) != 48:
        raise RuntimeError(f"expected 48 layers, got {len(census['per_layer'])}")
    checkpoint = model / "model.safetensors"
    actual_checkpoint_sha = _sha256(checkpoint)
    drop_page_cache(str(checkpoint))
    if actual_checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint sha256 changed: {actual_checkpoint_sha}")
    report["checkpoint"] = {
        "path": str(checkpoint),
        "sha256": actual_checkpoint_sha,
        "index_sha256": census["checkpoint_index_sha256"],
        "required_text_model_bytes": census["required_text_model_bytes"],
        "tensor_count": census["tensor_count"],
    }

    shared = {
        "id": "tied-embedding-lm-head",
        "kind": "tied-weight-shared-state",
        "tensor_keys": ["model.language_model.embed_tokens.weight"],
        "bytes": census["bytes_by_owner_category"]["embedding/input"],
        "materialization_policy": "single-cuda-tensor-used-for-input-and-output",
    }
    plan = freeze_dense_block_plan(
        census,
        [DenseBlockSpec(0, 48, True, True)],
        declared_shared_state=shared,
    )
    block = plan["blocks"][0]
    report["plan"] = {
        "spec": block["spec"],
        "owned_checkpoint_bytes": block["owned_checkpoint_bytes"],
        "allowed_tensor_key_count": len(block["allowed_tensor_keys"]),
        "declared_shared_state": shared,
    }

    try:
        runtime = GemmaDenseStage(
            role="single",
            model_path=str(model),
            adapter_data={
                **block,
                "declared_shared_state": shared,
                "runtime_capacity_tokens": RUNTIME_CAPACITY_TOKENS,
            },
        )
        generated: list[int] = []
        captured = {}
        nan_inf_count = 0
        for step in range(maximum):
            runtime.reset_session_state()
            replay = prompt + generated
            token, logits = runtime.prefill(replay, None, 0)
            row = logits[-1].detach().float().cpu()
            nan_inf_count += int(
                torch.isnan(row).sum().item() + torch.isinf(row).sum().item()
            )
            if step in CAPTURE_STEPS:
                captured[step] = row.clone()
            generated.append(int(token))
            del logits, row

        binary_out = out.with_name("single-gpu-logits-0-1-7.f32.bin")
        with binary_out.open("wb") as stream:
            for step in CAPTURE_STEPS:
                stream.write(captured[step].contiguous().numpy().tobytes())

        ref_diffs = _reference_diffs(reference, captured)
        dist_diffs = _distributed_diffs(Path(args.distributed_logits), captured)
        loader = runtime.report("P5_single_gpu_control_complete")
        invariants = {
            "all_required_checkpoint_bytes_processed": (
                loader["checkpoint_bytes_processed"] == EXPECTED_TEXT_BYTES
            ),
            "complete_layer_coverage": loader["global_layer_ids"] == list(range(48)),
            "unexpected_keys_zero": loader["unexpected_checkpoint_keys"] == [],
            "whole_checkpoint_fallback_calls_zero": (
                loader["whole_shard_sentinel_calls"] == 0
            ),
            "persistent_host_model_bytes_zero": (
                loader["persistent_host_model_bytes"] == 0
            ),
            "host_staging_current_bytes_zero": (
                loader["host_staging_current_bytes"] == 0
            ),
            "single_tied_embedding_materialization": (
                loader["tied_embedding_materializations"] == 1
                and loader["single_tied_embedding_storage"]
            ),
            "selected_bytes_exact": (
                loader["checkpoint_bytes_selected"] == EXPECTED_TEXT_BYTES
            ),
            "bounded_mappings_all_closed": (
                loader["safetensors_mapping_open_count"]
                == loader["safetensors_mapping_close_count"]
            ),
            "no_process_swap_after_control": (
                loader["process_current"]["VmSwap_kib"] == 0
            ),
            "no_cpu_owned_decoder_layers": (
                loader["cpu_owned_decoder_layers"] == 0
                and not loader["cpu_weight_offload"]
            ),
            "model_remained_resident": loader["resident_only"],
        }
        if not all(invariants.values()):
            raise RuntimeError(f"single-GPU invariant failure: {invariants}")
        aggregate_ref = max(ref_diffs.values())
        if aggregate_ref >= FROZEN_THRESHOLD:
            interpretation = "OUTCOME_A_FREE_TOKEN_NUMERICAL_DIFFERENCE"
        else:
            interpretation = "OUTCOME_B_DISTRIBUTION_DRIFT"
        report.update(
            {
                "status": "SINGLE_GPU_CONTROL_COMPLETE",
                "loader": loader,
                "invariants": invariants,
                "generated_token_ids": generated,
                "reference_generated_token_ids": reference["generated_token_ids"],
                "exact_generated_token_match": (
                    generated == reference["generated_token_ids"]
                ),
                "nan_inf_count": nan_inf_count,
                "comparisons": {
                    "transformers_vs_single_freetoken": {
                        "domain": "retained Transformers reference top-32",
                        "per_step_max_absdiff": ref_diffs,
                        "aggregate_max_absdiff": aggregate_ref,
                        "frozen_threshold": FROZEN_THRESHOLD,
                        "strict_pass": aggregate_ref < FROZEN_THRESHOLD,
                    },
                    "transformers_vs_distributed_freetoken": json.loads(
                        Path(args.distributed_comparator).read_text()
                    )["per_step_max_absdiff"],
                    "single_freetoken_vs_distributed_freetoken": {
                        "domain": "full vocabulary",
                        "per_step_max_absdiff": dist_diffs,
                        "aggregate_max_absdiff": max(dist_diffs.values()),
                    },
                },
                "captured_logits": {
                    "path": str(binary_out),
                    "sha256": _sha256(binary_out),
                    "dtype": "float32 little-endian",
                    "steps": list(CAPTURE_STEPS),
                    "vocab": captured[CAPTURE_STEPS[0]].numel(),
                },
                "interpretation": interpretation,
            }
        )
    except torch.OutOfMemoryError as exc:
        report.update(
            {
                "status": "SINGLE_GPU_REFERENCE_CAPACITY_BLOCKED",
                "error": str(exc),
                "cuda_allocated_bytes": torch.cuda.memory_allocated("cuda:0"),
                "cuda_peak_bytes": torch.cuda.max_memory_allocated("cuda:0"),
                "cuda_mem_get_info": list(torch.cuda.mem_get_info("cuda:0")),
            }
        )
    finally:
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", default="0", help="GPU index or UUID")
    parser.add_argument(
        "--reference", default="docs/inferswarm_r6/reference-generation.json"
    )
    parser.add_argument(
        "--distributed-comparator",
        default="docs/inferswarm_r6/lifecycle/secondary-comparator.json",
    )
    parser.add_argument(
        "--distributed-logits",
        default="docs/inferswarm_r6/lifecycle/distributed-logits-0-1-7.f32.bin",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = run_control(args)
    print(json.dumps({"status": report["status"]}, sort_keys=True))
    return 0 if report["status"] == "SINGLE_GPU_CONTROL_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
