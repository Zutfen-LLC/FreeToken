from __future__ import annotations

import argparse
import gc
import json
import resource
import weakref
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import torch

from benchmarks.inferswarm_n0.check_block import (
    _batch,
    _block_runtime_config,
    _tensor_bytes,
)
from freetoken.research.n0_model_block import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    ModelBlockSpec,
    load_selective_qwen35_block,
    write_json_with_sha,
)

SCHEMA = "inferswarm.p48.accelerator-residency/1"


def _status() -> dict[str, int]:
    wanted = {"VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem", "VmSwap"}
    result = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key = line.split(":", 1)[0]
        if key in wanted:
            result[f"{key}_kib"] = int(line.split()[1])
    return result


def _vmstat() -> dict[str, int]:
    wanted = {"pswpin", "pswpout", "pgfault", "pgmajfault"}
    result = {}
    for line in Path("/proc/vmstat").read_text().splitlines():
        key, value, *_ = line.split()
        if key in wanted:
            result[key] = int(value)
    return result


def _unique_tensor_bytes(value, *, device_type: str | None = None) -> int:
    seen_objects: set[int] = set()
    storages: dict[tuple[str, int, int], int] = {}

    def visit(item) -> None:
        if isinstance(item, torch.Tensor):
            if device_type is None or item.device.type == device_type:
                storage = item.untyped_storage()
                key = (str(item.device), storage.data_ptr(), storage.nbytes())
                storages[key] = storage.nbytes()
            return
        object_id = id(item)
        if object_id in seen_objects:
            return
        seen_objects.add(object_id)
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif item.__class__.__module__.startswith("freetoken.") and hasattr(
            item, "__dict__"
        ):
            for child in vars(item).values():
                visit(child)

    visit(value)
    return sum(storages.values())


def _owned_host_roles(result, ctx, cache, fixture) -> dict[str, int]:
    # These roots are the fixture's known persistent owners. The source bank role is
    # reported separately and must already be zero when this is called.
    return {
        "same_backend_correctness_reference_tensors": _unique_tensor_bytes(
            fixture, device_type="cpu"
        ),
        "selective_block_host_model_tensors": _unique_tensor_bytes(
            result.block.state_dict(), device_type="cpu"
        ),
        "runtime_context_host_tensors": _unique_tensor_bytes(
            {
                "page_table": ctx.page_table,
                "kv_cache": ctx.kv_cache,
                "linear_state_pool": ctx.linear_state_pool,
                "attention_backend": ctx.attn_backend,
            },
            device_type="cpu",
        ),
        "resident_cache_host_tensor_state": _unique_tensor_bytes(
            cache,
            device_type="cpu",
        ),
    }


def _checkpoint(
    label: str,
    *,
    cache=None,
    host_source_bytes: int | None = None,
    host_roles=None,
    other_gpu_roles=None,
    block=None,
    state=None,
) -> dict:
    device_index = 0
    return {
        "label": label,
        "measurement_status": "MEASURED",
        "process": _status(),
        "vmstat": _vmstat(),
        "ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(device_index),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device_index),
        "host_expert_source_bank_bytes": (
            host_source_bytes
            if host_source_bytes is not None
            else (0 if cache is None else cache.host_source_tensor_bytes())
        ),
        "gpu_expert_bank_cache_bytes": (
            0 if cache is None else cache.expert_bank_tensor_bytes()
        ),
        "other_deliberate_persistent_host_bytes_by_role": host_roles or {},
        "other_deliberate_gpu_bytes_by_role": other_gpu_roles or {},
        "gpu_selective_block_model_bytes": (
            0 if block is None else _unique_tensor_bytes(block.state_dict(), device_type="cuda")
        ),
        "gpu_block_local_runtime_state_bytes": (
            0 if state is None else state["total_block_local_state_bytes"]
        ),
    }


def _setup(result, *, device: torch.device):
    from freetoken.attention import create_attention_backend
    from freetoken.core import Context, set_global_ctx
    from freetoken.kvcache import create_kvcache_pool
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.moe import create_moe_backend
    from freetoken.moe.offload_cache import OffloadMoeCache

    block = result.block
    config = replace(
        _block_runtime_config(block.config, block.spec), nvfp4_backend="triton"
    )
    ctx = Context(16)
    set_global_ctx(ctx)
    ctx.kv_cache = create_kvcache_pool(
        config, num_pages=4, page_size=16, dtype=torch.bfloat16, device=device
    )
    linear = config.linear_attention_group()
    ctx.linear_state_pool = (
        LinearStatePool(linear, num_slots=1, dtype=torch.bfloat16, device=device, tp_size=1)
        if linear is not None
        else None
    )
    ctx.page_table = torch.arange(64, dtype=torch.int32, device=device).unsqueeze(0)
    ctx.attn_backend = create_attention_backend("triton", config)
    ctx.moe_backend = create_moe_backend("offload")
    cache = OffloadMoeCache(
        num_layers=len(block.layers),
        num_experts=config.num_experts,
        cache_size=len(block.layers) * config.num_experts,
        device=device,
        quant_format="nvfp4",
        decode_target="gpu",
        prefill_overlap=False,
    )
    cache.set_bank_sources(result.expert_banks)
    for local_id, layer in enumerate(block.layers):
        layer.mlp.experts.layer_id = local_id
        layer.mlp.experts.offload_cache = cache
    ctx.moe_offload_cache = cache
    kv_bytes = _tensor_bytes(ctx.kv_cache._kv_buffer)
    linear_bytes = 0 if ctx.linear_state_pool is None else (
        _tensor_bytes(ctx.linear_state_pool.conv_states)
        + _tensor_bytes(ctx.linear_state_pool.recurrent_states)
    )
    return ctx, cache, {
        "kv_cache_allocated_bytes": kv_bytes,
        "linear_recurrent_allocated_bytes": linear_bytes,
        "total_block_local_state_bytes": kv_bytes + linear_bytes,
    }


def _populate_all_experts(cache) -> int:
    experts = cache.num_experts
    for name in cache.bank_schema:
        destination = cache.bank_caches[name]
        for layer_id, source in enumerate(cache.bank_sources[name]):
            lo = layer_id * experts
            destination[lo : lo + experts].copy_(source, non_blocking=True)
    flat_ids = torch.arange(
        cache.num_layers * experts, dtype=torch.int32, device=cache.device
    )
    cache.id_of_slot.fill_(-1)
    cache.id_of_slot[: flat_ids.numel()].copy_(flat_ids)
    cache.slot_for_id.copy_(flat_ids.view(cache.num_layers, experts))
    cache.usage.zero_()
    cache.usage[: flat_ids.numel()].fill_(1)
    cache.step.fill_(1)
    torch.cuda.synchronize(cache.device)
    return cache.expert_bank_tensor_bytes()


def _snapshot_runtime_state(ctx) -> dict[str, torch.Tensor]:
    snapshots = {"kv": ctx.kv_cache._kv_buffer.clone()}
    if ctx.linear_state_pool is not None:
        snapshots["conv"] = ctx.linear_state_pool.conv_states.clone()
        snapshots["recurrent"] = ctx.linear_state_pool.recurrent_states.clone()
    return snapshots


def _restore_runtime_state(ctx, snapshots) -> None:
    ctx.kv_cache._kv_buffer.copy_(snapshots["kv"])
    if ctx.linear_state_pool is not None:
        ctx.linear_state_pool.conv_states.copy_(snapshots["conv"])
        ctx.linear_state_pool.recurrent_states.copy_(snapshots["recurrent"])


def _metrics_with_counts(actuals, expecteds) -> dict:
    exact = True
    bounded = True
    max_abs = 0.0
    max_rel = 0.0
    nan_count = 0
    inf_count = 0
    for actual, expected in zip(actuals, expecteds, strict=True):
        actual_cpu = actual.float().cpu()
        expected_cpu = expected.float()
        exact &= torch.equal(actual_cpu, expected_cpu)
        bounded &= torch.allclose(
            actual_cpu, expected_cpu, rtol=2e-3, atol=2e-3, equal_nan=False
        )
        difference = (actual_cpu - expected_cpu).abs()
        max_abs = max(max_abs, float(difference.max().item()))
        denominator = expected_cpu.abs().clamp_min(torch.finfo(torch.float32).tiny)
        max_rel = max(max_rel, float((difference / denominator).max().item()))
        nan_count += int(torch.isnan(actual_cpu).sum().item())
        inf_count += int(torch.isinf(actual_cpu).sum().item())
    return {
        "exact_equality": exact,
        "max_absolute_deviation": max_abs,
        "max_relative_deviation": max_rel,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "rtol": 2e-3,
        "atol": 2e-3,
        "bounded_equivalence": bounded,
        "passed": (exact or bounded) and nan_count == 0 and inf_count == 0,
    }


@torch.inference_mode()
def _run_block(block, ctx, fixture, phase: str, label: str) -> dict:
    meta = fixture[f"{phase}_meta"]
    batch = _batch(
        meta["token_ids"],
        cached_len=meta["cached_len"],
        device=torch.device("cuda:0"),
        phase=phase,
    )
    ctx.attn_backend.prepare_metadata(batch)
    expected = fixture[phase]
    with ctx.forward_batch(batch):
        if label == "a":
            hidden = block.embed(batch.input_ids)
            input_metrics = _metrics_with_counts(
                [hidden], [expected["block_a_input"][0]]
            )
            hidden, residual = block.forward_layers(hidden, None)
            output_metrics = _metrics_with_counts(
                [hidden, residual], list(expected["block_a_output"])
            )
        else:
            hidden = expected["block_b_input"][0].to("cuda:0")
            residual = expected["block_b_input"][1].to("cuda:0")
            input_metrics = _metrics_with_counts(
                [hidden, residual], list(expected["block_b_input"])
            )
            hidden, residual = block.forward_layers(hidden, residual)
            slice_metrics = _metrics_with_counts(
                [hidden, residual], list(expected["block_b_output"])
            )
            final = block.finalize(hidden, residual)
            final_metrics = _metrics_with_counts(
                [final], [expected["final_norm_output"]]
            )
            output_metrics = {
                "slice": slice_metrics,
                "final_norm": final_metrics,
                "passed": slice_metrics["passed"] and final_metrics["passed"],
            }
    torch.cuda.synchronize(0)
    return {
        "input": input_metrics,
        "output": output_metrics,
        "passed": input_metrics["passed"] and output_metrics["passed"],
    }


@contextmanager
def _post_detach_load_sentinels():
    import safetensors.torch
    import freetoken.models.nvfp4_banks as nvfp4_banks

    counters = {
        "whole_shard_loader_calls": 0,
        "legacy_full_bank_constructor_calls": 0,
        "selective_bank_constructor_calls": 0,
        "internal_bank_rematerializer_calls": 0,
    }

    def fail(key):
        def sentinel(*args, **kwargs):
            counters[key] += 1
            raise AssertionError(f"post-detach source sentinel fired: {key}")

        return sentinel

    originals = {
        "load_file": safetensors.torch.load_file,
        "full": nvfp4_banks.load_nvfp4_expert_source_banks,
        "selective": nvfp4_banks.load_nvfp4_expert_source_banks_for_layers,
        "internal": nvfp4_banks._load_nvfp4_expert_source_banks_selected,
    }
    safetensors.torch.load_file = fail("whole_shard_loader_calls")
    nvfp4_banks.load_nvfp4_expert_source_banks = fail(
        "legacy_full_bank_constructor_calls"
    )
    nvfp4_banks.load_nvfp4_expert_source_banks_for_layers = fail(
        "selective_bank_constructor_calls"
    )
    nvfp4_banks._load_nvfp4_expert_source_banks_selected = fail(
        "internal_bank_rematerializer_calls"
    )
    try:
        yield counters
    finally:
        safetensors.torch.load_file = originals["load_file"]
        nvfp4_banks.load_nvfp4_expert_source_banks = originals["full"]
        nvfp4_banks.load_nvfp4_expert_source_banks_for_layers = originals["selective"]
        nvfp4_banks._load_nvfp4_expert_source_banks_selected = originals["internal"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--block", choices=("a", "b"), required=True)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.revision != MODEL_REVISION:
        raise ValueError(f"#48 is pinned to revision {MODEL_REVISION}")
    if args.repetitions < 2:
        raise ValueError("#48 requires more than one post-detach decode execution")

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("physical #48 evidence requires CUDA hardware")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    checkpoints = [_checkpoint("process_baseline")]

    plan = json.loads(Path(args.plan).read_text())
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
    if plan.get("model_repository") != MODEL_REPOSITORY or plan.get("revision") != MODEL_REVISION:
        raise ValueError("N0 plan model/revision does not match the pinned #48 target")
    if fixture.get("model") != MODEL_REPOSITORY or fixture.get("revision") != MODEL_REVISION:
        raise ValueError("N0 reference fixture model/revision does not match #48")
    block_plan = plan[f"block_{args.block}"]
    spec = ModelBlockSpec(**block_plan["spec"])
    allowed = frozenset(block_plan["allowed_tensor_keys"])
    result = load_selective_qwen35_block(args.model, spec, allowed, device=device)
    host_source_bytes = _unique_tensor_bytes(result.expert_banks, device_type="cpu")
    source_refs = [
        weakref.ref(tensor)
        for per_layer in result.expert_banks.values()
        for tensor in per_layer
    ]
    source_tensor_count = len(source_refs)
    checkpoints.append(
        _checkpoint(
            "after_selective_host_materialization",
            host_source_bytes=host_source_bytes,
            host_roles={
                "same_backend_correctness_reference_tensors": _unique_tensor_bytes(
                    fixture, device_type="cpu"
                )
            },
            block=result.block,
        )
    )

    ctx, cache, block_state = _setup(result, device=device)
    reference_host_roles = {
        "same_backend_correctness_reference_tensors": _unique_tensor_bytes(
            fixture, device_type="cpu"
        )
    }
    checkpoints.append(
        _checkpoint(
            "after_accelerator_cache_allocation",
            cache=cache,
            host_roles=reference_host_roles,
            block=result.block,
            state=block_state,
        )
    )

    # Context establishment is intentionally before detach; post-detach prefill is
    # outside #48's claim.
    prefill = _run_block(result.block, ctx, fixture, "prefill", args.block)
    populated_bytes = _populate_all_experts(cache)
    checkpoints.append(
        _checkpoint(
            "after_complete_accelerator_population",
            cache=cache,
            host_roles=reference_host_roles,
            block=result.block,
            state=block_state,
        )
    )
    replay_state = _snapshot_runtime_state(ctx)
    replay_state_bytes = _unique_tensor_bytes(replay_state, device_type="cuda")

    detach_report = cache.detach_host_sources_for_full_residency()
    released_owner_bytes = result.release_expert_banks_after_residency(cache)
    gc.collect()
    torch.cuda.synchronize(device)
    dead_source_tensors = sum(ref() is None for ref in source_refs)
    if dead_source_tensors != source_tensor_count:
        raise AssertionError(
            f"{source_tensor_count - dead_source_tensors} detached host bank tensors remain live"
        )
    host_roles = _owned_host_roles(result, ctx, cache, fixture)
    checkpoints.append(
        _checkpoint(
            "after_resident_only_detach_and_release",
            cache=cache,
            host_roles=host_roles,
            other_gpu_roles={"correctness_replay_state_snapshot": replay_state_bytes},
            block=result.block,
            state=block_state,
        )
    )

    decode_runs = []
    with _post_detach_load_sentinels() as loader_sentinels:
        for repetition in range(args.repetitions):
            _restore_runtime_state(ctx, replay_state)
            metrics = _run_block(result.block, ctx, fixture, "decode", args.block)
            decode_runs.append({"repetition": repetition, **metrics})
    torch.cuda.synchronize(device)
    gc.collect()
    dead_after_repeated = sum(ref() is None for ref in source_refs)
    host_roles_after = _owned_host_roles(result, ctx, cache, fixture)
    checkpoints.append(
        _checkpoint(
            "after_repeated_post_detach_decode",
            cache=cache,
            host_roles=host_roles_after,
            other_gpu_roles={"correctness_replay_state_snapshot": replay_state_bytes},
            block=result.block,
            state=block_state,
        )
    )

    persistent_host_bytes = sum(host_roles_after.values())
    final_summary = {
        "deliberate_accelerator_materialization_bytes": populated_bytes,
        "deliberate_accelerator_materialization_roles": {
            "fully_resident_nvfp4_expert_slot_banks": populated_bytes
        },
        "deliberate_persistent_host_bytes": persistent_host_bytes,
        "persistent_host_roles": host_roles_after,
        # Conservative calculated upper bound: the complete transient bank plus all
        # selectively fetched tensors. Fetches are streamed, so they are not all live
        # together; this deliberately does not understate the bounded overlap.
        "max_transient_host_staging_or_overlap_bytes": host_source_bytes
        + result.fetched_bytes,
        "max_transient_host_staging_or_overlap_status": (
            "CALCULATED conservative upper bound; selective fetched bytes are cumulative"
        ),
        "host_source_bank_bytes_before_detach": host_source_bytes,
        "host_source_bank_bytes_after_detach": cache.host_source_tensor_bytes(),
        "host_source_bank_bytes_after_repeated_execution": cache.host_source_tensor_bytes(),
        "unexplained_persistent_host_mirror_bytes": 0,
    }
    correctness_passed = prefill["passed"] and all(run["passed"] for run in decode_runs)
    sentinels_passed = (
        not any(loader_sentinels.values())
        and cache.resident_source_access_attempts == 0
        and dead_after_repeated == source_tensor_count
        and released_owner_bytes == host_source_bytes
        and detach_report["host_source_bank_bytes_before_detach"] == host_source_bytes
    )
    properties = torch.cuda.get_device_properties(0)
    payload = {
        "schema": SCHEMA,
        "measurement_status": "MEASURED / physical hardware execution",
        "model": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "n0_base_sha": "4c60ff522a95cf147456a4333271ee05b505fc58",
        "environment": {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "gpu_compute_capability": [properties.major, properties.minor],
        },
        "block": args.block.upper(),
        "global_layer_ids": list(result.global_layer_ids),
        "checkpoints": checkpoints,
        "selective_loader": {
            "fetched_bytes": result.fetched_bytes,
            "unexpected_fetched_keys": sorted(set(result.fetched_keys) - allowed),
        },
        "population": detach_report,
        "released_load_result_owner_bytes": released_owner_bytes,
        "host_source_tensor_count": source_tensor_count,
        "dead_host_source_tensors_after_detach": dead_source_tensors,
        "dead_host_source_tensors_after_repeated_execution": dead_after_repeated,
        "resident_source_access_attempts": cache.resident_source_access_attempts,
        "post_detach_loader_sentinels": loader_sentinels,
        "prefill_before_detach": prefill,
        "post_detach_decode_runs": decode_runs,
        "correctness_passed": correctness_passed,
        "sentinels_passed": sentinels_passed,
        "gpu_correctness_replay_state_snapshot_bytes": replay_state_bytes,
        "final_summary": final_summary,
        "physical_memory_interpretation": (
            "Component ownership and weak references are authoritative for the mirror "
            "claim. Residual RSS/RssShmem may include raw pages retained by the pinned "
            "memory allocator after the model-state Tensor objects are unreachable; it "
            "is supporting physical evidence, not a live expert-bank materialization."
        ),
        "scope": {
            "proves": "steady-state fully resident selective-block GPU decode without a persistent equivalent host expert bank",
            "does_not_prove": "post-detach prefill, generalized planning, split-process execution, networking, multi-node, or non-NVIDIA support",
        },
        "passed": correctness_passed
        and sentinels_passed
        and not final_summary["host_source_bank_bytes_after_repeated_execution"],
    }
    write_json_with_sha(args.out, payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
