from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from benchmarks.inferswarm_n0.reference_fixtures import _batch
from freetoken.research.n0_model_block import (
    ModelBlockSpec,
    load_selective_qwen35_block,
    write_json_with_sha,
)


def _block_runtime_config(config, spec):
    groups = []
    owned = set(range(spec.start_layer, spec.end_layer))
    for group in config.attention_groups:
        ids = tuple(layer for layer in group.layer_ids if layer in owned)
        if ids:
            groups.append(replace(group, layer_ids=ids))
    return replace(config, attention_groups=tuple(groups))


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    return 0


def _setup(result, *, device):
    from freetoken.attention import create_attention_backend
    from freetoken.core import Context, set_global_ctx
    from freetoken.kvcache import create_kvcache_pool
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.moe import create_moe_backend
    from freetoken.moe.offload_cache import OffloadMoeCache

    block = result.block
    config = _block_runtime_config(block.config, block.spec)
    ctx = Context(16)
    set_global_ctx(ctx)
    ctx.kv_cache = create_kvcache_pool(
        config, num_pages=4, page_size=16, dtype=torch.bfloat16, device=device
    )
    linear = config.linear_attention_group()
    ctx.linear_state_pool = (
        LinearStatePool(linear, num_slots=1, dtype=torch.bfloat16, device=device, tp_size=1)
        if linear is not None else None
    )
    ctx.page_table = torch.arange(64, dtype=torch.int32, device=device).unsqueeze(0)
    ctx.attn_backend = create_attention_backend("triton", config)
    ctx.moe_backend = create_moe_backend("offload")
    cache = OffloadMoeCache(
        num_layers=len(block.layers), num_experts=config.num_experts,
        cache_size=config.num_experts, device=device, quant_format="nvfp4",
        decode_target="gpu", prefill_overlap=False,
    )
    cache.set_bank_sources(result.expert_banks)
    for local_id, layer in enumerate(block.layers):
        experts = layer.mlp.experts
        experts.layer_id = local_id
        experts.offload_cache = cache
    ctx.moe_offload_cache = cache
    kv_bytes = ctx.kv_cache._kv_buffer.numel() * ctx.kv_cache._kv_buffer.element_size()
    linear_bytes = 0 if ctx.linear_state_pool is None else (
        ctx.linear_state_pool.conv_states.numel() * ctx.linear_state_pool.conv_states.element_size()
        + ctx.linear_state_pool.recurrent_states.numel()
        * ctx.linear_state_pool.recurrent_states.element_size()
    )
    return ctx, cache, {"kv_cache_allocated_bytes": kv_bytes,
                        "linear_recurrent_allocated_bytes": linear_bytes,
                        "total_block_local_state_bytes": kv_bytes + linear_bytes,
                        "kv_global_layer_ids": list(config.kv_cache_group_specs()[0].layer_ids),
                        "linear_global_layer_ids": list(linear.layer_ids) if linear else []}


def _metrics(actuals, expecteds):
    exact = all(torch.equal(a.cpu(), e) for a, e in zip(actuals, expecteds, strict=True))
    max_abs = 0.0; max_rel = 0.0; has_nan = False; has_inf = False
    for actual, expected in zip(actuals, expecteds, strict=True):
        a = actual.float().cpu(); e = expected.float()
        diff = (a - e).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        denom = e.abs().clamp_min(torch.finfo(torch.float32).tiny)
        max_rel = max(max_rel, float((diff / denom).max().item()))
        has_nan |= bool(torch.isnan(a).any() or torch.isnan(e).any())
        has_inf |= bool(torch.isinf(a).any() or torch.isinf(e).any())
    bounded = all(torch.allclose(a.cpu(), e, rtol=2e-3, atol=2e-3, equal_nan=False)
                  for a, e in zip(actuals, expecteds, strict=True))
    return {"exact_equality": exact, "max_absolute_deviation": max_abs,
            "max_relative_deviation": max_rel, "nan": has_nan, "inf": has_inf,
            "rtol": 2e-3, "atol": 2e-3, "bounded_equivalence": bounded,
            "passed": exact or bounded}


@torch.inference_mode()
def _run(block, ctx, fixture, phase, label):
    meta = fixture[f"{phase}_meta"]
    batch = _batch(meta["token_ids"], cached_len=meta["cached_len"],
                   device=torch.device("cuda:0"), phase=phase)
    ctx.attn_backend.prepare_metadata(batch)
    expected = fixture[phase]
    with ctx.forward_batch(batch):
        if label == "a":
            hidden = block.embed(batch.input_ids)
            input_metrics = _metrics([hidden], [expected["block_a_input"][0]])
            residual = None
            hidden, residual = block.forward_layers(hidden, residual)
            output_metrics = _metrics(
                [hidden, residual], list(expected["block_a_output"])
            )
        else:
            hidden = expected["block_b_input"][0].to("cuda:0")
            residual = expected["block_b_input"][1].to("cuda:0")
            input_metrics = _metrics(
                [hidden, residual], list(expected["block_b_input"])
            )
            hidden, residual = block.forward_layers(hidden, residual)
            slice_metrics = _metrics(
                [hidden, residual], list(expected["block_b_output"])
            )
            final = block.finalize(hidden, residual)
            final_metrics = _metrics([final], [expected["final_norm_output"]])
            output_metrics = {"slice": slice_metrics, "final_norm": final_metrics,
                              "passed": slice_metrics["passed"] and final_metrics["passed"]}
    torch.cuda.synchronize(0)
    return {"input": input_metrics, "output": output_metrics,
            "passed": input_metrics["passed"] and output_metrics["passed"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--plan", required=True)
    parser.add_argument("--fixture", required=True); parser.add_argument("--block", choices=("a", "b"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text()); bp = plan[f"block_{args.block}"]
    spec = ModelBlockSpec(**bp["spec"]); allowed = frozenset(bp["allowed_tensor_keys"])
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
    result = load_selective_qwen35_block(args.model, spec, allowed, device=torch.device("cuda:0"))
    ctx, cache, state = _setup(result, device=torch.device("cuda:0"))
    prefill = _run(result.block, ctx, fixture, "prefill", args.block)
    decode = _run(result.block, ctx, fixture, "decode", args.block)
    unexpected = sorted(set(result.fetched_keys) - allowed)
    payload = {"schema": "inferswarm.n0.block-correctness/1", "block": args.block.upper(),
               "fixture_sha256": __import__("hashlib").sha256(Path(args.fixture).read_bytes()).hexdigest(),
               "global_layer_ids": list(result.global_layer_ids), "state_ownership": state,
               "prefill": prefill, "decode": decode, "unexpected_fetched_keys": unexpected,
               "passed": prefill["passed"] and decode["passed"] and not unexpected,
               "cuda_memory_after_bytes": torch.cuda.memory_allocated(0),
               "cuda_peak_bytes": torch.cuda.max_memory_allocated(0)}
    write_json_with_sha(args.out, payload)
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["passed"] else 2)


if __name__ == "__main__": main()
