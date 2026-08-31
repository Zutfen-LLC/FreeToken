from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch


def _status():
    out = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key = line.split(":", 1)[0]
        if key in {"VmPeak", "VmSize", "VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem"}:
            out[key + "_kib"] = int(line.split()[1])
    return out


def _vmstat():
    wanted = {"pswpin", "pswpout", "pgmajfault"}
    return {p[0]: int(p[1]) for line in Path("/proc/vmstat").read_text().splitlines()
            if (p := line.split())[0] in wanted}


def _setup_runtime(config, model, banks, *, device, page_size=16, num_pages=4):
    from freetoken.attention import create_attention_backend
    from freetoken.core import Context, set_global_ctx
    from freetoken.kvcache import create_kvcache_pool
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.moe import create_moe_backend
    from freetoken.moe.offload_cache import OffloadMoeCache, attach_offload_moe_cache

    ctx = Context(page_size)
    set_global_ctx(ctx)
    ctx.kv_cache = create_kvcache_pool(
        config, num_pages=num_pages, page_size=page_size, dtype=torch.bfloat16, device=device
    )
    linear = config.linear_attention_group()
    ctx.linear_state_pool = (
        LinearStatePool(linear, num_slots=1, dtype=torch.bfloat16, device=device, tp_size=1)
        if linear is not None else None
    )
    ctx.page_table = torch.arange(
        num_pages * page_size, dtype=torch.int32, device=device
    ).unsqueeze(0)
    ctx.attn_backend = create_attention_backend("triton", config)
    ctx.moe_backend = create_moe_backend("offload")
    cache = OffloadMoeCache(
        num_layers=config.num_moe_layers, num_experts=config.num_experts,
        cache_size=config.num_experts, device=device, quant_format=banks.quant_format,
        decode_target="gpu", prefill_overlap=False,
    )
    cache.set_bank_sources(banks.sources, layer_residency=banks.layer_residency)
    cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
    layers = attach_offload_moe_cache(model, cache)
    assert len(layers) == config.num_moe_layers
    ctx.moe_offload_cache = cache
    return ctx, cache


def _batch(token_ids, *, cached_len, device, phase):
    from freetoken.core import Batch, Req, SamplingParams

    req = Req(
        input_ids=torch.tensor(token_ids, dtype=torch.int32), table_idx=0,
        cached_len=cached_len, output_len=2, uid=31,
        sampling_params=SamplingParams(), cache_handle=None,
    )
    batch = Batch(reqs=[req], phase=phase)
    batch.padded_reqs = batch.reqs
    batch.input_ids = torch.tensor(token_ids[cached_len:], dtype=torch.int32, device=device)
    batch.positions = torch.arange(cached_len, len(token_ids), dtype=torch.int32, device=device)
    batch.out_loc = torch.arange(cached_len, len(token_ids), dtype=torch.int32, device=device)
    batch.linear_table_idx = torch.tensor([0], dtype=torch.int32, device=device)
    return batch


@torch.inference_mode()
def _run_and_capture(model, ctx, token_ids, *, cached_len, phase, split):
    batch = _batch(token_ids, cached_len=cached_len, device=torch.device("cuda:0"), phase=phase)
    ctx.attn_backend.prepare_metadata(batch)
    captures = {}
    with ctx.forward_batch(batch):
        hidden = model.model.embed_tokens.forward(batch.input_ids)
        residual = None
        captures["block_a_input"] = (hidden.detach().cpu(), None)
        for layer_id, layer in enumerate(model.model.layers.op_list):
            if layer_id == split:
                captures["block_b_input"] = (
                    hidden.detach().cpu(), residual.detach().cpu() if residual is not None else None
                )
            hidden, residual = layer.forward(hidden, residual)
            if layer_id == split - 1:
                captures["block_a_output"] = (
                    hidden.detach().cpu(), residual.detach().cpu()
                )
        captures["block_b_output"] = (hidden.detach().cpu(), residual.detach().cpu())
        final, _ = model.model.norm.forward_add_residual(hidden, residual)
        captures["final_norm_output"] = final.detach().cpu()
        logits = model.lm_head.forward(final)
        next_token = int(logits.argmax(dim=-1).item())
    return captures, next_token, {
        "phase": phase, "token_ids": token_ids, "cached_len": cached_len,
        "positions": batch.positions.cpu().tolist(), "out_loc": batch.out_loc.cpu().tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--fixture-out", required=True)
    parser.add_argument("--memory-out", required=True)
    args = parser.parse_args()
    from freetoken.distributed import set_tp_info
    from freetoken.engine.engine import _materialize_loaded_weight_state_dict
    from freetoken.layers.rotary import set_rope_device
    from freetoken.models import create_model, load_weight
    from freetoken.models.qwen3_5_moe.config import parse_config
    from freetoken.moe.expert_banks import load_expert_banks
    from freetoken.utils import cached_load_hf_config, torch_dtype
    from freetoken.research.n0_model_block import write_json_with_sha
    from transformers import AutoTokenizer

    before = _status(); vm_before = _vmstat(); started = time.monotonic()
    set_tp_info(0, 1)
    device = torch.device("cuda:0")
    config = replace(
        parse_config(cached_load_hf_config(args.model)),
        moe_backend="offload", nvfp4_backend="triton",
    )
    set_rope_device(device)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = create_model(config)
    state = _materialize_loaded_weight_state_dict(
        model.state_dict(), load_weight(args.model, device, include_moe_experts=False), device=device
    )
    model.load_state_dict(state)
    del state
    banks = load_expert_banks(
        args.model, config, device=device, dtype=torch.bfloat16,
        dummy=False, parallel=False, decode_target="gpu",
    )
    ctx, cache = _setup_runtime(config, model, banks, device=device)
    torch.cuda.synchronize(device)
    load_elapsed = time.monotonic() - started
    after_load = _status()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt_text = "InferSwarm N0 boundary fixture."
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    prefill, first_token, prefill_meta = _run_and_capture(
        model, ctx, prompt_ids, cached_len=0, phase="prefill", split=args.split
    )
    decode_ids = [*prompt_ids, first_token]
    decode, second_token, decode_meta = _run_and_capture(
        model, ctx, decode_ids, cached_len=len(prompt_ids), phase="decode", split=args.split
    )
    torch.cuda.synchronize(device)
    fixture = {
        "schema": "inferswarm.n0.reference-fixtures/1",
        "model": "nvidia/Qwen3.6-35B-A3B-NVFP4", "revision": args.revision,
        "split": args.split, "prompt_text": prompt_text, "prompt_ids": prompt_ids,
        "first_token": first_token, "second_token": second_token,
        "prefill_meta": prefill_meta, "decode_meta": decode_meta,
        "prefill": prefill, "decode": decode,
    }
    fixture_path = Path(args.fixture_out)
    torch.save(fixture, fixture_path)
    fixture_sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    fixture_path.with_suffix(fixture_path.suffix + ".sha256").write_text(
        f"{fixture_sha}  {fixture_path.name}\n"
    )
    after = _status(); vm_after = _vmstat()
    write_json_with_sha(args.memory_out, {
        "schema": "inferswarm.n0.full-loader-memory/1", "normal_loader": True,
        "model": "nvidia/Qwen3.6-35B-A3B-NVFP4", "revision": args.revision,
        "process_before": before, "process_after_load": after_load, "process_after": after,
        "ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "vmstat_delta": {k: vm_after[k] - vm_before[k] for k in vm_after},
        "cuda_memory_after_bytes": torch.cuda.memory_allocated(device),
        "cuda_peak_bytes": torch.cuda.max_memory_allocated(device),
        "load_elapsed_seconds": load_elapsed, "fixture_sha256": fixture_sha,
        "expert_bank_bytes": sum(t.numel() * t.element_size() for ls in banks.sources.values() for t in ls),
    })
    print(json.dumps({"fixture_sha256": fixture_sha, "prompt_tokens": len(prompt_ids),
                      "first_token": first_token, "second_token": second_token,
                      "ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "load_elapsed_seconds": load_elapsed}, indent=2))


if __name__ == "__main__":
    main()
