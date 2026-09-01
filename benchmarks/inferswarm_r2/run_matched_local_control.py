"""Run the noncanonical one-GPU, unsplit, R2-geometry correctness control."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from freetoken.research.n0_model_block import write_json_with_sha

from .correctness_support import NONCANONICAL_LABEL, tensor_record
from .qwen_split_adapter import _make_batch, tensor_sha256
from .run_correctness import _compare_logits, _prompt_ids


@dataclass
class _Graph:
    graph: torch.cuda.CUDAGraph
    batch: object
    input_ids: torch.Tensor
    logits: torch.Tensor
    token: torch.Tensor
    seam_hidden: torch.Tensor
    seam_residual: torch.Tensor
    replays: int = 0


class MatchedLocalRuntime:
    """Full model on one GPU; only expert capacity/offload differs from R2."""

    def __init__(self, model_path: str, *, capacity: int, cache_slots: int) -> None:
        from freetoken.distributed import set_tp_info
        from freetoken.engine.engine import _materialize_loaded_weight_state_dict
        from freetoken.layers.rotary import set_rope_device
        from freetoken.models import create_model, load_weight
        from freetoken.models.qwen3_5_moe.config import parse_config
        from freetoken.moe.expert_banks import load_expert_banks
        from freetoken.utils import cached_load_hf_config, torch_dtype

        set_tp_info(0, 1)
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.capacity = capacity
        self.config = replace(
            parse_config(cached_load_hf_config(model_path)),
            moe_backend="offload",
            nvfp4_backend="triton",
        )
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            self.model = create_model(self.config)
        state = _materialize_loaded_weight_state_dict(
            self.model.state_dict(),
            load_weight(model_path, self.device, include_moe_experts=False),
            device=self.device,
        )
        self.model.load_state_dict(state)
        del state
        self.banks = load_expert_banks(
            model_path,
            self.config,
            device=self.device,
            dtype=torch.bfloat16,
            dummy=False,
            parallel=False,
            decode_target="gpu",
        )
        self.ctx, self.cache = self._setup(cache_slots)
        self.graph = self._capture_graph()
        self.reset_session()

    def _setup(self, cache_slots: int):
        from freetoken.attention import create_attention_backend
        from freetoken.core import Context, set_global_ctx
        from freetoken.kvcache import create_kvcache_pool
        from freetoken.kvcache.linear_state_pool import LinearStatePool
        from freetoken.moe import create_moe_backend
        from freetoken.moe.offload_cache import (
            OffloadMoeCache,
            attach_offload_moe_cache,
        )

        ctx = Context(1)
        set_global_ctx(ctx)
        ctx.kv_cache = create_kvcache_pool(
            self.config,
            num_pages=self.capacity,
            page_size=1,
            dtype=torch.bfloat16,
            device=self.device,
        )
        linear = self.config.linear_attention_group()
        ctx.linear_state_pool = LinearStatePool(
            linear, num_slots=1, dtype=torch.bfloat16, device=self.device, tp_size=1
        )
        ctx.page_table = torch.arange(
            self.capacity, dtype=torch.int32, device=self.device
        ).unsqueeze(0)
        ctx.attn_backend = create_attention_backend("fi", self.config)
        ctx.moe_backend = create_moe_backend("offload")
        cache = OffloadMoeCache(
            num_layers=self.config.num_moe_layers,
            num_experts=self.config.num_experts,
            cache_size=cache_slots,
            device=self.device,
            quant_format=self.banks.quant_format,
            decode_target="gpu",
            prefill_overlap=False,
        )
        cache.set_bank_sources(
            self.banks.sources, layer_residency=self.banks.layer_residency
        )
        cache.set_alphas(self.banks.gate_up_alpha, self.banks.down_alpha)
        layers = attach_offload_moe_cache(self.model, cache)
        if len(layers) != self.config.num_moe_layers:
            raise RuntimeError("matched local control did not attach every MoE layer")
        cache.collect_stats = True
        ctx.moe_offload_cache = cache
        return ctx, cache

    def reset_session(self) -> None:
        self.ctx.kv_cache._kv_buffer.zero_()
        self.ctx.linear_state_pool.reset(0)
        self.cache.reset()
        torch.cuda.synchronize(self.device)

    def _forward(self, input_ids: torch.Tensor, *, seam_outputs=None):
        hidden = self.model.model.embed_tokens.forward(input_ids)
        residual = None
        for layer_id, layer in enumerate(self.model.model.layers.op_list):
            hidden, residual = layer.forward(hidden, residual)
            if layer_id == 18 and seam_outputs is not None:
                seam_outputs[0].copy_(hidden)
                seam_outputs[1].copy_(residual)
        final, _ = self.model.model.norm.forward_add_residual(hidden, residual)
        logits = self.model.lm_head.forward(final)
        return hidden, residual, final, logits

    def _capture_graph(self) -> _Graph:
        batch = _make_batch(start=0, token_count=1, phase="decode", device=self.device)
        self.ctx.attn_backend.init_capture_graph(max_seq_len=self.capacity, bs_list=[1])
        self.ctx.attn_backend.prepare_for_capture(batch)
        logits = torch.empty(
            (1, self.config.vocab_size), dtype=torch.float32, device=self.device
        )
        token = torch.empty((1,), dtype=torch.int32, device=self.device)
        seam_hidden = torch.empty(
            (1, self.config.hidden_size), dtype=torch.bfloat16, device=self.device
        )
        seam_residual = torch.empty_like(seam_hidden)

        def execute():
            *_, value = self._forward(
                batch.input_ids, seam_outputs=(seam_hidden, seam_residual)
            )
            logits.copy_(value)
            token.copy_(value.argmax(dim=-1).to(torch.int32))

        self.cache.reset()
        with self.ctx.forward_batch(batch):
            execute()
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.graph(graph, stream=stream), self.ctx.forward_batch(batch):
            execute()
        torch.cuda.synchronize(self.device)
        self.cache.reset()
        self.cache.reset_stats()
        return _Graph(
            graph, batch, batch.input_ids, logits, token, seam_hidden, seam_residual
        )

    def _prepare(self, start: int, count: int, phase: str):
        batch = _make_batch(
            start=start, token_count=count, phase=phase, device=self.device
        )
        self.ctx.attn_backend.prepare_metadata(batch)
        return batch

    def logical_state(self, used_tokens: int) -> dict:
        kv = self.ctx.kv_cache._kv_buffer
        kv_layers = [
            layer
            for group in self.config.kv_cache_group_specs()
            for layer in group.layer_ids
        ]
        kv_records = {
            str(global_id): tensor_record(kv[:, local_id, :used_tokens])
            for local_id, global_id in enumerate(kv_layers)
        }
        linear = self.config.linear_attention_group()
        linear_records = {
            str(global_id): {
                "conv": tensor_record(
                    self.ctx.linear_state_pool.conv_states[local_id, 0]
                ),
                "recurrent": tensor_record(
                    self.ctx.linear_state_pool.recurrent_states[local_id, 0]
                ),
            }
            for local_id, global_id in enumerate(linear.layer_ids)
        }
        return {
            "used_tokens": used_tokens,
            "kv_by_global_layer": kv_records,
            "linear_by_global_layer": linear_records,
        }

    @torch.inference_mode()
    def prefill(self, ids: list[int], start: int) -> tuple[int, torch.Tensor, dict]:
        batch = self._prepare(start, len(ids), "prefill")
        batch.input_ids.copy_(torch.tensor(ids, dtype=torch.int32, device=self.device))
        with self.ctx.forward_batch(batch):
            hidden = self.model.model.embed_tokens.forward(batch.input_ids)
            residual = None
            seam_hidden = seam_residual = None
            for layer_id, layer in enumerate(self.model.model.layers.op_list):
                hidden, residual = layer.forward(hidden, residual)
                if layer_id == 18:
                    seam_hidden, seam_residual = hidden.clone(), residual.clone()
            final, _ = self.model.model.norm.forward_add_residual(hidden, residual)
            logits = self.model.lm_head.forward(final)
        end = start + len(ids)
        diagnostic = {
            "layer_18_output": {
                "hidden": tensor_record(seam_hidden),
                "residual": tensor_record(seam_residual),
            },
            "mutable_state": self.logical_state(end),
            "block_b_output_hidden": tensor_record(hidden),
            "block_b_output_residual": tensor_record(residual),
            "final_norm": tensor_record(final),
            "logits": tensor_record(logits.float()),
        }
        return int(logits.argmax(dim=-1).item()), logits.detach(), diagnostic

    @torch.inference_mode()
    def decode(self, token_id: int, position: int) -> tuple[int, torch.Tensor]:
        graph = self.graph
        graph.input_ids.fill_(token_id)
        graph.batch.positions.fill_(position)
        graph.batch.out_loc.fill_(position)
        graph.batch.linear_table_idx.zero_()
        batch = self._prepare(position, 1, "decode")
        self.ctx.attn_backend.prepare_for_replay(batch)
        with self.ctx.forward_batch(batch):
            graph.graph.replay()
        graph.replays += 1
        return int(graph.token.item()), graph.logits


def _device_record() -> dict:
    fields = "index,uuid,name,memory.total"
    line = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()[0]
    return dict(
        zip(fields.split(","), [part.strip() for part in line.split(",")], strict=True)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--class-id", default="W4")
    parser.add_argument("--prefill-chunk", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=17152)
    parser.add_argument("--cache-slots", type=int, default=3774)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.name in {"correctness.json", "result.json"}:
        raise ValueError("matched local diagnostic refuses canonical artifact path")

    from inferswarm_phase0.manifest import load_manifest
    from transformers import AutoTokenizer

    manifest = load_manifest(args.manifest, canonical=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt_ids = _prompt_ids(tokenizer, manifest.by_class()[args.class_id])
    reference = json.loads(args.reference.read_text())
    expected = {row["class_id"]: row for row in reference["workloads"]}[args.class_id]
    runtime = MatchedLocalRuntime(
        args.model, capacity=args.capacity, cache_slots=args.cache_slots
    )
    generated = []
    selected_logits = {}
    chunks = []
    for start in range(0, len(prompt_ids), args.prefill_chunk):
        ids = prompt_ids[start : start + args.prefill_chunk]
        token, logits, diagnostic = runtime.prefill(ids, start)
        chunks.append({"start": start, "token_count": len(ids), **diagnostic})
        if start + len(ids) == len(prompt_ids):
            generated.append(token)
            selected_logits[0] = {
                "shape": list(logits.shape),
                "float32_sha256": tensor_sha256(logits.float()),
                "full_logits": logits.float().cpu().tolist(),
            }
    while len(generated) < args.max_new_tokens:
        step = len(generated)
        token, logits = runtime.decode(generated[-1], len(prompt_ids) + step - 1)
        generated.append(token)
        if step in {1, 15, 31}:
            selected_logits[step] = {
                "shape": list(logits.shape),
                "float32_sha256": tensor_sha256(logits.float()),
                "full_logits": logits.float().cpu().tolist(),
            }
    comparisons = [
        {
            "generated_step": step,
            **_compare_logits(value, expected["selected_logit_steps"][str(step)]),
        }
        for step, value in sorted(selected_logits.items())
    ]
    payload = {
        "schema": "inferswarm.r2.matched-local-control/1",
        "evidence_label": NONCANONICAL_LABEL,
        "treatment_removed": [
            "second_compute_unit",
            "process_split",
            "activation_transport",
            "r2_execution_edge",
        ],
        "producer_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "device": _device_record(),
        "runtime_configuration": {
            "attention": "fi",
            "nvfp4": "triton",
            "moe": "offload",
            "moe_cache_slots": args.cache_slots,
            "prefill_overlap": False,
            "page_size": 1,
            "runtime_capacity_tokens": args.capacity,
            "prefill_chunk_tokens": args.prefill_chunk,
            "concurrency": 1,
            "session_state_protocol": "zero-kv-and-linear-state-before-workload",
            "graph_policy": "one-full-model-bs1-decode-capture",
        },
        "workload": {
            "class_id": args.class_id,
            "prompt_token_ids": prompt_ids,
            "prompt_token_count": len(prompt_ids),
            "prefill_chunk_count": len(chunks),
            "generated_token_ids": generated,
            "expected_generated_token_ids": expected["generated_token_ids"][
                : args.max_new_tokens
            ],
            "exact_generated_sequence": generated
            == expected["generated_token_ids"][: args.max_new_tokens],
            "selected_logit_steps": selected_logits,
            "logit_checkpoints": comparisons,
            "prefill_checkpoints": chunks,
        },
        "nan_count": sum(row["nan_count"] for row in comparisons),
        "inf_count": sum(row["inf_count"] for row in comparisons),
    }
    write_json_with_sha(args.out, payload)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "generated_exact": payload["workload"]["exact_generated_sequence"],
                "max_abs": max(row["max_absolute_deviation"] for row in comparisons),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault(
        "CUDA_VISIBLE_DEVICES", "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
    )
    raise SystemExit(main())
