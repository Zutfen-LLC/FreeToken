from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch

from freetoken.research.n0_model_block import (
    ModelBlockSpec,
    load_selective_qwen35_block,
)
from freetoken.research.n1_local_boundary import BOUNDARY_PLANES, HIDDEN_SIZE


BlockRole = Literal["a", "b"]


def _status() -> dict[str, int]:
    wanted = {"VmPeak", "VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem"}
    result = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key = line.split(":", 1)[0]
        if key in wanted:
            result[key + "_kib"] = int(line.split()[1])
    return result


def _vmstat() -> dict[str, int]:
    wanted = {"pswpin", "pswpout", "pgmajfault"}
    return {
        fields[0]: int(fields[1])
        for line in Path("/proc/vmstat").read_text().splitlines()
        if (fields := line.split())[0] in wanted
    }


def _block_runtime_config(config, spec: ModelBlockSpec):
    owned = set(range(spec.start_layer, spec.end_layer))
    groups = []
    for group in config.attention_groups:
        layer_ids = tuple(layer for layer in group.layer_ids if layer in owned)
        if layer_ids:
            groups.append(replace(group, layer_ids=layer_ids))
    return replace(config, attention_groups=tuple(groups))


def _tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    return 0


def _make_batch(*, start: int, token_count: int, phase: str, device: torch.device):
    from freetoken.core import Batch, Req, SamplingParams

    end = start + token_count
    req = Req(
        input_ids=torch.zeros(end, dtype=torch.int32),
        table_idx=0,
        cached_len=start,
        output_len=65,
        uid=1,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    batch = Batch(reqs=[req], phase=phase)
    batch.padded_reqs = batch.reqs
    batch.input_ids = torch.zeros(token_count, dtype=torch.int32, device=device)
    batch.positions = torch.arange(start, end, dtype=torch.int32, device=device)
    batch.out_loc = torch.arange(start, end, dtype=torch.int32, device=device)
    batch.linear_table_idx = torch.zeros(1, dtype=torch.int32, device=device)
    return batch


@dataclass
class DecodeGraph:
    graph: torch.cuda.CUDAGraph
    batch: object
    token_input: torch.Tensor | None
    hidden_input: torch.Tensor | None
    residual_input: torch.Tensor | None
    hidden_output: torch.Tensor | None
    residual_output: torch.Tensor | None
    logits_output: torch.Tensor | None
    token_output: torch.Tensor | None
    captures: int = 1
    replays: int = 0
    recaptures: int = 0


class N1BlockRuntime:
    def __init__(
        self,
        *,
        role: BlockRole,
        model_path: str,
        plan_path: str,
        device_index: int,
        max_seq_len: int = 17152,
        attention_backend: str = "fi",
    ) -> None:
        self.role = role
        self.model_path = model_path
        self.plan_path = plan_path
        self.device = torch.device(f"cuda:{device_index}")
        self.max_seq_len = max_seq_len
        self.attention_backend_name = attention_backend
        self.process_before = _status()
        self.vmstat_before = _vmstat()
        self.started = time.monotonic()
        torch.cuda.set_device(self.device)

        plan = json.loads(Path(plan_path).read_text())
        block_plan = plan[f"block_{role}"]
        self.spec = ModelBlockSpec(**block_plan["spec"])
        self.allowed_keys = frozenset(block_plan["allowed_tensor_keys"])

        # Physical sentinels remain armed in the actual service process.
        import safetensors.torch
        from freetoken.models import nvfp4_banks

        self._whole_shard_original = safetensors.torch.load_file
        self._full_bank_original = nvfp4_banks.load_nvfp4_expert_source_banks
        safetensors.torch.load_file = self._whole_shard_sentinel
        nvfp4_banks.load_nvfp4_expert_source_banks = self._full_bank_sentinel
        self.whole_shard_sentinel_calls = 0
        self.full_bank_sentinel_calls = 0
        try:
            self.loaded = load_selective_qwen35_block(
                model_path, self.spec, self.allowed_keys, device=self.device
            )
        finally:
            safetensors.torch.load_file = self._whole_shard_original
            nvfp4_banks.load_nvfp4_expert_source_banks = self._full_bank_original

        self.block = self.loaded.block
        if self.block.config.hidden_size != HIDDEN_SIZE:
            raise RuntimeError(
                f"codec hidden size {HIDDEN_SIZE} != model {self.block.config.hidden_size}"
            )
        self.config = _block_runtime_config(self.block.config, self.spec)
        self.ctx, self.cache = self._setup_context()
        self.state_ownership = self._state_ownership()
        self.resident_population_count = 0
        self.resident_population_bytes = 0
        self.populate_all_experts()
        self.decode_graph = self._capture_decode_graph()
        self.reset_session_state()
        torch.cuda.synchronize(self.device)
        self.load_elapsed_seconds = time.monotonic() - self.started
        self.process_after_load = _status()

    def _whole_shard_sentinel(self, *args, **kwargs):
        self.whole_shard_sentinel_calls += 1
        raise AssertionError("whole-shard load_file() executed in N1 selective process")

    def _full_bank_sentinel(self, *args, **kwargs):
        self.full_bank_sentinel_calls += 1
        raise AssertionError("legacy full expert-bank constructor executed in N1 process")

    def _setup_context(self):
        from freetoken.attention import create_attention_backend
        from freetoken.core import Context, set_global_ctx
        from freetoken.kvcache import create_kvcache_pool
        from freetoken.kvcache.linear_state_pool import LinearStatePool
        from freetoken.moe import create_moe_backend
        from freetoken.moe.offload_cache import OffloadMoeCache

        ctx = Context(1)
        set_global_ctx(ctx)
        ctx.kv_cache = create_kvcache_pool(
            self.config,
            num_pages=self.max_seq_len,
            page_size=1,
            dtype=torch.bfloat16,
            device=self.device,
        )
        linear = self.config.linear_attention_group()
        ctx.linear_state_pool = (
            LinearStatePool(linear, num_slots=1, dtype=torch.bfloat16,
                            device=self.device, tp_size=1)
            if linear is not None else None
        )
        ctx.page_table = torch.arange(
            self.max_seq_len, dtype=torch.int32, device=self.device
        ).unsqueeze(0)
        ctx.attn_backend = create_attention_backend(
            self.attention_backend_name, self.config
        )
        ctx.moe_backend = create_moe_backend("offload")
        local_layers = len(self.block.layers)
        cache = OffloadMoeCache(
            num_layers=local_layers,
            num_experts=self.config.num_experts,
            cache_size=local_layers * self.config.num_experts,
            device=self.device,
            quant_format="nvfp4",
            decode_target="gpu",
            prefill_overlap=False,
        )
        cache.set_bank_sources(self.loaded.expert_banks)
        # Captured into the one block graph; reset after capture so retained counters
        # describe only real decode replays.
        cache.collect_stats = True
        for local_id, layer in enumerate(self.block.layers):
            layer.mlp.experts.layer_id = local_id
            layer.mlp.experts.offload_cache = cache
        ctx.moe_offload_cache = cache
        return ctx, cache

    def _state_ownership(self) -> dict:
        linear = self.config.linear_attention_group()
        kv_bytes = _tensor_bytes(self.ctx.kv_cache._kv_buffer)
        recurrent_bytes = (
            0
            if self.ctx.linear_state_pool is None
            else _tensor_bytes(self.ctx.linear_state_pool.conv_states)
            + _tensor_bytes(self.ctx.linear_state_pool.recurrent_states)
        )
        return {
            "kv_cache_allocated_bytes": kv_bytes,
            "linear_recurrent_allocated_bytes": recurrent_bytes,
            "total_block_local_state_bytes": kv_bytes + recurrent_bytes,
            "kv_global_layer_ids": list(self.config.kv_cache_group_specs()[0].layer_ids),
            "linear_global_layer_ids": list(linear.layer_ids) if linear else [],
        }

    def populate_all_experts(self) -> None:
        """One startup/post-prefill H2D population; decode subsequently has only hits."""
        experts = self.config.num_experts
        for name in self.cache.bank_schema:
            destination = self.cache.bank_caches[name]
            for local_layer, source in enumerate(self.cache.bank_sources[name]):
                lo = local_layer * experts
                destination[lo:lo + experts].copy_(source, non_blocking=True)
        flat_ids = torch.arange(
            self.cache.num_layers * experts, dtype=torch.int32, device=self.device
        )
        self.cache.id_of_slot.copy_(flat_ids)
        self.cache.slot_for_id.copy_(flat_ids.view(self.cache.num_layers, experts))
        self.cache.usage.fill_(1)
        self.cache.step.fill_(1)
        torch.cuda.synchronize(self.device)
        self.resident_population_count += 1
        self.resident_population_bytes += self.cache.expert_bank_tensor_bytes()

    def reset_session_state(self) -> None:
        if self.ctx.linear_state_pool is not None:
            self.ctx.linear_state_pool.reset(0)
        self.ctx.kv_cache._kv_buffer.zero_()
        torch.cuda.synchronize(self.device)

    def _capture_decode_graph(self) -> DecodeGraph:
        batch = _make_batch(start=0, token_count=1, phase="decode", device=self.device)
        self.ctx.attn_backend.init_capture_graph(max_seq_len=self.max_seq_len, bs_list=[1])
        self.ctx.attn_backend.prepare_for_capture(batch)
        graph = torch.cuda.CUDAGraph()
        token_input = hidden_input = residual_input = None
        hidden_output = residual_output = logits_output = token_output = None
        if self.role == "a":
            token_input = batch.input_ids
            hidden_output = torch.empty((1, HIDDEN_SIZE), dtype=torch.bfloat16,
                                        device=self.device)
            residual_output = torch.empty_like(hidden_output)

            def execute():
                hidden = self.block.embed(token_input)
                hidden, residual = self.block.forward_layers(hidden, None)
                hidden_output.copy_(hidden)
                residual_output.copy_(residual)
        else:
            hidden_input = torch.zeros((1, HIDDEN_SIZE), dtype=torch.bfloat16,
                                       device=self.device)
            residual_input = torch.zeros_like(hidden_input)
            logits_output = torch.empty(
                (1, self.config.vocab_size), dtype=torch.float32, device=self.device
            )
            token_output = torch.empty((1,), dtype=torch.int32, device=self.device)

            def execute():
                hidden, residual = self.block.forward_layers(hidden_input, residual_input)
                final = self.block.finalize(hidden, residual)
                logits = self.block.lm_head.forward(final)
                logits_output.copy_(logits)
                token_output.copy_(torch.argmax(logits, dim=-1).to(torch.int32))

        with self.ctx.forward_batch(batch):
            execute()
        torch.cuda.synchronize(self.device)
        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.graph(graph, stream=stream):
            with self.ctx.forward_batch(batch):
                execute()
        torch.cuda.synchronize(self.device)
        self.cache.reset_stats()
        return DecodeGraph(
            graph, batch, token_input, hidden_input, residual_input,
            hidden_output, residual_output, logits_output, token_output,
        )

    def _prepare_eager(self, *, start: int, token_count: int, phase: str):
        batch = _make_batch(start=start, token_count=token_count, phase=phase, device=self.device)
        self.ctx.attn_backend.prepare_metadata(batch)
        return batch

    @torch.inference_mode()
    def prefill_a(self, token_ids: list[int], *, start: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.role != "a":
            raise RuntimeError("prefill_a called on Block B")
        batch = self._prepare_eager(start=start, token_count=len(token_ids), phase="prefill")
        batch.input_ids.copy_(torch.tensor(token_ids, dtype=torch.int32, device=self.device))
        with self.ctx.forward_batch(batch):
            hidden = self.block.embed(batch.input_ids)
            hidden, residual = self.block.forward_layers(hidden, None)
        return hidden, residual

    @torch.inference_mode()
    def prefill_b(
        self, hidden: torch.Tensor, residual: torch.Tensor, *, start: int
    ) -> tuple[int, torch.Tensor]:
        if self.role != "b":
            raise RuntimeError("prefill_b called on Block A")
        batch = self._prepare_eager(start=start, token_count=hidden.shape[0], phase="prefill")
        with self.ctx.forward_batch(batch):
            hidden, residual = self.block.forward_layers(hidden, residual)
            final = self.block.finalize(hidden, residual)
            logits = self.block.lm_head.forward(final)
            token = int(torch.argmax(logits, dim=-1).item())
        return token, logits.detach()

    def _prepare_graph_replay(self, position: int):
        graph = self.decode_graph
        graph.batch.positions.fill_(position)
        graph.batch.out_loc.fill_(position)
        graph.batch.linear_table_idx.zero_()
        replay_batch = _make_batch(
            start=position, token_count=1, phase="decode", device=self.device
        )
        self.ctx.attn_backend.prepare_metadata(replay_batch)
        self.ctx.attn_backend.prepare_for_replay(replay_batch)
        return replay_batch

    @torch.inference_mode()
    def decode_a(self, token_id: int, *, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.role != "a":
            raise RuntimeError("decode_a called on Block B")
        graph = self.decode_graph
        graph.token_input.fill_(token_id)
        replay_batch = self._prepare_graph_replay(position)
        with self.ctx.forward_batch(replay_batch):
            graph.graph.replay()
        graph.replays += 1
        return graph.hidden_output, graph.residual_output

    @torch.inference_mode()
    def decode_b(
        self, hidden: torch.Tensor, residual: torch.Tensor, *, position: int
    ) -> tuple[int, torch.Tensor]:
        if self.role != "b":
            raise RuntimeError("decode_b called on Block A")
        graph = self.decode_graph
        graph.hidden_input.copy_(hidden)
        graph.residual_input.copy_(residual)
        replay_batch = self._prepare_graph_replay(position)
        with self.ctx.forward_batch(replay_batch):
            graph.graph.replay()
        graph.replays += 1
        token = int(graph.token_output.item())
        return token, graph.logits_output

    def boundary_payload(self, hidden: torch.Tensor, residual: torch.Tensor) -> bytes:
        if hidden.shape != residual.shape or hidden.shape[-1] != HIDDEN_SIZE:
            raise RuntimeError("invalid residual-pair boundary shape")
        pair = torch.empty(
            (BOUNDARY_PLANES, *hidden.shape), dtype=torch.bfloat16, pin_memory=True
        )
        pair[0].copy_(hidden, non_blocking=True)
        pair[1].copy_(residual, non_blocking=True)
        torch.cuda.synchronize(self.device)
        return pair.view(torch.uint8).numpy().tobytes()

    def receive_boundary(self, payload: bytes, token_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        expected = BOUNDARY_PLANES * token_count * HIDDEN_SIZE * 2
        if len(payload) != expected:
            raise RuntimeError(f"boundary bytes {len(payload)} != {expected}")
        host = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(
            BOUNDARY_PLANES, token_count, HIDDEN_SIZE
        )
        return host[0].to(self.device), host[1].to(self.device)

    def state_hashes(self) -> dict:
        result = {}
        for name, tensor in (
            ("kv", self.ctx.kv_cache._kv_buffer),
            ("conv", self.ctx.linear_state_pool.conv_states),
            ("recurrent", self.ctx.linear_state_pool.recurrent_states),
        ):
            raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            result[name] = hashlib.sha256(raw).hexdigest()
        return result

    def report(self) -> dict:
        torch.cuda.synchronize(self.device)
        vm_after = _vmstat()
        lru = self.cache.lru_stats.detach().cpu()
        return {
            "pid": os.getpid(),
            "role": self.role.upper(),
            "global_layer_ids": list(self.loaded.global_layer_ids),
            "fetched_keys": len(set(self.loaded.fetched_keys)),
            "fetched_bytes": self.loaded.fetched_bytes,
            "unexpected_fetched_keys": sorted(set(self.loaded.fetched_keys) - self.allowed_keys),
            "whole_shard_sentinel_calls": self.whole_shard_sentinel_calls,
            "full_bank_sentinel_calls": self.full_bank_sentinel_calls,
            "process_before": self.process_before,
            "process_after_load": self.process_after_load,
            "process_current": _status(),
            "ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cuda_allocated_bytes": torch.cuda.memory_allocated(self.device),
            "cuda_peak_bytes": torch.cuda.max_memory_allocated(self.device),
            "state_ownership": self.state_ownership,
            "resident_expert_bytes": self.cache.expert_bank_tensor_bytes(),
            "resident_population_count": self.resident_population_count,
            "resident_population_bytes": self.resident_population_bytes,
            "decode_cache_lru_stats": lru.tolist(),
            "decode_cache_stats": self.cache.decode_miss_stats(),
            "decode_cache_stats_per_layer": self.cache.decode_miss_stats_per_layer(),
            "decode_graph": {
                "captures": self.decode_graph.captures,
                "replays": self.decode_graph.replays,
                "recaptures": self.decode_graph.recaptures,
                "python_per_layer_launches_per_replay": 0,
            },
            "fallback_count": 0,
            "failure_count": 0,
            "steady_expert_weight_movement_bytes": (
                self.cache.decode_miss_stats()["fetches"]
                * self.cache.expert_bytes_per_identity()
            ),
            "vmstat_delta": {key: vm_after[key] - self.vmstat_before[key] for key in vm_after},
            "load_elapsed_seconds": self.load_elapsed_seconds,
        }


__all__ = ["N1BlockRuntime"]
