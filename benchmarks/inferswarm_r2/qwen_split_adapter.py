"""Pinned Qwen R2 adapter; generic R1/R2 code remains model-opaque."""

from __future__ import annotations

import gc
import hashlib
import os
import resource
import time
import weakref
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

import torch
from freetoken.research.n0_model_block import (
    ModelBlockSpec,
    load_selective_qwen35_block,
)

from benchmarks.inferswarm_p48.run_resident_block import _unique_tensor_bytes
from benchmarks.inferswarm_r2.correctness_support import tensor_record

HIDDEN_SIZE = 2048
BOUNDARY_PLANES = 2


def _status() -> dict[str, int]:
    wanted = {"VmPeak", "VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap"}
    return {
        key + "_kib": int(fields[1])
        for line in Path("/proc/self/status").read_text().splitlines()
        if (fields := line.split()) and (key := fields[0].rstrip(":")) in wanted
    }


def _vmstat() -> dict[str, int]:
    wanted = {"pswpin", "pswpout", "pgmajfault"}
    return {
        fields[0]: int(fields[1])
        for line in Path("/proc/vmstat").read_text().splitlines()
        if (fields := line.split())[0] in wanted
    }


def _tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    return 0


def _block_runtime_config(config, spec: ModelBlockSpec):
    owned = set(range(spec.start_layer, spec.end_layer))
    groups = []
    for group in config.attention_groups:
        layer_ids = tuple(layer for layer in group.layer_ids if layer in owned)
        if layer_ids:
            groups.append(replace(group, layer_ids=layer_ids))
    return replace(config, attention_groups=tuple(groups))


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


class QwenSplitRuntime:
    """One process-local, fully resident, captured execution block."""

    def __init__(self, *, role: str, model_path: str, adapter_data: dict) -> None:
        if role not in ("a", "b"):
            raise ValueError("role must be a or b")
        self.role = role
        self.device = torch.device("cuda:0")
        self.max_seq_len = int(adapter_data["runtime_capacity_tokens"])
        self.process_before = _status()
        self.vmstat_before = _vmstat()
        started = time.perf_counter()
        torch.cuda.set_device(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.spec = ModelBlockSpec(**adapter_data["spec"])
        self.allowed_keys = frozenset(adapter_data["allowed_tensor_keys"])
        self.whole_shard_sentinel_calls = 0
        self.full_bank_sentinel_calls = 0

        import safetensors.torch
        from freetoken.models import nvfp4_banks

        whole_original = safetensors.torch.load_file
        full_original = nvfp4_banks.load_nvfp4_expert_source_banks

        def whole_sentinel(*args, **kwargs):
            self.whole_shard_sentinel_calls += 1
            raise AssertionError("whole-shard loader executed in selective R2 process")

        def full_sentinel(*args, **kwargs):
            self.full_bank_sentinel_calls += 1
            raise AssertionError("full expert-bank constructor executed in R2 process")

        safetensors.torch.load_file = whole_sentinel
        nvfp4_banks.load_nvfp4_expert_source_banks = full_sentinel
        try:
            self.loaded = load_selective_qwen35_block(
                model_path, self.spec, self.allowed_keys, device=self.device
            )
        finally:
            safetensors.torch.load_file = whole_original
            nvfp4_banks.load_nvfp4_expert_source_banks = full_original
        self.block = self.loaded.block
        if self.block.config.hidden_size != HIDDEN_SIZE:
            raise RuntimeError("pinned strategy hidden width changed")
        self.non_routed_bytes = _unique_tensor_bytes(
            self.block.state_dict(), device_type="cuda"
        )
        self.host_source_bytes_before_release = _unique_tensor_bytes(
            self.loaded.expert_banks, device_type="cpu"
        )
        self.source_refs = [
            weakref.ref(tensor)
            for per_layer in self.loaded.expert_banks.values()
            for tensor in per_layer
        ]
        self.config = _block_runtime_config(self.block.config, self.spec)
        self.ctx, self.cache, self.state_ownership = self._setup_context()
        self.populate_count = 0
        self.populate_all_experts()
        self.routed_bytes = self.cache.expert_bank_tensor_bytes()
        before_graph = torch.cuda.memory_allocated(self.device)
        self.decode_graph = self._capture_decode_graph()
        self.graph_allocation_delta_bytes = (
            torch.cuda.memory_allocated(self.device) - before_graph
        )
        self.reset_session_state()
        torch.cuda.synchronize(self.device)
        self.cuda_allocated_after_realization = torch.cuda.memory_allocated(self.device)
        self.graph_backend_bytes = (
            self.cuda_allocated_after_realization
            - self.non_routed_bytes
            - self.routed_bytes
            - self.state_ownership["total_block_local_state_bytes"]
        )
        self.load_elapsed_seconds = time.perf_counter() - started
        self.process_after_realization = _status()
        self.detached = False
        self.detach_report: dict = {}

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
            LinearStatePool(
                linear, num_slots=1, dtype=torch.bfloat16, device=self.device, tp_size=1
            )
            if linear is not None
            else None
        )
        ctx.page_table = torch.arange(
            self.max_seq_len, dtype=torch.int32, device=self.device
        ).unsqueeze(0)
        ctx.attn_backend = create_attention_backend("fi", self.config)
        ctx.moe_backend = create_moe_backend("offload")
        cache = OffloadMoeCache(
            num_layers=len(self.block.layers),
            num_experts=self.config.num_experts,
            cache_size=len(self.block.layers) * self.config.num_experts,
            device=self.device,
            quant_format="nvfp4",
            decode_target="gpu",
            prefill_overlap=False,
        )
        cache.set_bank_sources(self.loaded.expert_banks)
        cache.collect_stats = True
        for local_id, layer in enumerate(self.block.layers):
            layer.mlp.experts.layer_id = local_id
            layer.mlp.experts.offload_cache = cache
        ctx.moe_offload_cache = cache
        kv = _tensor_bytes(ctx.kv_cache._kv_buffer)
        linear_bytes = (
            0
            if ctx.linear_state_pool is None
            else (
                _tensor_bytes(ctx.linear_state_pool.conv_states)
                + _tensor_bytes(ctx.linear_state_pool.recurrent_states)
            )
        )
        kv_layers = sorted(
            {
                layer
                for group in self.config.kv_cache_group_specs()
                for layer in group.layer_ids
            }
        )
        return (
            ctx,
            cache,
            {
                "kv_cache_allocated_bytes": kv,
                "linear_recurrent_allocated_bytes": linear_bytes,
                "total_block_local_state_bytes": kv + linear_bytes,
                "kv_global_layer_ids": kv_layers,
                "linear_global_layer_ids": list(linear.layer_ids) if linear else [],
            },
        )

    def populate_all_experts(self) -> None:
        experts = self.config.num_experts
        for name in self.cache.bank_schema:
            destination = self.cache.bank_caches[name]
            for local_layer, source in enumerate(self.cache.bank_sources[name]):
                lo = local_layer * experts
                destination[lo : lo + experts].copy_(source, non_blocking=True)
        flat_ids = torch.arange(
            self.cache.num_layers * experts, dtype=torch.int32, device=self.device
        )
        self.cache.id_of_slot.copy_(flat_ids)
        self.cache.slot_for_id.copy_(flat_ids.view(self.cache.num_layers, experts))
        self.cache.usage.fill_(1)
        self.cache.step.fill_(1)
        torch.cuda.synchronize(self.device)
        self.populate_count += 1

    def detach_host_staging(self) -> dict:
        if self.detached:
            raise RuntimeError("host staging already detached")
        self.detach_report = self.cache.detach_host_sources_for_full_residency()
        released = self.loaded.release_expert_banks_after_residency(self.cache)
        gc.collect()
        torch.cuda.synchronize(self.device)
        dead = sum(ref() is None for ref in self.source_refs)
        if released != self.host_source_bytes_before_release or dead != len(
            self.source_refs
        ):
            raise RuntimeError("host routed staging did not release completely")
        if self.cache.host_source_tensor_bytes() != 0 or not self.cache.resident_only:
            raise RuntimeError("resident-only finalization failed")
        self.detached = True
        return {"released_bytes": released, "dead_staging_tensors": dead}

    def reset_session_state(self) -> None:
        if self.ctx.linear_state_pool is not None:
            self.ctx.linear_state_pool.reset(0)
        self.ctx.kv_cache._kv_buffer.zero_()
        torch.cuda.synchronize(self.device)

    def logical_state_records(self, used_tokens: int) -> dict:
        """Hash logical, used state by global layer; never expose allocator identity."""

        kv = self.ctx.kv_cache._kv_buffer
        kv_records = {}
        for local_id, global_id in enumerate(
            self.state_ownership["kv_global_layer_ids"]
        ):
            kv_records[str(global_id)] = tensor_record(kv[:, local_id, :used_tokens])
        linear_records = {}
        if self.ctx.linear_state_pool is not None:
            pool = self.ctx.linear_state_pool
            for local_id, global_id in enumerate(
                self.state_ownership["linear_global_layer_ids"]
            ):
                linear_records[str(global_id)] = {
                    "conv": tensor_record(pool.conv_states[local_id, 0]),
                    "recurrent": tensor_record(pool.recurrent_states[local_id, 0]),
                }
        return {
            "used_tokens": used_tokens,
            "kv_by_global_layer": kv_records,
            "linear_by_global_layer": linear_records,
        }

    def _capture_decode_graph(self) -> DecodeGraph:
        batch = _make_batch(start=0, token_count=1, phase="decode", device=self.device)
        self.ctx.attn_backend.init_capture_graph(
            max_seq_len=self.max_seq_len, bs_list=[1]
        )
        self.ctx.attn_backend.prepare_for_capture(batch)
        graph = torch.cuda.CUDAGraph()
        token_input = hidden_input = residual_input = None
        hidden_output = residual_output = logits_output = token_output = None
        if self.role == "a":
            token_input = batch.input_ids
            hidden_output = torch.empty(
                (1, HIDDEN_SIZE), dtype=torch.bfloat16, device=self.device
            )
            residual_output = torch.empty_like(hidden_output)

            def execute():
                hidden = self.block.embed(token_input)
                hidden, residual = self.block.forward_layers(hidden, None)
                hidden_output.copy_(hidden)
                residual_output.copy_(residual)
        else:
            hidden_input = torch.zeros(
                (1, HIDDEN_SIZE), dtype=torch.bfloat16, device=self.device
            )
            residual_input = torch.zeros_like(hidden_input)
            logits_output = torch.empty(
                (1, self.config.vocab_size), dtype=torch.float32, device=self.device
            )
            token_output = torch.empty((1,), dtype=torch.int32, device=self.device)

            def execute():
                hidden, residual = self.block.forward_layers(
                    hidden_input, residual_input
                )
                final = self.block.finalize(hidden, residual)
                logits = self.block.lm_head.forward(final)
                logits_output.copy_(logits)
                token_output.copy_(torch.argmax(logits, dim=-1).to(torch.int32))

        with self.ctx.forward_batch(batch):
            execute()
        torch.cuda.synchronize(self.device)
        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.graph(graph, stream=stream), self.ctx.forward_batch(batch):
            execute()
        torch.cuda.synchronize(self.device)
        self.cache.reset_stats()
        return DecodeGraph(
            graph,
            batch,
            token_input,
            hidden_input,
            residual_input,
            hidden_output,
            residual_output,
            logits_output,
            token_output,
        )

    def _prepare(self, *, start: int, token_count: int, phase: str):
        batch = _make_batch(
            start=start, token_count=token_count, phase=phase, device=self.device
        )
        self.ctx.attn_backend.prepare_metadata(batch)
        return batch

    @torch.inference_mode()
    def prefill_a(self, token_ids: list[int], start: int):
        batch = self._prepare(start=start, token_count=len(token_ids), phase="prefill")
        batch.input_ids.copy_(
            torch.tensor(token_ids, dtype=torch.int32, device=self.device)
        )
        with self.ctx.forward_batch(batch):
            hidden = self.block.embed(batch.input_ids)
            return self.block.forward_layers(hidden, None)

    @torch.inference_mode()
    def prefill_b(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        start: int,
        *,
        capture_diagnostics: bool = False,
    ):
        batch = self._prepare(start=start, token_count=hidden.shape[0], phase="prefill")
        with self.ctx.forward_batch(batch):
            hidden, residual = self.block.forward_layers(hidden, residual)
            final = self.block.finalize(hidden, residual)
            logits = self.block.lm_head.forward(final)
        diagnostic = None
        if capture_diagnostics:
            diagnostic = {
                "block_output_hidden": tensor_record(hidden),
                "block_output_residual": tensor_record(residual),
                "final_norm": tensor_record(final),
                "logits": tensor_record(logits.float()),
            }
        return int(torch.argmax(logits, dim=-1).item()), logits.detach(), diagnostic

    def _prepare_replay(self, position: int):
        graph = self.decode_graph
        graph.batch.positions.fill_(position)
        graph.batch.out_loc.fill_(position)
        graph.batch.linear_table_idx.zero_()
        batch = _make_batch(
            start=position, token_count=1, phase="decode", device=self.device
        )
        self.ctx.attn_backend.prepare_metadata(batch)
        self.ctx.attn_backend.prepare_for_replay(batch)
        return batch

    @torch.inference_mode()
    def decode_a(self, token_id: int, position: int):
        graph = self.decode_graph
        graph.token_input.fill_(token_id)
        batch = self._prepare_replay(position)
        with self.ctx.forward_batch(batch):
            graph.graph.replay()
        graph.replays += 1
        return graph.hidden_output, graph.residual_output

    @torch.inference_mode()
    def decode_b(self, hidden: torch.Tensor, residual: torch.Tensor, position: int):
        graph = self.decode_graph
        graph.hidden_input.copy_(hidden)
        graph.residual_input.copy_(residual)
        batch = self._prepare_replay(position)
        with self.ctx.forward_batch(batch):
            graph.graph.replay()
        graph.replays += 1
        return int(graph.token_output.item()), graph.logits_output

    def report(self) -> dict:
        torch.cuda.synchronize(self.device)
        stats = self.cache.decode_miss_stats()
        vm_after = _vmstat()
        return {
            "pid": os.getpid(),
            "role": self.role,
            "global_layer_ids": list(self.loaded.global_layer_ids),
            "fetched_keys": len(set(self.loaded.fetched_keys)),
            "fetched_bytes": self.loaded.fetched_bytes,
            "unexpected_checkpoint_keys": sorted(
                set(self.loaded.fetched_keys) - self.allowed_keys
            ),
            "whole_shard_sentinel_calls": self.whole_shard_sentinel_calls,
            "full_bank_sentinel_calls": self.full_bank_sentinel_calls,
            "state_ownership": self.state_ownership,
            "non_routed_device_bytes": self.non_routed_bytes,
            "routed_device_bytes": self.routed_bytes,
            "graph_backend_bytes": self.graph_backend_bytes,
            "graph_allocation_delta_bytes": self.graph_allocation_delta_bytes,
            "host_staging_before_release_bytes": self.host_source_bytes_before_release,
            "host_staging_current_bytes": self.cache.host_source_tensor_bytes(),
            "unexplained_persistent_host_mirror_bytes": 0
            if self.detached and self.cache.host_source_tensor_bytes() == 0
            else self.cache.host_source_tensor_bytes(),
            "resident_only": self.cache.resident_only,
            "populate_count": self.populate_count,
            "decode_graph": {
                "captures": self.decode_graph.captures,
                "replays": self.decode_graph.replays,
                "recaptures": self.decode_graph.recaptures,
            },
            "host_expert_fetches": stats["fetches"],
            "resident_source_accesses": self.cache.resident_source_access_attempts,
            "fallbacks": 0,
            "steady_model_state_movement_bytes": stats["fetches"]
            * self.cache.expert_bytes_per_identity(),
            "cuda_allocated_bytes": torch.cuda.memory_allocated(self.device),
            "cuda_peak_bytes": torch.cuda.max_memory_allocated(self.device),
            "process_before": self.process_before,
            "process_after_realization": self.process_after_realization,
            "process_current": _status(),
            "ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "vmstat_delta": {
                key: vm_after[key] - self.vmstat_before[key] for key in vm_after
            },
            "load_elapsed_seconds": self.load_elapsed_seconds,
        }


class QwenSplitResearchAdapter:
    """R1 materialization adapter for exactly one frozen R2 participant."""

    _representations: ClassVar[set[str]] = {
        "checkpoint-native-host",
        "freetoken-native-device",
        "freetoken-nvfp4-slot-banks",
        "freetoken-block-runtime-device",
        "freetoken-captured-backend-device",
    }

    def __init__(self, *, role: str, model_path: str):
        self.role = role
        self.model_path = model_path
        self.plan = None
        self.runtime: QwenSplitRuntime | None = None
        self.runtime_authorities: list[dict] = []

    def supports_representation(self, logical_state_id, representation):
        return representation in self._representations

    def supports_execution(self, execution):
        return execution.get("strategy_unit") == "opaque-contiguous-block-v1"

    def begin(self, plan, environment):
        self.plan = plan
        self.environment = environment
        self.runtime_authorities = []

    def _ensure(self) -> QwenSplitRuntime:
        if self.runtime is None:
            self.runtime = QwenSplitRuntime(
                role=self.role,
                model_path=self.model_path,
                adapter_data=self.plan["adapter_data"],
            )
        return self.runtime

    def realize_materialization(self, item):
        runtime = self._ensure()
        suffix = item["logical_state_id"].rsplit(".", 1)[-1]
        observed = {
            "non-routed": runtime.non_routed_bytes,
            "routed": runtime.host_source_bytes_before_release
            if item["role"] == "staging"
            else runtime.routed_bytes,
            "mutable-runtime": runtime.state_ownership["total_block_local_state_bytes"],
            "graph-backend": runtime.graph_backend_bytes,
        }[suffix]
        record = {
            "actual_representation": item["representation"],
            "actual_memory_resource_id": item["memory_resource_id"],
            "observed_bytes": observed,
            "lifecycle_state": "live",
            "status": "PLANNED_AND_REALIZED",
        }
        if item["role"] == "mutable_authority":
            authority = next(
                value
                for value in self.plan["authorities"]
                if value["materialization_id"] == item["id"]
            )
            self.runtime_authorities.append(
                {
                    "logical_state_id": item["logical_state_id"],
                    "materialization_id": item["id"],
                    "lineage": authority["lineage"],
                    "memory_resource_id": item["memory_resource_id"],
                }
            )
        return record

    def release_materialization(self, item):
        released = self._ensure().detach_host_staging()
        return {
            "observed_bytes_after_release": 0,
            "released_bytes": released["released_bytes"],
        }

    def activate_execution(self, execution):
        runtime = self._ensure()
        if not runtime.detached or not runtime.cache.resident_only:
            raise RuntimeError("execution activation before resident-only finalization")
        return {
            "execution_id": execution["id"],
            "compute_unit_id": execution["compute_unit_id"],
            "status": "ACTIVE_BACKEND_NATIVE",
        }

    def observe_authorities(self):
        return list(self.runtime_authorities)


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


__all__ = [
    "BOUNDARY_PLANES",
    "HIDDEN_SIZE",
    "QwenSplitResearchAdapter",
    "QwenSplitRuntime",
    "tensor_sha256",
]
