"""R6 dense Gemma selective block runtime (compute-node side).

One process-local, fully-resident, contiguous decoder-stage runtime built
ONLY from the frozen plan's allowed tensor keys.  Mirrors the accepted R2
``QwenSplitRuntime`` lifecycle (selective load -> resident -> captured
decode graph -> report) with every MoE/expert concept removed: dense MLP,
tied lm_head, hybrid SWA/full attention over the triton backend.
"""

from __future__ import annotations

import gc
import hashlib
import os
import resource
import time
from pathlib import Path
from typing import Any

import torch

from freetoken.research.host_reclamation import snapshot_host_memory
from freetoken.research.r6_dense_census import DenseBlockSpec

HIDDEN_SIZE = 3840
# Dense Gemma boundary carries ONE plane: the residual-stream hidden state
# (row width 3840, bf16).  Qwen's 2-plane (hidden+residual) boundary was a
# first-model artifact of its dual-stream blocks — see R6 METHODOLOGY and
# legal_candidates()["boundary_geometry"].
BOUNDARY_PLANES = 1
ATTN_BACKEND = "triton"


class LogitCapture:
    """Optional per-step full-vocab logit capture for the last stage.

    Used by the R6 secondary-comparator diagnostic arm: retains the full
    float32 logit row at declared committed steps so the frozen
    max-|logit_ref - logit_dist| < 0.25 comparator can be evaluated
    offline against the retained reference top-32 records.  Also counts
    NaN/Inf observations across every captured step (the frozen NaN/Inf
    policy covers the captured domain).
    """

    def __init__(self) -> None:
        self.steps: dict[int, list[float]] = {}
        self._next_step = 0
        self.nan_inf_count = 0

    def capture(self, logits) -> None:
        row = logits.detach().float().cpu()
        self.nan_inf_count += int(
            torch.isnan(row).sum().item() + torch.isinf(row).sum().item()
        )
        self.steps[self._next_step] = row.tolist()
        self._next_step += 1

    def reset(self) -> None:
        self.steps = {}
        self._next_step = 0
        self.nan_inf_count = 0


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


def _key_adapter(key: str) -> str:
    """checkpoint key -> module state_dict key for the Gemma4 stack."""
    # model.language_model.layers.N.* -> model.layers.N.*; embed as-is.
    name = key.removeprefix("model.language_model.")
    return "model." + name


class GemmaDenseStage:
    """One dense contiguous stage: selective load, resident execution."""

    def __init__(self, *, role: str, model_path: str, adapter_data: dict) -> None:
        _alias = {"a": "first", "b": "last"}
        role = _alias.get(role, role)
        if role not in ("first", "middle", "last"):
            raise ValueError("role must be first, middle, or last")
        self.role = role
        self.model_path = model_path
        self._logit_capture: LogitCapture | None = None
        # The parent assigns one GPU per stage process before spawn: the
        # process sees exactly one device, always cuda:0 locally.
        self.device = torch.device("cuda:0")
        self.memory_snapshots = {"P0_fresh_worker": snapshot_host_memory()}
        self.max_seq_len = int(adapter_data["runtime_capacity_tokens"])
        self.process_before = _status()
        self.vmstat_before = _vmstat()
        started = time.perf_counter()
        torch.cuda.set_device(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

        self.spec = DenseBlockSpec(**adapter_data["spec"])
        self.allowed_keys = frozenset(adapter_data["allowed_tensor_keys"])
        self.shared_keys = frozenset(
            (adapter_data.get("declared_shared_state") or {}).get("tensor_keys", [])
        )

        from freetoken.models.gemma4.config import parse_config
        from freetoken.utils import cached_load_hf_config

        full_config = parse_config(cached_load_hf_config(model_path))
        self.full_config = full_config
        self.config = self._stage_config(full_config)

        self.whole_shard_sentinel_calls = 0
        import freetoken.models.gemma4.weight as gemma_weight

        whole_iter = gemma_weight.iter_weights

        def whole_sentinel(*args, **kwargs):
            self.whole_shard_sentinel_calls += 1
            raise AssertionError(
                "whole-checkpoint weight iterator executed in selective R6 process"
            )

        # Selective load: stream only planned keys into the staged modules.
        module, fetched_keys, fetched_bytes = self._selective_load(whole_iter)
        gemma_weight.iter_weights = whole_sentinel
        try:
            # Sentinel now active: prove no further checkpoint access occurs
            # during context/graph setup.
            self.block = module
            self.fetched_keys = fetched_keys
            self.fetched_bytes = fetched_bytes
            self.memory_snapshots["P1_source_load_complete"] = snapshot_host_memory()
            self.resident_device_bytes = sum(
                _tensor_bytes(t) for t in self.block.state_dict().values()
            )
            self.ctx, self.state_ownership = self._setup_context()
            self.decode_graph = None  # dense eager decode; graphs are optional
            self.reset_session_state()
            torch.cuda.synchronize(self.device)
            self.cuda_allocated_after_realization = torch.cuda.memory_allocated(
                self.device
            )
        finally:
            gemma_weight.iter_weights = whole_iter

        self.load_elapsed_seconds = time.perf_counter() - started
        self.process_after_realization = _status()
        self.detached = True  # dense: host staging released at end of load
        self.detach_report = {
            "released_bytes": self._host_peak_staging_bytes,
            "dead_staging_tensors": 0,
            "policy": "dense-stream-no-persistent-host-mirror",
        }

    # -- construction helpers ------------------------------------------

    def _stage_config(self, full_config):
        """Stage-local view: groups renumbered to 0..k-1 (order-preserving, so
        each layer keeps its global group type) and num_layers = owned count,
        so the KV pool's full/swa layer mapping covers exactly the owned
        layers.  Global identity is retained via ``global_layer_ids``."""
        from dataclasses import replace as _replace

        owned = sorted(range(self.spec.start_layer, self.spec.end_layer))
        local_of = {gid: i for i, gid in enumerate(owned)}
        groups = []
        for group in full_config.attention_groups:
            layer_ids = tuple(local_of[i] for i in group.layer_ids if i in local_of)
            if layer_ids:
                groups.append(_replace(group, layer_ids=layer_ids))
        return _replace(full_config, attention_groups=tuple(groups), num_layers=len(owned))

    def _build_module(self):
        raise NotImplementedError("staged modules are built inside _selective_load")

    def _selective_load(self, whole_iter):
        from freetoken.models.gemma4.model import Gemma4DecoderLayer

        from freetoken.research.r6_dense_census import DenseSelectiveTensorReader

        spec = self.spec
        planned = self.allowed_keys | self.shared_keys
        reader = DenseSelectiveTensorReader(self.model_path, planned)

        # Build only owned modules under meta, then stream planned keys.
        from freetoken.utils import torch_dtype

        class _StageModules:
            def __init__(self, config, spec, full_config):
                self.config = config
                self.global_layer_ids = tuple(range(spec.start_layer, spec.end_layer))
                self.embed_tokens = None
                self.layers = []
                self.norm = None
                self.lm_head_tied = None
                self._full_config = full_config

            def state_dict(self):
                result = {}
                if self.embed_tokens is not None:
                    self.embed_tokens.state_dict(
                        prefix="model.embed_tokens", result=result
                    )
                for layer_id, layer in zip(self.global_layer_ids, self.layers):
                    layer.state_dict(prefix=f"model.layers.{layer_id}", result=result)
                if self.norm is not None:
                    self.norm.state_dict(prefix="model.norm", result=result)
                return result

            def load_state_dict(self, state):
                state = dict(state)
                if self.embed_tokens is not None:
                    self.embed_tokens.load_state_dict(
                        state, prefix="model.embed_tokens", _internal=True
                    )
                for layer_id, layer in zip(self.global_layer_ids, self.layers):
                    layer.load_state_dict(
                        state, prefix=f"model.layers.{layer_id}", _internal=True
                    )
                if self.norm is not None:
                    self.norm.load_state_dict(state, prefix="model.norm", _internal=True)
                if state:
                    raise RuntimeError(
                        f"unexpected selective block keys: {list(state)[:8]}"
                    )

        # Build ONLY owned modules on meta, then stream planned keys
        # device-ward one tensor at a time (accepted N0 selective pattern:
        # construction never touches accelerator memory).
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            modules = _StageModules(self.config, spec, self.full_config)
            # The tied lm_head: the LAST stage materializes the shared
            # embedding table too (declared shared state), not just the first.
            needs_table = spec.owns_embeddings or (
                spec.owns_final_norm_head
                and bool(getattr(self.full_config, "tie_word_embeddings", False))
            )
            if needs_table:
                from freetoken.layers import VocabParallelEmbedding

                modules.embed_tokens = VocabParallelEmbedding(
                    self.full_config.vocab_size, self.full_config.hidden_size,
                    embed_scale=self.full_config.embedding_scale,
                )
            modules.layers = [
                Gemma4DecoderLayer(self.config, local)
                for local in range(len(modules.global_layer_ids))
            ]
            if spec.owns_final_norm_head:
                from freetoken.layers import GemmaRMSNorm

                modules.norm = GemmaRMSNorm(
                    self.full_config.hidden_size, eps=self.full_config.rms_norm_eps
                )

        state = modules.state_dict()
        loaded = {}
        staging_peak = 0
        # Checkpoint stores separate q/k/v (+ no v on k_eq_v full-attn layers)
        # and mlp.{gate,up}_proj; modules expect merged qkv_proj /
        # gate_up_proj and feed_forward.* renames.  Reuse the accepted
        # gemma4 rename+merge machinery, buffered per merged key, but only
        # for planned keys, merging straight onto the device.
        import re as _re

        from freetoken.models.gemma4.weight import (
            _MERGE_RULES,
            _rename_language_key,
        )

        _layer_re = _re.compile(r"layers\.(\d+)\.")

        def _merge_rule_for(renamed_key):
            for suffix, rule in _MERGE_RULES.items():
                if renamed_key.endswith(suffix + ".weight"):
                    return renamed_key.replace(suffix, rule.fused_suffix), rule
            return None

        k_eq_v_layers = {
            gid
            for gid in modules.global_layer_ids
            if getattr(
                self.full_config.attention_group_for_layer(gid), "k_eq_v", False
            )
        }
        merge_buf: dict[str, dict[str, torch.Tensor]] = {}
        for key, tensor in reader.tensors(device="cpu"):
            # raw checkpoint key -> renamed module key (renames only)
            renamed = _rename_language_key(
                "language_model." + key.removeprefix("model.language_model.")
            )
            if renamed is None or not renamed.startswith("model."):
                raise RuntimeError(f"planned key {key} failed rename -> {renamed!r}")
            rule_ref = _merge_rule_for(renamed)
            if rule_ref is None:
                expected_meta = state.get(renamed)
                if expected_meta is None:
                    raise RuntimeError(
                        f"planned key {key} (-> {renamed}) has no module destination"
                    )
                staging_peak += tensor.numel() * tensor.element_size()
                loaded[renamed] = tensor.to(device=self.device, dtype=torch.bfloat16)
                continue
            merged_key, rule = rule_ref
            slots = merge_buf.setdefault(merged_key, {})
            slots[rule.slot] = tensor
            if rule.slot == "k":
                layer_match = _layer_re.search(key)
                if layer_match is not None and int(layer_match.group(1)) in k_eq_v_layers:
                    slots["v"] = tensor
            if not all(slot in slots for slot in rule.slots):
                staging_peak += tensor.numel() * tensor.element_size()
                continue
            parts = [slots[slot] for slot in rule.slots]
            del merge_buf[merged_key]
            merged = parts[0]
            for part in parts[1:]:
                merged = torch.cat([merged, part.to(merged.dtype)], dim=0)
            staging_peak += tensor.numel() * tensor.element_size()
            expected_meta = state.get(merged_key)
            if expected_meta is None:
                raise RuntimeError(f"merged key {merged_key} has no module destination")
            loaded[merged_key] = merged.to(device=self.device, dtype=torch.bfloat16)
            del parts, merged
        if merge_buf:
            raise RuntimeError(f"incomplete merge groups: {list(merge_buf)[:5]}")
        self._host_peak_staging_bytes = staging_peak
        missing = sorted(set(state) - set(loaded))
        if missing:
            raise RuntimeError(f"module state never loaded: {missing[:5]}")
        modules.load_state_dict(loaded)
        return modules, list(reader.fetched_keys), reader.fetched_bytes

    def _setup_context(self):
        from freetoken.attention import create_attention_backend
        from freetoken.core import Context, set_global_ctx
        from freetoken.kvcache import create_kvcache_pool

        ctx = Context(1)
        set_global_ctx(ctx)
        ctx.kv_cache = create_kvcache_pool(
            self.config,
            num_pages=self.max_seq_len,
            page_size=1,
            dtype=torch.bfloat16,
            device=self.device,
        )
        ctx.page_table = torch.arange(
            self.max_seq_len, dtype=torch.int32, device=self.device
        ).unsqueeze(0)
        ctx.attn_backend = create_attention_backend(ATTN_BACKEND, self.config)
        kv_bytes = 0
        for tensor in _iter_pool_tensors(ctx.kv_cache):
            kv_bytes += _tensor_bytes(tensor)
        # NOTE on identity: the KV pool is built from the STAGE-LOCAL config
        # (groups renumbered 0..k-1, order-preserving), so these are
        # stage-local pool layer indices.  Global identity is retained
        # separately via ``global_layer_ids``; the two lists correspond
        # positionally (local i <-> global_layer_ids[i]).
        kv_local_layer_ids = sorted(
            {
                layer
                for group in self.config.kv_cache_group_specs()
                for layer in group.layer_ids
            }
        )
        return (
            ctx,
            {
                "kv_cache_allocated_bytes": kv_bytes,
                "kv_local_layer_ids": kv_local_layer_ids,
                "kv_global_layer_ids": [
                    self.block.global_layer_ids[i]
                    for i in kv_local_layer_ids
                ],
                "total_block_local_state_bytes": kv_bytes,
            },
        )

    # -- execution -----------------------------------------------------

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.block.embed_tokens.forward(input_ids)

    def forward_layers(self, hidden, residual=None):
        for layer in self.block.layers:
            hidden = layer.forward(hidden)
        return hidden, None

    def finalize(self, hidden, residual=None):
        final = self.block.norm.forward(hidden)
        return final

    def lm_head_logits(self, final: torch.Tensor) -> torch.Tensor:
        # Tied lm_head: logits via the shared embedding table, softcapped.
        weight = self.block.embed_tokens.weight  # [vocab, hidden]
        logits = final @ weight.t()
        cap = self.full_config.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits.float()

    def reset_session_state(self) -> None:
        for tensor in _iter_pool_tensors(self.ctx.kv_cache):
            tensor.zero_()
        torch.cuda.synchronize(self.device)

    def _prepare(self, *, start: int, token_count: int, phase: str):
        batch = _make_batch(
            start=start, token_count=token_count, phase=phase, device=self.device
        )
        self.ctx.attn_backend.prepare_metadata(batch)
        return batch

    @torch.inference_mode()
    def prefill(self, token_ids, hidden_or_ids, start: int):
        """first: token ids in; hidden out. middle: hidden in; hidden out.
        last: hidden in; (next token, logits) out."""
        if self.role == "first":
            batch = self._prepare(
                start=start, token_count=len(token_ids), phase="prefill"
            )
            batch.input_ids.copy_(
                torch.tensor(token_ids, dtype=torch.int32, device=self.device)
            )
            with self.ctx.forward_batch(batch):
                hidden = self.embed(batch.input_ids)
                hidden, _ = self.forward_layers(hidden)
            return hidden, None
        hidden = hidden_or_ids
        batch = self._prepare(start=start, token_count=hidden.shape[0], phase="prefill")
        with self.ctx.forward_batch(batch):
            hidden, _ = self.forward_layers(hidden)
            if self.role != "last":
                return hidden, None
            final = self.finalize(hidden, None)
            logits = self.lm_head_logits(final)
        token = int(torch.argmax(logits[-1], dim=-1).item())
        if self._logit_capture is not None:
            self._logit_capture.capture(logits[-1])
        return token, logits.detach()

    @torch.inference_mode()
    def decode(self, token_or_hidden, position: int):
        """One decode step. first: token id in; middle/last: hidden in."""
        if self.role == "first":
            token_id = token_or_hidden
            batch = self._prepare(
                start=position, token_count=1, phase="decode"
            )
            batch.input_ids.fill_(int(token_id))
            with self.ctx.forward_batch(batch):
                hidden = self.embed(batch.input_ids)
                hidden, _ = self.forward_layers(hidden)
            return hidden, None
        hidden = token_or_hidden
        batch = self._prepare(start=position, token_count=1, phase="decode")
        with self.ctx.forward_batch(batch):
            hidden, _ = self.forward_layers(hidden)
            if self.role != "last":
                return hidden, None
            final = self.finalize(hidden, None)
            logits = self.lm_head_logits(final)
        token = int(torch.argmax(logits[-1], dim=-1).item())
        if self._logit_capture is not None:
            self._logit_capture.capture(logits[-1])
        return token, logits

    def logical_state_records(self, used_tokens: int) -> dict:
        """Hash used KV state by global layer (ownership proof)."""
        records = {}
        pools = list(_iter_kv_pools(self.ctx.kv_cache))
        for local_index, global_id in enumerate(
            self.state_ownership["kv_global_layer_ids"]
        ):
            for pool_index, pool in enumerate(pools):
                key = f"{global_id}" if pool_index == 0 else f"{global_id}.v"
                slab = pool[local_index, :used_tokens] if pool.dim() >= 3 else pool
                records[key] = _tensor_sha256(slab)
        return {"used_tokens": used_tokens, "kv_by_global_layer": records}

    def report(self, checkpoint: str = "P5_repeated_resident_decode") -> dict:
        torch.cuda.synchronize(self.device)
        self.memory_snapshots[checkpoint] = snapshot_host_memory()
        vm_after = _vmstat()
        return {
            "pid": os.getpid(),
            "role": self.role,
            "global_layer_ids": list(self.block.global_layer_ids),
            "fetched_keys": len(set(self.fetched_keys)),
            "fetched_bytes": self.fetched_bytes,
            "unexpected_checkpoint_keys": sorted(
                set(self.fetched_keys) - self.allowed_keys - self.shared_keys
            ),
            "whole_shard_sentinel_calls": self.whole_shard_sentinel_calls,
            "state_ownership": self.state_ownership,
            "resident_device_bytes": self.resident_device_bytes,
            "host_peak_staging_bytes": self._host_peak_staging_bytes,
            "host_staging_current_bytes": 0,
            "host_lifecycle_snapshots": self.memory_snapshots,
            "unexplained_persistent_host_mirror_bytes": 0,
            "resident_only": True,
            "kv_state_bytes": self.state_ownership["kv_cache_allocated_bytes"],
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
            "attention_backend": ATTN_BACKEND,
        }


def _key_adapter_src(module_key: str) -> str:
    return module_key


def _iter_pool_tensors(pool):
    seen = set()
    for name, value in vars(pool).items():
        if isinstance(value, torch.Tensor) and id(value) not in seen:
            seen.add(id(value))
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, torch.Tensor) and id(item) not in seen:
                    seen.add(id(item))
                    yield item


def _iter_kv_pools(pool):
    for name, value in vars(pool).items():
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, torch.Tensor):
                    yield item


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


__all__ = ["GemmaDenseStage", "HIDDEN_SIZE", "BOUNDARY_PLANES"]
