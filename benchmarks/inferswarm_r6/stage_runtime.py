"""R6 dense Gemma selective block runtime (compute-node side).

One process-local, fully-resident, contiguous decoder-stage runtime built
ONLY from the frozen plan's allowed tensor keys.  Mirrors the accepted R2
``QwenSplitRuntime`` lifecycle (selective load -> resident -> captured
decode graph -> report) with every MoE/expert concept removed: dense MLP,
tied lm_head, hybrid SWA/full attention over the triton backend.
"""

from __future__ import annotations

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


# --- final-logit softcap: frozen legacy vs in-place candidate ----------------
#
# Legacy (frozen R6 semantics, out-of-place):
#     logits = torch.tanh(logits / cap) * cap
# Candidate (SINGLE_GPU_CONTROL_AMENDMENT-003, in-place; allocation-lifetime
# change ONLY): the identical elementwise operation order (divide, tanh,
# multiply) with the identical cap value applied directly into the full
# [sequence, vocab] BF16 logits tensor, eliminating the transient
# full-matrix BF16 temporaries the out-of-place expression materializes
# (one [rows, vocab] temporary per intermediate result).
#
# The full BF16 GEMM, the full [sequence, vocab] BF16 materialization, the
# softcap math, the cap value, and the final-row FP32 promotion are all
# unchanged; bit-equivalence is proven physically on the real Gemma CUDA
# path before the production default flips.
#
# The optional ``probe`` callback receives phase names (softcap_div_complete,
# ...) for OOM localization.  Probes are diagnostics only: they never alter
# semantics, and probe failures are swallowed so a broken diagnostic can
# never mask or replace the real execution result.


def softcap_legacy(logits: torch.Tensor, cap, probe=None) -> torch.Tensor:
    """Legacy out-of-place softcap: ``torch.tanh(logits / cap) * cap``.

    Retained verbatim as the frozen reference transformation for the
    equivalence proof and for explicit ``softcap_mode="legacy"`` runs.
    """
    try:
        scaled = logits / cap
    except Exception:
        _notify(probe, "softcap_div_failed")
        raise
    _notify(probe, "softcap_div_complete")
    try:
        activated = torch.tanh(scaled)
    except Exception:
        _notify(probe, "softcap_tanh_failed")
        raise
    _notify(probe, "softcap_tanh_complete")
    try:
        out = activated * cap
    except Exception:
        _notify(probe, "softcap_mul_failed")
        raise
    _notify(probe, "softcap_mul_complete")
    return out


def softcap_inplace(logits: torch.Tensor, cap, probe=None) -> torch.Tensor:
    """Candidate in-place softcap: div_/tanh_/mul_ into ``logits``.

    Same elementwise operation order and cap as :func:`softcap_legacy`;
    differs only in temporary-allocation lifetime (no full-matrix
    intermediates).  Returns the mutated ``logits`` for expression parity.
    """
    try:
        logits.div_(cap)
    except Exception:
        _notify(probe, "softcap_div_failed")
        raise
    _notify(probe, "softcap_div_complete")
    try:
        logits.tanh_()
    except Exception:
        _notify(probe, "softcap_tanh_failed")
        raise
    _notify(probe, "softcap_tanh_complete")
    try:
        logits.mul_(cap)
    except Exception:
        _notify(probe, "softcap_mul_failed")
        raise
    _notify(probe, "softcap_mul_complete")
    return logits


def _notify(probe, phase: str) -> None:
    if probe is None:
        return
    try:
        probe(phase)
    except Exception:
        pass


def cuda_phase_probe(
    phase: str,
    *,
    step: int | None = None,
    replay_rows: int | None = None,
    generated_tokens: int | None = None,
    device: str = "cuda:0",
) -> dict:
    """One named-phase CUDA allocation snapshot for OOM localization.

    Reads only host-side allocator bookkeeping (allocated/reserved/peak)
    plus ``torch.cuda.mem_get_info`` (free/total); none of these force a
    device synchronization, so arming the diagnostic does not serialize
    execution.  ``mem_get_info`` free bytes may transiently lag asynchronous
    frees and is best-effort; allocation/peak deltas used for attribution
    are bookkeeping-exact.  Never raises.
    """
    record: dict = {"phase": phase}
    if step is not None:
        record["step"] = step
    if replay_rows is not None:
        record["replay_rows"] = replay_rows
    if generated_tokens is not None:
        record["generated_tokens"] = generated_tokens
    if not torch.cuda.is_available():
        record["cuda_available"] = False
        return record
    try:
        record["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
        record["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved(device))
        record["cuda_peak_bytes"] = int(torch.cuda.max_memory_allocated(device))
        free, total = torch.cuda.mem_get_info(device)
        record["cuda_free_bytes"] = int(free)
        record["cuda_total_bytes"] = int(total)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
        record["probe_error"] = f"{type(exc).__name__}: {exc}"
    return record


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


def _rusage_counters() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "minor_page_faults": usage.ru_minflt,
        "major_page_faults": usage.ru_majflt,
        "swaps": usage.ru_nswap,
    }


def _tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    return 0


_SUPPORTED_FLOAT_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)


def _copy_checkpoint_tensor(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    raw_key: str,
) -> None:
    if tuple(destination.shape) != tuple(source.shape):
        raise RuntimeError(
            f"shape mismatch for {raw_key}: source {tuple(source.shape)} != "
            f"destination {tuple(destination.shape)}"
        )
    if source.dtype != destination.dtype and not (
        source.dtype in _SUPPORTED_FLOAT_DTYPES
        and destination.dtype in _SUPPORTED_FLOAT_DTYPES
    ):
        raise RuntimeError(
            f"unsupported dtype conversion for {raw_key}: "
            f"{source.dtype} -> {destination.dtype}"
        )
    destination.copy_(source, non_blocking=False)


def materialize_dense_state(
    *,
    expected_state: dict[str, torch.Tensor],
    reader,
    device: torch.device | str,
    rename_key,
    merge_rule_for,
    k_eq_v_k_keys: frozenset[str] = frozenset(),
) -> tuple[dict[str, torch.Tensor], int]:
    """Materialize planned state directly into its final device tensors.

    Merge groups are validated entirely from safetensors headers before any
    data mapping opens.  Q/K/V and gate/up sources are copied one at a time
    into final destination slices; no source tensor survives its context and
    no CPU concatenation is constructed.
    """
    standalone: dict[str, str] = {}
    groups: dict[str, dict[str, Any]] = {}
    destinations: dict[str, str] = {}

    for raw_key in sorted(reader.allowed_keys):
        renamed = rename_key(raw_key)
        if renamed is None or not renamed.startswith("model."):
            raise RuntimeError(f"planned key {raw_key} failed rename -> {renamed!r}")
        rule_ref = merge_rule_for(renamed)
        if rule_ref is None:
            if renamed not in expected_state:
                raise RuntimeError(
                    f"planned key {raw_key} (-> {renamed}) has no module destination"
                )
            if renamed in destinations:
                raise RuntimeError(f"duplicate destination component {renamed}")
            destinations[renamed] = raw_key
            standalone[raw_key] = renamed
            continue

        merged_key, rule = rule_ref
        if merged_key not in expected_state:
            raise RuntimeError(f"merged key {merged_key} has no module destination")
        group = groups.setdefault(merged_key, {"rule": rule, "sources": {}})
        if (
            group["rule"].fused_suffix != rule.fused_suffix
            or group["rule"].slots != rule.slots
        ):
            raise RuntimeError(f"inconsistent merge rules for {merged_key}")
        sources = group["sources"]
        if rule.slot in sources:
            raise RuntimeError(f"duplicate merge component {merged_key}:{rule.slot}")
        sources[rule.slot] = raw_key

    copies_by_raw: dict[str, list[tuple[str, str, int, tuple[int, ...]]]] = {}
    for merged_key, group in groups.items():
        rule = group["rule"]
        sources = dict(group["sources"])
        k_source = sources.get("k")
        if k_source in k_eq_v_k_keys:
            if "v" in sources:
                raise RuntimeError(
                    f"duplicate K=V merge component {merged_key}: checkpoint has v"
                )
            sources["v"] = k_source
        missing = [slot for slot in rule.slots if slot not in sources]
        unexpected = sorted(set(sources) - set(rule.slots))
        if missing or unexpected:
            raise RuntimeError(
                f"incomplete merge group {merged_key}: missing={missing}, "
                f"unexpected={unexpected}"
            )

        expected = expected_state[merged_key]
        offset = 0
        for slot in rule.slots:
            raw_key = sources[slot]
            record = reader.record(raw_key)
            if not record.shape:
                raise RuntimeError(f"merge component {raw_key} must have rank >= 1")
            if tuple(record.shape[1:]) != tuple(expected.shape[1:]):
                raise RuntimeError(
                    f"shape mismatch for {raw_key}: trailing dimensions "
                    f"{record.shape[1:]} != {tuple(expected.shape[1:])}"
                )
            rows = record.shape[0]
            copies_by_raw.setdefault(raw_key, []).append(
                (merged_key, slot, offset, record.shape)
            )
            offset += rows
        if offset != expected.shape[0]:
            raise RuntimeError(
                f"shape mismatch for {merged_key}: merged rows {offset} != "
                f"destination rows {expected.shape[0]}"
            )
        if merged_key in destinations:
            raise RuntimeError(f"duplicate final destination {merged_key}")
        destinations[merged_key] = "merge"

    missing_destinations = sorted(set(expected_state) - set(destinations))
    if missing_destinations:
        raise RuntimeError(f"module state never planned: {missing_destinations[:5]}")

    loaded: dict[str, torch.Tensor] = {}
    filled: set[tuple[str, str]] = set()
    fusion_source_peak_live_bytes = 0
    for raw_key in sorted(reader.allowed_keys):
        with reader.open_tensor(raw_key) as source:
            if raw_key in standalone:
                destination_key = standalone[raw_key]
                expected = expected_state[destination_key]
                destination = torch.empty(
                    tuple(expected.shape), dtype=expected.dtype, device=device
                )
                _copy_checkpoint_tensor(destination, source, raw_key=raw_key)
                loaded[destination_key] = destination
                continue

            source_bytes = source.numel() * source.element_size()
            fusion_source_peak_live_bytes = max(
                fusion_source_peak_live_bytes, source_bytes
            )
            for merged_key, slot, offset, shape in copies_by_raw.get(raw_key, []):
                expected = expected_state[merged_key]
                destination = loaded.get(merged_key)
                if destination is None:
                    destination = torch.empty(
                        tuple(expected.shape), dtype=expected.dtype, device=device
                    )
                    loaded[merged_key] = destination
                target = destination.narrow(0, offset, shape[0])
                _copy_checkpoint_tensor(target, source, raw_key=raw_key)
                marker = (merged_key, slot)
                if marker in filled:
                    raise RuntimeError(f"duplicate merge copy {merged_key}:{slot}")
                filled.add(marker)

    reader.assert_all_fetched()
    for merged_key, group in groups.items():
        missing = [
            slot for slot in group["rule"].slots if (merged_key, slot) not in filled
        ]
        if missing:
            raise RuntimeError(
                f"merge destination {merged_key} missing copies {missing}"
            )
    if reader.host_staging_current_bytes != 0:
        raise RuntimeError("bounded reader retained host staging after realization")
    return loaded, fusion_source_peak_live_bytes


def execute_dense_layer_sequence(layers, hidden: torch.Tensor) -> torch.Tensor:
    """Shared ordered decoder execution used by staged and single roles."""
    for layer in layers:
        hidden = layer.forward(hidden)
    return hidden


def semantic_boundaries_for_role(role: str):
    """A complete single role has no inter-stage semantic boundary."""
    return [] if role == "single" else None


def _make_batch(*, start: int, token_count: int, phase: str, device: torch.device):
    from freetoken.core import Batch, Req, SamplingParams

    end = start + token_count
    req = Req(
        input_ids=torch.zeros(end, dtype=torch.int32),
        table_idx=0,
        cached_len=start,
        output_len=64,
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


def _host_resident_bytes(state_dict: dict[str, torch.Tensor]) -> tuple[int, list[str]]:
    """Mechanically prove complete CUDA residency from realized module state.

    Any tensor whose device is not CUDA counts as a persistent host mirror;
    this is derived from the actual materialized tensors, never asserted.
    """
    host_bytes = 0
    host_keys: list[str] = []
    for key, tensor in state_dict.items():
        if tensor.device.type != "cuda":
            host_bytes += _tensor_bytes(tensor)
            host_keys.append(key)
    return host_bytes, host_keys


def _cpu_owned_decoder_layers(global_layer_ids, layers) -> int:
    """Count decoder layers with any parameter not resident on CUDA."""
    count = 0
    for layer_id, layer in zip(global_layer_ids, layers):
        layer_state = layer.state_dict(prefix=f"model.layers.{layer_id}")
        host_bytes, _ = _host_resident_bytes(layer_state)
        if host_bytes > 0:
            count += 1
    return count


class _StageModules:
    """Owned modules for one selective block: embedding/layers/norm only.

    Module-level (not nested in ``_selective_load``) so the exact production
    ``load_state_dict`` consumption contract is directly unit-testable
    without constructing a full ``GemmaDenseStage``.
    """

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
            self.embed_tokens.state_dict(prefix="model.embed_tokens", result=result)
        for layer_id, layer in zip(self.global_layer_ids, self.layers):
            layer.state_dict(prefix=f"model.layers.{layer_id}", result=result)
        if self.norm is not None:
            self.norm.state_dict(prefix="model.norm", result=result)
        return result

    def load_state_dict(self, state):
        # NOTE: this must mutate the CALLER's dict in place (BaseOP.load_state_dict
        # consumes keys via state.pop(...)), so the caller can verify complete
        # consumption by checking its own dict is empty afterward.  A defensive
        # `state = dict(state)` copy here would silently break that contract:
        # every sub-load would drain the copy while the caller's dict stayed
        # full, making any "unconsumed state" check fire unconditionally.
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
            raise RuntimeError(f"unexpected selective block keys: {list(state)[:8]}")


class GemmaDenseStage:
    """One dense contiguous stage: selective load, resident execution."""

    # Class-level defaults so the helper seam is well-defined on every
    # instance (including minimal stubs built via object.__new__ in tests).
    _softcap_mode = "inplace"
    _phase_probe = None

    def __init__(self, *, role: str, model_path: str, adapter_data: dict) -> None:
        _alias = {"a": "first", "b": "last"}
        role = _alias.get(role, role)
        if role not in ("first", "middle", "last", "single"):
            raise ValueError("role must be first, middle, last, or single")
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
        self.rusage_before = _rusage_counters()
        started = time.perf_counter()
        torch.cuda.set_device(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.cuda_allocated_before_load = torch.cuda.memory_allocated(self.device)
        free_before, total_memory = torch.cuda.mem_get_info(self.device)
        self.cuda_total_memory_bytes = total_memory
        self.cuda_free_before_load_bytes = free_before

        self.spec = DenseBlockSpec(**adapter_data["spec"])
        self.allowed_keys = frozenset(adapter_data["allowed_tensor_keys"])
        self.shared_keys = frozenset(
            (adapter_data.get("declared_shared_state") or {}).get("tensor_keys", [])
        )
        if self.role == "single" and not (
            self.spec.owns_embeddings and self.spec.owns_final_norm_head
        ):
            raise ValueError("single role must own embeddings and final norm/head")

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
        module, reader = self._selective_load(whole_iter)
        torch.cuda.synchronize(self.device)
        self.cuda_allocated_after_weights = torch.cuda.memory_allocated(self.device)
        self.process_after_source_load = _status()
        gemma_weight.iter_weights = whole_sentinel
        try:
            # Sentinel now active: prove no further checkpoint access occurs
            # during context/graph setup.
            self.block = module
            self.fetched_keys = list(reader.fetched_keys)
            self.fetched_bytes = reader.fetched_bytes
            self.checkpoint_bytes_selected = reader.selected_bytes
            self.largest_raw_tensor_bytes = reader.largest_raw_tensor_bytes
            self.host_staging_total_bytes_processed = reader.fetched_bytes
            self.host_staging_peak_live_tensor_bytes = (
                reader.host_staging_peak_live_tensor_bytes
            )
            self.host_staging_current_bytes = reader.host_staging_current_bytes
            self.page_cache_advisory_calls = reader.page_cache_advisory_calls
            self.safetensors_mapping_open_count = reader.mapping_open_count
            self.safetensors_mapping_close_count = reader.mapping_close_count
            self.memory_snapshots["P1_source_load_complete"] = snapshot_host_memory()
            block_state = self.block.state_dict()
            self.resident_device_bytes = sum(
                _tensor_bytes(t) for t in block_state.values()
            )
            (
                self.persistent_host_model_bytes,
                self._host_resident_tensor_keys,
            ) = _host_resident_bytes(block_state)
            self.cpu_owned_decoder_layers = _cpu_owned_decoder_layers(
                self.block.global_layer_ids, self.block.layers
            )
            self.ctx, self.state_ownership = self._setup_context()
            self.decode_graph = None  # dense eager decode; graphs are optional
            self.reset_session_state()
            torch.cuda.synchronize(self.device)
            self.cuda_allocated_after_realization = torch.cuda.memory_allocated(
                self.device
            )
            self.cuda_allocated_after_runtime_initialization = (
                self.cuda_allocated_after_realization
            )
            self.cuda_free_after_runtime_initialization = torch.cuda.mem_get_info(
                self.device
            )[0]
        finally:
            gemma_weight.iter_weights = whole_iter

        self.load_elapsed_seconds = time.perf_counter() - started
        self.process_after_realization = _status()
        self.detached = True  # dense: host staging released at end of load
        self.detach_report = {
            "released_bytes": self.host_staging_total_bytes_processed,
            "dead_staging_tensors": 0,
            "policy": "bounded-per-tensor-source-no-persistent-host-mirror",
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
        return _replace(
            full_config, attention_groups=tuple(groups), num_layers=len(owned)
        )

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
                    self.full_config.vocab_size,
                    self.full_config.hidden_size,
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
            if getattr(self.full_config.attention_group_for_layer(gid), "k_eq_v", False)
        }

        def _rename(key: str) -> str | None:
            return _rename_language_key(
                "language_model." + key.removeprefix("model.language_model.")
            )

        k_eq_v_k_keys = frozenset(
            key
            for key in planned
            if key.endswith(".self_attn.k_proj.weight")
            and (match := _layer_re.search(key)) is not None
            and int(match.group(1)) in k_eq_v_layers
        )
        loaded, self.fusion_source_peak_live_bytes = materialize_dense_state(
            expected_state=state,
            reader=reader,
            device=self.device,
            rename_key=_rename,
            merge_rule_for=_merge_rule_for,
            k_eq_v_k_keys=k_eq_v_k_keys,
        )
        modules.load_state_dict(loaded)
        if loaded:
            raise RuntimeError(f"unconsumed materialized state: {list(loaded)[:5]}")
        return modules, reader

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
                    self.block.global_layer_ids[i] for i in kv_local_layer_ids
                ],
                "total_block_local_state_bytes": kv_bytes,
            },
        )

    # -- execution -----------------------------------------------------

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.block.embed_tokens.forward(input_ids)

    def forward_layers(self, hidden, residual=None):
        return execute_dense_layer_sequence(self.block.layers, hidden), None

    def finalize(self, hidden, residual=None):
        final = self.block.norm.forward(hidden)
        return final

    def full_bf16_logits(self, final: torch.Tensor) -> torch.Tensor:
        # Tied lm_head: full [sequence, vocab] BF16 logits via the shared
        # embedding table, softcapped.  The GEMM shape, dtype, and softcap
        # placement are frozen R6 semantics and must not change: every
        # consumer that needs logits computes the SAME full BF16 tensor.
        weight = self.block.embed_tokens.weight  # [vocab, hidden]
        logits = final @ weight.t()
        cap = self.full_config.final_logit_softcapping
        if cap is not None:
            if self._softcap_mode == "inplace":
                logits = softcap_inplace(logits, cap, probe=self._phase_probe)
            elif self._softcap_mode == "legacy":
                logits = softcap_legacy(logits, cap, probe=self._phase_probe)
            else:
                raise ValueError(f"unknown softcap mode: {self._softcap_mode!r}")
        return logits

    def final_row_logits(self, final: torch.Tensor) -> torch.Tensor:
        # Consumer-facing final-token logits: identical BF16 GEMM + softcap
        # as lm_head_logits(), but only the final row is promoted to FP32.
        # BF16->FP32 conversion is elementwise and exact, so the returned
        # row is bit-identical to lm_head_logits(final)[-1] while the
        # unused earlier rows are never materialized as a [seq, vocab]
        # FP32 tensor (a full 32-row promotion costs 32 MiB at vocab
        # 262,144 and is a pure allocation-lifetime waste: the R6 control
        # consumes only the last row).
        return self.full_bf16_logits(final)[-1].float()

    def lm_head_logits(self, final: torch.Tensor) -> torch.Tensor:
        # Legacy whole-tensor promotion [sequence, vocab] FP32.  Retained
        # unchanged for direct callers and equivalence proof against
        # final_row_logits(); the prefill/decode execution paths no longer
        # route through it.
        return self.full_bf16_logits(final).float()

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
        """Execute one matched prefill over this role's complete ownership."""
        if self.role in ("first", "single"):
            batch = self._prepare(
                start=start, token_count=len(token_ids), phase="prefill"
            )
            batch.input_ids.copy_(
                torch.tensor(token_ids, dtype=torch.int32, device=self.device)
            )
            with self.ctx.forward_batch(batch):
                _notify(self._phase_probe, "embedding_complete")
                hidden = self.embed(batch.input_ids)
                hidden, _ = self.forward_layers(hidden)
                _notify(self._phase_probe, "layers_complete")
                if self.role == "single":
                    final = self.finalize(hidden, None)
                    _notify(self._phase_probe, "final_norm_complete")
                    logits = self.final_row_logits(final)
                    _notify(self._phase_probe, "lm_head_gemm_complete")
            if self.role == "single":
                _notify(self._phase_probe, "final_row_fp32_complete")
                token = int(torch.argmax(logits, dim=-1).item())
                _notify(self._phase_probe, "argmax_complete")
                if self._logit_capture is not None:
                    self._logit_capture.capture(logits)
                return token, logits.detach()
            return hidden, None
        hidden = hidden_or_ids
        batch = self._prepare(start=start, token_count=hidden.shape[0], phase="prefill")
        with self.ctx.forward_batch(batch):
            hidden, _ = self.forward_layers(hidden)
            if self.role != "last":
                return hidden, None
            final = self.finalize(hidden, None)
            logits = self.final_row_logits(final)
        token = int(torch.argmax(logits, dim=-1).item())
        if self._logit_capture is not None:
            self._logit_capture.capture(logits)
        return token, logits.detach()

    @torch.inference_mode()
    def decode(self, token_or_hidden, position: int):
        """Execute one matched decode step over this role's ownership."""
        if self.role in ("first", "single"):
            token_id = token_or_hidden
            batch = self._prepare(start=position, token_count=1, phase="decode")
            batch.input_ids.fill_(int(token_id))
            with self.ctx.forward_batch(batch):
                _notify(self._phase_probe, "embedding_complete")
                hidden = self.embed(batch.input_ids)
                hidden, _ = self.forward_layers(hidden)
                _notify(self._phase_probe, "layers_complete")
                if self.role == "single":
                    final = self.finalize(hidden, None)
                    _notify(self._phase_probe, "final_norm_complete")
                    logits = self.final_row_logits(final)
                    _notify(self._phase_probe, "lm_head_gemm_complete")
            if self.role == "single":
                token = int(torch.argmax(logits, dim=-1).item())
                if self._logit_capture is not None:
                    self._logit_capture.capture(logits)
                return token, logits
            return hidden, None
        hidden = token_or_hidden
        batch = self._prepare(start=position, token_count=1, phase="decode")
        with self.ctx.forward_batch(batch):
            hidden, _ = self.forward_layers(hidden)
            if self.role != "last":
                return hidden, None
            final = self.finalize(hidden, None)
            logits = self.final_row_logits(final)
        token = int(torch.argmax(logits, dim=-1).item())
        if self._logit_capture is not None:
            self._logit_capture.capture(logits)
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
        rusage_after = _rusage_counters()
        return {
            "pid": os.getpid(),
            "role": self.role,
            "global_layer_ids": list(self.block.global_layer_ids),
            "fetched_keys": len(set(self.fetched_keys)),
            "fetched_bytes": self.fetched_bytes,
            "checkpoint_bytes_selected": self.checkpoint_bytes_selected,
            "checkpoint_bytes_processed": self.fetched_bytes,
            "largest_individual_raw_tensor_bytes": self.largest_raw_tensor_bytes,
            "unexpected_checkpoint_keys": sorted(
                set(self.fetched_keys) - self.allowed_keys - self.shared_keys
            ),
            "whole_shard_sentinel_calls": self.whole_shard_sentinel_calls,
            "state_ownership": self.state_ownership,
            "resident_device_bytes": self.resident_device_bytes,
            # Historical reports retain host_peak_staging_bytes unchanged.
            # New reports define it truthfully as simultaneous live source bytes.
            "host_peak_staging_bytes": self.host_staging_peak_live_tensor_bytes,
            "host_staging_total_bytes_processed": (
                self.host_staging_total_bytes_processed
            ),
            "host_staging_peak_live_tensor_bytes": (
                self.host_staging_peak_live_tensor_bytes
            ),
            "fusion_source_peak_live_bytes": self.fusion_source_peak_live_bytes,
            "host_staging_current_bytes": self.host_staging_current_bytes,
            "host_lifecycle_snapshots": self.memory_snapshots,
            # Mechanically derived from the realized module state_dict's tensor
            # devices (see _host_resident_bytes), not asserted.
            "persistent_host_model_bytes": self.persistent_host_model_bytes,
            "host_resident_tensor_keys": self._host_resident_tensor_keys,
            "unexplained_persistent_host_mirror_bytes": (
                self.persistent_host_model_bytes + self.host_staging_current_bytes
            ),
            "resident_only": (
                self.persistent_host_model_bytes == 0
                and self.host_staging_current_bytes == 0
            ),
            "cpu_weight_offload": self.persistent_host_model_bytes > 0,
            "cpu_owned_decoder_layers": self.cpu_owned_decoder_layers,
            "semantic_boundaries": semantic_boundaries_for_role(self.role),
            "tied_embedding_materializations": (
                1 if self.block.embed_tokens is not None else 0
            ),
            "single_tied_embedding_storage": (
                self.role == "single" and self.block.embed_tokens is not None
            ),
            "kv_state_bytes": self.state_ownership["kv_cache_allocated_bytes"],
            "cuda_allocated_bytes": torch.cuda.memory_allocated(self.device),
            "cuda_allocated_before_load_bytes": self.cuda_allocated_before_load,
            "cuda_allocated_after_weights_bytes": self.cuda_allocated_after_weights,
            "cuda_allocated_after_runtime_initialization_bytes": (
                self.cuda_allocated_after_runtime_initialization
            ),
            "cuda_peak_bytes": torch.cuda.max_memory_allocated(self.device),
            "cuda_total_memory_bytes": self.cuda_total_memory_bytes,
            "cuda_free_before_load_bytes": self.cuda_free_before_load_bytes,
            "cuda_free_after_runtime_initialization_bytes": (
                self.cuda_free_after_runtime_initialization
            ),
            "safetensors_mapping_open_count": self.safetensors_mapping_open_count,
            "safetensors_mapping_close_count": self.safetensors_mapping_close_count,
            "page_cache_advisory_calls": self.page_cache_advisory_calls,
            "process_before": self.process_before,
            "process_after_source_load": self.process_after_source_load,
            "process_after_realization": self.process_after_realization,
            "process_current": _status(),
            "ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "vmstat_delta": {
                key: vm_after[key] - self.vmstat_before[key] for key in vm_after
            },
            "process_fault_swap_delta": {
                key: rusage_after[key] - self.rusage_before[key] for key in rusage_after
            },
            "load_elapsed_seconds": self.load_elapsed_seconds,
            "attention_backend": ATTN_BACKEND,
        }


def _key_adapter_src(module_key: str) -> str:
    return module_key


def _iter_pool_tensors(pool):
    seen = set()
    for value in vars(pool).values():
        if isinstance(value, torch.Tensor) and id(value) not in seen:
            seen.add(id(value))
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, torch.Tensor) and id(item) not in seen:
                    seen.add(id(item))
                    yield item


def _iter_kv_pools(pool):
    for value in vars(pool).values():
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


__all__ = [
    "BOUNDARY_PLANES",
    "HIDDEN_SIZE",
    "GemmaDenseStage",
    "cuda_phase_probe",
    "execute_dense_layer_sequence",
    "materialize_dense_state",
    "semantic_boundaries_for_role",
    "softcap_inplace",
    "softcap_legacy",
]
