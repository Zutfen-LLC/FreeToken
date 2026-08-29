"""InferSwarm Phase-1 P4 overlapped, explicitly host-staged remote decode."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from .inferswarm_resident_bank import SecondaryResidentExpertBank

REMOTE_DECODE_SCHEMA = "freetoken.inferswarm-remote-decode/2"
SPLIT_GPU0_DIAGNOSTIC_SCHEMA = "freetoken.inferswarm-c3-split-gpu0/1"
TRANSPORT_MODE = "host_staged"
REMOTE_MODES = ("overlap", "serialized")
F1_THRESHOLD = 0.25
F2_THRESHOLD = 0.20
F5_THRESHOLD = 0.01


def build_remote_slot_lookup(
    placement, primary_device: torch.device, *, torch_module=torch
) -> torch.Tensor:
    """Build ``[layer, expert] -> remote slot|-1`` from P2's authority."""
    lookup = torch_module.full(
        (placement.num_layers, placement.num_experts),
        -1,
        dtype=torch_module.int32,
        device="cpu",
    )
    for identity in placement.identities_in_rank_order:
        lookup[identity.layer_id, identity.expert_id] = identity.remote_slot
    if int((lookup >= 0).sum().item()) != placement.remote_slots:
        raise RuntimeError(
            "P4 route lookup does not contain every P2 resident slot exactly once"
        )
    return lookup.to(primary_device)


def validate_remote_decode_runtime(config, cache, resident, secondary) -> None:
    """Refuse any resolved runtime outside the narrow canonical P4 experiment."""
    tp_info = getattr(config, "tp_info", None)
    if tp_info is None or int(getattr(tp_info, "size", -1)) != 1:
        raise ValueError("--inferswarm-remote-decode requires tensor parallel size 1")
    if getattr(config, "cuda_graph_max_bs", None) != 0:
        raise ValueError("--inferswarm-remote-decode requires --cuda-graph-max-bs 0")
    if getattr(config, "moe_backend", None) != "offload":
        raise ValueError(
            "--inferswarm-remote-decode requires resolved --moe-backend offload"
        )
    if getattr(config, "inferswarm_remote_mode", "overlap") not in REMOTE_MODES:
        raise ValueError(
            "--inferswarm-remote-mode must be one of " + ", ".join(REMOTE_MODES)
        )
    if getattr(cache, "decode_target", None) != "gpu":
        raise ValueError(
            "--inferswarm-remote-decode requires GPU decode target; CPU/hybrid decode is unsupported"
        )
    if getattr(cache, "cpu_layer_ids", frozenset()):
        raise ValueError("--inferswarm-remote-decode requires zero CPU MoE layers")
    if getattr(cache, "quant_format", None) != "nvfp4":
        raise ValueError(
            "--inferswarm-remote-decode requires native nvfp4 bank layout and Triton backend"
        )
    layout = resident.report.layout
    if (
        layout.quant_format != "nvfp4"
        or layout.nvfp4_backend != "triton"
        or layout.bank_layout != "native_modelopt_nvfp4"
    ):
        raise ValueError(
            "--inferswarm-remote-decode requires the P2 native NVFP4/Triton resident layout"
        )
    if tuple(cache.bank_schema) != tuple(layout.bank_schema):
        raise ValueError(
            "P2 resident bank schema disagrees with the local production layout"
        )
    local_views, remote_views = cache.bank_views(), resident.bank_views()
    if len(local_views) != len(remote_views):
        raise ValueError(
            "P2 resident bank count disagrees with the local production layout"
        )
    for name, local, remote in zip(layout.bank_schema, local_views, remote_views):
        if local.dtype != remote.dtype or tuple(local.shape[1:]) != tuple(
            remote.shape[1:]
        ):
            raise ValueError(
                f"P2 resident bank {name!r} disagrees with the local production row layout"
            )
        if remote.device.type != "cuda" or remote.device.index != int(
            secondary.secondary.visible_ordinal
        ):
            raise ValueError(
                f"P2 resident bank {name!r} is on {remote.device}, not the configured secondary"
            )


@dataclass
class LayerCounters:
    total_router_selections: int = 0
    selected_for_gpu1: int = 0
    executed_on_gpu1: int = 0
    executed_on_gpu0: int = 0
    remote_selected_layer_calls: int = 0
    expected_remote_dispatches: int = 0
    remote_dispatches: int = 0
    dispatch_mismatch_layer_calls: int = 0
    hypothetical_streamed_remote_weight_bytes: int = 0
    explicit_failure: int = 0
    fallback_elsewhere: int = 0
    combine_operations: int = 0

    def as_dict(self, layer_id: int) -> dict[str, int]:
        return {"layer_id": layer_id, **self.__dict__}


@dataclass
class RemoteDecodeCounters:
    num_layers: int
    layers: list[LayerCounters] = field(init=False)
    failure_events: int = 0
    prefill_remote_dispatches: int = 0

    def __post_init__(self) -> None:
        self.layers = [LayerCounters() for _ in range(self.num_layers)]

    def reset(self) -> None:
        self.__post_init__()
        self.failure_events = 0
        self.prefill_remote_dispatches = 0

    def aggregate(self) -> dict[str, int]:
        keys = tuple(LayerCounters.__dataclass_fields__)
        out = {key: sum(getattr(layer, key) for layer in self.layers) for key in keys}
        out.update(
            failure_events=self.failure_events,
            prefill_remote_dispatches=self.prefill_remote_dispatches,
        )
        return out


@dataclass
class TransferByteCounters:
    gpu0_to_host_activation: int = 0
    gpu0_to_host_routing_weights: int = 0
    gpu0_to_host_routing_ids: int = 0
    host_to_gpu1_activation: int = 0
    host_to_gpu1_routing_weights: int = 0
    host_to_gpu1_routing_ids: int = 0
    host_to_gpu1_expert_weights: int = 0
    gpu1_to_host_returned_partial: int = 0
    host_to_gpu0_returned_partial: int = 0

    def reset(self) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "gpu0_to_host": {
                "activation": self.gpu0_to_host_activation,
                "routing_weights": self.gpu0_to_host_routing_weights,
                "routing_ids": self.gpu0_to_host_routing_ids,
            },
            "host_to_gpu1": {
                "activation": self.host_to_gpu1_activation,
                "routing_weights": self.host_to_gpu1_routing_weights,
                "routing_ids": self.host_to_gpu1_routing_ids,
                "expert_weights": self.host_to_gpu1_expert_weights,
            },
            "gpu1_to_host": {"returned_partial": self.gpu1_to_host_returned_partial},
            "host_to_gpu0": {"returned_partial": self.host_to_gpu0_returned_partial},
        }


@dataclass
class PendingRemoteOperation:
    slot_index: int
    generation: int
    tokens: int
    layer_id: int
    completion_event: Any
    diagnostics: Any = None
    completion_recorded: bool = False
    finished: bool = False
    released: bool = False
    timing_values: dict[str, dict[str, Any]] = field(default_factory=dict)
    transfer_bytes: dict[str, Any] = field(default_factory=dict)


@dataclass
class _BufferSlot:
    host_activation: torch.Tensor
    host_slots: torch.Tensor
    host_weights: torch.Tensor
    host_partial: torch.Tensor
    gpu1_activation: torch.Tensor
    gpu1_slots: torch.Tensor
    gpu1_weights: torch.Tensor
    gpu0_partial: torch.Tensor
    stage_ready_event: Any
    completion_event: Any
    return_consumed_event: Any
    timing_events: dict[str, Any]
    generation: int = 0
    inflight: bool = False
    return_pending: bool = False


class RemoteTransport(Protocol):
    def submit(self, layer, cache, hidden_states, routing_weights, remote_slot_ids): ...

    def finish(
        self,
        pending,
        *,
        before_return_copy: Callable[[], None] | None = None,
        after_return_copy: Callable[[], None] | None = None,
    ) -> torch.Tensor: ...

    def drain(self, pending) -> None: ...

    def release(self, pending) -> None: ...

    def report(self) -> dict[str, Any]: ...


class HostStagedRemoteTransport:
    """Persistent two-slot host-staged transport with one GPU1 CUDA stream."""

    _RING_SIZE = 2

    def __init__(
        self,
        *,
        primary_device: torch.device,
        secondary_device: torch.device,
        max_tokens: int,
        hidden_size: int,
        top_k: int,
        hidden_dtype: torch.dtype,
        resident_bank: SecondaryResidentExpertBank,
        timing_enabled: bool = False,
        torch_module=torch,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("P4 staging capacity must be positive")
        self.primary_device, self.secondary_device = primary_device, secondary_device
        self.primary_ordinal, self.secondary_ordinal = (
            int(primary_device.index),
            int(secondary_device.index),
        )
        self.max_tokens, self.hidden_size, self.top_k = max_tokens, hidden_size, top_k
        self.hidden_dtype, self.resident_bank = hidden_dtype, resident_bank
        self.timing_enabled, self._torch = bool(timing_enabled), torch_module
        self.transfer_bytes = TransferByteCounters()
        self._next_slot = self._buffer_reuse_waits = 0
        cuda, primary, secondary = (
            torch_module.cuda,
            self.primary_ordinal,
            self.secondary_ordinal,
        )
        before_allocated = int(cuda.memory_allocated(secondary))
        before_reserved = int(cuda.memory_reserved(secondary))
        self._slots: list[_BufferSlot] = []
        try:
            cuda.set_device(secondary)
            self.stream = cuda.Stream(device=secondary_device)
            for _ in range(self._RING_SIZE):
                host_activation = torch_module.empty(
                    (max_tokens, hidden_size), dtype=hidden_dtype, pin_memory=True
                )
                host_slots = torch_module.empty(
                    (max_tokens, top_k), dtype=torch_module.int32, pin_memory=True
                )
                host_weights = torch_module.empty(
                    (max_tokens, top_k), dtype=torch_module.float32, pin_memory=True
                )
                host_partial = torch_module.empty(
                    (max_tokens, hidden_size), dtype=hidden_dtype, pin_memory=True
                )
                gpu1_activation = torch_module.empty(
                    (max_tokens, hidden_size),
                    dtype=hidden_dtype,
                    device=secondary_device,
                )
                gpu1_slots = torch_module.empty(
                    (max_tokens, top_k),
                    dtype=torch_module.int32,
                    device=secondary_device,
                )
                gpu1_weights = torch_module.empty(
                    (max_tokens, top_k),
                    dtype=torch_module.float32,
                    device=secondary_device,
                )
                cuda.set_device(primary)
                gpu0_partial = torch_module.empty(
                    (max_tokens, hidden_size), dtype=hidden_dtype, device=primary_device
                )
                cuda.set_device(secondary)
                timing_events = (
                    {
                        name: cuda.Event(enable_timing=True)
                        for name in (
                            "stage_start",
                            "stage_end",
                            "gpu1_branch_start",
                            "gpu1_h2d_end",
                            "gpu1_exec_end",
                            "gpu1_d2h_end",
                        )
                    }
                    if self.timing_enabled
                    else {}
                )
                self._slots.append(
                    _BufferSlot(
                        host_activation,
                        host_slots,
                        host_weights,
                        host_partial,
                        gpu1_activation,
                        gpu1_slots,
                        gpu1_weights,
                        gpu0_partial,
                        timing_events.get("stage_end") or cuda.Event(),
                        timing_events.get("gpu1_d2h_end") or cuda.Event(),
                        cuda.Event(),
                        timing_events,
                    )
                )
            self._gpu1_allocated_after_init = int(cuda.memory_allocated(secondary))
            self._gpu1_reserved_after_init = int(cuda.memory_reserved(secondary))
        finally:
            cuda.set_device(primary)
        if int(cuda.current_device()) != primary:
            raise RuntimeError(
                "P4 transport initialization failed to restore the primary device"
            )
        self._gpu1_allocated_before_init = before_allocated
        self._gpu1_reserved_before_init = before_reserved

    @staticmethod
    def _active(tensor: torch.Tensor, tokens: int) -> torch.Tensor:
        return tensor[:tokens]

    @staticmethod
    def _timing(value: float, source: str) -> dict[str, Any]:
        return {"status": "valid", "value_ms": float(value), "source": source}

    @classmethod
    def _elapsed(cls, start, end, source: str) -> dict[str, Any]:
        return cls._timing(float(start.elapsed_time(end)), source)

    def _payload_bytes(self, tokens: int) -> tuple[int, int, int, int]:
        activation = tokens * self.hidden_size * self.hidden_dtype.itemsize
        return (
            activation,
            tokens * self.top_k * torch.float32.itemsize,
            tokens * self.top_k * torch.int32.itemsize,
            activation,
        )

    def _validate_payload(self, hidden, weights, slots) -> int:
        tokens = int(hidden.shape[0])
        if not 1 <= tokens <= self.max_tokens:
            raise RuntimeError(
                f"P4 remote payload has {tokens} token rows; capacity is {self.max_tokens}"
            )
        if tuple(hidden.shape) != (tokens, self.hidden_size):
            raise RuntimeError("P4 activation shape disagrees with configured geometry")
        if tuple(weights.shape) != (tokens, self.top_k) or tuple(slots.shape) != (
            tokens,
            self.top_k,
        ):
            raise RuntimeError(
                "P4 routing payload shape disagrees with configured geometry"
            )
        if (
            hidden.device != self.primary_device
            or weights.device != self.primary_device
            or slots.device != self.primary_device
        ):
            raise RuntimeError(
                "P4 payload is not on the configured primary CUDA device"
            )
        if hidden.dtype != self.hidden_dtype:
            raise RuntimeError(
                "P4 activation dtype disagrees with configured runtime dtype"
            )
        if weights.dtype != torch.float32 or slots.dtype != torch.int32:
            raise RuntimeError(
                "P4 routing payload requires float32 weights and int32 slot IDs"
            )
        return tokens

    def _acquire_slot(self) -> tuple[int, _BufferSlot]:
        index = self._next_slot
        self._next_slot = (index + 1) % len(self._slots)
        slot = self._slots[index]
        if slot.inflight:
            raise RuntimeError(
                "P4 transport permits one operation per buffer generation"
            )
        if slot.return_pending:
            if not slot.return_consumed_event.query():
                self._buffer_reuse_waits += 1
                slot.return_consumed_event.synchronize()
            slot.return_pending = False
        slot.generation += 1
        slot.inflight = True
        return index, slot

    def submit(self, layer, cache, hidden_states, routing_weights, remote_slot_ids):
        tokens = self._validate_payload(hidden_states, routing_weights, remote_slot_ids)
        index, slot = self._acquire_slot()
        cuda, primary, secondary = (
            self._torch.cuda,
            self.primary_ordinal,
            self.secondary_ordinal,
        )
        act_b, weight_b, id_b, partial_b = self._payload_bytes(tokens)
        layer_id = int(getattr(layer, "layer_id", -1))
        diagnostics = getattr(layer, "inferswarm_correctness_diagnostics", None)
        pending = PendingRemoteOperation(
            index,
            slot.generation,
            tokens,
            layer_id,
            slot.completion_event,
            diagnostics,
        )
        pending.transfer_bytes = {
            "gpu0_to_host": {"activation": 0, "routing_weights": 0, "routing_ids": 0},
            "host_to_gpu1": {
                "activation": 0,
                "routing_weights": 0,
                "routing_ids": 0,
                "expert_weights": 0,
            },
            "gpu1_to_host": {"returned_partial": 0},
            "host_to_gpu0": {"returned_partial": 0},
        }
        secondary_enqueued = False
        try:
            cuda.set_device(primary)
            pstream = cuda.current_stream(self.primary_device)
            if diagnostics is not None:
                diagnostics.capture_transport_tensor(
                    layer_id, "gpu0_source_hidden_activation", hidden_states
                )
                diagnostics.capture_transport_tensor(
                    layer_id, "gpu0_source_routing_weights", routing_weights
                )
                diagnostics.capture_transport_tensor(
                    layer_id, "expected_remote_slot_ids", remote_slot_ids
                )
            if self.timing_enabled:
                slot.timing_events["stage_start"].record(pstream)
            self._active(slot.host_activation, tokens).copy_(
                hidden_states, non_blocking=True
            )
            self._active(slot.host_weights, tokens).copy_(
                routing_weights, non_blocking=True
            )
            self._active(slot.host_slots, tokens).copy_(
                remote_slot_ids, non_blocking=True
            )
            slot.stage_ready_event.record(pstream)
            self.transfer_bytes.gpu0_to_host_activation += act_b
            self.transfer_bytes.gpu0_to_host_routing_weights += weight_b
            self.transfer_bytes.gpu0_to_host_routing_ids += id_b
            pending.transfer_bytes["gpu0_to_host"].update(
                activation=act_b, routing_weights=weight_b, routing_ids=id_b
            )
            tic = time.perf_counter_ns()
            slot.stage_ready_event.synchronize()
            if diagnostics is not None:
                diagnostics.capture_transport_tensor(
                    layer_id,
                    "pinned_host_staged_activation",
                    self._active(slot.host_activation, tokens),
                )
                diagnostics.capture_transport_tensor(
                    layer_id,
                    "pinned_host_routing_weights",
                    self._active(slot.host_weights, tokens),
                )
                diagnostics.capture_transport_tensor(
                    layer_id,
                    "pinned_host_remote_slot_ids",
                    self._active(slot.host_slots, tokens),
                )
            pending.timing_values["gpu0_to_host_staging_host_wait"] = self._timing(
                (time.perf_counter_ns() - tic) / 1e6, "host_monotonic"
            )
            if self.timing_enabled:
                pending.timing_values["gpu0_to_host_activation_routing"] = (
                    self._elapsed(
                        slot.timing_events["stage_start"],
                        slot.timing_events["stage_end"],
                        "cuda_event_gpu0",
                    )
                )
            tic = time.perf_counter_ns()
            cuda.set_device(secondary)
            with cuda.stream(self.stream):
                events = slot.timing_events
                if self.timing_enabled:
                    events["gpu1_branch_start"].record(self.stream)
                activation = self._active(slot.gpu1_activation, tokens)
                weights = self._active(slot.gpu1_weights, tokens)
                slots = self._active(slot.gpu1_slots, tokens)
                activation.copy_(
                    self._active(slot.host_activation, tokens), non_blocking=True
                )
                weights.copy_(
                    self._active(slot.host_weights, tokens), non_blocking=True
                )
                slots.copy_(self._active(slot.host_slots, tokens), non_blocking=True)
                if diagnostics is not None:
                    diagnostics.capture_transport_tensor(
                        layer_id, "gpu1_activation_after_h2d", activation
                    )
                    diagnostics.capture_transport_tensor(
                        layer_id, "gpu1_routing_weights", weights
                    )
                    diagnostics.capture_transport_tensor(
                        layer_id, "gpu1_remote_slot_ids", slots
                    )
                secondary_enqueued = True
                self.transfer_bytes.host_to_gpu1_activation += act_b
                self.transfer_bytes.host_to_gpu1_routing_weights += weight_b
                self.transfer_bytes.host_to_gpu1_routing_ids += id_b
                pending.transfer_bytes["host_to_gpu1"].update(
                    activation=act_b, routing_weights=weight_b, routing_ids=id_b
                )
                if self.timing_enabled:
                    events["gpu1_h2d_end"].record(self.stream)
                partial = layer._expert_gemm(
                    cache,
                    activation,
                    weights,
                    slots,
                    views=self.resident_bank.bank_views(),
                    n=None,
                    alphas=self.resident_bank.alpha_views(),
                    is_prefill=False,
                )
                if tuple(partial.shape) != (tokens, self.hidden_size):
                    raise RuntimeError(
                        "P4 remote result shape disagrees with configured geometry"
                    )
                if (
                    partial.dtype != self.hidden_dtype
                    or partial.device != self.secondary_device
                ):
                    raise RuntimeError(
                        "P4 remote result dtype/device disagrees with secondary"
                    )
                if self.timing_enabled:
                    events["gpu1_exec_end"].record(self.stream)
                if diagnostics is not None:
                    diagnostics.capture_transport_tensor(
                        layer_id, "gpu1_remote_partial_before_d2h", partial
                    )
                self._active(slot.host_partial, tokens).copy_(
                    partial, non_blocking=True
                )
                self.transfer_bytes.gpu1_to_host_returned_partial += partial_b
                pending.transfer_bytes["gpu1_to_host"]["returned_partial"] = partial_b
                slot.completion_event.record(self.stream)
                pending.completion_recorded = True
            pending.timing_values["host_remote_submit_control"] = self._timing(
                (time.perf_counter_ns() - tic) / 1e6, "host_monotonic_enqueue_cost"
            )
            return pending
        except Exception:
            try:
                cuda.set_device(secondary)
                if secondary_enqueued:
                    self.stream.synchronize()
            finally:
                slot.inflight = False
                cuda.set_device(primary)
            raise
        finally:
            cuda.set_device(primary)

    def finish(
        self,
        pending,
        *,
        before_return_copy: Callable[[], None] | None = None,
        after_return_copy: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        slot = self._slots[pending.slot_index]
        if (
            pending.generation != slot.generation
            or not slot.inflight
            or pending.finished
        ):
            raise RuntimeError("P4 pending operation has a stale generation")
        cuda, primary, secondary = (
            self._torch.cuda,
            self.primary_ordinal,
            self.secondary_ordinal,
        )
        partial_b = self._payload_bytes(pending.tokens)[3]
        primary_return_enqueued = False
        try:
            cuda.set_device(secondary)
            tic = time.perf_counter_ns()
            pending.completion_event.synchronize()
            diagnostics = pending.diagnostics
            if diagnostics is not None:
                diagnostics.capture_transport_tensor(
                    pending.layer_id,
                    "pinned_host_returned_partial",
                    self._active(slot.host_partial, pending.tokens),
                )
            pending.timing_values["host_remote_join_wait"] = self._timing(
                (time.perf_counter_ns() - tic) / 1e6, "host_monotonic"
            )
            if self.timing_enabled:
                e = slot.timing_events
                pending.timing_values.update(
                    host_to_gpu1_payload_h2d=self._elapsed(
                        e["gpu1_branch_start"], e["gpu1_h2d_end"], "cuda_event_gpu1"
                    ),
                    remote_expert_execution=self._elapsed(
                        e["gpu1_h2d_end"], e["gpu1_exec_end"], "cuda_event_gpu1"
                    ),
                    gpu1_to_host_partial_d2h=self._elapsed(
                        e["gpu1_exec_end"], e["gpu1_d2h_end"], "cuda_event_gpu1"
                    ),
                    complete_gpu1_branch=self._elapsed(
                        e["gpu1_branch_start"], e["gpu1_d2h_end"], "cuda_event_gpu1"
                    ),
                )
            cuda.set_device(primary)
            if before_return_copy:
                before_return_copy()
            out = self._active(slot.gpu0_partial, pending.tokens)
            out.copy_(
                self._active(slot.host_partial, pending.tokens), non_blocking=True
            )
            if diagnostics is not None:
                diagnostics.capture_transport_tensor(
                    pending.layer_id, "gpu0_returned_remote_partial", out
                )
            primary_return_enqueued = True
            self.transfer_bytes.host_to_gpu0_returned_partial += partial_b
            pending.transfer_bytes["host_to_gpu0"]["returned_partial"] = partial_b
            if after_return_copy:
                after_return_copy()
            # The host source is consumed when the asynchronous H2D copy completes, but
            # gpu0_partial is still read by the caller's combine. ``release`` records the
            # reusable event only after that final consumer has been enqueued.
            pending.finished = True
            return out
        except Exception:
            try:
                if primary_return_enqueued:
                    # Failure cleanup may block. Once the returned-partial H2D has been
                    # enqueued, prove GPU0 no longer consumes the slot buffers before making
                    # the generation reusable. The successful path remains asynchronous.
                    cuda.set_device(primary)
                    cuda.current_stream(self.primary_device).synchronize()
                else:
                    cuda.set_device(secondary)
                    self.stream.synchronize()
            finally:
                slot.inflight = False
                cuda.set_device(primary)
            raise
        finally:
            cuda.set_device(primary)

    def drain(self, pending) -> None:
        slot, cuda = self._slots[pending.slot_index], self._torch.cuda
        try:
            if pending.finished:
                # A returned partial may still have an H2D copy or consumer queued on
                # GPU0. Failure cleanup is allowed to block so no buffer escapes live.
                cuda.set_device(self.primary_ordinal)
                cuda.current_stream(self.primary_device).synchronize()
            else:
                cuda.set_device(self.secondary_ordinal)
                (
                    pending.completion_event.synchronize()
                    if pending.completion_recorded
                    else self.stream.synchronize()
                )
        finally:
            slot.inflight, pending.finished, pending.released = False, True, True
            cuda.set_device(self.primary_ordinal)

    def release(self, pending) -> None:
        """Release a finished slot after its GPU0 result consumer is enqueued."""
        slot, cuda = self._slots[pending.slot_index], self._torch.cuda
        if pending.generation != slot.generation or pending.released:
            raise RuntimeError("P4 pending operation release has a stale generation")
        if not pending.finished or not slot.inflight:
            raise RuntimeError("P4 pending operation cannot be released before finish")
        try:
            cuda.set_device(self.primary_ordinal)
            slot.return_consumed_event.record(cuda.current_stream(self.primary_device))
            slot.return_pending = True
            slot.inflight = False
            pending.released = True
        finally:
            cuda.set_device(self.primary_ordinal)

    def execute(self, layer, cache, hidden_states, routing_weights, remote_slot_ids):
        """Serialized P3-equivalent compatibility wrapper."""
        pending = self.submit(
            layer, cache, hidden_states, routing_weights, remote_slot_ids
        )
        result = self.finish(pending)
        self.release(pending)
        return result

    def reset_counters(self) -> None:
        self.transfer_bytes.reset()
        self._buffer_reuse_waits = 0

    def report(self) -> dict[str, Any]:
        payload = {
            "activation": self.max_tokens
            * self.hidden_size
            * self.hidden_dtype.itemsize,
            "routing_ids": self.max_tokens * self.top_k * torch.int32.itemsize,
            "routing_weights": self.max_tokens * self.top_k * torch.float32.itemsize,
            "returned_partial": self.max_tokens
            * self.hidden_size
            * self.hidden_dtype.itemsize,
        }
        return {
            "mode": TRANSPORT_MODE,
            "persistent_secondary_stream": True,
            "secondary_stream_device": str(self.secondary_device),
            "ring_slots": len(self._slots),
            "buffer_reuse_rule": "generation guard plus GPU0 return-consumed event",
            "buffer_reuse_waits": self._buffer_reuse_waits,
            "persistent_capacity_tokens": self.max_tokens,
            "payload_capacity_bytes_per_slot": payload,
            "pinned_host_staging_bytes": sum(payload.values()) * len(self._slots),
            "gpu1_persistent_payload_bytes": sum(
                payload[k] for k in ("activation", "routing_ids", "routing_weights")
            )
            * len(self._slots),
            "gpu0_persistent_return_bytes": payload["returned_partial"]
            * len(self._slots),
            "steady_state_transfer_bytes": self.transfer_bytes.as_dict(),
            "gpu1_allocator": {
                "allocated_before_init": self._gpu1_allocated_before_init,
                "allocated_after_init": self._gpu1_allocated_after_init,
                "allocated_delta": self._gpu1_allocated_after_init
                - self._gpu1_allocated_before_init,
                "reserved_before_init": self._gpu1_reserved_before_init,
                "reserved_after_init": self._gpu1_reserved_after_init,
                "reserved_delta": self._gpu1_reserved_after_init
                - self._gpu1_reserved_before_init,
            },
        }


def evaluate_mechanism_gates(
    *,
    gpu0_expert_cache_bytes: int,
    gpu1_expert_cache_bytes: int,
    counters: dict[str, int],
    transfer_bytes: dict[str, Any],
) -> dict[str, Any]:
    """Pure F1/F2/F3/F5/F6 evaluation for one reset-delimited window."""
    combined = gpu0_expert_cache_bytes + gpu1_expert_cache_bytes
    f1 = gpu1_expert_cache_bytes / combined if combined else None
    total, remote = counters["total_router_selections"], counters["executed_on_gpu1"]
    f2 = remote / total if total else None
    h2g = transfer_bytes["host_to_gpu1"]
    steady = sum(
        int(h2g[name])
        for name in ("activation", "routing_weights", "routing_ids", "expert_weights")
    )
    hypothetical = counters["hypothetical_streamed_remote_weight_bytes"]
    f5 = steady / hypothetical if hypothetical else None
    f3_pass = (
        counters["dispatch_mismatch_layer_calls"] == 0
        and counters["expected_remote_dispatches"] == counters["remote_dispatches"]
    )
    f6_pass = (
        counters["selected_for_gpu1"] == counters["executed_on_gpu1"]
        and counters["explicit_failure"] == 0
        and counters["fallback_elsewhere"] == 0
    )
    return {
        "F1": {
            "threshold": F1_THRESHOLD,
            "gpu1_expert_cache_bytes": gpu1_expert_cache_bytes,
            "combined_expert_cache_bytes": combined,
            "gpu1_expert_byte_fraction": f1,
            "passed": f1 is not None and f1 >= F1_THRESHOLD,
        },
        "F2": {
            "threshold": F2_THRESHOLD,
            "executed_on_gpu1": remote,
            "total_router_selections": total,
            "gpu1_execution_fraction": f2,
            "passed": f2 is not None and f2 >= F2_THRESHOLD,
            "scope": "current reset-delimited workload window; classes are not averaged",
        },
        "F3": {
            "expected_dispatches": counters["expected_remote_dispatches"],
            "actual_dispatches": counters["remote_dispatches"],
            "per_layer_call_mismatches": counters["dispatch_mismatch_layer_calls"],
            "passed": f3_pass,
        },
        "F4": {"status": "evaluated_by_correctness", "passed": None},
        "F5": {
            "threshold": F5_THRESHOLD,
            "steady_state_host_to_gpu1_bytes": steady,
            "steady_state_expert_weight_bytes_host_to_gpu1": h2g["expert_weights"],
            "hypothetical_streamed_remote_weight_bytes": hypothetical,
            "ratio": f5,
            "passed": f5 is not None and f5 < F5_THRESHOLD,
            "denominator_rule": "unique remote identities per layer/step x bytes_per_identity",
        },
        "F6": {
            "selected_for_gpu1": counters["selected_for_gpu1"],
            "executed_on_gpu1": counters["executed_on_gpu1"],
            "explicit_failure": counters["explicit_failure"],
            "fallback_elsewhere": counters["fallback_elsewhere"],
            "passed": f6_pass,
        },
    }


class SplitGpu0DiagnosticExecutor:
    """Correctness-only complementary route split using two production GPU0 GEMMs.

    This is deliberately not a distributed executor. It consumes the frozen placement only
    as a classifier, services every selected raw expert through the ordinary GPU0 cache,
    makes exactly two calls to ``OffloadMoELayer._expert_gemm``, and combines once. It owns
    no transport, secondary device, or F-gate counters.
    """

    mode = "DIAGNOSTIC_SPLIT_GPU0"

    def __init__(
        self,
        *,
        placement,
        primary_device: torch.device,
        diagnostics,
        route_lookup: torch.Tensor | None = None,
    ) -> None:
        self.placement = placement
        self.primary_device = primary_device
        self.diagnostics = diagnostics
        self.route_lookup = (
            route_lookup
            if route_lookup is not None
            else build_remote_slot_lookup(placement, primary_device)
        )
        expected = (placement.num_layers, placement.num_experts)
        if (
            tuple(self.route_lookup.shape) != expected
            or self.route_lookup.dtype != torch.int32
        ):
            raise ValueError(
                "DIAGNOSTIC_SPLIT_GPU0 route lookup has invalid geometry or dtype"
            )
        if self.route_lookup.device != primary_device:
            raise ValueError("DIAGNOSTIC_SPLIT_GPU0 route lookup is not on GPU0")
        self.layer_calls = 0
        self.production_gemm_calls = 0
        self.combine_operations = 0
        self.gpu1_dispatches = 0

    def begin_decode_step(self, step: int) -> None:
        del step

    def decode(
        self, layer, cache, hidden_states, topk_weights, topk_ids
    ) -> torch.Tensor:
        layer_id = int(layer.layer_id)
        raw_ids = topk_ids.clone()
        num_experts = int(self.placement.num_experts)
        invalid = (raw_ids < 0) | (raw_ids >= num_experts)
        if bool(invalid.any().item()):
            values = torch.unique(raw_ids[invalid]).detach().cpu().tolist()
            raise RuntimeError(
                "invalid DIAGNOSTIC_SPLIT_GPU0 raw expert routing: "
                f"expert IDs {values} are outside [0, {num_experts})"
            )
        remote_slot_ids = self.route_lookup[layer_id][raw_ids.long()]
        remote_mask = remote_slot_ids >= 0
        local_mask = ~remote_mask
        self.diagnostics.capture_ownership(
            layer_id,
            local_mask=local_mask,
            remote_mask=remote_mask,
            remote_slot_ids=remote_slot_ids,
        )
        self.diagnostics.capture_selected_expert_weights(layer_id, raw_ids, cache)

        cache.record_decode_routing(layer_id, raw_ids)
        gpu0_slot_ids = raw_ids.clone()
        cache.ensure_experts(layer_id, gpu0_slot_ids, record_routing=False)
        cache.copy_missing()
        self.diagnostics.capture_gpu0_slot_ids(layer_id, gpu0_slot_ids)
        views = cache.bank_views()
        alphas = cache.alphas_for_slots(layer_id)
        zero = topk_weights.new_zeros(())
        local_weights = torch.where(local_mask, topk_weights, zero).contiguous()
        remote_weights = torch.where(remote_mask, topk_weights, zero).contiguous()
        local_partial = layer._expert_gemm(
            cache,
            hidden_states,
            local_weights,
            gpu0_slot_ids,
            views=views,
            n=None,
            alphas=alphas,
            is_prefill=False,
        )
        remote_partial = layer._expert_gemm(
            cache,
            hidden_states,
            remote_weights,
            gpu0_slot_ids,
            views=views,
            n=None,
            alphas=alphas,
            is_prefill=False,
        )
        combined = local_partial + remote_partial
        self.diagnostics.capture_candidate_partials(
            layer_id,
            local_partial=local_partial,
            remote_partial=remote_partial,
            combined_partial=combined,
        )
        self.layer_calls += 1
        self.production_gemm_calls += 2
        self.combine_operations += 1
        return combined

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": SPLIT_GPU0_DIAGNOSTIC_SCHEMA,
            "enabled": True,
            "diagnostic_label": self.mode,
            "correctness_only": True,
            "performance_compatible": False,
            "performance_fields_collected": False,
            "placement_sha256": self.placement.artifact_sha256,
            "uses_gpu1": False,
            "gpu1_dispatches": self.gpu1_dispatches,
            "all_selected_experts_serviced_by_gpu0_cache": True,
            "production_gemm_calls": self.production_gemm_calls,
            "combine_operations": self.combine_operations,
            "layer_calls": self.layer_calls,
            "f_gate_evidence_eligible": False,
            "f_gate_counters_incremented": False,
        }

    def reset(self) -> None:
        self.layer_calls = 0
        self.production_gemm_calls = 0
        self.combine_operations = 0
        self.gpu1_dispatches = 0


def absent_split_gpu0_diagnostic_report() -> dict[str, Any]:
    return {
        "schema": SPLIT_GPU0_DIAGNOSTIC_SCHEMA,
        "enabled": False,
        "diagnostic_label": None,
        "correctness_only": True,
        "performance_compatible": False,
        "performance_fields_collected": False,
        "placement_sha256": None,
        "uses_gpu1": False,
        "gpu1_dispatches": 0,
        "all_selected_experts_serviced_by_gpu0_cache": None,
        "production_gemm_calls": 0,
        "combine_operations": 0,
        "layer_calls": 0,
        "f_gate_evidence_eligible": False,
        "f_gate_counters_incremented": False,
    }


class InferSwarmRemoteDecodeExecutor:
    """Authoritative partition, submit/local/join scheduling, and one combine."""

    def __init__(
        self,
        *,
        resident_bank: SecondaryResidentExpertBank,
        secondary_device,
        primary_device: torch.device,
        transport: RemoteTransport,
        route_lookup: torch.Tensor | None = None,
        mode: str = "overlap",
        mechanism_max_steps: int = 256,
    ) -> None:
        if mode not in REMOTE_MODES:
            raise ValueError(f"invalid InferSwarm remote mode {mode!r}")
        if mechanism_max_steps < 0:
            raise ValueError("InferSwarm mechanism trace capacity cannot be negative")
        self.resident_bank = resident_bank
        self.secondary_device = secondary_device
        self.primary_device = primary_device
        self.transport = transport
        self.mode = mode
        self.mechanism_max_steps = int(mechanism_max_steps)
        self.route_lookup = (
            route_lookup
            if route_lookup is not None
            else build_remote_slot_lookup(resident_bank.placement, primary_device)
        )
        expected = (
            resident_bank.placement.num_layers,
            resident_bank.placement.num_experts,
        )
        if (
            tuple(self.route_lookup.shape) != expected
            or self.route_lookup.dtype != torch.int32
        ):
            raise ValueError("P4 route lookup has invalid geometry or dtype")
        if self.route_lookup.device != primary_device:
            raise ValueError("P4 route lookup is not on the configured primary device")
        self.counters = RemoteDecodeCounters(resident_bank.placement.num_layers)
        self._current_step: int | None = None
        self._steps_observed = 0
        self._mechanism_records: list[dict[str, Any]] = []
        self._overflow_layer_calls = 0

    def begin_decode_step(self, step: int) -> None:
        self._current_step = int(step)
        self._steps_observed = max(self._steps_observed, int(step) + 1)

    @staticmethod
    def _valid_timing(value: float, source: str) -> dict[str, Any]:
        return {"status": "valid", "value_ms": float(value), "source": source}

    @staticmethod
    def _timing(cache):
        return getattr(cache, "layer_timing", None)

    @staticmethod
    def _mark(timing, layer_id: int, marker: str, *, begin: bool = False) -> None:
        if timing is not None:
            timing.mark(layer_id, marker, begin_layer=begin)

    def _run_local(
        self,
        *,
        layer,
        cache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        raw_ids: torch.Tensor,
        local_mask: torch.Tensor,
        total_count: int,
        timing,
    ) -> torch.Tensor:
        layer_id = int(layer.layer_id)
        self._mark(timing, layer_id, "local_start")
        # Remote positions duplicate one actually-selected local identity. The fixed-shape
        # ensure therefore sees exactly the local unique set, with no dynamic compaction.
        placeholder = torch.where(
            local_mask,
            raw_ids,
            raw_ids.new_full((), self.resident_bank.placement.num_experts),
        ).amin()
        local_slots = torch.where(local_mask, raw_ids, placeholder).contiguous()
        cache.ensure_experts(layer_id, local_slots, record_routing=False)
        diagnostics = getattr(layer, "inferswarm_correctness_diagnostics", None)
        if diagnostics is not None:
            diagnostics.capture_gpu0_slot_ids(layer_id, local_slots)
        self._mark(timing, layer_id, "cache_service_end")
        if timing is not None:
            timing.record_cache_metadata(layer_id, cache, total_routes=total_count)
        cache.copy_missing()
        self._mark(timing, layer_id, "weight_fetch_end")
        local_weights = torch.where(
            local_mask, topk_weights, topk_weights.new_zeros(())
        ).contiguous()
        local_partial = layer._expert_gemm(
            cache,
            hidden_states,
            local_weights,
            local_slots,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(layer_id),
            is_prefill=False,
        )
        self._mark(timing, layer_id, "local_expert_end")
        self._mark(timing, layer_id, "local_branch_end")
        return local_partial

    def _classify(self, layer_id: int, raw_ids: torch.Tensor):
        """Return routing facts through one bounded host read in the valid hot path."""
        num_experts = int(self.resident_bank.placement.num_experts)
        invalid_mask = (raw_ids < 0) | (raw_ids >= num_experts)
        safe_ids = raw_ids.clamp(0, num_experts - 1)
        try:
            remote_slots = self.route_lookup[layer_id][safe_ids.long()]
        except (IndexError, RuntimeError) as exc:
            raise RuntimeError(f"P4 routing classification failed: {exc}") from exc
        remote_mask = (remote_slots >= 0) & ~invalid_mask
        flags = torch.zeros(num_experts, dtype=torch.int32, device=raw_ids.device)
        flags.scatter_add_(
            0,
            safe_ids.reshape(-1).long(),
            remote_mask.reshape(-1).to(torch.int32),
        )
        facts = torch.stack((invalid_mask.sum(), remote_mask.sum(), (flags > 0).sum()))
        tic = time.perf_counter_ns()
        invalid_count, remote_count, unique_remote = (int(v) for v in facts.tolist())
        wait_ms = (time.perf_counter_ns() - tic) / 1e6
        if invalid_count:
            invalid_values = torch.unique(raw_ids[invalid_mask]).detach().cpu().tolist()
            raise RuntimeError(
                f"invalid P4 raw expert routing: expert IDs {invalid_values} are outside [0, {num_experts})"
            )
        return remote_slots, remote_mask, remote_count, unique_remote, wait_ms

    def _record_trace(self, record: dict[str, Any]) -> None:
        step = record["decode_step"]
        if step is not None and 0 <= step < self.mechanism_max_steps:
            self._mechanism_records.append(record)
        else:
            self._overflow_layer_calls += 1

    @staticmethod
    def _zero_transfer_record() -> dict[str, Any]:
        return {
            "gpu0_to_host": {"activation": 0, "routing_weights": 0, "routing_ids": 0},
            "host_to_gpu1": {
                "activation": 0,
                "routing_weights": 0,
                "routing_ids": 0,
                "expert_weights": 0,
            },
            "gpu1_to_host": {"returned_partial": 0},
            "host_to_gpu0": {"returned_partial": 0},
        }

    @staticmethod
    def _per_dispatch_bytes(hidden_states, topk_weights) -> dict[str, Any]:
        activation = hidden_states.numel() * hidden_states.element_size()
        weights = topk_weights.numel() * topk_weights.element_size()
        ids = topk_weights.numel() * torch.int32.itemsize
        return {
            "gpu0_to_host": {
                "activation": activation,
                "routing_weights": weights,
                "routing_ids": ids,
            },
            "host_to_gpu1": {
                "activation": activation,
                "routing_weights": weights,
                "routing_ids": ids,
                "expert_weights": 0,
            },
            "gpu1_to_host": {"returned_partial": activation},
            "host_to_gpu0": {"returned_partial": activation},
        }

    def _annotate_timing(
        self,
        timing,
        *,
        step: int | None,
        layer_id: int,
        total: int,
        local: int,
        remote: int,
        unique_remote: int,
        dispatch: int,
        transfer: dict[str, Any],
        durations: dict[str, Any],
    ) -> None:
        if timing is None or step is None:
            return
        timing.annotate(
            step,
            layer_id,
            {
                "candidate": True,
                "identity": {
                    "decode_step": step,
                    "layer_id": layer_id,
                    "total_route_selections": total,
                    "gpu0_owned_selections": local,
                    "gpu1_owned_selections": remote,
                    "unique_gpu1_expert_identities": unique_remote,
                    "dispatch_count": dispatch,
                },
                "transfer_bytes": {
                    leg: {
                        name: {"status": "measured", "bytes": value}
                        for name, value in values.items()
                    }
                    for leg, values in transfer.items()
                },
                "durations": durations,
            },
        )

    def _trace_record(
        self,
        *,
        step,
        layer_id,
        total,
        local,
        remote,
        unique_remote,
        expected,
        actual,
        hypothetical,
        transfer,
        failure_stage,
    ) -> dict[str, Any]:
        return {
            "decode_step": step,
            "layer_id": layer_id,
            "total_route_selections": total,
            "gpu0_owned_selections": local,
            "gpu1_owned_selections": remote,
            "remote_unique_expert_count": unique_remote,
            "expected_dispatch_count": expected,
            "actual_dispatch_count": actual,
            "hypothetical_streamed_remote_weight_bytes": hypothetical,
            "transfer_bytes": transfer,
            "failure_stage": failure_stage,
        }

    def decode(
        self, layer, cache, hidden_states, topk_weights, topk_ids
    ) -> torch.Tensor:
        layer_id, timing = int(layer.layer_id), self._timing(cache)
        diagnostics = getattr(layer, "inferswarm_correctness_diagnostics", None)
        raw_ids = topk_ids.clone()
        remote_slots, remote_mask, remote_count, unique_remote, control_ms = (
            self._classify(layer_id, raw_ids)
        )
        total_count, local_count = raw_ids.numel(), raw_ids.numel() - remote_count
        if diagnostics is not None:
            diagnostics.capture_ownership(
                layer_id,
                local_mask=~remote_mask,
                remote_mask=remote_mask,
                remote_slot_ids=remote_slots,
            )
        counts = self.counters.layers[layer_id]
        counts.total_router_selections += total_count
        counts.selected_for_gpu1 += remote_count
        expected, actual = int(remote_count > 0), 0
        counts.expected_remote_dispatches += expected
        hypothetical = 0
        if expected:
            counts.remote_selected_layer_calls += 1
            hypothetical = unique_remote * int(
                self.resident_bank.placement.bytes_per_slot
            )
            counts.hypothetical_streamed_remote_weight_bytes += hypothetical
        step = self._current_step
        durations = {
            "classification_control_host_wait": self._valid_timing(
                control_ms, "host_monotonic_single_control_read"
            )
        }
        transfer = self._zero_transfer_record()

        if not remote_count:
            self._mark(timing, layer_id, "local_start")
            cache.ensure_experts(layer_id, topk_ids)
            self._mark(timing, layer_id, "cache_service_end")
            if timing is not None:
                timing.record_cache_metadata(layer_id, cache, total_routes=total_count)
            cache.copy_missing()
            if diagnostics is not None:
                diagnostics.capture_gpu0_slot_ids(layer_id, topk_ids)
            self._mark(timing, layer_id, "weight_fetch_end")
            out = layer._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=cache.bank_views(),
                n=None,
                alphas=cache.alphas_for_slots(layer_id),
                is_prefill=False,
            )
            self._mark(timing, layer_id, "local_expert_end")
            self._mark(timing, layer_id, "local_branch_end")
            self._mark(timing, layer_id, "complete_end")
            counts.executed_on_gpu0 += local_count
            if diagnostics is not None:
                diagnostics.capture_candidate_partials(
                    layer_id,
                    local_partial=out,
                    remote_partial=torch.zeros_like(out),
                    combined_partial=out,
                )
        else:
            cache.record_decode_routing(layer_id, raw_ids)
            remote_weights = torch.where(
                remote_mask, topk_weights, topk_weights.new_zeros(())
            ).contiguous()
            payload_slots = (
                torch.where(remote_mask, remote_slots, remote_slots.new_zeros(()))
                .to(torch.int32)
                .contiguous()
            )
            pending = None
            remote_succeeded = False
            failure_stage = "remote_submit"
            try:
                pending = self.transport.submit(
                    layer, cache, hidden_states, remote_weights, payload_slots
                )
                actual = 1
                counts.remote_dispatches += 1
                transfer = getattr(
                    pending,
                    "transfer_bytes",
                    self._per_dispatch_bytes(hidden_states, topk_weights),
                )
                durations.update(pending.timing_values)
                if self.mode == "serialized":
                    failure_stage = "remote_join_or_return"
                    remote_partial = self.transport.finish(
                        pending,
                        before_return_copy=lambda: self._mark(
                            timing, layer_id, "returned_partial_h2d_start"
                        ),
                        after_return_copy=lambda: self._mark(
                            timing, layer_id, "returned_partial_h2d_end"
                        ),
                    )
                    durations.update(pending.timing_values)
                    transfer = getattr(pending, "transfer_bytes", transfer)
                    counts.executed_on_gpu1 += remote_count
                    remote_succeeded = True

                failure_stage = "gpu0_local_service"
                if local_count:
                    local_partial = self._run_local(
                        layer=layer,
                        cache=cache,
                        hidden_states=hidden_states,
                        topk_weights=topk_weights,
                        raw_ids=raw_ids,
                        local_mask=~remote_mask,
                        total_count=total_count,
                        timing=timing,
                    )
                    counts.executed_on_gpu0 += local_count
                else:
                    local_partial = torch.zeros_like(hidden_states)

                if self.mode == "overlap":
                    failure_stage = "remote_join_or_return"
                    remote_partial = self.transport.finish(
                        pending,
                        before_return_copy=lambda: self._mark(
                            timing, layer_id, "returned_partial_h2d_start"
                        ),
                        after_return_copy=lambda: self._mark(
                            timing, layer_id, "returned_partial_h2d_end"
                        ),
                    )
                    durations.update(pending.timing_values)
                    transfer = getattr(pending, "transfer_bytes", transfer)
                    counts.executed_on_gpu1 += remote_count
                    remote_succeeded = True

                failure_stage = "combine"
                self._mark(timing, layer_id, "combine_start")
                out = local_partial + remote_partial
                if diagnostics is not None:
                    diagnostics.capture_candidate_partials(
                        layer_id,
                        local_partial=local_partial,
                        remote_partial=remote_partial,
                        combined_partial=out,
                    )
                self._mark(timing, layer_id, "combine_end")
                self.transport.release(pending)
                self._mark(timing, layer_id, "complete_end")
                counts.combine_operations += 1
            except Exception:
                if pending is not None and not pending.released:
                    try:
                        self.transport.drain(pending)
                    except Exception as drain_error:
                        raise RuntimeError(
                            "P4 failed to drain the in-flight GPU1 branch after a local/remote error"
                        ) from drain_error
                # F6 must fail for every surfaced failure, including a GPU0-local
                # failure discovered after serialized GPU1 work completed.  When the
                # remote selections did not execute, count those selections; otherwise
                # retain one explicit failure observation for the failed layer call.
                counts.explicit_failure += remote_count if not remote_succeeded else 1
                self.counters.failure_events += 1
                if actual != expected:
                    counts.dispatch_mismatch_layer_calls += 1
                record = self._trace_record(
                    step=step,
                    layer_id=layer_id,
                    total=total_count,
                    local=local_count,
                    remote=remote_count,
                    unique_remote=unique_remote,
                    expected=expected,
                    actual=actual,
                    hypothetical=hypothetical,
                    transfer=transfer,
                    failure_stage=failure_stage,
                )
                self._record_trace(record)
                self._annotate_timing(
                    timing,
                    step=step,
                    layer_id=layer_id,
                    total=total_count,
                    local=local_count,
                    remote=remote_count,
                    unique_remote=unique_remote,
                    dispatch=actual,
                    transfer=transfer,
                    durations=durations,
                )
                raise

        if actual != expected:
            counts.dispatch_mismatch_layer_calls += 1
        self._record_trace(
            self._trace_record(
                step=step,
                layer_id=layer_id,
                total=total_count,
                local=local_count,
                remote=remote_count,
                unique_remote=unique_remote,
                expected=expected,
                actual=actual,
                hypothetical=hypothetical,
                transfer=transfer,
                failure_stage=None,
            )
        )
        self._annotate_timing(
            timing,
            step=step,
            layer_id=layer_id,
            total=total_count,
            local=local_count,
            remote=remote_count,
            unique_remote=unique_remote,
            dispatch=actual,
            transfer=transfer,
            durations=durations,
        )
        return out

    def configuration_report(self) -> dict[str, Any]:
        placement = self.resident_bank.placement
        primary, secondary = (
            self.secondary_device.primary,
            self.secondary_device.secondary,
        )
        return {
            "schema": REMOTE_DECODE_SCHEMA,
            "enabled": True,
            "execution_mode": self.mode,
            "overlap_active": self.mode == "overlap",
            "serialized_diagnostic_only": self.mode == "serialized",
            "transport": TRANSPORT_MODE,
            "primary": {
                "uuid": primary.uuid,
                "visible_cuda_ordinal": primary.visible_ordinal,
            },
            "secondary": {
                "uuid": secondary.uuid,
                "visible_cuda_ordinal": secondary.visible_ordinal,
            },
            "placement_sha256": placement.artifact_sha256,
            "resolved_quant_format": self.resident_bank.report.layout.quant_format,
            "resolved_nvfp4_backend": self.resident_bank.report.layout.nvfp4_backend,
            "resolved_bank_layout": self.resident_bank.report.layout.bank_layout,
            "counter_source": "POST /v1/moe/instrumentation idle snapshot",
            "mechanism_trace_capacity_steps": self.mechanism_max_steps,
            "expert_weight_traffic": {
                "startup_bytes_host_to_gpu1": self.resident_bank.report.total_live_resident_bytes,
                "steady_state_bytes_host_to_gpu1": self.transport.transfer_bytes.host_to_gpu1_expert_weights,
                "source": "transport/storage execution-boundary observed counter",
            },
            "transport_buffers": self.transport.report(),
        }

    def snapshot(
        self, *, gpu0_expert_cache_slots: int = 0, gpu0_expert_cache_bytes: int = 0
    ) -> dict[str, Any]:
        aggregate, transfer = (
            self.counters.aggregate(),
            self.transport.transfer_bytes.as_dict(),
        )
        gpu1_bytes = int(self.resident_bank.report.expert_bank_tensor_bytes)
        gates = evaluate_mechanism_gates(
            gpu0_expert_cache_bytes=gpu0_expert_cache_bytes,
            gpu1_expert_cache_bytes=gpu1_bytes,
            counters=aggregate,
            transfer_bytes=transfer,
        )
        return {
            **self.configuration_report(),
            "residency": {
                "gpu0": {
                    "configured_expert_cache_slots": int(gpu0_expert_cache_slots),
                    "actual_expert_bank_tensor_bytes": int(gpu0_expert_cache_bytes),
                },
                "gpu1": {
                    "frozen_resident_slots": int(
                        self.resident_bank.placement.remote_slots
                    ),
                    "actual_expert_bank_tensor_bytes": gpu1_bytes,
                },
                "combined_expert_cache_bytes": gates["F1"][
                    "combined_expert_cache_bytes"
                ],
                "gpu1_expert_byte_fraction": gates["F1"]["gpu1_expert_byte_fraction"],
                "aliasing_rule": "GPU0 prefill views alias slot storage and are not recounted",
            },
            "aggregate": aggregate,
            "per_layer": [
                item.as_dict(layer_id)
                for layer_id, item in enumerate(self.counters.layers)
            ],
            "steady_state_transfer_bytes": transfer,
            "mechanism_trace": {
                "capacity_steps": self.mechanism_max_steps,
                "steps_observed": self._steps_observed,
                "steps_retained": min(self._steps_observed, self.mechanism_max_steps),
                "records_retained": len(self._mechanism_records),
                "truncated": bool(self._overflow_layer_calls),
                "overflow_layer_calls": self._overflow_layer_calls,
                "records": list(self._mechanism_records),
            },
            "gates": gates,
            "ownership": {
                "successful_selection_arithmetic_exact": (
                    aggregate["explicit_failure"] == 0
                    and aggregate["executed_on_gpu0"] + aggregate["executed_on_gpu1"]
                    == aggregate["total_router_selections"]
                ),
                "selected_accounted_exactly": (
                    aggregate["selected_for_gpu1"]
                    == aggregate["executed_on_gpu1"] + aggregate["explicit_failure"]
                ),
            },
        }

    def reset(self) -> None:
        self.counters.reset()
        self.transport.reset_counters()
        self._current_step = None
        self._steps_observed = 0
        self._mechanism_records.clear()
        self._overflow_layer_calls = 0


def absent_remote_decode_report() -> dict[str, Any]:
    return {
        **absent_remote_decode_configuration_report(),
        "residency": None,
        "aggregate": {
            **LayerCounters().__dict__,
            "failure_events": 0,
            "prefill_remote_dispatches": 0,
        },
        "per_layer": [],
        "steady_state_transfer_bytes": TransferByteCounters().as_dict(),
        "mechanism_trace": {
            "capacity_steps": 0,
            "steps_observed": 0,
            "steps_retained": 0,
            "records_retained": 0,
            "truncated": False,
            "overflow_layer_calls": 0,
            "records": [],
        },
        "gates": None,
        "ownership": None,
    }


def absent_remote_decode_configuration_report() -> dict[str, Any]:
    return {
        "schema": REMOTE_DECODE_SCHEMA,
        "enabled": False,
        "execution_mode": None,
        "overlap_active": False,
        "serialized_diagnostic_only": False,
        "transport": None,
        "primary": None,
        "secondary": None,
        "placement_sha256": None,
        "resolved_quant_format": None,
        "resolved_nvfp4_backend": None,
        "resolved_bank_layout": None,
        "counter_source": "POST /v1/moe/instrumentation idle snapshot",
        "mechanism_trace_capacity_steps": 0,
        "expert_weight_traffic": {
            "startup_bytes_host_to_gpu1": 0,
            "steady_state_bytes_host_to_gpu1": 0,
            "source": "not_configured",
        },
        "transport_buffers": None,
    }
