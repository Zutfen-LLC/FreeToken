"""Experimental D2 graph-compatible two-GPU decode.

This module is deliberately separate from :mod:`inferswarm_remote_decode`.  The
canonical Phase-1 flag keeps its eager, host-staged implementation.  D2 is a narrowly
gated architecture-search path for batch-one decode whose GPU0->GPU1->GPU0 fork/join is
captured into FreeToken's existing whole-forward CUDA graph.
"""

from __future__ import annotations

from typing import Any

import torch

from freetoken.kernel import moe_sum_reduce_triton

from .inferswarm_remote_decode import build_remote_slot_lookup
from .inferswarm_resident_bank import SecondaryResidentExpertBank

D2_SCHEMA = "freetoken.inferswarm-d2-graph-remote/1"
D2_TOPOLOGY = "unified_multidevice_whole_model_graph"
D2_DEPENDENCY = "cuda_capture_internal_cross_device_event_fork_join"
FANOUT_SHAPE = "CONCURRENT_BOUNDED"


def build_local_fallback_ids(placement, primary_device: torch.device) -> torch.Tensor:
    """One placement-excluded dummy identity per layer for zero-weight local routes."""
    values: list[int] = []
    for layer in placement.per_layer:
        remote = set(layer.expert_ids)
        local = next((expert for expert in range(placement.num_experts) if expert not in remote), None)
        if local is None:
            raise ValueError(f"D2 layer {layer.layer_id} has no GPU0-owned fallback expert")
        values.append(local)
    return torch.tensor(values, dtype=torch.int32, device=primary_device)


def validate_d2_runtime(config, cache, resident, secondary) -> None:
    """Refuse every configuration outside the frozen batch-one graph experiment."""
    if bool(getattr(config, "inferswarm_remote_decode", False)):
        raise ValueError(
            "--inferswarm-experimental-d2-graph-remote is mutually exclusive with "
            "--inferswarm-remote-decode"
        )
    tp_info = getattr(config, "tp_info", None)
    if tp_info is None or int(getattr(tp_info, "size", -1)) != 1:
        raise ValueError("D2 graph remote requires tensor parallel size 1")
    if int(getattr(config, "cuda_graph_max_bs", -1) or -1) != 1:
        raise ValueError("D2 graph remote requires --cuda-graph-max-bs 1")
    if int(getattr(config, "max_running_req", -1)) != 1:
        raise ValueError("D2 graph remote requires --max-running-requests 1")
    if getattr(config, "moe_backend", None) != "offload":
        raise ValueError("D2 graph remote requires resolved --moe-backend offload")
    if getattr(cache, "decode_target", None) != "gpu":
        raise ValueError("D2 graph remote requires GPU decode")
    if getattr(cache, "cpu_layer_ids", frozenset()):
        raise ValueError("D2 graph remote requires zero CPU MoE layers")
    if getattr(cache, "quant_format", None) != "nvfp4":
        raise ValueError("D2 graph remote requires native NVFP4/Triton")
    layout = resident.report.layout
    if (
        layout.quant_format != "nvfp4"
        or layout.nvfp4_backend != "triton"
        or layout.bank_layout != "native_modelopt_nvfp4"
    ):
        raise ValueError("D2 graph remote requires the frozen native NVFP4 resident layout")
    if resident.report.secondary_visible_ordinal != int(secondary.secondary.visible_ordinal):
        raise ValueError("D2 resident bank is not on the configured secondary")


class InferSwarmD2GraphRemoteExecutor:
    """Persistent fixed-shape graph-captured route-contribution fork/join.

    Every layer runs both fixed-width branches.  Routes outside a branch carry exactly
    zero weight and use a valid dummy slot.  Consequently route order never changes and
    the only combine is an elementwise route contribution addition followed by the one
    canonical sum reduction.
    """

    def __init__(
        self,
        *,
        resident_bank: SecondaryResidentExpertBank,
        secondary_device,
        primary_device: torch.device,
        route_lookup: torch.Tensor,
        local_fallback_ids: torch.Tensor,
        hidden_size: int,
        top_k: int,
        hidden_dtype: torch.dtype,
        num_layers: int,
        intermediate_size: int,
        torch_module=torch,
    ) -> None:
        self.resident_bank = resident_bank
        self.secondary_device = secondary_device
        self.primary_device = primary_device
        self.secondary_torch_device = torch.device(
            "cuda", int(secondary_device.secondary.visible_ordinal)
        )
        self.route_lookup = route_lookup
        self.local_fallback_ids = local_fallback_ids
        self.hidden_size = int(hidden_size)
        self.top_k = int(top_k)
        self.hidden_dtype = hidden_dtype
        self.num_layers = int(num_layers)
        self.intermediate_size = int(intermediate_size)
        self._torch = torch_module
        self._captured_bs: tuple[int, ...] = ()
        self._capture_complete = False
        self._failure_count = 0
        self._steady_state_host_sync_count = 0
        self._graph_recapture_count = 0

        cuda = torch_module.cuda
        primary = int(primary_device.index)
        secondary = int(self.secondary_torch_device.index)
        try:
            cuda.set_device(primary)
            self.host_activation = torch_module.empty(
                (1, self.hidden_size), dtype=hidden_dtype, pin_memory=True
            )
            self.host_slots = torch_module.empty(
                (1, self.top_k), dtype=torch_module.int32, pin_memory=True
            )
            self.host_weights = torch_module.empty(
                (1, self.top_k), dtype=torch_module.float32, pin_memory=True
            )
            self.host_return = torch_module.empty(
                (1, self.top_k, self.hidden_size), dtype=hidden_dtype, pin_memory=True
            )
            self.gpu0_remote_slots = torch_module.empty(
                (1, self.top_k), dtype=torch_module.int32, device=primary_device
            )
            self.gpu0_remote_weights = torch_module.empty(
                (1, self.top_k), dtype=torch_module.float32, device=primary_device
            )
            self.gpu0_local_ids = torch_module.empty_like(self.gpu0_remote_slots)
            self.gpu0_local_weights = torch_module.empty_like(self.gpu0_remote_weights)
            self.gpu0_local_routes = torch_module.empty(
                (1, self.top_k, self.hidden_size), dtype=hidden_dtype, device=primary_device
            )
            self.gpu0_return_routes = torch_module.empty_like(self.gpu0_local_routes)
            self.gpu0_combined_routes = torch_module.empty_like(self.gpu0_local_routes)
            self.gpu0_output = torch_module.empty(
                (1, self.hidden_size), dtype=hidden_dtype, device=primary_device
            )
            self.gpu0_gate_up = torch_module.empty(
                (1, self.top_k, 2 * self.intermediate_size),
                dtype=hidden_dtype,
                device=primary_device,
            )
            self.gpu0_activation = torch_module.empty(
                (self.top_k, self.intermediate_size),
                dtype=hidden_dtype,
                device=primary_device,
            )
            self.device_counts = torch_module.zeros(
                (self.num_layers, 4), dtype=torch_module.int64, device=primary_device
            )
            self.ready_events = [cuda.Event() for _ in range(self.num_layers)]
            self.done_events = [cuda.Event() for _ in range(self.num_layers)]

            cuda.set_device(secondary)
            self.remote_stream = cuda.Stream(device=self.secondary_torch_device)
            self.gpu1_activation = torch_module.empty(
                (1, self.hidden_size), dtype=hidden_dtype, device=self.secondary_torch_device
            )
            self.gpu1_slots = torch_module.empty(
                (1, self.top_k), dtype=torch_module.int32, device=self.secondary_torch_device
            )
            self.gpu1_weights = torch_module.empty(
                (1, self.top_k), dtype=torch_module.float32, device=self.secondary_torch_device
            )
            self.gpu1_routes = torch_module.empty(
                (1, self.top_k, self.hidden_size),
                dtype=hidden_dtype,
                device=self.secondary_torch_device,
            )
            self.gpu1_gate_up = torch_module.empty(
                (1, self.top_k, 2 * self.intermediate_size),
                dtype=hidden_dtype,
                device=self.secondary_torch_device,
            )
            self.gpu1_activation_out = torch_module.empty(
                (self.top_k, self.intermediate_size),
                dtype=hidden_dtype,
                device=self.secondary_torch_device,
            )
        finally:
            cuda.set_device(primary)

    def _validate(self, layer, hidden, weights, ids) -> int:
        layer_id = int(layer.layer_id)
        if not 0 <= layer_id < self.num_layers:
            raise RuntimeError(f"D2 layer id {layer_id} is out of range")
        if tuple(hidden.shape) != (1, self.hidden_size):
            raise RuntimeError("D2 graph remote accepts batch-one decode only")
        if tuple(weights.shape) != (1, self.top_k) or tuple(ids.shape) != (1, self.top_k):
            raise RuntimeError("D2 routing geometry is not the captured [1, top_k] shape")
        if (
            hidden.device != self.primary_device
            or weights.device != self.primary_device
            or ids.device != self.primary_device
        ):
            raise RuntimeError("D2 inputs must remain on the primary CUDA device")
        if (
            hidden.dtype != self.hidden_dtype
            or weights.dtype != torch.float32
            or ids.dtype != torch.int32
        ):
            raise RuntimeError("D2 input dtypes disagree with the captured contract")
        return layer_id

    def decode(self, layer, cache, hidden_states, topk_weights, topk_ids):
        layer_id = self._validate(layer, hidden_states, topk_weights, topk_ids)
        timing = getattr(cache, "layer_timing", None)
        cuda = self._torch.cuda
        primary = int(self.primary_device.index)
        secondary = int(self.secondary_torch_device.index)
        try:
            cuda.set_device(primary)
            primary_stream = cuda.current_stream(self.primary_device)

            # Fixed-width device-side classification.  No CUDA value reaches Python.
            slots = self.route_lookup[layer_id][topk_ids.long()]
            self.gpu0_remote_slots.copy_(slots)
            remote_mask = self.gpu0_remote_slots >= 0
            torch.where(
                remote_mask,
                topk_weights,
                topk_weights.new_zeros(()),
                out=self.gpu0_remote_weights,
            )
            self.gpu0_remote_slots.clamp_min_(0)
            torch.where(
                remote_mask,
                self.local_fallback_ids[layer_id],
                topk_ids,
                out=self.gpu0_local_ids,
            )
            torch.where(
                remote_mask,
                topk_weights.new_zeros(()),
                topk_weights,
                out=self.gpu0_local_weights,
            )
            remote_count = remote_mask.sum()
            self.device_counts[layer_id, 0].add_(self.top_k)
            self.device_counts[layer_id, 1].add_(remote_count)
            self.device_counts[layer_id, 2].add_(self.top_k - remote_count)
            self.device_counts[layer_id, 3].add_(1)

            # GPU0 dispatch segment.  The event is internal to global capture: its
            # record/wait becomes a graph dependency edge, not a replay-time host wait.
            self.host_activation.copy_(hidden_states, non_blocking=True)
            self.host_slots.copy_(self.gpu0_remote_slots, non_blocking=True)
            self.host_weights.copy_(self.gpu0_remote_weights, non_blocking=True)
            self.ready_events[layer_id].record(primary_stream)

            # GPU1 fixed resident sequence.  Switching devices occurs only while the
            # model graph is initially captured or on the eager capture warmup, never
            # during steady-state graph replay.
            cuda.set_device(secondary)
            with cuda.stream(self.remote_stream):
                self.ready_events[layer_id].wait(self.remote_stream)
                self.gpu1_activation.copy_(self.host_activation, non_blocking=True)
                self.gpu1_slots.copy_(self.host_slots, non_blocking=True)
                self.gpu1_weights.copy_(self.host_weights, non_blocking=True)
                layer._expert_route_contributions(
                    cache,
                    self.gpu1_activation,
                    self.gpu1_weights,
                    self.gpu1_slots,
                    views=self.resident_bank.bank_views(),
                    alphas=self.resident_bank.alpha_views(),
                    out=self.gpu1_routes,
                    gate_up_out=self.gpu1_gate_up,
                    activation_out=self.gpu1_activation_out,
                )
                self.host_return.copy_(self.gpu1_routes, non_blocking=True)
                self.done_events[layer_id].record(self.remote_stream)

            # GPU0 local branch can overlap the GPU1 branch.  Remote routes use an
            # owned dummy identity and zero weight, so GPU1-owned weights never enter
            # GPU0's cache service/copy plan.
            cuda.set_device(primary)
            if timing is not None:
                timing.mark(layer_id, "local_start")
            cache.ensure_experts(layer_id, self.gpu0_local_ids)
            if timing is not None:
                timing.mark(layer_id, "cache_service_end")
                timing.record_cache_metadata(
                    layer_id, cache, total_routes=topk_ids.numel()
                )
            cache.copy_missing()
            if timing is not None:
                timing.mark(layer_id, "weight_fetch_end")
            layer._expert_route_contributions(
                cache,
                hidden_states,
                self.gpu0_local_weights,
                self.gpu0_local_ids,
                views=cache.bank_views(),
                alphas=cache.alphas_for_slots(layer_id),
                out=self.gpu0_local_routes,
                gate_up_out=self.gpu0_gate_up,
                activation_out=self.gpu0_activation,
            )
            if timing is not None:
                timing.mark(layer_id, "local_expert_end")
                timing.mark(layer_id, "local_branch_end")

            # Captured GPU-to-GPU dependency join, pinned return H2D, exact route-order
            # reconstruction, and the one canonical final route reduction.
            self.done_events[layer_id].wait(primary_stream)
            if timing is not None:
                timing.mark(layer_id, "returned_route_contributions_h2d_start")
            self.gpu0_return_routes.copy_(self.host_return, non_blocking=True)
            if timing is not None:
                timing.mark(layer_id, "returned_route_contributions_h2d_end")
                timing.mark(layer_id, "route_reconstruction_start")
            torch.add(
                self.gpu0_local_routes,
                self.gpu0_return_routes,
                out=self.gpu0_combined_routes,
            )
            if timing is not None:
                timing.mark(layer_id, "route_reconstruction_end")
                timing.mark(layer_id, "final_sum_reduce_start")
            moe_sum_reduce_triton(self.gpu0_combined_routes, self.gpu0_output)
            if timing is not None:
                timing.mark(layer_id, "final_sum_reduce_end")
                timing.mark(layer_id, "complete_end")
            return self.gpu0_output
        except Exception:
            self._failure_count += 1
            # Synchronization is permitted only for failure cleanup.
            try:
                cuda.set_device(secondary)
                self.remote_stream.synchronize()
            finally:
                cuda.set_device(primary)
            raise
        finally:
            cuda.set_device(primary)

    def set_graph_state(self, captured_bs: list[int]) -> None:
        captured = tuple(sorted(int(bs) for bs in captured_bs))
        if captured != (1,):
            raise RuntimeError(
                "D2 refused silent eager fallback: expected exactly CUDA graph BS1, "
                f"captured {list(captured)}"
            )
        if self._capture_complete:
            self._graph_recapture_count += 1
        self._captured_bs = captured
        self._capture_complete = True
        self.reset_counters()

    def reset_counters(self) -> None:
        self.device_counts.zero_()

    def configuration_report(self) -> dict[str, Any]:
        placement = self.resident_bank.placement
        primary = self.secondary_device.primary
        secondary = self.secondary_device.secondary
        active = self._capture_complete and self._captured_bs == (1,)
        return {
            "schema": D2_SCHEMA,
            "enabled": True,
            "experimental": True,
            "gpu0_graph_active": active,
            "gpu0_graph_topology": D2_TOPOLOGY,
            "gpu0_graph_segments_per_token": 1 if active else 0,
            "gpu0_graph_replays_per_token": 1 if active else 0,
            "gpu1_graph_active": active,
            "gpu1_graph_topology": "embedded_gpu1_nodes_in_unified_multidevice_graph",
            "gpu1_graph_replays_per_token": 1 if active else 0,
            "remote_operations_per_token": self.num_layers if active else 0,
            "steady_state_host_sync_count": self._steady_state_host_sync_count,
            "eager_gpu0_fallback": not active,
            "cross_device_dependency": D2_DEPENDENCY,
            "fanout_shape": FANOUT_SHAPE,
            "captured_batch_sizes": list(self._captured_bs),
            "graph_recapture_count": self._graph_recapture_count,
            "primary": {"uuid": primary.uuid, "visible_cuda_ordinal": primary.visible_ordinal},
            "secondary": {"uuid": secondary.uuid, "visible_cuda_ordinal": secondary.visible_ordinal},
            "placement_sha256": placement.artifact_sha256,
            "resident_slots_gpu1": placement.remote_slots,
            "steady_state_expert_weight_bytes_host_to_gpu1": 0,
            "fallback_count": 0,
            "failure_count": self._failure_count,
            "fixed_allocations": True,
            "stable_tensor_addresses": True,
            "per_token_recapture": False,
            "route_reconstruction": "elementwise_same_route_add_then_one_canonical_sum",
        }

    def snapshot(self) -> dict[str, Any]:
        # Called only at the engine's synchronized idle instrumentation boundary.
        counts = self.device_counts.detach().cpu()
        total = int(counts[:, 0].sum().item())
        remote = int(counts[:, 1].sum().item())
        local = int(counts[:, 2].sum().item())
        calls = int(counts[:, 3].sum().item())
        return {
            **self.configuration_report(),
            "ownership": {
                "total_router_selections": total,
                "executed_on_gpu1": remote,
                "executed_on_gpu0": local,
                "layer_calls": calls,
                "selection_arithmetic_exact": total == remote + local,
                "no_route_dropped_or_duplicated": total == remote + local,
                "per_layer": [
                    {
                        "layer_id": layer_id,
                        "total_router_selections": int(row[0].item()),
                        "executed_on_gpu1": int(row[1].item()),
                        "executed_on_gpu0": int(row[2].item()),
                        "layer_calls": int(row[3].item()),
                    }
                    for layer_id, row in enumerate(counts)
                ],
            },
        }


def absent_d2_graph_remote_report() -> dict[str, Any]:
    return {
        "schema": D2_SCHEMA,
        "enabled": False,
        "experimental": True,
        "gpu0_graph_active": False,
        "gpu0_graph_topology": None,
        "gpu0_graph_segments_per_token": 0,
        "gpu0_graph_replays_per_token": 0,
        "gpu1_graph_active": False,
        "gpu1_graph_topology": None,
        "gpu1_graph_replays_per_token": 0,
        "steady_state_host_sync_count": 0,
        "eager_gpu0_fallback": False,
        "fanout_shape": "UNKNOWN",
    }


__all__ = [
    "D2_DEPENDENCY",
    "D2_SCHEMA",
    "D2_TOPOLOGY",
    "FANOUT_SHAPE",
    "InferSwarmD2GraphRemoteExecutor",
    "absent_d2_graph_remote_report",
    "build_local_fallback_ids",
    "build_remote_slot_lookup",
    "validate_d2_runtime",
]
