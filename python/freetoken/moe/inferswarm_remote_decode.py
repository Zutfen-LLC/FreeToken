"""InferSwarm Phase-1 P3 serialized, correctness-first remote decode.

This is deliberately a two-device experiment, not a worker/fabric abstraction.  Raw
decode routes are classified by the already-validated P2 placement before GPU0 cache
service.  One fixed-shape payload is explicitly staged through pinned host memory for
each participating layer call, and the resident GPU1 tensors are executed through the
same :meth:`OffloadMoELayer._expert_gemm` dispatch used by local offload decode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from .inferswarm_resident_bank import SecondaryResidentExpertBank

REMOTE_DECODE_SCHEMA = "freetoken.inferswarm-remote-decode/1"
EXECUTION_MODE = "correctness_first_serialized"
TRANSPORT_MODE = "host_staged"


def build_remote_slot_lookup(
    placement,
    primary_device: torch.device,
    *,
    torch_module=torch,
) -> torch.Tensor:
    """Build ``[layer, expert] -> remote slot|-1`` from P2's authority.

    No artifact bytes are parsed here.  ``identities_in_rank_order`` is the mapping P2
    already mechanically reconciled against every redundant placement representation.
    """
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
            "P3 route lookup does not contain every P2 resident slot exactly once"
        )
    return lookup.to(primary_device)


def validate_remote_decode_runtime(config, cache, resident, secondary) -> None:
    """Refuse any resolved runtime outside the narrow canonical P3 experiment."""
    tp_info = getattr(config, "tp_info", None)
    if tp_info is None or int(getattr(tp_info, "size", -1)) != 1:
        raise ValueError("--inferswarm-remote-decode requires tensor parallel size 1")
    if getattr(config, "cuda_graph_max_bs", None) != 0:
        raise ValueError("--inferswarm-remote-decode requires --cuda-graph-max-bs 0")
    if getattr(config, "moe_backend", None) != "offload":
        raise ValueError(
            "--inferswarm-remote-decode requires resolved --moe-backend offload"
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
    local_views = cache.bank_views()
    remote_views = resident.bank_views()
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
    remote_dispatches: int = 0
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
        self.layers = [LayerCounters() for _ in range(self.num_layers)]
        self.failure_events = 0
        self.prefill_remote_dispatches = 0

    def aggregate(self) -> dict[str, int]:
        keys = tuple(LayerCounters.__dataclass_fields__)
        out = {key: sum(getattr(layer, key) for layer in self.layers) for key in keys}
        out["failure_events"] = self.failure_events
        out["prefill_remote_dispatches"] = self.prefill_remote_dispatches
        return out


class RemoteTransport(Protocol):
    def execute(
        self,
        layer,
        cache,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        remote_slot_ids: torch.Tensor,
    ) -> torch.Tensor: ...

    def report(self) -> dict[str, Any]: ...


class HostStagedRemoteTransport:
    """Persistent GPU0/host/GPU1/host/GPU0 payload buffers.

    Every cross-device leg is split into two explicit copies with a persistent pinned
    host tensor between them.  This code never asks CUDA to infer a GPU-to-GPU path.
    P3 intentionally synchronizes every stage.
    """

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
        torch_module=torch,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("P3 staging capacity must be positive")
        self.primary_device = primary_device
        self.secondary_device = secondary_device
        self.primary_ordinal = int(primary_device.index)
        self.secondary_ordinal = int(secondary_device.index)
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.hidden_dtype = hidden_dtype
        self.resident_bank = resident_bank
        self._torch = torch_module
        cuda = torch_module.cuda
        primary = self.primary_ordinal
        secondary = self.secondary_ordinal
        before_allocated = int(cuda.memory_allocated(secondary))
        before_reserved = int(cuda.memory_reserved(secondary))
        try:
            # Pinned host tensors are persistent for the executor lifetime.
            self.host_activation = torch_module.empty(
                (max_tokens, hidden_size), dtype=hidden_dtype, pin_memory=True
            )
            self.host_slots = torch_module.empty(
                (max_tokens, top_k), dtype=torch_module.int32, pin_memory=True
            )
            self.host_weights = torch_module.empty(
                (max_tokens, top_k), dtype=torch_module.float32, pin_memory=True
            )
            self.host_partial = torch_module.empty(
                (max_tokens, hidden_size), dtype=hidden_dtype, pin_memory=True
            )
            cuda.set_device(secondary)
            self.gpu1_activation = torch_module.empty(
                (max_tokens, hidden_size), dtype=hidden_dtype, device=secondary_device
            )
            self.gpu1_slots = torch_module.empty(
                (max_tokens, top_k), dtype=torch_module.int32, device=secondary_device
            )
            self.gpu1_weights = torch_module.empty(
                (max_tokens, top_k), dtype=torch_module.float32, device=secondary_device
            )
            cuda.set_device(primary)
            self.gpu0_partial = torch_module.empty(
                (max_tokens, hidden_size), dtype=hidden_dtype, device=primary_device
            )
            cuda.synchronize(primary)
            cuda.set_device(secondary)
            cuda.synchronize(secondary)
            self._gpu1_allocated_after_init = int(cuda.memory_allocated(secondary))
            self._gpu1_reserved_after_init = int(cuda.memory_reserved(secondary))
        finally:
            cuda.set_device(primary)
        if int(cuda.current_device()) != primary:
            raise RuntimeError(
                "P3 transport initialization failed to restore the primary device"
            )
        self._gpu1_allocated_before_init = before_allocated
        self._gpu1_reserved_before_init = before_reserved

    def _active(self, tensor: torch.Tensor, tokens: int) -> torch.Tensor:
        return tensor[:tokens]

    def execute(
        self,
        layer,
        cache,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        remote_slot_ids: torch.Tensor,
    ) -> torch.Tensor:
        tokens = int(hidden_states.shape[0])
        expected_hidden = (tokens, self.hidden_size)
        expected_routes = (tokens, self.top_k)
        if tokens < 1 or tokens > self.max_tokens:
            raise RuntimeError(
                f"P3 remote payload has {tokens} token rows; capacity is {self.max_tokens}"
            )
        if tuple(hidden_states.shape) != expected_hidden:
            raise RuntimeError(
                f"P3 activation shape mismatch: expected {expected_hidden}, got {tuple(hidden_states.shape)}"
            )
        if (
            tuple(routing_weights.shape) != expected_routes
            or tuple(remote_slot_ids.shape) != expected_routes
        ):
            raise RuntimeError(
                "P3 routing payload shape disagrees with configured token/top-k geometry"
            )
        if hidden_states.device != self.primary_device:
            raise RuntimeError(
                "P3 activation is not on the configured primary CUDA device"
            )
        if (
            routing_weights.device != self.primary_device
            or remote_slot_ids.device != self.primary_device
        ):
            raise RuntimeError(
                "P3 routing tensors are not on the configured primary CUDA device"
            )
        if hidden_states.dtype != self.hidden_dtype:
            raise RuntimeError(
                "P3 activation dtype disagrees with the configured runtime dtype"
            )
        if (
            routing_weights.dtype != torch.float32
            or remote_slot_ids.dtype != torch.int32
        ):
            raise RuntimeError(
                "P3 routing payload requires float32 weights and int32 slot IDs"
            )

        cuda = self._torch.cuda
        primary = self.primary_ordinal
        secondary = self.secondary_ordinal
        try:
            cuda.set_device(primary)
            # GPU0 -> pinned host.  Synchronize before the host buffers are consumed.
            self._active(self.host_activation, tokens).copy_(
                hidden_states, non_blocking=True
            )
            self._active(self.host_weights, tokens).copy_(
                routing_weights, non_blocking=True
            )
            self._active(self.host_slots, tokens).copy_(
                remote_slot_ids, non_blocking=True
            )
            cuda.synchronize(primary)

            # Pinned host -> GPU1.
            cuda.set_device(secondary)
            activation = self._active(self.gpu1_activation, tokens)
            weights = self._active(self.gpu1_weights, tokens)
            slots = self._active(self.gpu1_slots, tokens)
            activation.copy_(
                self._active(self.host_activation, tokens), non_blocking=True
            )
            weights.copy_(self._active(self.host_weights, tokens), non_blocking=True)
            slots.copy_(self._active(self.host_slots, tokens), non_blocking=True)
            cuda.synchronize(secondary)

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
            if tuple(partial.shape) != expected_hidden:
                raise RuntimeError(
                    f"P3 remote result shape mismatch: expected {expected_hidden}, got {tuple(partial.shape)}"
                )
            if (
                partial.dtype != self.hidden_dtype
                or partial.device != self.secondary_device
            ):
                raise RuntimeError(
                    "P3 remote result dtype/device disagrees with the configured secondary runtime"
                )

            # GPU1 -> pinned host.
            self._active(self.host_partial, tokens).copy_(partial, non_blocking=True)
            cuda.synchronize(secondary)

            # Pinned host -> GPU0.
            cuda.set_device(primary)
            out = self._active(self.gpu0_partial, tokens)
            out.copy_(self._active(self.host_partial, tokens), non_blocking=True)
            cuda.synchronize(primary)
            return out
        finally:
            cuda.set_device(primary)

    def report(self) -> dict[str, Any]:
        payload_bytes = {
            "activation": self.max_tokens
            * self.hidden_size
            * self.hidden_dtype.itemsize,
            "slot_ids": self.max_tokens * self.top_k * torch.int32.itemsize,
            "routing_weights": self.max_tokens * self.top_k * torch.float32.itemsize,
            "returned_partial": self.max_tokens
            * self.hidden_size
            * self.hidden_dtype.itemsize,
        }
        host_bytes = sum(payload_bytes.values())
        gpu1_bytes = (
            payload_bytes["activation"]
            + payload_bytes["slot_ids"]
            + payload_bytes["routing_weights"]
        )
        return {
            "mode": TRANSPORT_MODE,
            "persistent_capacity_tokens": self.max_tokens,
            "payload_capacity_bytes": payload_bytes,
            "pinned_host_staging_bytes": host_bytes,
            "gpu1_persistent_payload_bytes": gpu1_bytes,
            "gpu0_persistent_return_bytes": payload_bytes["returned_partial"],
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


class InferSwarmRemoteDecodeExecutor:
    """Authoritative route partition, exact ownership, and one final combine."""

    def __init__(
        self,
        *,
        resident_bank: SecondaryResidentExpertBank,
        secondary_device,
        primary_device: torch.device,
        transport: RemoteTransport,
        route_lookup: torch.Tensor | None = None,
    ) -> None:
        self.resident_bank = resident_bank
        self.secondary_device = secondary_device
        self.primary_device = primary_device
        self.transport = transport
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
            raise ValueError("P3 route lookup has invalid geometry or dtype")
        if self.route_lookup.device != primary_device:
            raise ValueError("P3 route lookup is not on the configured primary device")
        self.counters = RemoteDecodeCounters(resident_bank.placement.num_layers)

    def decode(
        self,
        layer,
        cache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        layer_id = int(layer.layer_id)
        raw_ids = topk_ids.clone()
        num_experts = int(self.resident_bank.placement.num_experts)
        invalid_mask = (raw_ids < 0) | (raw_ids >= num_experts)
        if bool(invalid_mask.any().item()):
            invalid_values = torch.unique(raw_ids[invalid_mask]).detach().cpu().tolist()
            raise RuntimeError(
                "invalid P3 raw expert routing: "
                f"expert IDs {invalid_values} are outside [0, {num_experts})"
            )
        try:
            remote_slots = self.route_lookup[layer_id][raw_ids.long()]
        except (IndexError, RuntimeError) as exc:
            raise RuntimeError(
                f"P3 routing contains an invalid raw expert ID: {exc}"
            ) from exc
        remote_mask = remote_slots >= 0
        remote_count = int(remote_mask.sum().item())
        total_count = raw_ids.numel()
        local_count = total_count - remote_count
        layer_counts = self.counters.layers[layer_id]
        layer_counts.total_router_selections += total_count
        layer_counts.selected_for_gpu1 += remote_count

        if remote_count == 0:
            # Preserve the ordinary offload path directly when no destination dispatch exists.
            cache.ensure_experts(layer_id, topk_ids)
            cache.copy_missing()
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
            layer_counts.executed_on_gpu0 += local_count
            return out

        # Record the raw router output exactly once before the compact local ensure.
        cache.record_decode_routing(layer_id, raw_ids)
        remote_weights = torch.where(
            remote_mask, topk_weights, topk_weights.new_zeros(())
        ).contiguous()
        remote_payload_slots = (
            torch.where(remote_mask, remote_slots, remote_slots.new_zeros(()))
            .to(torch.int32)
            .contiguous()
        )
        try:
            remote_partial = self.transport.execute(
                layer,
                cache,
                hidden_states,
                remote_weights,
                remote_payload_slots,
            )
        except Exception:
            layer_counts.explicit_failure += remote_count
            self.counters.failure_events += 1
            raise
        layer_counts.executed_on_gpu1 += remote_count
        layer_counts.remote_dispatches += 1

        local_mask = ~remote_mask
        if local_count:
            # Only locally owned raw identities enter LRU service and its copy plan.
            compact_local = raw_ids[local_mask].contiguous()
            cache.ensure_experts(layer_id, compact_local, record_routing=False)
            cache.copy_missing()
            # compact_local now holds known-valid resident slots.  Use its first slot as
            # the zero-weight placeholder; masked positions never read uninitialized data.
            local_slots = compact_local[0].expand_as(raw_ids).clone()
            local_slots.masked_scatter_(local_mask, compact_local)
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
            layer_counts.executed_on_gpu0 += local_count
        else:
            # Remote-only: no GPU0 cache service and no local expert kernel.
            local_partial = torch.zeros_like(hidden_states)

        combined = local_partial + remote_partial
        layer_counts.combine_operations += 1
        return combined

    def configuration_report(self) -> dict[str, Any]:
        """Static readiness provenance; dynamic counters live only in ``snapshot``."""
        placement = self.resident_bank.placement
        primary = self.secondary_device.primary
        secondary = self.secondary_device.secondary
        return {
            "schema": REMOTE_DECODE_SCHEMA,
            "enabled": True,
            "execution_mode": EXECUTION_MODE,
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
            "expert_weight_traffic": {
                "startup_bytes_host_to_gpu1": self.resident_bank.report.total_live_resident_bytes,
                "steady_state_bytes_host_to_gpu1": 0,
            },
            "transport_buffers": self.transport.report(),
        }

    def snapshot(self) -> dict[str, Any]:
        aggregate = self.counters.aggregate()
        return {
            **self.configuration_report(),
            "aggregate": aggregate,
            "per_layer": [
                counters.as_dict(layer_id)
                for layer_id, counters in enumerate(self.counters.layers)
            ],
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
        """Clear P3 measurement counters without touching either expert cache/bank."""
        self.counters.reset()


def absent_remote_decode_report() -> dict[str, Any]:
    return {
        **absent_remote_decode_configuration_report(),
        "aggregate": {
            **LayerCounters().__dict__,
            "failure_events": 0,
            "prefill_remote_dispatches": 0,
        },
        "per_layer": [],
        "ownership": None,
    }


def absent_remote_decode_configuration_report() -> dict[str, Any]:
    return {
        "schema": REMOTE_DECODE_SCHEMA,
        "enabled": False,
        "execution_mode": None,
        "transport": None,
        "primary": None,
        "secondary": None,
        "placement_sha256": None,
        "resolved_quant_format": None,
        "resolved_nvfp4_backend": None,
        "resolved_bank_layout": None,
        "counter_source": "POST /v1/moe/instrumentation idle snapshot",
        "expert_weight_traffic": {
            "startup_bytes_host_to_gpu1": 0,
            "steady_state_bytes_host_to_gpu1": 0,
        },
        "transport_buffers": None,
    }
