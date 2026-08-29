"""CUDA-graph-safe timestamp and metadata markers for MoE diagnostics.

The kernels in this module are imported only when ``--moe-layer-timing-max-steps``
is non-zero.  Ordinary FreeToken execution therefore does not import Triton or
allocate timing storage because of this facility.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from flashlib.kernels.slot_cache import Stat


@triton.jit
def _record_timestamp_kernel(
    timestamps,
    layer_steps,
    capacity: tl.constexpr,
    num_layers: tl.constexpr,
    num_markers: tl.constexpr,
    layer_id: tl.constexpr,
    marker_id: tl.constexpr,
    begin_layer: tl.constexpr,
):
    if begin_layer:
        step = tl.atomic_add(layer_steps + layer_id, 1)
    else:
        step = tl.load(layer_steps + layer_id) - 1
    slot = tl.minimum(step, capacity)
    offset = (slot * num_layers + layer_id) * num_markers + marker_id
    tl.store(timestamps + offset, tl.extra.cuda.globaltimer())


@triton.jit
def _record_cache_metadata_kernel(
    metadata,
    layer_steps,
    lru_stats,
    num_indices,
    capacity: tl.constexpr,
    num_layers: tl.constexpr,
    num_metrics: tl.constexpr,
    layer_id: tl.constexpr,
    active_stat: tl.constexpr,
    miss_stat: tl.constexpr,
    total_routes: tl.constexpr,
    bytes_per_identity: tl.constexpr,
):
    # flashlib's LRU kernel accumulates these values in the same launch that
    # performs admission.  Recording the cumulative values here is graph-safe;
    # MoeLayerTiming.snapshot derives reset-delimited per-replay deltas after the
    # existing decode completion boundary.
    active_cumulative = tl.load(lru_stats + active_stat)
    miss_cumulative = tl.load(lru_stats + miss_stat)
    fetched = tl.load(num_indices)
    step = tl.load(layer_steps + layer_id) - 1
    slot = tl.minimum(step, capacity)
    base = (slot * num_layers + layer_id) * num_metrics
    tl.store(metadata + base + 0, total_routes)
    tl.store(metadata + base + 1, active_cumulative)
    tl.store(metadata + base + 2, miss_cumulative)
    tl.store(metadata + base + 3, fetched)
    tl.store(metadata + base + 4, fetched * bytes_per_identity)


def record_timestamp(
    timestamps: torch.Tensor,
    layer_steps: torch.Tensor,
    *,
    capacity: int,
    num_layers: int,
    num_markers: int,
    layer_id: int,
    marker_id: int,
    begin_layer: bool,
) -> None:
    _record_timestamp_kernel[(1,)](
        timestamps,
        layer_steps,
        capacity=capacity,
        num_layers=num_layers,
        num_markers=num_markers,
        layer_id=layer_id,
        marker_id=marker_id,
        begin_layer=begin_layer,
    )


def record_cache_metadata(
    metadata: torch.Tensor,
    layer_steps: torch.Tensor,
    lru_stats: torch.Tensor,
    num_indices: torch.Tensor,
    *,
    capacity: int,
    num_layers: int,
    num_metrics: int,
    layer_id: int,
    total_routes: int,
    bytes_per_identity: int,
) -> None:
    _record_cache_metadata_kernel[(1,)](
        metadata,
        layer_steps,
        lru_stats,
        num_indices,
        capacity=capacity,
        num_layers=num_layers,
        num_metrics=num_metrics,
        layer_id=layer_id,
        active_stat=int(Stat.ACTIVE),
        miss_stat=int(Stat.MISS),
        total_routes=total_routes,
        bytes_per_identity=bytes_per_identity,
    )
