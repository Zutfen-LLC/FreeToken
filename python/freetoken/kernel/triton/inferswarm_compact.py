"""D5-only fixed-capacity stable route compaction and route-order scatter."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compact_routes_kernel(ids, weights, lookup_a, lookup_b,
                           local_ids, local_weights, local_pos, local_count,
                           a_ids, a_weights, a_pos, a_count,
                           b_ids, b_weights, b_pos, b_count,
                           lookup_offset: tl.constexpr, K: tl.constexpr,
                           HAS_A: tl.constexpr, HAS_B: tl.constexpr):
    p = tl.arange(0, K)
    raw = tl.load(ids + p)
    weight = tl.load(weights + p)
    slot_a = tl.load(lookup_a + lookup_offset + raw, mask=HAS_A, other=-1)
    slot_b = tl.load(lookup_b + lookup_offset + raw, mask=HAS_B, other=-1)
    own_a = slot_a >= 0
    own_b = slot_b >= 0
    own_local = ~(own_a | own_b)
    rank_a = tl.cumsum(own_a.to(tl.int32), axis=0) - 1
    rank_b = tl.cumsum(own_b.to(tl.int32), axis=0) - 1
    rank_local = tl.cumsum(own_local.to(tl.int32), axis=0) - 1
    # One program owns every output, so deterministic tail initialization and compact
    # writes are race-free. Tail ids/positions are valid zero values and weights zero.
    tl.store(local_ids + p, 0); tl.store(local_weights + p, 0.0); tl.store(local_pos + p, 0)
    tl.store(a_ids + p, 0); tl.store(a_weights + p, 0.0); tl.store(a_pos + p, 0)
    tl.store(b_ids + p, 0); tl.store(b_weights + p, 0.0); tl.store(b_pos + p, 0)
    tl.store(local_ids + rank_local, raw, mask=own_local)
    tl.store(local_weights + rank_local, weight, mask=own_local)
    tl.store(local_pos + rank_local, p, mask=own_local)
    tl.store(a_ids + rank_a, slot_a, mask=own_a)
    tl.store(a_weights + rank_a, weight, mask=own_a)
    tl.store(a_pos + rank_a, p, mask=own_a)
    tl.store(b_ids + rank_b, slot_b, mask=own_b)
    tl.store(b_weights + rank_b, weight, mask=own_b)
    tl.store(b_pos + rank_b, p, mask=own_b)
    tl.store(local_count, tl.sum(own_local.to(tl.int32), axis=0))
    tl.store(a_count, tl.sum(own_a.to(tl.int32), axis=0))
    tl.store(b_count, tl.sum(own_b.to(tl.int32), axis=0))


def compact_routes(ids: torch.Tensor, weights: torch.Tensor, lookup_a: torch.Tensor,
                   lookup_b: torch.Tensor, layer_id: int, outputs: tuple[torch.Tensor, ...],
                   *, has_a: bool, has_b: bool) -> None:
    k = ids.shape[1]
    _compact_routes_kernel[(1,)](ids, weights, lookup_a, lookup_b, *outputs,
                                 lookup_offset=layer_id * lookup_a.shape[1], K=k,
                                 HAS_A=has_a, HAS_B=has_b, num_warps=1)


@triton.jit
def _scatter_compact_kernel(routes, positions, count, reconstruction,
                            H: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    route = tl.program_id(0); block = tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    active = route < tl.load(count)
    pos = tl.load(positions + route, mask=active, other=0)
    values = tl.load(routes + route * H + cols, mask=active & (cols < H), other=0.0)
    tl.store(reconstruction + pos * H + cols, values, mask=active & (cols < H))


def scatter_compact(routes: torch.Tensor, positions: torch.Tensor, count: torch.Tensor,
                    reconstruction: torch.Tensor) -> None:
    k, h = routes.shape[-2:]
    _scatter_compact_kernel[(k, triton.cdiv(h, 256))](routes, positions, count,
                                                       reconstruction, H=h, K=k,
                                                       BLOCK=256, num_warps=4)
