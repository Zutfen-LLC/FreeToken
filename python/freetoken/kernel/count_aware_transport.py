"""D6 device-packed mapped-host transport with device-side active counts."""
from __future__ import annotations

from functools import lru_cache

import torch

from .utils import load_jit, make_cpp_args


@lru_cache(maxsize=None)
def _module(top_k: int, hidden_size: int):
    args = make_cpp_args(top_k, hidden_size)
    return load_jit(
        "d6_count_aware_transport_v2", *args,
        cuda_files=["count_aware_transport.cuh"],
        cuda_wrappers=[
            ("pack", f"&d6_transport::CountAwareTransport<{args}>::pack"),
            ("scatter", f"&d6_transport::CountAwareTransport<{args}>::scatter"),
            ("pack_metadata", f"&d6_transport::CountAwareTransport<{args}>::pack_metadata"),
            ("unpack_metadata", f"&d6_transport::CountAwareTransport<{args}>::unpack_metadata"),
        ],
    )


def pack_active_routes(mapped_host: torch.Tensor, routes: torch.Tensor,
                       active_count: torch.Tensor) -> None:
    _module(routes.shape[-2], routes.shape[-1]).pack(mapped_host, routes, active_count)


def scatter_active_routes(reconstruction: torch.Tensor, mapped_host: torch.Tensor,
                          positions: torch.Tensor, active_count: torch.Tensor) -> None:
    _module(reconstruction.shape[-2], reconstruction.shape[-1]).scatter(
        reconstruction, mapped_host, positions, active_count)


def pack_active_metadata(mapped_slots: torch.Tensor, mapped_weights: torch.Tensor,
                         slots: torch.Tensor, weights: torch.Tensor,
                         active_count: torch.Tensor) -> None:
    _module(slots.shape[-1], 2048).pack_metadata(
        mapped_slots, mapped_weights, slots, weights, active_count)


def unpack_active_metadata(slots: torch.Tensor, weights: torch.Tensor,
                           mapped_slots: torch.Tensor, mapped_weights: torch.Tensor,
                           active_count: torch.Tensor) -> None:
    _module(slots.shape[-1], 2048).unpack_metadata(
        slots, weights, mapped_slots, mapped_weights, active_count)
