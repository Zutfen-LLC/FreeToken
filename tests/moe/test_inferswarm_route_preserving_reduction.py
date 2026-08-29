"""Regression for the frozen SPLIT_REDUCTION_TOPOLOGY failure mode."""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel import moe_sum_reduce_triton
from freetoken.moe.inferswarm_remote_decode import InferSwarmRemoteDecodeExecutor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs production Triton reduce")
def test_reconstructed_bf16_routes_match_full_reduce_when_split_partials_do_not():
    device = torch.device("cuda")
    # Fixed seed-1 values retain the diagnosed 5-local/3-remote shape. Reducing the
    # independently rounded subset totals changes the result by one BF16 ULP.
    values = torch.tensor(
        [
            0.032958984375,
            0.01336669921875,
            0.003082275390625,
            0.031005859375,
            -0.0225830078125,
            -0.00830078125,
            -0.076171875,
            0.01904296875,
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    hidden_size = 64
    full_routes = values.view(1, 8, 1).expand(1, 8, hidden_size).contiguous()
    remote_mask = torch.tensor(
        [[False, True, False, True, False, True, False, False]],
        dtype=torch.bool,
        device=device,
    )
    local_routes = torch.where(
        remote_mask.unsqueeze(-1), torch.zeros_like(full_routes), full_routes
    )
    remote_routes = torch.where(
        remote_mask.unsqueeze(-1), full_routes, torch.zeros_like(full_routes)
    )

    full_out = torch.empty((1, hidden_size), dtype=torch.bfloat16, device=device)
    local_out = torch.empty_like(full_out)
    remote_out = torch.empty_like(full_out)
    moe_sum_reduce_triton(full_routes, full_out)
    moe_sum_reduce_triton(local_routes, local_out)
    moe_sum_reduce_triton(remote_routes, remote_out)
    split_out = local_out + remote_out

    assert not torch.equal(full_out, split_out)
    assert int((full_out[0, 0].view(torch.int16) - split_out[0, 0].view(torch.int16)).abs()) == 1

    reconstructed = InferSwarmRemoteDecodeExecutor._reconstruct_route_contributions(
        hidden_states=torch.empty_like(full_out),
        remote_mask=remote_mask,
        local_routes=local_routes,
        remote_routes=remote_routes,
        local_count=5,
        remote_count=3,
    )
    reconstructed_out = torch.empty_like(full_out)
    moe_sum_reduce_triton(reconstructed, reconstructed_out)
    assert torch.equal(reconstructed, full_routes)
    assert torch.equal(reconstructed_out, full_out)

