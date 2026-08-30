from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from freetoken.kernel.triton.inferswarm_compact import compact_routes, scatter_compact
from freetoken.moe.inferswarm_d5_compact_routes import InferSwarmD5CompactRoutesExecutor


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("owners", ["llllllll", "aaaaaaaa", "bbbbbbbb",
                                     "ablablab", "abllllll", "aabbblll"])
def test_stable_compaction_counts_positions_and_zero_tails(owners):
    device = torch.device("cuda", 0); k = 8; experts = 16
    ids = torch.arange(k, dtype=torch.int32, device=device).reshape(1, k)
    weights = torch.arange(1, k + 1, dtype=torch.float32, device=device).reshape(1, k)
    lookups = []
    for label in "ab":
        table = torch.full((1, experts), -1, dtype=torch.int32, device=device)
        for pos, owner in enumerate(owners):
            if owner == label: table[0, pos] = 10 + pos
        lookups.append(table)
    outputs = []
    for _label in ("local", "a", "b"):
        outputs += [torch.empty((1, k), dtype=torch.int32, device=device),
                    torch.empty((1, k), dtype=torch.float32, device=device),
                    torch.empty((1, k), dtype=torch.int32, device=device),
                    torch.empty((), dtype=torch.int32, device=device)]
    compact_routes(ids, weights, *lookups, 0, tuple(outputs), has_a=True, has_b=True)
    torch.cuda.synchronize(device)
    for branch, label in enumerate("lab"):
        out_ids, out_weights, out_pos, count = outputs[branch * 4:(branch + 1) * 4]
        expected = [i for i, owner in enumerate(owners) if owner == label]
        assert int(count.item()) == len(expected)
        assert out_pos[0, :len(expected)].tolist() == expected
        assert out_weights[0, :len(expected)].tolist() == [float(i + 1) for i in expected]
        assert torch.count_nonzero(out_weights[0, len(expected):]).item() == 0
        expected_ids = expected if label == "l" else [10 + i for i in expected]
        assert out_ids[0, :len(expected)].tolist() == expected_ids


def test_compact_scatter_reconstructs_route_order_across_stale_tail_transitions():
    device = torch.device("cuda", 0); k, h = 8, 32
    reconstruction = torch.empty((1, k, h), dtype=torch.float16, device=device)
    routes = torch.full((1, k, h), 99, dtype=torch.float16, device=device)
    positions = torch.arange(k, dtype=torch.int32, device=device).reshape(1, k)
    count = torch.tensor(8, dtype=torch.int32, device=device)
    scatter_compact(routes, positions, count, reconstruction)
    reconstruction.zero_(); routes.fill_(7); count.fill_(1); positions.fill_(0)
    scatter_compact(routes, positions, count, reconstruction); torch.cuda.synchronize(device)
    assert torch.all(reconstruction[0, 0] == 7)
    assert torch.count_nonzero(reconstruction[0, 1:]).item() == 0


def test_decode_contains_no_host_value_extraction_or_route_count_branch():
    tree = ast.parse(textwrap.dedent(inspect.getsource(InferSwarmD5CompactRoutesExecutor.decode)))
    forbidden = {"item", "cpu", "tolist"}
    assert not [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr in forbidden]
