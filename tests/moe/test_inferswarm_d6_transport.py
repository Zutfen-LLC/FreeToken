from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from freetoken.moe.inferswarm_d6_count_aware_transport import (
    InferSwarmD6CountAwareTransportExecutor,
    transport_byte_geometry,
)


@pytest.mark.parametrize("count", range(9))
def test_active_count_byte_arithmetic(count):
    geometry = transport_byte_geometry(count, top_k=8, hidden_size=2048, element_size=2)
    assert geometry["d5_total_path"] == 69700
    assert geometry["d6_total_path"] == 4100 + count * 8200
    assert geometry["bytes_saved"] == (8 - count) * 8200
    assert geometry["d6_actual_return_per_leg"] == count * 4096


def test_active_count_byte_arithmetic_rejects_invalid_counts():
    with pytest.raises(ValueError): transport_byte_geometry(-1, top_k=8, hidden_size=2048, element_size=2)
    with pytest.raises(ValueError): transport_byte_geometry(9, top_k=8, hidden_size=2048, element_size=2)


def test_d6_decode_has_no_host_reads_sync_or_token_time_selection():
    tree = ast.parse(textwrap.dedent(inspect.getsource(InferSwarmD6CountAwareTransportExecutor.decode)))
    forbidden = {"item", "cpu", "tolist", "elapsed_time"}
    assert not [node.attr for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and node.attr in forbidden]
    source = inspect.getsource(InferSwarmD6CountAwareTransportExecutor.decode)
    assert source.index("except Exception:") < source.index(".synchronize()")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_count_aware_mapped_return_dynamic_replay_and_stale_tail():
    from freetoken.kernel.count_aware_transport import (
        pack_active_metadata, pack_active_routes, scatter_active_routes, unpack_active_metadata,
    )
    from freetoken.kernel.pinned import alloc_pinned_tensor

    device = torch.device("cuda", 0); torch.cuda.set_device(device); k, h = 8, 2048
    routes = torch.arange(k * h, device=device, dtype=torch.bfloat16).reshape(1, k, h)
    mapped = alloc_pinned_tensor(1, k, h, dtype=torch.bfloat16)
    positions = torch.arange(k, dtype=torch.int32, device=device).reshape(1, k)
    count = torch.tensor(k, dtype=torch.int32, device=device)
    output = torch.empty_like(routes); output.zero_()
    source_slots = torch.arange(k, dtype=torch.int32, device=device).reshape(1, k)
    source_weights = torch.arange(k, dtype=torch.float32, device=device).reshape(1, k)
    mapped_slots = alloc_pinned_tensor(1, k, dtype=torch.int32)
    mapped_weights = alloc_pinned_tensor(1, k, dtype=torch.float32)
    worker_slots = torch.full_like(source_slots, -99); worker_weights = torch.full_like(source_weights, -99)
    pack_active_routes(mapped, routes, count); scatter_active_routes(output, mapped, positions, count)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pack_active_metadata(mapped_slots, mapped_weights, source_slots, source_weights, count)
        unpack_active_metadata(worker_slots, worker_weights, mapped_slots, mapped_weights, count)
        pack_active_routes(mapped, routes, count)
        output.zero_()
        scatter_active_routes(output, mapped, positions, count)
    for active in (0, 1, 2, 4, 6, 8, 1, 0):
        count.fill_(active); routes.add_(1); graph.replay(); torch.cuda.synchronize(device)
        assert torch.equal(output[:, :active], routes[:, :active])
        assert torch.count_nonzero(output[:, active:]).item() == 0
        assert torch.equal(worker_slots[:, :active], source_slots[:, :active])
        assert torch.equal(worker_weights[:, :active], source_weights[:, :active])
