from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

from freetoken.moe.inferswarm_d3_graph_multiworker import (
    D3_DEPENDENCY, D3_FANOUT_SHAPE, D3_TOPOLOGY,
    InferSwarmD3GraphMultiworkerExecutor, absent_d3_graph_multiworker_report,
    build_d3_local_fallback_ids, build_d3_route_lookups,
)


def _placement():
    # A deterministic 40x256 ownership fixture with the production 6000/4240 split.
    nlayer, nexpert = 40, 256
    a = {layer * nexpert + expert for layer in range(nlayer) for expert in range(75)}
    b = {layer * nexpert + expert for layer in range(nlayer) for expert in range(75, 150)}
    local = set(range(nlayer * nexpert)) - a - b
    def worker(ids):
        per_layer = []
        for layer in range(nlayer):
            experts = [e for e in range(nexpert) if layer * nexpert + e in ids]
            slots = [sum(x < layer * nexpert + e for x in ids) for e in experts]
            per_layer.append(SimpleNamespace(layer_id=layer, expert_ids=tuple(experts), remote_slots=tuple(slots)))
        return SimpleNamespace(num_layers=nlayer, num_experts=nexpert, flat_ids_in_rank_order=tuple(sorted(ids)), per_layer=tuple(per_layer))
    return SimpleNamespace(worker_a=worker(a), worker_b=worker(b), local_remainder=tuple(sorted(local)))


def test_lookup_partition_and_fallbacks_cover_every_identity():
    placement = _placement()
    lookup_a, lookup_b = build_d3_route_lookups(placement, torch.device("cpu"))
    assert not bool(torch.any((lookup_a >= 0) & (lookup_b >= 0)))
    assert int((lookup_a >= 0).sum()) == 3000
    assert int((lookup_b >= 0).sum()) == 3000
    assert int(((lookup_a < 0) & (lookup_b < 0)).sum()) == 4240
    fallback = build_d3_local_fallback_ids(placement, torch.device("cpu"))
    assert len(fallback) == 40
    assert all(layer * 256 + int(expert) in set(placement.local_remainder) for layer, expert in enumerate(fallback))


@pytest.mark.parametrize("active, remote, local_count", [(("a",), "a", 7240), (("b",), "b", 7240), (("a", "b"), "ab", 4240)])
def test_fixed_shape_runtime_ownership_and_fallback_domain(active, remote, local_count):
    placement = _placement()
    a, b = build_d3_route_lookups(placement, torch.device("cpu"), active)
    assert (a is not None) == ("a" in active)
    assert (b is not None) == ("b" in active)
    remote_ids = set()
    if a is not None: remote_ids.update(torch.nonzero(a >= 0)[:, 0].tolist())
    # Table cardinality, instead of a fabricated inactive remote table.
    assert sum(int((x >= 0).sum()) for x in (a, b) if x is not None) == 3000 * len(active)
    fallback = build_d3_local_fallback_ids(placement, torch.device("cpu"), active)
    local = set(placement.local_remainder)
    if "a" not in active: local.update(placement.worker_a.flat_ids_in_rank_order)
    if "b" not in active: local.update(placement.worker_b.flat_ids_in_rank_order)
    assert len(local) == local_count
    assert all(layer * 256 + int(expert) in local for layer, expert in enumerate(fallback))


def test_fixed_width_classification_has_one_owner_and_zero_weight_dummies():
    placement = _placement(); a, b = build_d3_route_lookups(placement, torch.device("cpu")); fallback = build_d3_local_fallback_ids(placement, torch.device("cpu"))
    # A, B, and local routes mixed in original order for layer zero.
    ids = torch.tensor([[0, 100, 200]], dtype=torch.int64); weights = torch.tensor([[.2, .3, .5]])
    aslot, bslot = a[0][ids], b[0][ids]; amask, bmask = aslot >= 0, bslot >= 0; local = ~(amask | bmask)
    aw, bw, lw = torch.where(amask, weights, torch.zeros(())), torch.where(bmask, weights, torch.zeros(())), torch.where(local, weights, torch.zeros(()))
    assert amask.tolist() == [[True, False, False]] and bmask.tolist() == [[False, True, False]] and local.tolist() == [[False, False, True]]
    assert aw.tolist() == [[pytest.approx(.2), 0.0, 0.0]]
    assert bw.tolist() == [[0.0, pytest.approx(.3), 0.0]]
    assert lw.tolist() == [[0.0, 0.0, pytest.approx(.5)]]
    assert fallback[0].item() in range(256) and 0 * 256 + fallback[0].item() in set(placement.local_remainder)
    assert aslot.clamp_min(0).min() >= 0 and bslot.clamp_min(0).min() >= 0
    assert int(amask.sum() + bmask.sum() + local.sum()) == 3


def test_reconstruction_preserves_route_order_before_one_sum():
    local = torch.tensor([[[1., 0.], [0., 2.], [3., 0.]]])
    a = torch.tensor([[[0., 4.], [0., 0.], [0., 0.]]])
    b = torch.tensor([[[0., 0.], [5., 0.], [0., 0.]]])
    routes = local + a + b
    assert torch.equal(routes, torch.tensor([[[1., 4.], [5., 2.], [3., 0.]]]))
    assert torch.equal(routes.sum(dim=1), torch.tensor([[9., 6.]]))


def test_graph_contract_and_decode_host_firewall_are_explicit():
    executor = object.__new__(InferSwarmD3GraphMultiworkerExecutor)
    executor._capture_complete = False; executor._graph_recapture_count = 0; executor.device_counts = torch.zeros((1, 5), dtype=torch.int64)
    for captured in ([], [2], [1, 2]):
        with pytest.raises(RuntimeError, match="silent eager fallback"):
            executor.set_graph_state(captured)
    executor.set_graph_state([1]); executor.set_graph_state([1])
    assert executor._graph_recapture_count == 1
    tree = ast.parse(textwrap.dedent(inspect.getsource(InferSwarmD3GraphMultiworkerExecutor.decode)))
    forbidden = {"cpu", "item", "tolist"}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden]
    syncs = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "synchronize"]
    assert syncs and all(any(n in set(ast.walk(h)) for h in ast.walk(tree) if isinstance(h, ast.ExceptHandler)) for n in syncs)


def test_report_constants_and_absent_state():
    assert D3_TOPOLOGY == "unified_three_device_whole_model_graph_independent_ab_fanout"
    assert D3_DEPENDENCY == "cuda_capture_internal_gpu0_ready_ab_independent_done_fanin"
    assert D3_FANOUT_SHAPE == "CONCURRENT_BOUNDED_TWO_WORKER"
    report = absent_d3_graph_multiworker_report()
    assert report["enabled"] is False and report["graph_active"] is False and report["eager_fallback"] is False
