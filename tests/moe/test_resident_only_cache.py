import gc
import weakref

import pytest
import torch


def _cache(*, num_layers=2, num_experts=3, cache_size=None, **kwargs):
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache_size = cache_size or num_layers * num_experts
    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        device=torch.device("cpu"),
        **kwargs,
    )
    sources = {
        "gate_up": [
            torch.full((num_experts, 4, 2), 10 * layer + 1.0)
            for layer in range(num_layers)
        ],
        "down": [
            torch.full((num_experts, 2, 2), 10 * layer + 2.0)
            for layer in range(num_layers)
        ],
    }
    cache.set_bank_sources(sources)
    return cache, sources


def _populate(cache, slots=None):
    required = cache.num_layers * cache.num_experts
    for destination in cache.bank_caches.values():
        destination.zero_()
    if slots is None:
        slots = torch.arange(required, dtype=torch.int32)
    slots = slots.view(cache.num_layers, cache.num_experts)
    cache.slot_for_id.copy_(slots)
    cache.id_of_slot.fill_(-1)
    identities = torch.arange(required, dtype=torch.int32)
    cache.id_of_slot[slots.view(-1).long()] = identities
    for name in cache.bank_schema:
        destination = cache.bank_caches[name]
        for layer_id, source in enumerate(cache.bank_sources[name]):
            for expert_id in range(cache.num_experts):
                destination[int(slots[layer_id, expert_id])] = source[expert_id]
    cache.usage[slots.view(-1).long()] = 1
    cache.step.fill_(1)


def test_resident_only_rejects_incomplete_residency_without_releasing_sources():
    cache, sources = _cache()

    with pytest.raises(RuntimeError, match="missing"):
        cache.detach_host_sources_for_full_residency()

    assert cache.resident_only is False
    assert cache.bank_sources["gate_up"][0] is sources["gate_up"][0]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"decode_target": "cpu"}, "decode_target"),
        ({"decode_target": "hybrid"}, "decode_target"),
        ({"prefill_overlap": True, "cache_size": 12}, "prefill overlap"),
    ],
)
def test_resident_only_rejects_source_dependent_execution_modes(kwargs, message):
    cache_size = kwargs.pop("cache_size", None)
    cache, _ = _cache(cache_size=cache_size, **kwargs)
    _populate(cache)

    with pytest.raises(RuntimeError, match=message):
        cache.detach_host_sources_for_full_residency()


@pytest.mark.parametrize(
    "corruption", ["duplicate_slot", "missing_identity", "bad_inverse"]
)
def test_full_residency_validation_catches_inconsistent_maps(corruption):
    cache, _ = _cache(cache_size=7)
    _populate(cache, torch.tensor([5, 1, 3, 0, 4, 2], dtype=torch.int32))
    if corruption == "duplicate_slot":
        cache.slot_for_id[1, 2] = cache.slot_for_id[0, 0]
    elif corruption == "missing_identity":
        cache.id_of_slot[int(cache.slot_for_id[1, 2])] = -1
    else:
        cache.id_of_slot[int(cache.slot_for_id[1, 2])] = 0

    with pytest.raises(RuntimeError, match="duplicate|missing|mutual inverses"):
        cache.detach_host_sources_for_full_residency()


def test_successful_detach_clears_sources_and_descriptors_but_keeps_cache_views():
    cache, sources = _cache(cache_size=7)
    _populate(cache, torch.tensor([5, 1, 3, 0, 4, 2], dtype=torch.int32))
    expected_views = tuple(tensor.clone() for tensor in cache.bank_views())
    host_bytes = cache.host_source_tensor_bytes()
    refs = [weakref.ref(t) for layers in sources.values() for t in layers]

    # CPU caches do not normally build fused descriptors. Seed every source-plan
    # field so this test proves the detach teardown rather than an absent plan.
    cache._copy_src_ptrs = [torch.tensor([123])]
    cache._copy_src_ptrs_host = [[123]]
    cache._copy_feat_bytes = torch.tensor([8])
    cache._copy_feat_bytes_host = [8]
    cache._copy_dst_ptrs = torch.tensor([456])
    cache._copy_dst_ptrs_host = [456]

    report = cache.detach_host_sources_for_full_residency()

    assert report["host_source_bank_bytes_before_detach"] == host_bytes
    assert report["host_source_bank_bytes_after_detach"] == 0
    assert cache.bank_sources == {} and cache.banks == []
    assert cache._copy_src_ptrs is None and cache._copy_src_ptrs_host == []
    assert cache._copy_feat_bytes is None and cache._copy_feat_bytes_host == []
    assert cache._copy_dst_ptrs is None and cache._copy_dst_ptrs_host == []
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(cache.bank_views(), expected_views)
    )

    del sources
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_resident_expert_mapping_is_frozen_deterministic_and_decode_uses_it(
    monkeypatch,
):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    cache, _ = _cache(num_layers=1, num_experts=4, cache_size=6)
    _populate(cache, torch.tensor([4, 1, 5, 2], dtype=torch.int32))
    cache.detach_host_sources_for_full_residency()
    # The validated resident map is a private frozen clone, not an alias of the
    # ordinary mutable offload bookkeeping.
    cache.slot_for_id.fill_(-1)
    cache.id_of_slot.fill_(-1)

    layer = OffloadMoELayer(0, 4, 2, 2, 2)
    layer.offload_cache = cache
    routed = torch.tensor([[3, 0], [1, 2]], dtype=torch.int32)
    captured = {}

    def expert_gemm(cache_arg, hidden, weights, ids, **kwargs):
        captured["ids"] = ids.clone()
        captured["views"] = kwargs["views"]
        return hidden

    monkeypatch.setattr(layer, "_expert_gemm", expert_gemm)
    monkeypatch.setattr(
        cache,
        "ensure_experts",
        lambda *args, **kwargs: pytest.fail("legacy cache service ran"),
    )
    monkeypatch.setattr(cache, "copy_missing", lambda: pytest.fail("host copy ran"))
    hidden = torch.randn(2, 2)
    out = layer._decode_routed(hidden, torch.ones(2, 2), routed)

    assert out is hidden
    assert captured["ids"].tolist() == [[2, 4], [1, 5]]
    assert captured["views"] == cache.bank_views()
    assert cache.resident_source_access_attempts == 0


def test_resident_only_prefill_maps_to_frozen_slots_without_host_sources(monkeypatch):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    cache, _ = _cache(num_layers=2, num_experts=3, cache_size=7)
    _populate(cache, torch.tensor([5, 1, 3, 0, 4, 2], dtype=torch.int32))
    cache.detach_host_sources_for_full_residency()
    layer = OffloadMoELayer(1, 3, 2, 2, 2)
    layer.offload_cache = cache
    routed = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32)
    captured = {}

    def expert_gemm(cache_arg, hidden, weights, ids, **kwargs):
        captured.update(ids=ids.clone(), **kwargs)
        return hidden

    monkeypatch.setattr(layer, "_expert_gemm", expert_gemm)
    monkeypatch.setattr(
        cache,
        "materialize_layer",
        lambda *args: pytest.fail("host materialization ran"),
    )
    monkeypatch.setattr(cache, "copy_missing", lambda: pytest.fail("host copy ran"))
    hidden = torch.randn(2, 2)
    out = layer._prefill_routed(hidden, torch.ones(2, 2), routed)

    assert out is hidden
    assert captured["ids"].tolist() == [[2, 0], [4, 2]]
    assert captured["views"] == cache.bank_views()
    assert captured["n"] is None
    assert captured["is_prefill"] is False
    assert cache.resident_source_access_attempts == 0


def test_resident_only_prefill_uses_layer_aliases_for_canonical_population(monkeypatch):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    cache, _ = _cache(num_layers=2, num_experts=3)
    _populate(cache)
    cache.detach_host_sources_for_full_residency()
    layer = OffloadMoELayer(1, 3, 2, 2, 2)
    layer.offload_cache = cache
    routed = torch.tensor([[2, 0]], dtype=torch.int32)
    captured = {}

    def expert_gemm(cache_arg, hidden, weights, ids, **kwargs):
        captured.update(ids=ids.clone(), **kwargs)
        return hidden

    monkeypatch.setattr(layer, "_expert_gemm", expert_gemm)
    hidden = torch.randn(1, 2)
    out = layer._prefill_routed(hidden, torch.ones(1, 2), routed)

    assert out is hidden
    assert captured["ids"].tolist() == [[2, 0]]
    assert captured["n"] == 3 and captured["is_prefill"] is True
    for actual, full in zip(captured["views"], cache.bank_views(), strict=True):
        assert actual.data_ptr() == full[3:].data_ptr()
        assert actual.shape[0] == 3
    assert cache.resident_source_access_attempts == 0


def test_source_dependent_operations_fail_closed_after_detach():
    cache, _ = _cache()
    _populate(cache)
    cache.detach_host_sources_for_full_residency()

    operations = [
        lambda: cache.reset(),
        lambda: cache.rebuild(cache.cache_size),
        lambda: cache.ensure_experts(0, torch.tensor([0], dtype=torch.int32)),
        lambda: cache.copy_missing(),
        lambda: cache.materialize_layer(0),
        lambda: cache.begin_prefill(),
    ]
    for operation in operations:
        with pytest.raises(RuntimeError, match="resident-only"):
            operation()
    assert cache.resident_source_access_attempts == len(operations)


def test_selective_load_result_releases_its_source_owner_only_after_detach():
    from freetoken.research.n0_model_block import SelectiveBlockLoadResult

    cache, sources = _cache()
    result = SelectiveBlockLoadResult(None, sources, (0, 1), frozenset(), [], 0)
    with pytest.raises(RuntimeError, match="after cache resident-only"):
        result.release_expert_banks_after_residency(cache)

    expected = sum(
        t.numel() * t.element_size() for layers in sources.values() for t in layers
    )
    _populate(cache)
    cache.detach_host_sources_for_full_residency()

    assert result.release_expert_banks_after_residency(cache) == expected
    assert result.expert_banks == {}


def test_ordinary_offload_still_retains_host_banks_and_can_rebuild():
    cache, sources = _cache()
    before = cache.host_source_tensor_bytes()

    cache.rebuild(cache.cache_size + 1)

    assert cache.resident_only is False
    assert cache.host_source_tensor_bytes() == before
    assert cache.bank_sources["down"][1] is sources["down"][1]
    assert cache.resident_source_access_attempts == 0
