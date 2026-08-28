"""Public MoE instrumentation behavior that does not require CUDA."""

from __future__ import annotations

import torch
import pytest

from freetoken.moe.offload_cache import OffloadMoeCache


def _cache(*, trace_steps: int = 0, device: torch.device | None = None) -> OffloadMoeCache:
    return OffloadMoeCache(
        num_layers=2,
        num_experts=6,
        cache_size=6,
        device=device or torch.device("cpu"),
        trace_max_steps=trace_steps,
        trace_max_tokens_per_step=2,
        trace_top_k=3,
    )


def test_instrumentation_is_disabled_by_default():
    cache = _cache()

    snapshot = cache.instrumentation_snapshot()

    assert snapshot["schema"] == "freetoken.moe-instrumentation/1"
    assert snapshot["collection"]["stats_enabled"] is False
    assert snapshot["trace"]["enabled"] is False
    assert snapshot["trace"]["capacity_steps"] == 0


def test_exact_trace_preserves_step_layer_token_and_topk_order():
    cache = _cache(trace_steps=2)
    cache.record_decode_routing(0, torch.tensor([[4, 1, 5], [0, 2, 3]], dtype=torch.int32))
    cache.record_decode_routing(1, torch.tensor([[3, 2, 1], [5, 4, 0]], dtype=torch.int32))
    cache.record_decode_routing(0, torch.tensor([[1, 0, 2]], dtype=torch.int32))
    cache.record_decode_routing(1, torch.tensor([[2, 4, 5]], dtype=torch.int32))

    trace = cache.instrumentation_snapshot()["trace"]

    assert trace["truncated"] is False
    assert trace["steps_recorded"] == 2
    assert trace["records"] == [
        {
            "step": 0,
            "layers": [
                {"layer": 0, "token_routes": [[4, 1, 5], [0, 2, 3]]},
                {"layer": 1, "token_routes": [[3, 2, 1], [5, 4, 0]]},
            ],
        },
        {
            "step": 1,
            "layers": [
                {"layer": 0, "token_routes": [[1, 0, 2]]},
                {"layer": 1, "token_routes": [[2, 4, 5]]},
            ],
        },
    ]


def test_exact_trace_reports_step_overflow_instead_of_silently_accepting_it():
    cache = _cache(trace_steps=1)
    route = torch.tensor([[4, 1, 5]], dtype=torch.int32)
    for layer in range(2):
        cache.record_decode_routing(layer, route)
        cache.record_decode_routing(layer, route)

    trace = cache.instrumentation_snapshot()["trace"]

    assert trace["truncated"] is True
    assert trace["overflow_layer_calls"] == 2
    assert trace["steps_observed"] == 2
    assert len(trace["records"]) == 1


def test_snapshot_reuses_authoritative_counters_and_routing_histogram():
    cache = _cache(trace_steps=1)
    cache.collect_stats = True
    cache.collect_decode_freq = True
    # Columns are ACTIVE, MISS, CALLS from flashlib's authoritative LRU stats.
    cache.lru_stats.copy_(torch.tensor([[3, 1, 1], [2, 0, 1]], dtype=torch.int64))
    cache.record_decode_routing(0, torch.tensor([[4, 1, 4]], dtype=torch.int32))
    cache.record_decode_routing(1, torch.tensor([[2, 3, 5]], dtype=torch.int32))

    snapshot = cache.instrumentation_snapshot()

    assert snapshot["aggregate"] == {
        "decode_steps": 1,
        "layer_calls": 2,
        "active_selections": 5,
        "hits": 4,
        "misses": 1,
        "fetches": 1,
        "active_per_layer": 2.5,
        "missing_per_layer": 0.5,
        "miss_rate": 0.2,
        "fetched_per_layer": 0.5,
        "cpu_per_layer": 0.0,
        "fetch_rate": 1.0,
        "prefill_hit_rows": 0,
        "prefill_rows": 0,
    }
    assert snapshot["routing"]["histogram"][0] == [0, 1, 0, 0, 2, 0]
    assert snapshot["routing"]["histogram"][1] == [0, 0, 1, 1, 0, 1]


def test_residency_comes_from_authoritative_slot_map_and_reset_preserves_it():
    cache = _cache(trace_steps=2)
    # Flat ids encode layer * num_experts + expert id.
    cache.id_of_slot.copy_(torch.tensor([4, 7, -1, 11, -1, -1], dtype=torch.int32))
    cache.record_decode_routing(0, torch.tensor([[1, 2, 3]], dtype=torch.int32))

    before = cache.instrumentation_snapshot()["residency"]
    cache.reset_instrumentation()
    after = cache.instrumentation_snapshot()

    assert before["source"] == "id_of_slot_authoritative_slot_map"
    assert before["configured_slots"] == 6
    assert before["actual_resident_slots"] == 3
    assert before["per_layer"] == [
        {"layer": 0, "resident_expert_ids": [4]},
        {"layer": 1, "resident_expert_ids": [1, 5]},
    ]
    assert after["residency"] == before
    assert after["trace"]["records"] == []
    assert after["aggregate"]["active_selections"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="on-device trace indexing needs CUDA")
def test_exact_trace_accumulates_on_cuda_until_snapshot_boundary():
    cache = _cache(trace_steps=1, device=torch.device("cuda"))
    cache.record_decode_routing(
        0, torch.tensor([[5, 3, 1]], dtype=torch.int32, device="cuda")
    )
    cache.record_decode_routing(
        1, torch.tensor([[2, 4, 0]], dtype=torch.int32, device="cuda")
    )

    trace = cache.instrumentation_snapshot()["trace"]

    assert trace["truncated"] is False
    assert trace["records"][0]["layers"][0]["token_routes"] == [[5, 3, 1]]
