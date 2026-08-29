from __future__ import annotations

import pytest
import torch
from freetoken.moe.layer_timing import (
    MARKER_ID,
    METRIC_ID,
    MoeLayerTiming,
    absent_moe_layer_timing_report,
)


def _timing(*, role="baseline", max_steps=2, overlap=False):
    return MoeLayerTiming(
        max_steps=max_steps,
        num_layers=1,
        device=torch.device("cpu"),
        bytes_per_identity=100,
        role=role,
        graph_requested=True,
        remote_overlap_active=overlap,
    )


def _fill_baseline_record(timing, step=0):
    timing.begin_decode_step(step, batch_size=1, padded_batch_size=1, graph_replay=True)
    timing.layer_steps[0] = step + 1
    base = 1_000_000_000 + step * 100_000_000
    values = {
        "complete_start": base,
        "local_start": base + 1_000,
        "cache_service_end": base + 2_000_000,
        "weight_fetch_end": base + 5_000_000,
        "local_expert_end": base + 9_000_000,
        "local_branch_end": base + 9_001_000,
        "complete_end": base + 10_000_000,
    }
    for name, value in values.items():
        timing.timestamps[step, 0, MARKER_ID[name]] = value
    timing.metadata[step, 0, METRIC_ID["total_route_selections"]] = 2
    timing.metadata[step, 0, METRIC_ID["local_unique_experts"]] = 2
    timing.metadata[step, 0, METRIC_ID["local_cache_misses"]] = 1
    timing.metadata[step, 0, METRIC_ID["local_fetched_experts"]] = 1
    timing.metadata[step, 0, METRIC_ID["host_to_gpu0_expert_weight_bytes"]] = 100


def test_timing_disabled_report_is_explicit_and_empty():
    report = absent_moe_layer_timing_report()
    assert report["enabled"] is False
    assert report["capacity_steps"] == 0
    assert report["records"] == []
    assert report["timer"] is None


def test_common_baseline_schema_has_complete_wall_components_and_remote_na():
    timing = _timing()
    timing.set_graph_state([1])
    _fill_baseline_record(timing)
    report = timing.snapshot()
    record = report["records"][0]
    assert report["graph"] == {
        "requested": True,
        "captured_batch_sizes": [1],
        "active": True,
    }
    assert report["timer"]["host_sync_per_layer"] is False
    assert record["durations"]["complete_layer"]["value_ms"] == 10.0
    assert (
        record["durations"]["gpu0_branch"]["host_to_gpu0_expert_fetch_copy"]["value_ms"]
        == 3.0
    )
    assert (
        record["durations"]["gpu0_branch"]["local_expert_execution"]["value_ms"] == 4.0
    )
    assert (
        record["durations"]["gpu1_branch"]["gpu1_route_contribution_execution"][
            "status"
        ]
        == "not_applicable"
    )
    assert (
        record["durations"]["remote_dispatch_control"][
            "gpu0_to_host_staging_host_wait"
        ]["status"]
        == "not_applicable"
    )
    assert record["transfer_bytes"]["host_to_gpu1"]["activation"]["status"] == (
        "not_applicable"
    )
    assert report["validity"]["complete_layer_timing_valid"] is True


def test_candidate_uses_same_schema_and_does_not_sum_concurrent_branches():
    timing = _timing(role="candidate", overlap=True)
    _fill_baseline_record(timing)
    base = int(timing.timestamps[0, 0, MARKER_ID["complete_start"]])
    for name, value in {
        "returned_route_contributions_h2d_start": base + 9_100_000,
        "returned_route_contributions_h2d_end": base + 9_300_000,
        "route_reconstruction_start": base + 9_301_000,
        "route_reconstruction_end": base + 9_500_000,
        "final_sum_reduce_start": base + 9_501_000,
        "final_sum_reduce_end": base + 9_800_000,
    }.items():
        timing.timestamps[0, 0, MARKER_ID[name]] = value
    timing.annotate(
        0,
        0,
        {
            "candidate": True,
            "identity": {
                "decode_step": 0,
                "layer_id": 0,
                "total_route_selections": 2,
                "gpu0_owned_selections": 1,
                "gpu1_owned_selections": 1,
                "unique_gpu1_expert_identities": 1,
                "dispatch_count": 1,
            },
            "transfer_bytes": {},
            "durations": {
                "classification_control_host_wait": {
                    "status": "valid",
                    "value_ms": 0.2,
                    "source": "host_monotonic",
                },
                "gpu0_to_host_activation_routing": {
                    "status": "valid",
                    "value_ms": 1.0,
                    "source": "cuda_event_gpu0",
                },
                "gpu0_to_host_staging_host_wait": {
                    "status": "valid",
                    "value_ms": 0.5,
                    "source": "host_monotonic",
                },
                "host_remote_submit_control": {
                    "status": "valid",
                    "value_ms": 0.1,
                    "source": "host_monotonic",
                },
                "host_to_gpu1_payload_h2d": {
                    "status": "valid",
                    "value_ms": 2.0,
                    "source": "cuda_event_gpu1",
                },
                "gpu1_route_contribution_execution": {
                    "status": "valid",
                    "value_ms": 5.0,
                    "source": "cuda_event_gpu1",
                },
                "gpu1_to_host_route_contributions_d2h": {
                    "status": "valid",
                    "value_ms": 2.0,
                    "source": "cuda_event_gpu1",
                },
                "complete_gpu1_branch": {
                    "status": "valid",
                    "value_ms": 9.0,
                    "source": "cuda_event_gpu1",
                },
                "host_remote_join_wait": {
                    "status": "valid",
                    "value_ms": 1.0,
                    "source": "host_monotonic",
                },
            },
        },
    )
    record = timing.snapshot()["records"][0]
    assert record["durations"]["complete_layer"]["value_ms"] == 10.0
    assert record["durations"]["gpu0_branch"]["complete_local_branch"]["value_ms"] > 0
    assert record["durations"]["gpu1_branch"]["complete_gpu1_branch"]["value_ms"] == 9.0
    join = record["durations"]["join_reconstruct_reduce"]
    assert join["host_to_gpu0_returned_route_contributions"]["value_ms"] == 0.2
    assert join["route_reconstruction"]["value_ms"] == pytest.approx(0.199)
    assert join["final_moe_sum_reduce"]["value_ms"] == pytest.approx(0.299)
    assert "sum" not in record["durations"]


def test_timing_capacity_truncation_and_reset_preserve_object():
    timing = _timing(max_steps=1)
    _fill_baseline_record(timing)
    timing.steps_observed = 2
    timing.layer_steps[0] = 2
    report = timing.snapshot()
    assert report["steps_observed"] == 2
    assert report["steps_retained"] == 1
    assert report["truncated"] is True
    storage = timing.timestamps
    timing.reset()
    assert timing.timestamps is storage
    assert timing.layer_steps.tolist() == [0]
    assert timing.snapshot()["records"] == []


def test_cumulative_lru_metadata_becomes_per_replay_deltas():
    timing = _timing(max_steps=2)
    _fill_baseline_record(timing, step=0)
    _fill_baseline_record(timing, step=1)
    # Device marker values are cumulative because flashlib updates its LRU stats
    # in the graph-replayed admission kernel.
    timing.metadata[0, 0, METRIC_ID["local_unique_experts"]] = 2
    timing.metadata[0, 0, METRIC_ID["local_cache_misses"]] = 2
    timing.metadata[1, 0, METRIC_ID["local_unique_experts"]] = 5
    timing.metadata[1, 0, METRIC_ID["local_cache_misses"]] = 3

    records = timing.snapshot()["records"]
    assert records[0]["cache_service"]["local_unique_experts"] == 2
    assert records[0]["cache_service"]["local_cache_misses"] == 2
    assert records[1]["cache_service"]["local_unique_experts"] == 3
    assert records[1]["cache_service"]["local_cache_misses"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph timing test")
def test_globaltimer_markers_survive_capture_and_update_each_replay():
    device = torch.device("cuda", torch.cuda.current_device())
    timing = MoeLayerTiming(
        max_steps=3,
        num_layers=1,
        device=device,
        bytes_per_identity=100,
        role="baseline",
        graph_requested=True,
        remote_overlap_active=False,
    )
    value = torch.ones(256, device=device)

    # Compile every marker outside capture, as GraphRunner's eager pre-capture pass does.
    timing.mark(0, "complete_start", begin_layer=True)
    timing.mark(0, "local_start")
    timing.mark(0, "cache_service_end")
    timing.mark(0, "weight_fetch_end")
    value.add_(1)
    timing.mark(0, "local_expert_end")
    timing.mark(0, "local_branch_end")
    timing.mark(0, "complete_end")
    torch.cuda.synchronize(device)
    timing.reset()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        timing.mark(0, "complete_start", begin_layer=True)
        timing.mark(0, "local_start")
        timing.mark(0, "cache_service_end")
        timing.mark(0, "weight_fetch_end")
        value.add_(1)
        timing.mark(0, "local_expert_end")
        timing.mark(0, "local_branch_end")
        timing.mark(0, "complete_end")
    timing.reset()

    for step in range(2):
        timing.begin_decode_step(
            step, batch_size=1, padded_batch_size=1, graph_replay=True
        )
        graph.replay()
        # Mimic the scheduler's already-required output completion boundary.
        done = torch.cuda.Event()
        done.record()
        done.synchronize()

    starts = timing.timestamps[:2, 0, MARKER_ID["complete_start"]].tolist()
    report = timing.snapshot()
    assert starts[0] > 0 and starts[1] > starts[0]
    assert report["steps_retained"] == 2
    assert report["validity"]["complete_layer_timing_valid"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA timer calibration test")
def test_globaltimer_nanoseconds_crosscheck_same_stream_cuda_event():
    device = torch.device("cuda", torch.cuda.current_device())
    timing = MoeLayerTiming(
        max_steps=1,
        num_layers=1,
        device=device,
        bytes_per_identity=100,
        role="baseline",
        graph_requested=False,
        remote_overlap_active=False,
    )
    # Compile markers before the measured interval.
    timing.mark(0, "complete_start", begin_layer=True)
    timing.mark(0, "complete_end")
    torch.cuda.synchronize(device)
    timing.reset()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    timing.mark(0, "complete_start", begin_layer=True)
    torch.cuda._sleep(20_000_000)
    timing.mark(0, "complete_end")
    end.record()
    end.synchronize()

    marker_ms = timing.snapshot()["records"][0]["durations"]["complete_layer"][
        "value_ms"
    ]
    event_ms = start.elapsed_time(end)
    # CUDA events enclose both marker kernels; the marker interval excludes their outer
    # launch edges. The same-stream values should nevertheless agree in timer scale.
    assert marker_ms > 0
    assert marker_ms == pytest.approx(event_ms, rel=0.20, abs=0.5)
