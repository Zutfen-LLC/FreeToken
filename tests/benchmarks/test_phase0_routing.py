from __future__ import annotations

import json

import pytest

from inferswarm_phase0.manifest import load_manifest
from inferswarm_phase0.routing import (
    RoutingSettings,
    build_routing_plan,
    cache_sweep_points,
    generation_evidence,
)

from .fakes import FAKE_UUID, write_manifest


def test_cache_fraction_points_are_predeclared_quartiles_and_deduplicated():
    assert cache_sweep_points(256, 900, num_layers=40, num_experts=256) == [
        {"resolved_slots": 256, "cache_fraction": 0.025},
        {"resolved_slots": 417, "cache_fraction": 417 / 10240},
        {"resolved_slots": 578, "cache_fraction": 578 / 10240},
        {"resolved_slots": 739, "cache_fraction": 739 / 10240},
        {"resolved_slots": 900, "cache_fraction": 900 / 10240},
    ]
    assert [p["resolved_slots"] for p in cache_sweep_points(5, 7, 2, 4)] == [5, 6, 7]


def test_dry_run_separates_eager_trace_from_graph_safe_pressure(tmp_path):
    manifest = load_manifest(write_manifest(tmp_path), canonical=True)
    settings = RoutingSettings(model_path="/models/qwen", gpu=FAKE_UUID, trace_max_steps=512)

    plan = build_routing_plan(
        manifest, settings, session_id="s1", warmups=1, repetitions=2, canonical=True
    )

    exact = plan["server_modes"]["exact_trace"]["command"]
    pressure = plan["server_modes"]["cache_pressure"]["command"]
    assert exact[exact.index("--cuda-graph-max-bs") + 1] == "0"
    assert "--moe-trace-max-steps" in exact
    assert pressure[pressure.index("--cuda-graph-max-bs") + 1] == "1"
    assert "--moe-trace-max-steps" not in pressure
    assert plan["workload_classes"] == ["W1", "W2", "W3", "W4"]
    assert plan["canonical"] is True


def test_canonical_plan_rejects_a_workload_subset(tmp_path):
    manifest = load_manifest(
        write_manifest(tmp_path, classes=("W1", "W2"), canonical=False), canonical=False
    )
    with pytest.raises(ValueError, match="all frozen W1-W4"):
        build_routing_plan(
            manifest,
            RoutingSettings(model_path="/models/qwen"),
            session_id="s",
            warmups=1,
            repetitions=1,
            canonical=True,
        )


def test_generation_artifact_contains_hashes_and_counts_but_no_text():
    body = {"messages": [{"role": "user", "content": "secret prompt"}], "max_tokens": 3}
    result = {
        "text": "secret output",
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        "response_id": "chatcmpl-7",
    }

    artifact = generation_evidence(body, result, fixture_sha256="a" * 64)
    encoded = json.dumps(artifact)

    assert artifact["prompt_tokens"] == 11
    assert artifact["completion_tokens"] == 3
    assert artifact["fixture_sha256"] == "a" * 64
    assert len(artifact["request_sha256"]) == 64
    assert len(artifact["output_sha256"]) == 64
    assert "secret prompt" not in encoded
    assert "secret output" not in encoded
    assert "text" not in artifact
