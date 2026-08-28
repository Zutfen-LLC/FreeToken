from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inferswarm_phase0 import routing as routing_mod
from inferswarm_phase0 import validity as V
from inferswarm_phase0.manifest import load_manifest
from inferswarm_phase0.routing import (
    RoutingCampaign,
    RoutingSettings,
    build_routing_plan,
    cache_sweep_points,
    generation_evidence,
)

from .fakes import FAKE_UUID, SHA40, instrumentation_doc, runtime_config, write_manifest


def _routing_campaign(tmp_path, *, canonical=True):
    return RoutingCampaign(
        manifest=load_manifest(write_manifest(tmp_path, canonical=canonical), canonical=canonical),
        settings=RoutingSettings(
            model_path=str(tmp_path / "model"),
            model_revision=SHA40,
            gpu=FAKE_UUID,
            python_executable="python",
            trace_max_steps=512,
        ),
        out_root=tmp_path / "runs",
        short_name="routing-unit",
        session_id="routing-session",
        inferswarm_commit="b" * 40,
        warmups=0,
        repetitions=1,
        canonical=canonical,
        echo_server_output=False,
    )


@pytest.fixture
def routed_server(monkeypatch, resolved_gpu):
    """A complete valid routing campaign; tests break exactly one public contract fact."""
    state = {
        "mode": None,
        "slots": 1024,
        "last_completion": 0,
        "generation": 0,
        "prompt_override": None,
        "completion_delta": 0,
        "trace_truncated": False,
        "trace_step_delta": 0,
        "runtime_missing": False,
        "runtime_backend": "offload",
        "gpu_rows": [{
            "index": 0,
            "uuid": FAKE_UUID,
            "name": "NVIDIA GeForce RTX 3060",
            "total_bytes": 12 << 30,
        }],
        "point_policy": "lru",
        "point_slot_delta": 0,
        "fail_generation": None,
    }

    monkeypatch.setattr(routing_mod.prov, "validate_revision", lambda *a, **k: None)
    monkeypatch.setattr(routing_mod.prov, "validate_inferswarm_commit", lambda *a, **k: None)
    monkeypatch.setattr(routing_mod.prov, "check_snapshot_revision", lambda *a, **k: None)
    monkeypatch.setattr(routing_mod.prov, "check_clean_working_tree", lambda *a, **k: None)
    monkeypatch.setattr(routing_mod.prov, "git_commit", lambda *a, **k: {"value": "a" * 40})
    monkeypatch.setattr(
        routing_mod.prov,
        "software_provenance",
        lambda *a, **k: {"freetoken_commit": "a" * 40, "inferswarm_commit": "b" * 40},
    )
    monkeypatch.setattr(
        routing_mod.prov,
        "model_provenance",
        lambda *a, **k: {"repository": routing_mod.CANONICAL_MODEL_REPOSITORY, "revision": SHA40},
    )
    monkeypatch.setattr(
        routing_mod.prov,
        "host_provenance",
        lambda: {"os": "Linux", "cpu_model": "test", "ram_total_bytes": 64 << 30},
    )
    monkeypatch.setattr(
        routing_mod.prov,
        "gpu_provenance",
        lambda *a, **k: {"gpus": state["gpu_rows"], "selected": {"resolved_uuid": FAKE_UUID}},
    )
    monkeypatch.setattr(routing_mod.prov, "missing_required", lambda provenance: [])

    def start_mode(self, root, *, exact):
        state["mode"] = "exact" if exact else "pressure"
        command = routing_mod._serve_command(self.settings, port=1234, exact=exact, gpu=FAKE_UUID)
        return SimpleNamespace(proc=None), f"http://{state['mode']}", command

    monkeypatch.setattr(RoutingCampaign, "_start_mode", start_mode)
    monkeypatch.setattr(routing_mod, "stop_server", lambda handle: None)
    monkeypatch.setattr(routing_mod, "_model_id", lambda origin: "qwen-test")
    monkeypatch.setattr(routing_mod.gpu_mod, "engine_gpus", lambda origin: list(state["gpu_rows"]))

    def live_runtime(origin):
        if state["runtime_missing"]:
            return {"unavailable": "simulated missing runtime identity"}
        exact = origin.endswith("exact")
        config = runtime_config(
            model={"is_moe": True, "top_k": 8},
            moe={
                "backend_resolved": state["runtime_backend"],
                "decode_target": "gpu",
                "collect_stats": True,
                "trace_enabled": exact,
                "trace_max_steps": 512 if exact else 0,
            },
            cache={"moe_cache_policy": "lru"},
            runtime={"cuda_graph_max_bs": 0 if exact else 1},
        )
        return instrumentation_doc(config=config)

    monkeypatch.setattr(routing_mod, "fetch_instrumentation", live_runtime)

    def get_json(url, timeout=10):
        if url.endswith("/v1/cache/status"):
            return {
                "geometry": {
                    "moe_cache_size": 1024,
                    "num_moe_layers": 40,
                    "num_experts": 256,
                    "limits": {"moe_experts": {"min": 256}},
                }
            }
        if url.endswith("/v1/instrumentation"):
            return live_runtime(url)
        raise AssertionError(url)

    monkeypatch.setattr(routing_mod, "get_json", get_json)

    def rebuild(origin, slots):
        state["slots"] = slots
        return {"status": "ok", "moe_cache_size": slots}

    monkeypatch.setattr(routing_mod, "_rebuild", rebuild)

    def snapshot(origin, operation):
        exact = origin.endswith("exact")
        completion = state["last_completion"]
        steps = max(0, completion - 1) + state["trace_step_delta"]
        live_slots = state["slots"] + (state["point_slot_delta"] if not exact else 0)
        return {
            "schema": "freetoken.moe-instrumentation/1",
            "geometry": {
                "resolved_cache_slots": live_slots,
                "num_moe_layers": 40,
                "num_routed_experts": 256,
                "cache_policy": state["point_policy"],
                "decode_target": "gpu",
                "prefill_overlap_active": False if not exact else True,
            },
            "collection": {
                "stats_enabled": True,
                "routing_histogram_enabled": exact,
                "trace_enabled": exact,
            },
            "aggregate": {
                "decode_steps": steps,
                "active_selections": steps * 8 * 40,
                "misses": 0,
                "fetches": 0,
                "miss_rate": 0.0,
            },
            "per_layer": [],
            "routing": {"histogram": [], "derived_concentration": {}},
            "trace": {
                "enabled": exact,
                "capacity_steps": 512 if exact else 0,
                "truncated": state["trace_truncated"] if exact else False,
                "steps_observed": steps if exact else 0,
                "steps_recorded": steps if exact else 0,
                "records": [],
            },
            "residency": {
                "configured_slots": live_slots,
                "actual_resident_slots": 0,
                "per_layer": [],
                "slot_map": [],
            },
            "boundary": {"residency_preserved_by_reset": operation == "reset"},
        }

    monkeypatch.setattr(routing_mod, "_control", snapshot)

    def generation(origin, body, timeout=3600.0):
        state["generation"] += 1
        if state["fail_generation"] == state["generation"]:
            raise OSError("simulated interrupted observation")
        class_id = body["messages"][0]["content"].rsplit(" ", 1)[-1]
        prompt = {"W1": 1800, "W2": 900, "W3": 16000, "W4": 128}[class_id]
        completion = body["max_tokens"] + state["completion_delta"]
        state["last_completion"] = completion
        return {
            "text": "not serialized",
            "usage": {
                "prompt_tokens": state["prompt_override"] or prompt,
                "completion_tokens": completion,
            },
            "response_id": f"response-{state['generation']}",
        }

    monkeypatch.setattr(routing_mod, "stream_generation", generation)
    return state


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
    assert "--disable-moe-prefill-overlap" not in exact
    assert pressure[pressure.index("--cuda-graph-max-bs") + 1] == "1"
    assert "--moe-trace-max-steps" not in pressure
    assert "--disable-moe-prefill-overlap" in pressure
    assert plan["workload_classes"] == ["W1", "W2", "W3", "W4"]
    assert plan["canonical_intent"] is True
    assert plan["planned_validity"] == "PENDING_RUNTIME_VALIDATION"


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


def test_complete_routing_campaign_uses_phase0_validity_vocabulary(
    tmp_path, routed_server
):
    doc = _routing_campaign(tmp_path).execute()

    assert doc["canonical_intent"] is True
    assert doc["execution_status"] == V.EXECUTION_COMPLETE
    assert doc["validity"] == V.VALIDITY_VALID
    assert doc["headline"] == "VALID CANONICAL CAMPAIGN"
    assert "canonical" not in doc


@pytest.mark.parametrize(
    "mutation,reason_code",
    [
        ({"prompt_override": 99999}, V.PROMPT_SHAPE_VIOLATION),
        ({"completion_delta": -1}, V.COMPLETION_LENGTH_MISMATCH),
        ({"runtime_missing": True}, V.INSTRUMENTATION_UNAVAILABLE),
        ({"runtime_backend": "hybrid"}, V.ROUTING_RUNTIME_IDENTITY_MISMATCH),
        ({"gpu_rows": []}, V.GPU_UNPROVEN),
        ({"gpu_rows": [{"uuid": "GPU-99999999-9999-9999-9999-999999999999"}]}, V.GPU_MISMATCH),
        ({"trace_truncated": True}, "routing.trace_truncated"),
        ({"trace_step_delta": -1}, "routing.trace_incomplete"),
    ],
)
def test_runtime_contract_violations_preserve_observations_and_invalidate(
    tmp_path, routed_server, mutation, reason_code
):
    routed_server.update(mutation)

    doc = _routing_campaign(tmp_path).execute()

    assert doc["execution_status"] == V.EXECUTION_COMPLETE
    assert doc["validity"] == V.VALIDITY_INVALID
    assert doc["headline"] == "INVALID CANONICAL ATTEMPT"
    assert reason_code in doc["campaign_invalidation_codes"]
    exact = Path(doc["run_directory"]) / "exact-routing.jsonl"
    assert any(json.loads(line)["record_type"] == "measured_repetition" for line in exact.read_text().splitlines())


def test_missing_planned_observation_is_incomplete(tmp_path, routed_server):
    routed_server["fail_generation"] = 2

    doc = _routing_campaign(tmp_path).execute()

    assert doc["execution_status"] == V.EXECUTION_INCOMPLETE
    assert doc["headline"] == "INCOMPLETE RUN"
    assert doc["observations"]["observed"] < doc["observations"]["expected"]


def test_interrupted_campaign_leaves_an_incomplete_run_document(
    tmp_path, routed_server, monkeypatch
):
    def interrupted(self, root, *, exact):
        raise KeyboardInterrupt

    monkeypatch.setattr(RoutingCampaign, "_start_mode", interrupted)

    with pytest.raises(KeyboardInterrupt):
        _routing_campaign(tmp_path).execute()

    run_json = next((tmp_path / "runs").glob("*/run.json"))
    doc = json.loads(run_json.read_text())
    assert doc["execution_status"] == V.EXECUTION_INCOMPLETE
    assert doc["headline"] == "INCOMPLETE RUN"
    assert doc["observations"]["expected"] == 4
    assert doc["observations"]["observed"] == 0


def test_developer_smoke_campaign_remains_non_canonical(tmp_path, routed_server):
    doc = _routing_campaign(tmp_path, canonical=False).execute()

    assert doc["canonical_intent"] is False
    assert doc["execution_status"] == V.EXECUTION_COMPLETE
    assert doc["validity"] == V.VALIDITY_NON_CANONICAL
    assert doc["headline"] == "NON-CANONICAL DEVELOPER RUN"


@pytest.mark.parametrize(
    "mutation",
    [
        {"point_slot_delta": 1},
        {"point_policy": "fifo"},
    ],
)
def test_live_pressure_point_mismatch_invalidates_and_refuses_the_point(
    tmp_path, routed_server, mutation
):
    routed_server.update(mutation)

    doc = _routing_campaign(tmp_path).execute()

    assert doc["validity"] == V.VALIDITY_INVALID
    assert doc["execution_status"] == V.EXECUTION_INCOMPLETE
    assert "routing.cache_pressure_point_contract" in doc["campaign_invalidation_codes"]
    rows = [
        json.loads(line)
        for line in (Path(doc["run_directory"]) / "cache-pressure.jsonl").read_text().splitlines()
    ]
    point = next(row for row in rows if row["record_type"] == "point_pre_warmup")
    assert point["point_runtime_provenance"]["geometry"]["prefill_overlap_active"] is False
    assert point["point_contract"]["valid"] is False


def test_pressure_point_provenance_pins_live_prefill_policy_at_the_minimum(
    tmp_path, routed_server
):
    doc = _routing_campaign(tmp_path).execute()

    rows = [
        json.loads(line)
        for line in (Path(doc["run_directory"]) / "cache-pressure.jsonl").read_text().splitlines()
    ]
    minimum = next(
        row for row in rows
        if row["record_type"] == "point_pre_warmup"
        and row["cache_point"]["resolved_slots"] == 256
    )
    assert minimum["cache_point"]["resolved_slots"] == 256
    assert minimum["point_runtime_provenance"]["geometry"]["prefill_overlap_active"] is False
    assert minimum["point_contract"]["valid"] is True


def test_quantity_labels_match_the_data_each_routing_mode_actually_emits(
    tmp_path, routed_server
):
    doc = _routing_campaign(tmp_path).execute()
    root = Path(doc["run_directory"])

    def first_measured(name):
        return next(
            json.loads(line)
            for line in (root / name).read_text().splitlines()
            if json.loads(line)["record_type"] == "measured_repetition"
        )

    exact = first_measured("exact-routing.jsonl")["quantity_labels"]
    pressure = first_measured("cache-pressure.jsonl")["quantity_labels"]

    assert set(exact["measured"]) == {
            "active_selections",
            "misses",
            "fetches",
            "decode_steps",
            "resident_expert_ids",
            "routing_histogram",
            "exact_routes",
    }
    assert set(exact["derived"]) == {
        "miss_rate", "cache_fraction", "routing_concentration"
    }
    assert set(pressure["measured"]) == {
            "active_selections",
            "misses",
            "fetches",
            "decode_steps",
            "resident_expert_ids",
    }
    assert set(pressure["derived"]) == {"miss_rate", "cache_fraction"}
