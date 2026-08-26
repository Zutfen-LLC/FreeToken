"""The session-level ``ft bench bw`` prerequisite.

The failure this file exists to prevent: a campaign that produced 200 clean-looking
repetitions while B2's fetch split and B3's ``auto`` backend pick were resolved from a
bandwidth profile the campaign never produced -- or from another card's.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .fakes import FAKE_UUID, SHA40, good_bench_bw_record, instrumentation_doc, write_manifest

from inferswarm_phase0 import bench_bw as bench_bw_mod
from inferswarm_phase0 import runner as runner_mod
from inferswarm_phase0 import validity as V
from inferswarm_phase0.baselines import BASELINE_ARMS, BASELINE_ARMS_BY_ID
from inferswarm_phase0.gpu import GpuSelection
from inferswarm_phase0.manifest import load_manifest
from inferswarm_phase0.protocol import build_protocol
from inferswarm_phase0.runner import Campaign, ServeSettings

SELECTION = GpuSelection(requested=FAKE_UUID, resolved_uuid=FAKE_UUID, physical_index=0)


# --- which arms consume the profile ---------------------------------------------------------

def test_b2_and_b3_both_consume_the_profile():
    """B2 resolves its fetch split from it; B3's --moe-backend auto reads the same profile
    to decide whether to upgrade offload to hybrid (engine._adjust_config)."""
    assert bench_bw_mod.consuming_arms(BASELINE_ARMS) == ["B2", "B3"]
    assert BASELINE_ARMS_BY_ID["B2"].requires_bench_bw is True
    # B3 is not named by the criteria table, but it reads the profile all the same
    assert BASELINE_ARMS_BY_ID["B3"].requires_bench_bw is False
    assert BASELINE_ARMS_BY_ID["B3"].consumes_bench_bw is True


# --- capturing the exact profile the engine will read ------------------------------------------

def test_the_profile_is_pinned_by_content_hash(tmp_path):
    profile = tmp_path / "benchbw.json"
    contents = {"gpu": {"uuid": FAKE_UUID, "name": "NVIDIA GeForce RTX 3060"},
                "dtypes": {"nvfp4": "hybrid"}}
    profile.write_text(json.dumps(contents))
    block = bench_bw_mod.capture_profile(str(profile), SELECTION)
    assert block["gpu_matches"] is True
    assert block["contents"] == contents
    assert len(block["sha256"]) == 64
    # the hash is over the raw bytes, not over a re-serialization
    import hashlib
    assert block["sha256"] == hashlib.sha256(profile.read_bytes()).hexdigest()


def test_a_missing_profile_is_an_explicit_reason(tmp_path):
    block = bench_bw_mod.capture_profile(str(tmp_path / "nope.json"), SELECTION)
    assert "sha256" not in block
    assert "could not be read" in block["unavailable"]


def test_an_unparseable_profile_is_an_explicit_reason(tmp_path):
    profile = tmp_path / "benchbw.json"
    profile.write_text("{not json")
    block = bench_bw_mod.capture_profile(str(profile), SELECTION)
    assert "not readable JSON" in block["unavailable"]


def test_a_profile_benched_on_another_gpu_is_detected(tmp_path):
    profile = tmp_path / "benchbw.json"
    profile.write_text(json.dumps({"gpu": {"uuid": "GPU-99999999-0000-0000-0000-000000000000"}}))
    block = bench_bw_mod.capture_profile(str(profile), SELECTION)
    assert block["gpu_matches"] is False
    assert "benched on" in block["gpu_mismatch"]


def test_the_command_names_the_resolved_uuid_not_the_raw_selector(monkeypatch, tmp_path):
    captured = {}

    class _Completed:
        returncode = 0
        stdout = f"FTBENCH_OUT {tmp_path / 'p.json'}\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return _Completed()

    (tmp_path / "p.json").write_text(json.dumps({"gpu": {"uuid": FAKE_UUID}}))
    monkeypatch.setattr(bench_bw_mod.subprocess, "run", fake_run)
    result = bench_bw_mod.run_bench_bw(
        python_executable="python",
        selection=GpuSelection(requested="0", resolved_uuid=FAKE_UUID, physical_index=0),
    )
    assert captured["cmd"][captured["cmd"].index("--gpu") + 1] == FAKE_UUID
    # FTBENCH_OUT is only printed when progress reporting is on, and it is how the exact
    # written path is learned rather than re-derived
    assert captured["env"]["FREETOKEN_BENCH_PROGRESS"] == "1"
    assert result.record["profile_path_source"] == "FTBENCH_OUT"
    assert result.profile_usable


# --- the campaign gate ---------------------------------------------------------------------

def _campaign(tmp_path, *, arms, canonical=True, reverse=False, refresh=True):
    manifest_path = write_manifest(tmp_path, canonical=canonical)
    return Campaign(
        arms=arms,
        manifest=load_manifest(manifest_path, canonical=canonical),
        protocol=build_protocol(
            warmups=0, repetitions=1, session_id="s", reverse_order=reverse,
            dev_smoke=not canonical,
        ) if not canonical else build_protocol(
            warmups=None, repetitions=None, session_id="s", reverse_order=reverse,
            dev_smoke=False,
        ),
        settings=ServeSettings(
            model_path=str(tmp_path / "model"),
            model_repository="nvidia/Qwen3.6-35B-A3B-NVFP4",
            model_revision=SHA40,
            gpu=FAKE_UUID,
            python_executable="python",
        ),
        out_root=tmp_path / "runs",
        short_name="unit",
        inferswarm_commit="b" * 40,
        canonical=canonical,
        refresh_bench_bw=refresh,
        echo_server_output=False,
    )


@pytest.fixture
def campaign_env(monkeypatch, resolved_gpu, tmp_path):
    """Everything except the bench-bw call itself; each test supplies that."""
    calls = {"order": [], "started": []}

    monkeypatch.setattr(
        runner_mod.prov, "git_commit",
        lambda repo_dir: {"value": "a" * 40, "dirty": False, "dirty_paths": []},
    )
    monkeypatch.setattr(
        runner_mod.prov, "gpu_provenance",
        lambda selector=None, resolved_uuid=None: {
            "gpus": [{"index": "0", "uuid": FAKE_UUID}], "topology": "GPU0\tX",
        },
    )
    monkeypatch.setattr(
        runner_mod.prov, "host_provenance",
        lambda: {"os": "Linux", "cpu_model": "cpu", "ram_total_bytes": 1 << 36, "environment": {}},
    )
    monkeypatch.setattr(runner_mod.prov, "_torch_versions", lambda: {"torch": "2.11.0"})

    def fake_start_server(command, origin, log_path, **kwargs):
        flags = list(command)
        arm = {
            ("offload", "auto"): "B1", ("hybrid", "triton"): "B2", ("auto", "auto"): "B3",
            ("offload", "triton"): "B4", ("cpu", "triton"): "B5",
        }[(flags[flags.index("--moe-backend") + 1], flags[flags.index("--nvfp4-backend") + 1])]
        calls["order"].append(("serve", arm))
        calls["started"].append(arm)
        Path(log_path).write_text("log\n")
        return object()

    monkeypatch.setattr(runner_mod, "start_server", fake_start_server)
    monkeypatch.setattr(runner_mod, "stop_server", lambda h: None)
    monkeypatch.setattr(
        runner_mod, "fetch_instrumentation", lambda origin, limit=8: instrumentation_doc()
    )
    monkeypatch.setattr(runner_mod, "prefill_seq_floor", lambda origin: 0)
    monkeypatch.setattr(runner_mod, "_model_id", lambda origin: "qwen-test")

    def fake_measure(origin, body, *, prefill_seq_floor, store_text, **kwargs):
        class_id = body["messages"][0]["content"].rsplit(" ", 1)[-1]
        prompt_tokens = {"W1": 1800, "W2": 900, "W3": 16000, "W4": 128}[class_id]
        return {
            "ttft_ms": 1.0, "decode_tok_s": 20.0, "prompt_tokens": prompt_tokens,
            "completion_tokens": body["max_tokens"], "decode_steps": body["max_tokens"] - 1,
            "inter_token_ms": [1.0], "output_sha256": "x", "output_text": None,
            "output_text_stored": False, "vram_bytes": 1,
            "prefill": {"gpu_ms": 1.0, "new_tokens": prompt_tokens, "prefill_tok_s": 1.0},
            "prefill_status": {"ok": True, "code": "ok", "attribution": "uid"},
        }

    monkeypatch.setattr(runner_mod, "measure_generation", fake_measure)
    return calls


def _patch_bench(monkeypatch, calls, record):
    def fake(*, python_executable, selection, dtype="nvfp4", **kwargs):
        calls["order"].append(("bench_bw", selection.resolved_uuid))
        return bench_bw_mod.BenchBwResult(record)

    monkeypatch.setattr(runner_mod.bench_bw_mod, "run_bench_bw", fake)


def test_the_refresh_runs_before_the_first_server_in_forward_order(tmp_path, campaign_env, monkeypatch):
    _patch_bench(monkeypatch, campaign_env, good_bench_bw_record(FAKE_UUID))
    arms = [BASELINE_ARMS_BY_ID[i] for i in ("B1", "B2", "B3")]
    _campaign(tmp_path, arms=arms).execute()
    assert campaign_env["order"][0] == ("bench_bw", FAKE_UUID)
    assert campaign_env["started"] == ["B1", "B2", "B3"]


def test_the_refresh_runs_before_b3_in_a_reversed_session(tmp_path, campaign_env, monkeypatch):
    """Session 2 traverses in reverse, so B3 runs before B2. A B2-local refresh would let
    B3's --moe-backend auto consume a stale profile."""
    _patch_bench(monkeypatch, campaign_env, good_bench_bw_record(FAKE_UUID))
    arms = [BASELINE_ARMS_BY_ID[i] for i in ("B1", "B2", "B3")]
    _campaign(tmp_path, arms=arms, reverse=True).execute()
    assert campaign_env["started"] == ["B3", "B2", "B1"]
    assert campaign_env["order"][0] == ("bench_bw", FAKE_UUID)
    bench_index = campaign_env["order"].index(("bench_bw", FAKE_UUID))
    b3_index = campaign_env["order"].index(("serve", "B3"))
    assert bench_index < b3_index


def test_a_failed_bench_bw_aborts_before_any_server_starts(tmp_path, campaign_env, monkeypatch):
    failed = {**good_bench_bw_record(FAKE_UUID), "ok": False, "returncode": 1,
              "stderr_tail": "RuntimeError: benchbw needs a CUDA device"}
    _patch_bench(monkeypatch, campaign_env, failed)
    arms = [BASELINE_ARMS_BY_ID[i] for i in ("B1", "B2", "B3")]
    with pytest.raises(ValueError, match="aborted before any generation"):
        _campaign(tmp_path, arms=arms).execute()
    assert campaign_env["started"] == []


def test_an_unreadable_profile_aborts_a_canonical_campaign(tmp_path, campaign_env, monkeypatch):
    record = good_bench_bw_record(FAKE_UUID)
    record["profile"] = {"path": "/cache/x.json", "unavailable": "profile file could not be read"}
    _patch_bench(monkeypatch, campaign_env, record)
    arms = [BASELINE_ARMS_BY_ID[i] for i in ("B1", "B2", "B3")]
    with pytest.raises(ValueError, match="aborted before any generation"):
        _campaign(tmp_path, arms=arms).execute()
    assert campaign_env["started"] == []


def test_a_profile_from_another_gpu_aborts_a_canonical_campaign(tmp_path, campaign_env, monkeypatch):
    record = good_bench_bw_record("GPU-99999999-0000-0000-0000-000000000000")
    record["profile"]["gpu_matches"] = False
    record["profile"]["gpu_mismatch"] = "profile was benched on another card"
    _patch_bench(monkeypatch, campaign_env, record)
    arms = [BASELINE_ARMS_BY_ID[i] for i in ("B1", "B2", "B3")]
    with pytest.raises(ValueError, match="aborted before any generation"):
        _campaign(tmp_path, arms=arms).execute()
    assert campaign_env["started"] == []


def test_skipping_the_refresh_is_refused_for_a_canonical_campaign(tmp_path, campaign_env):
    arms = [BASELINE_ARMS_BY_ID[i] for i in ("B1", "B2", "B3")]
    with pytest.raises(ValueError, match="canonical run refused"):
        _campaign(tmp_path, arms=arms, refresh=False).execute()
    assert campaign_env["started"] == []


def test_a_smoke_run_may_skip_the_refresh_but_records_it_as_invalidating(tmp_path, campaign_env):
    arms = [BASELINE_ARMS_BY_ID[i] for i in ("B1", "B2", "B3")]
    doc = _campaign(tmp_path, arms=arms, canonical=False, refresh=False).execute()
    assert doc["validity"] == V.VALIDITY_NON_CANONICAL
    assert V.BENCH_BW_SKIPPED in doc["campaign_invalidation_codes"]
    assert campaign_env["started"] == ["B1", "B2", "B3"]


def test_the_profile_used_is_recorded_in_the_artifact(tmp_path, campaign_env, monkeypatch):
    _patch_bench(monkeypatch, campaign_env, good_bench_bw_record(FAKE_UUID))
    arms = [BASELINE_ARMS_BY_ID[i] for i in ("B1", "B2", "B3")]
    doc = _campaign(tmp_path, arms=arms).execute()
    bench = doc["bench_bw"]
    assert bench["ok"] is True
    assert bench["consuming_arms"] == ["B2", "B3"]
    assert bench["gpu_resolved_uuid"] == FAKE_UUID
    assert bench["started_at"] and bench["finished_at"]
    assert bench["returncode"] == 0
    assert bench["command"]
    profile = bench["profile"]
    assert profile["path"] and profile["sha256"] and profile["contents"]
    assert profile["gpu_matches"] is True
    summary = (Path(doc["run_directory"]) / "SUMMARY.md").read_text()
    assert profile["sha256"] in summary


def test_a_campaign_with_no_consuming_arm_needs_no_refresh(tmp_path, campaign_env):
    doc = _campaign(tmp_path, arms=[BASELINE_ARMS_BY_ID["B1"]]).execute()
    assert doc["bench_bw"]["skipped"] is True
    assert doc["validity"] == V.VALIDITY_VALID
