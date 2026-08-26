"""End-to-end campaign behaviour with the server, GPU and HTTP layer mocked out.

Nothing here needs CUDA, a checkpoint, or a network. What is exercised is the part that can
fail silently: whether every repetition reaches the artifacts, whether a run that lost
repetitions can still look successful, and whether a non-canonical run is labelled as one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferswarm_phase0.artifacts import STATUS_COMPLETE, STATUS_INCOMPLETE, block_stats
from inferswarm_phase0.baselines import BASELINE_ARMS_BY_ID, correctness_reference_arm
from inferswarm_phase0.manifest import load_manifest, sha256_text
from inferswarm_phase0.protocol import build_protocol
from inferswarm_phase0.runner import Campaign, ServeSettings
from inferswarm_phase0 import runner as runner_mod

SHA40 = "0" * 40


def _manifest(tmp_path, classes=("W1", "W2", "W3", "W4"), canonical=True):
    from inferswarm_phase0.manifest import CLASS_SPECS

    entries = []
    for c in classes:
        content = f"prompt for {c}"
        entries.append({
            "class_id": c,
            "content": content,
            "content_sha256": sha256_text(content),
            "output_tokens": CLASS_SPECS[c].output_tokens,
            "ignore_eos": True,
            "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
            "seed": None,
            "chat_template_kwargs": {},
            "role": "user",
        })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema": "inferswarm.phase0.workload-manifest/1",
        "manifest_id": "test-manifest",
        "canonical": canonical,
        "workloads": entries,
    }))
    return load_manifest(path, canonical=canonical)


ALL_CLASSES = ("W1", "W2", "W3", "W4")


def _campaign(tmp_path, *, arms=None, classes=ALL_CLASSES, canonical=True, **protocol_kw):
    kw = dict(warmups=None, repetitions=None, session_id="session-1",
              reverse_order=False, dev_smoke=not canonical)
    kw.update(protocol_kw)
    return Campaign(
        arms=arms or [BASELINE_ARMS_BY_ID["B1"]],
        manifest=_manifest(tmp_path, classes, canonical=canonical),
        protocol=build_protocol(**kw),
        settings=ServeSettings(
            model_path=str(tmp_path / "model"),
            model_repository="nvidia/Qwen3.6-35B-A3B-NVFP4",
            model_revision=SHA40,
            gpu="GPU-fake",
            python_executable="python",
        ),
        out_root=tmp_path / "runs",
        short_name="unit",
        inferswarm_commit="b" * 40,
        canonical=canonical,
        refresh_bench_bw=False,
        echo_server_output=False,
    )


@pytest.fixture(autouse=True)
def complete_provenance(monkeypatch):
    """CI has no GPU, and a canonical run rightly refuses a document with holes in it.

    Supply the two host-dependent blocks so the tests below exercise the campaign rather
    than re-testing the provenance gate (which has its own tests, including the refusal)."""
    monkeypatch.setattr(
        runner_mod.prov, "gpu_provenance",
        lambda selector=None: {
            "gpus": [{"uuid": "GPU-fake", "name": "NVIDIA GeForce RTX 3060", "selected": True}],
            "topology": "GPU0\tX",
            "selected": selector,
        },
    )
    monkeypatch.setattr(
        runner_mod.prov, "host_provenance",
        lambda: {"os": "Linux 6.0", "cpu_model": "Test CPU", "ram_total_bytes": 64 << 30,
                 "environment": {}},
    )
    monkeypatch.setattr(runner_mod.prov, "_torch_versions", lambda: {"torch": "2.11.0", "cuda": "13.0"})


class _FakeHandle:
    def __init__(self):
        self.proc = None


@pytest.fixture
def mocked_server(monkeypatch):
    """Replace every process/HTTP boundary; record what the runner asked for."""
    calls = {"generations": [], "instrumentation": 0, "started": [], "stopped": 0}

    def fake_start_server(command, origin, log_path, **kwargs):
        calls["started"].append({"command": list(command), "env": kwargs.get("env_overrides")})
        Path(log_path).write_text("fake server log\n")
        return _FakeHandle()

    def fake_fetch_instrumentation(origin, limit=8):
        calls["instrumentation"] += 1
        return {
            "schema": "freetoken.instrumentation/1",
            "runtime_config": {
                "moe": {"backend_requested": "auto", "backend_resolved": "offload"},
                "nvfp4": {"requested": "auto", "resolved": "triton", "inert": False},
                "marlin_cache_cap": {"applicable": False, "bound": False},
            },
            "prefill": {"enabled": True, "observed": 0, "records": []},
        }

    def fake_measure(origin, body, *, prefill_seq_floor, store_text, **kwargs):
        calls["generations"].append(dict(body))
        return {
            "ttft_ms": 100.0 + len(calls["generations"]),
            "decode_tok_s": 20.0 + len(calls["generations"]),
            "prompt_tokens": 900,
            "completion_tokens": body["max_tokens"],
            "decode_steps": body["max_tokens"] - 1,
            "inter_token_ms": [50.0, 51.0],
            "output_sha256": "deadbeef",
            "output_text": "hello" if store_text else None,
            "output_text_stored": bool(store_text),
            "vram_bytes": 11 << 30,
            "prefill": {"gpu_ms": 40.0, "new_tokens": 900, "prefill_tok_s": 22500.0},
        }

    monkeypatch.setattr(runner_mod, "start_server", fake_start_server)
    monkeypatch.setattr(runner_mod, "stop_server", lambda h: calls.__setitem__("stopped", calls["stopped"] + 1))
    monkeypatch.setattr(runner_mod, "fetch_instrumentation", fake_fetch_instrumentation)
    monkeypatch.setattr(runner_mod, "measure_generation", fake_measure)
    monkeypatch.setattr(runner_mod, "prefill_seq_floor", lambda origin: 0)
    monkeypatch.setattr(runner_mod, "_model_id", lambda origin: "qwen-test")
    return calls


def test_a_complete_run_preserves_every_repetition(tmp_path, mocked_server):
    doc = _campaign(tmp_path).execute()
    assert doc["status"] == STATUS_COMPLETE
    # 4 classes x 10 measured, per criteria section 9 rule 1 ("all four must be run")
    assert doc["measured_repetition_count"] == 40

    reps = [json.loads(l) for l in (Path(doc["run_directory"]) / "repetitions.jsonl").read_text().splitlines()]
    # warmups are recorded too -- tagged, never dropped
    assert len(reps) == 48
    assert [r["phase"] for r in reps[:12]] == ["warmup"] * 2 + ["measured"] * 10
    assert [r["execution_index"] for r in reps] == list(range(48))
    assert {r["class_id"] for r in reps} == set(ALL_CLASSES)
    # the raw per-token timings survive, so variance/CV/bootstrap remain computable later
    assert all(r["inter_token_ms"] for r in reps)
    assert all(r["seed"] is None and r["seed_unavailable"] for r in reps)
    assert all(r["batch_size"] == 1 for r in reps)


def test_every_generation_sends_the_frozen_prompt_settings(tmp_path, mocked_server):
    _campaign(tmp_path).execute()
    bodies = mocked_server["generations"]
    assert len(bodies) == 48
    expected_max_tokens = {"W1": 512, "W2": 512, "W3": 256, "W4": 128}
    for body in bodies:
        assert body["ignore_eos"] is True
        assert body["temperature"] == 0.0 and body["top_p"] == 1.0 and body["top_k"] == -1
        content = body["messages"][0]["content"]
        class_id = content.rsplit(" ", 1)[-1]
        assert body["max_tokens"] == expected_max_tokens[class_id]


def test_the_run_records_the_resolved_configuration_not_just_the_flags(tmp_path, mocked_server):
    doc = _campaign(tmp_path).execute()
    resolved = doc["resolved_configuration"]["B1"]["instrumentation"]["runtime_config"]
    assert resolved["moe"]["backend_resolved"] == "offload"
    assert resolved["nvfp4"]["resolved"] == "triton"
    assert "marlin_cache_cap" in resolved


def test_a_failed_generation_cannot_produce_a_complete_run(tmp_path, mocked_server, monkeypatch):
    from inferswarm_phase0.client import GenerationError

    original = runner_mod.measure_generation
    state = {"n": 0}

    def flaky(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 5:
            raise GenerationError("simulated stream drop")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner_mod, "measure_generation", flaky)
    doc = _campaign(tmp_path).execute()

    assert doc["status"] == STATUS_INCOMPLETE
    assert doc["label"] == "INCOMPLETE"
    assert doc["failure_count"] == 1
    assert doc["measured_repetition_count"] == 39
    assert doc["incomplete_blocks"]
    summary = (Path(doc["run_directory"]) / "SUMMARY.md").read_text()
    assert "INCOMPLETE" in summary.splitlines()[0]
    assert "**NO**" in summary
    failures = (Path(doc["run_directory"]) / "failures.jsonl").read_text().strip().splitlines()
    assert len(failures) == 1 and "simulated stream drop" in failures[0]


def test_a_dead_server_fails_every_step_of_that_arm(tmp_path, mocked_server, monkeypatch):
    from inferswarm_phase0.client import ServerError

    def dead(*args, **kwargs):
        raise ServerError("server exited with code 1 during startup", "traceback tail")

    monkeypatch.setattr(runner_mod, "start_server", dead)
    doc = _campaign(tmp_path).execute()
    assert doc["status"] == STATUS_INCOMPLETE
    assert doc["measured_repetition_count"] == 0
    assert doc["failure_count"] == 48  # nothing is silently skipped


def test_a_non_canonical_run_is_labelled_everywhere(tmp_path, mocked_server):
    doc = _campaign(tmp_path, canonical=False, warmups=1, repetitions=2).execute()
    assert doc["canonical"] is False
    assert doc["canonical_blockers"]
    summary = (Path(doc["run_directory"]) / "SUMMARY.md").read_text()
    assert "NON-CANONICAL developer smoke test" in summary
    assert "must not be published as one" in summary


def test_a_canonical_run_refuses_missing_provenance(tmp_path, mocked_server):
    campaign = _campaign(tmp_path)
    campaign.inferswarm_commit = None
    with pytest.raises(ValueError, match="required provenance is missing"):
        campaign.execute()


def test_the_summary_never_computes_a_cross_configuration_ratio(tmp_path, mocked_server):
    doc = _campaign(
        tmp_path, arms=[BASELINE_ARMS_BY_ID["B1"], BASELINE_ARMS_BY_ID["B4"]]
    ).execute()
    summary = (Path(doc["run_directory"]) / "SUMMARY.md").read_text()
    assert "does **not** select `CANONICAL_PERFORMANCE_BASELINE`" in summary
    for banned in ("speedup", "faster than", "winner", "R_agg"):
        assert banned not in summary.lower() or banned == "winner" and "winner" not in summary


def test_correctness_reference_output_text_is_always_retained(tmp_path, mocked_server):
    arm = correctness_reference_arm("triton", 512)
    campaign = _campaign(tmp_path, arms=[arm])
    campaign.store_output_text = True
    doc = campaign.execute()
    reps = [json.loads(l) for l in (Path(doc["run_directory"]) / "repetitions.jsonl").read_text().splitlines()]
    assert all(r["output_text_stored"] for r in reps)
    assert all(r["arm_role"] == "correctness" for r in reps)


def test_prefill_instrumentation_is_requested_for_every_arm(tmp_path, mocked_server):
    _campaign(tmp_path, arms=[BASELINE_ARMS_BY_ID["B1"], BASELINE_ARMS_BY_ID["B5"]]).execute()
    assert [s["env"] for s in mocked_server["started"]] == [
        {"FREETOKEN_INSTRUMENT_PREFILL": "1"}
    ] * 2


def test_rerunning_into_the_same_directory_does_not_blend_campaigns(tmp_path, mocked_server):
    first = _campaign(tmp_path).execute()
    second = _campaign(tmp_path).execute()
    assert first["run_directory"] == second["run_directory"]
    reps = (Path(second["run_directory"]) / "repetitions.jsonl").read_text().strip().splitlines()
    assert len(reps) == 48


def test_block_stats_reports_dispersion_not_just_a_mean():
    stats = block_stats([10.0, 11.0, 12.0, 40.0])
    assert stats["n"] == 4
    assert stats["min"] == 10.0 and stats["max"] == 40.0
    assert stats["median"] == 11.5
    assert stats["cv_percent"] > 0
    assert "iqr" in stats
    # the outlier is described, never removed
    assert stats["max"] == 40.0


def test_block_stats_on_no_data_says_so():
    assert block_stats([]) == {"n": 0}
    assert block_stats([None, None]) == {"n": 0}


def test_the_resolved_weight_format_is_backfilled_from_the_arms(tmp_path, mocked_server, monkeypatch):
    """The expert format is only knowable once an engine has loaded the banks, so it must
    not stay "server not started yet" for the life of the artifact."""
    monkeypatch.setattr(
        runner_mod, "fetch_instrumentation",
        lambda origin, limit=8: {
            "runtime_config": {"model": {"expert_quant": "nvfp4"}},
            "prefill": {"enabled": True, "observed": 0, "records": []},
        },
    )
    doc = _campaign(tmp_path, arms=[BASELINE_ARMS_BY_ID["B1"], BASELINE_ARMS_BY_ID["B4"]]).execute()
    quant = doc["model_expert_quant_resolved"]
    assert quant["value"] == "nvfp4"
    assert quant["consistent_across_arms"] is True
    assert quant["per_arm"] == {"B1": "nvfp4", "B4": "nvfp4"}


def test_arms_disagreeing_on_the_weight_format_is_recorded_as_invalidating(tmp_path, mocked_server, monkeypatch):
    """Criteria section 3 rule 4 holds the weight format constant across arms."""
    formats = iter(["nvfp4", "fp8_block"])

    monkeypatch.setattr(
        runner_mod, "fetch_instrumentation",
        lambda origin, limit=8: {
            "runtime_config": {"model": {"expert_quant": next(formats)}},
            "prefill": {"enabled": True, "observed": 0, "records": []},
        },
    )
    doc = _campaign(tmp_path, arms=[BASELINE_ARMS_BY_ID["B1"], BASELINE_ARMS_BY_ID["B4"]]).execute()
    quant = doc["model_expert_quant_resolved"]
    assert quant["value"] is None
    assert quant["consistent_across_arms"] is False
    assert "campaign is invalid" in quant["unavailable"]
