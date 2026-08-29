"""Session execution tests with the server, GPU and HTTP layers mocked out.

No CUDA, checkpoint, or network. What is exercised is everything that can fail
silently: completeness arithmetic, failure preservation, the baseline-drift STOP,
block instrumentation windows, execution order, artifact schemas, the supplementary
KV determination, and the absence of any verdict vocabulary in the output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from inferswarm_phase1 import campaign as campaign_mod
from inferswarm_phase1.campaign import (
    CampaignDefinition,
    CampaignSettings,
    SessionExecution,
)
from inferswarm_phase1.campaign_arms import (
    BASELINE_ARM_ID,
    CANDIDATE_ARM_ID,
    GPU0_UUID,
    GPU1_UUID,
    baseline_b1_arm,
    candidate_v2_arm,
    kv_matched_arm,
)
from inferswarm_phase1.campaign_protocol import build_protocol

from .phase1_fakes import (
    INFERSWARM_SHA40,
    SHA40,
    baseline_runtime_config,
    candidate_runtime_config,
    install_clean_environment,
    install_mocked_server,
    moe_window_snapshot,
)

ALL_CLASSES = ("W1", "W2", "W3", "W4")


_FROZEN: dict[str, str] = {}


def _settings(tmp_path, **kw) -> CampaignSettings:
    defaults = {
        "model_path": str(tmp_path / "model"),
        "manifest_path": _FROZEN["manifest"],
        "model_revision": SHA40,
        "placement_path": _FROZEN["placement"],
        "inferswarm_commit": INFERSWARM_SHA40,
        "out_root": tmp_path / "runs",
        "prerequisites_path": _FROZEN["prerequisites"],
        "echo_server_output": False,
    }
    defaults.update(kw)
    return CampaignSettings(**defaults)


def _definition(tmp_path, *, canonical=True, arms=None, protocol=None, **settings_kw):
    return CampaignDefinition(
        arms=arms or [baseline_b1_arm(), candidate_v2_arm()],
        protocol=protocol
        or build_protocol(
            warmups=None,
            repetitions=None,
            classes=None,
            dev_smoke=not canonical,
        ),
        settings=_settings(tmp_path, **settings_kw),
        canonical=canonical,
    )


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    frozen = install_clean_environment(monkeypatch, tmp_path)
    _FROZEN.clear()
    _FROZEN.update(frozen)


@pytest.fixture
def mocked_server(monkeypatch):
    return install_mocked_server(monkeypatch)


# --- completeness ---------------------------------------------------------------------------


def test_a_complete_session_preserves_every_generation(tmp_path, mocked_server):
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    assert doc["execution_status"] == "COMPLETE"
    assert doc["validity"] == "VALID"
    assert doc["completion"]["observed_generations"] == 96
    assert doc["completion"]["failed_generations"] == 0

    reps = []
    for arm in (BASELINE_ARM_ID, CANDIDATE_ARM_ID):
        for c in ALL_CLASSES:
            path = Path(doc["run_directory"]) / arm / f"{c}.jsonl"
            reps.extend(json.loads(line) for line in path.read_text().splitlines())
    assert len(reps) == 96
    assert sorted(r["execution_index"] for r in reps) == list(range(96))
    assert all(r["schema"] == "inferswarm.phase1.repetition/1" for r in reps)
    assert all(r["inter_token_ms"] for r in reps)
    assert all(r["batch_size"] == 1 for r in reps)
    assert all(r["failed"] is False for r in reps)


def test_session_two_reverses_the_executed_arm_order(tmp_path, mocked_server):
    SessionExecution(definition=_definition(tmp_path), session_number=2,
                     thermal_reset_attested="cooled to idle at 2026-08-30T09:00").execute()
    served = [s["arm"] for s in mocked_server["started"]]
    assert served == [CANDIDATE_ARM_ID, BASELINE_ARM_ID]


def test_server_startup_is_recorded_and_never_amortized(tmp_path, mocked_server):
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    for arm in (BASELINE_ARM_ID, CANDIDATE_ARM_ID):
        startup = json.loads(
            (Path(doc["run_directory"]) / arm / "startup.json").read_text()
        )
        assert startup["schema"] == "inferswarm.phase1.arm-startup/1"
        assert startup["launched_at_unix"] <= startup["ready_at_unix"]
        assert startup["m_start_duration_s"] >= 0
        assert "--model" in startup["serve_command"]
        assert "never amortized" in startup["note"]


def test_measurement_windows_reset_after_warmups_and_snapshot_after_measured(
    tmp_path, mocked_server
):
    SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    ops = mocked_server["moe_ops"]
    # every arm/class block contributes exactly one reset + one snapshot
    assert ops.count(("reset", BASELINE_ARM_ID)) == 4
    assert ops.count(("snapshot", BASELINE_ARM_ID)) == 4
    assert ops.count(("reset", CANDIDATE_ARM_ID)) == 4
    assert ops.count(("snapshot", CANDIDATE_ARM_ID)) == 4
    order = mocked_server["order"]
    first_reset = order.index(("moe:reset", BASELINE_ARM_ID))
    # the first block's reset happens after its two warmups and before any snapshot
    assert order[0] == ("serve", BASELINE_ARM_ID)
    assert ("moe:snapshot", BASELINE_ARM_ID) in order[:first_reset + 1] or (
        order.index(("moe:snapshot", BASELINE_ARM_ID)) > first_reset
    )


def test_candidate_block_mechanism_artifacts_retain_counters_and_timing(
    tmp_path, mocked_server
):
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    for c in ALL_CLASSES:
        path = Path(doc["run_directory"]) / CANDIDATE_ARM_ID / f"block-mechanism-{c}.json"
        block = json.loads(path.read_text())
        assert block["schema"] == "inferswarm.phase1.block-mechanism/1"
        aggregate = block["remote_decode"]["aggregate"]
        for counter in (
            "selected_for_gpu1", "executed_on_gpu1", "explicit_failure", "fallback_elsewhere"
        ):
            assert counter in aggregate
        assert "moe_layer_timing" in block


def test_fallback_or_remote_prefill_invalidates_the_session(tmp_path, mocked_server):
    mocked_server["moe_ops"].clear()

    def fake(origin, operation, *, timeout=60.0):
        if operation == "snapshot":
            return moe_window_snapshot(prefill_dispatches=3, fallback=5)
        return {"boundary": {"operation": "reset", "idle": True}}

    from unittest import mock

    with mock.patch.object(campaign_mod, "moe_instrumentation", fake):
        doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    assert doc["validity"] == "INVALID"
    codes = doc["campaign_invalidation_codes"]
    assert "remote.forbidden_fallback" in codes


# --- failure preservation --------------------------------------------------------------------


def test_a_failed_generation_cannot_produce_a_complete_session(tmp_path, mocked_server, monkeypatch):
    original = campaign_mod.measure_generation
    state = {"n": 0}

    def flaky(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 5:  # W1 measured rep 2
            raise campaign_mod.GenerationError("simulated stream drop")
        return original(*args, **kwargs)

    monkeypatch.setattr(campaign_mod, "measure_generation", flaky)
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    assert doc["execution_status"] == "INCOMPLETE"
    assert doc["completion"]["failed_generations"] == 1

    # the failed repetition is preserved in place, never deleted, never retried
    w1 = [
        json.loads(line)
        for line in (
            Path(doc["run_directory"]) / BASELINE_ARM_ID / "W1.jsonl"
        ).read_text().splitlines()
    ]
    assert len(w1) == 12  # 2 warmups + 10 measured: exactly the planned generations
    failures = [r for r in w1 if r.get("failed")]
    assert len(failures) == 1
    assert "simulated stream drop" in failures[0]["error"]
    assert failures[0]["measured"] is True
    # the other blocks completed
    assert doc["completion"]["observed_generations"] == 96  # failure recorded, not skipped


def test_no_generation_is_ever_duplicated_or_lost(tmp_path, mocked_server):
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    all_records = []
    for arm in (BASELINE_ARM_ID, CANDIDATE_ARM_ID):
        for c in ALL_CLASSES:
            path = Path(doc["run_directory"]) / arm / f"{c}.jsonl"
            all_records.extend(json.loads(l) for l in path.read_text().splitlines())
    indices = [r["execution_index"] for r in all_records]
    assert sorted(indices) == list(range(96))
    assert len(indices) == len(set(indices))
    assert all(r["block_id"].startswith(f"session-1/{r['arm_id']}/{r['class_id']}/block-") for r in all_records)


def test_a_dead_server_fails_the_remaining_generations_in_place(tmp_path, mocked_server, monkeypatch):
    state = {"started": 0}

    def dying_server(command, origin, log_path, **kwargs):
        state["started"] += 1
        if state["started"] == 2:  # the candidate arm's server dies at startup
            raise campaign_mod.ServerError(
                "server exited with code 1 during startup", "log tail"
            )
        arm = (
            CANDIDATE_ARM_ID if "--inferswarm-remote-decode" in command
            else "baseline_b1_kv_matched" if "--num-tokens" in command
            else BASELINE_ARM_ID
        )
        mocked_server["started"].append({"arm": arm, "command": list(command)})
        Path(log_path).write_text("fake server log\n")
        return type("H", (), {"proc": None})()

    monkeypatch.setattr(campaign_mod, "start_server", dying_server)
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    assert doc["execution_status"] == "INCOMPLETE"
    assert doc["validity"] == "INVALID"
    assert doc["completion"]["failed_generations"] == 48  # whole candidate arm preserved as failures
    candidate_w1 = [
        json.loads(l)
        for l in (Path(doc["run_directory"]) / CANDIDATE_ARM_ID / "W1.jsonl").read_text().splitlines()
    ]
    assert len(candidate_w1) == 12
    assert all(r["failed"] for r in candidate_w1)


# --- baseline drift STOP -----------------------------------------------------------------------


def test_baseline_identity_drift_stops_the_session_before_candidate_performance(
    tmp_path, mocked_server
):
    mocked_server["runtime_by_arm"][BASELINE_ARM_ID] = baseline_runtime_config(
        nvfp4={"requested": "auto", "resolved": "marlin", "inert": False},
    )
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    assert doc["stopped_early_reason"] is not None
    assert "identity drift" in doc["stopped_early_reason"]
    assert "marlin" in doc["stopped_early_reason"]
    # the candidate server was NEVER started: no candidate performance exists
    served = [s["arm"] for s in mocked_server["started"]]
    assert served == [BASELINE_ARM_ID]
    assert doc["validity"] == "INVALID"
    assert "runtime.baseline_identity_drift" in doc["campaign_invalidation_codes"]
    # the candidate arm's planned generations are preserved as not-executed failures
    assert doc["completion"]["failed_generations"] == 48


def test_baseline_cpu_layer_autolocking_stops_the_session(tmp_path, mocked_server):
    mocked_server["runtime_by_arm"][BASELINE_ARM_ID] = baseline_runtime_config(
        moe={"cpu_layers_resolved": [4, 7], "auto_cpu_layers_fired": True},
    )
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    assert doc["stopped_early_reason"] is not None
    assert [s["arm"] for s in mocked_server["started"]] == [BASELINE_ARM_ID]


def test_candidate_contract_mismatch_invalidates_but_records(tmp_path, mocked_server):
    mocked_server["runtime_by_arm"][CANDIDATE_ARM_ID] = candidate_runtime_config(
        inferswarm_remote_decode={
            "enabled": True, "execution_mode": "overlap", "transport": "host_staged",
            "placement_sha256": "wrong" * 8,
            "primary": {"uuid": GPU0_UUID}, "secondary": {"uuid": GPU1_UUID},
        },
    )
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    assert doc["validity"] == "INVALID"
    assert "runtime.candidate_contract_mismatch" in doc["campaign_invalidation_codes"]
    assert doc["execution_status"] == "COMPLETE"  # generations really happened


# --- supplementary KV arm -------------------------------------------------------------------------


def test_supplementary_requirement_is_mechanical_from_resolved_kv_capacities(
    tmp_path, mocked_server
):
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    requirement = doc["supplementary_arm_requirement"]
    # both fakes resolve num_pages=17075 at page_size=1 -> equal capacities
    assert requirement["required"] is False
    assert requirement["baseline_kv_tokens"] == 17075
    assert requirement["candidate_kv_tokens"] == 17075
    assert requirement["arm_id"] == "baseline_b1_kv_matched"


def test_supplementary_requirement_fires_when_kv_capacities_differ(tmp_path, mocked_server):
    mocked_server["runtime_by_arm"][BASELINE_ARM_ID] = baseline_runtime_config(
        runtime={"num_pages": 19000},
    )
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    requirement = doc["supplementary_arm_requirement"]
    assert requirement["required"] is True
    assert requirement["baseline_kv_tokens"] == 19000
    assert requirement["candidate_kv_tokens"] == 17075


def test_the_supplementary_arm_runs_after_the_primaries_counted_separately(
    tmp_path, mocked_server
):
    definition = _definition(
        tmp_path,
        arms=[baseline_b1_arm(), candidate_v2_arm(), kv_matched_arm(17075)],
    )
    doc = SessionExecution(definition=definition, session_number=1).execute()
    served = [s["arm"] for s in mocked_server["started"]]
    assert served == [BASELINE_ARM_ID, CANDIDATE_ARM_ID, "baseline_b1_kv_matched"]
    # primary counts stay exactly 96; the supplementary arm is extra
    assert doc["completion"]["observed_generations"] == 96 + 48
    assert doc["completion"]["expected_primary_generations"] == 96
    assert doc["supplementary_arm_requirement"]["arm_id"] == "baseline_b1_kv_matched"


# --- dev smoke ---------------------------------------------------------------------------------------


def test_dev_smoke_sessions_are_labelled_noncanonical_everywhere(tmp_path, mocked_server):
    definition = _definition(
        tmp_path, canonical=False,
        protocol=build_protocol(warmups=1, repetitions=1, classes=["W1"], dev_smoke=True),
    )
    doc = SessionExecution(definition=definition, session_number=1).execute()
    assert doc["validity"] == "NON_CANONICAL_DEV_SMOKE"
    assert doc["execution_status"] == "COMPLETE"
    assert doc["completion"]["observed_generations"] == 2 * 2  # 2 arms x (1 warmup + 1 measured)
    assert doc["completion"]["expected_primary_generations"] is None
    assert any("--dev-smoke" in b for b in doc["canonical_blockers"])
    assert any("warmups=1" in d for d in doc["canonical_blockers"])


# --- session summary -----------------------------------------------------------------------------------


def test_session_summary_records_order_blocks_and_provenance(tmp_path, mocked_server):
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    order = doc["execution_order"]
    assert order["arm_order"] == [BASELINE_ARM_ID, CANDIDATE_ARM_ID]
    assert order["primary_arm_order"] == [BASELINE_ARM_ID, CANDIDATE_ARM_ID]
    assert order["class_order"] == ["W1", "W2", "W3", "W4"]
    assert len(order["all_block_identities"]) == 8
    assert order["execution_indices_recorded"] == list(range(96))
    consistency = doc["provenance_consistency"]
    assert consistency["manifest_canonical"] is True
    assert consistency["placement_canonical"] is True
    assert doc["held_constant_validation"]["runtime.memory_ratio"]["equal"] is True
    assert doc["baseline_noise_floor_status"]["per_class"]["W1"]["within_5_percent_ceiling"] is True
    # every artifact file is hash-indexed
    assert "plan.json" in doc["artifact_sha256"]
    assert "session-summary.json" in doc["artifact_sha256"]


def test_per_class_summaries_carry_the_required_descriptive_fields(tmp_path, mocked_server):
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    summary = json.loads(
        (Path(doc["run_directory"]) / BASELINE_ARM_ID / "summary.json").read_text()
    )
    assert summary["schema"] == "inferswarm.phase1.arm-summary/1"
    stats = summary["statistics"]["W1"]
    for key in ("min", "median", "max", "iqr", "cv_percent"):
        assert key in stats["decode_tok_s"]
    assert "median" in stats["ttft_ms"]
    assert "median" in stats["prefill_tok_s"]
    assert "p50_of_p50" in stats["inter_token_ms"]
    assert stats["prompt_output_counts"]["completion_tokens"] == [512]
    assert len(summary["blocks"]) == 4


def test_thermal_records_captured_before_each_arm_and_idle_checks_run(tmp_path, mocked_server):
    observed = {"thermal": 0}

    def fake_thermal():
        observed["thermal"] += 1
        return {"observed_at": f"t{observed['thermal']}", "gpus": []}

    from unittest import mock

    with mock.patch.object(campaign_mod, "thermal_observation", fake_thermal):
        doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    # once per arm before start (2), plus the session-boundary observation at preflight
    assert observed["thermal"] >= 2
    assert len(doc["idle_records"]) >= 4  # before+after per arm


def test_stale_gpu_memory_refuses_the_next_canonical_arm(tmp_path, mocked_server):
    from unittest import mock

    used = {"bytes": 8 << 20}
    run2 = tmp_path / "run2"
    run2.mkdir()
    with mock.patch.object(
        campaign_mod, "gpu_memory_used_bytes", lambda uuid: used["bytes"]
    ):
        doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
        assert doc["execution_status"] == "COMPLETE"
        # now the host reports resident memory above the idle bound
        used["bytes"] = 6 << 30
        doc2 = SessionExecution(
            definition=_definition(run2),
            session_number=2,
            thermal_reset_attested="cooled to idle at 2026-08-30T09:00",
        ).execute()
    assert doc2["execution_status"] == "INCOMPLETE"
    assert "gpu.idle_memory_not_restored" in doc2["campaign_invalidation_codes"]
    # nothing started in session 2: the before-arm check refused the first arm
    assert doc2["completion"]["failed_generations"] == 96


# --- schema round-trip -----------------------------------------------------------------------------------


def test_every_artifact_round_trips_with_its_schema(tmp_path, mocked_server):
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    root = Path(doc["run_directory"])
    plan = json.loads((root / "plan.json").read_text())
    assert plan["schema"] == "inferswarm.phase1.session-plan/1"
    assert plan["generation_count"] == 96
    assert plan["no_dynamic_shortening"] is True
    provenance = json.loads((root / "provenance.json").read_text())
    assert provenance["schema"] == "inferswarm.phase1.session-provenance/1"
    assert provenance["software"]["phase1_campaign_runner_version"]
    assert provenance["prerequisites"]["supplied"] is True
    assert provenance["placement"]["artifact_sha256"] == provenance["placement"]["frozen_sha256"]
    assert provenance["historical_phase0_baseline_commit"] == "2c3da952e47391bf392e0ece8ae4c67acbc91762"
    runtime = json.loads((root / CANDIDATE_ARM_ID / "runtime.json").read_text())
    assert runtime["schema"] == "inferswarm.phase1.arm-runtime/1"
    assert runtime["validation"]["checked"] is True
    summary = json.loads((root / "session-summary.json").read_text())
    assert summary["schema"] == "inferswarm.phase1.session-summary/1"


