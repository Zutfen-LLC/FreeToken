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
    predeclared_kv_matched_arm,
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
    write_session_one_gate,
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
        arms=arms
        or [baseline_b1_arm(), candidate_v2_arm(), predeclared_kv_matched_arm()],
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
    write_session_one_gate(tmp_path)
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
            all_records.extend(json.loads(line) for line in path.read_text().splitlines())
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
        json.loads(line)
        for line in (Path(doc["run_directory"]) / CANDIDATE_ARM_ID / "W1.jsonl").read_text().splitlines()
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
    assert "identity failure" in doc["stopped_early_reason"]
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


def test_baseline_inferswarm_leakage_is_a_b1_identity_failure(tmp_path, mocked_server):
    """InferSwarm treatment on a baseline arm is an identity failure, not an
    invalidation that still benchmarks: the session stops before any candidate
    generation, exactly like a resolution drift."""
    mocked_server["runtime_by_arm"][BASELINE_ARM_ID] = baseline_runtime_config(
        inferswarm_remote_decode={
            "enabled": True, "execution_mode": "overlap", "transport": "host_staged",
            "placement_sha256": None, "primary": {"uuid": GPU0_UUID},
            "secondary": {"uuid": GPU1_UUID},
        },
    )
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    assert doc["stopped_early_reason"] is not None
    assert "InferSwarm leakage" in doc["stopped_early_reason"]
    assert [s["arm"] for s in mocked_server["started"]] == [BASELINE_ARM_ID]
    assert doc["validity"] == "INVALID"
    assert "runtime.baseline_inferswarm_present" in doc["campaign_invalidation_codes"]
    assert doc["baseline_identity_gate"]["passed"] is False
    # no candidate performance exists
    assert doc["completion"]["failed_generations"] == 48
    assert ("reset", CANDIDATE_ARM_ID) not in mocked_server["moe_ops"]


def test_wrong_engine_gpu_stops_the_arm_before_warmups(tmp_path, mocked_server, monkeypatch):
    """The engine reporting a different physical GPU stops the arm before any
    generation: nothing at all is measured, the B1 gate cannot pass, and the
    remaining arms are preserved as not-executed evidence."""
    from unittest import mock

    def wrong_engine_gpus(origin):
        return [{
            "index": 0,
            "uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "name": "NVIDIA GeForce RTX 3060",
            "total_bytes": 12 << 30,
        }]

    with mock.patch.object(campaign_mod.gpu_mod, "engine_gpus", wrong_engine_gpus):
        doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    assert doc["stopped_early_reason"] is not None
    assert "physical GPU" in doc["stopped_early_reason"]
    assert doc["validity"] == "INVALID"
    assert "gpu.mismatch" in doc["campaign_invalidation_codes"]
    assert doc["baseline_identity_gate"]["passed"] is False
    # nothing was measured anywhere: no generation, no instrumentation window
    assert mocked_server["generations"] == []
    assert mocked_server["moe_ops"] == []
    assert [s["arm"] for s in mocked_server["started"]] == [BASELINE_ARM_ID]
    # the candidate arm's planned generations are preserved as not-executed evidence
    assert doc["completion"]["failed_generations"] == 48
    candidate_w1 = [
        json.loads(line)
        for line in (Path(doc["run_directory"]) / CANDIDATE_ARM_ID / "W1.jsonl").read_text().splitlines()
    ]
    assert all(r["failed"] and "not executed" in r["error"] for r in candidate_w1)


def test_supplementary_arm_contract_failure_aborts_it_before_warmups(
    tmp_path, mocked_server
):
    """A required supplementary arm that fails the B1 contract is aborted with its
    generations preserved as not-executed evidence; the primary arms stand."""
    mocked_server["runtime_by_arm"][BASELINE_ARM_ID] = baseline_runtime_config(
        runtime={"num_pages": 19000},
    )
    mocked_server["runtime_by_arm"]["baseline_b1_kv_matched"] = baseline_runtime_config(
        runtime={"num_pages": 17075},
        moe={"cpu_layers_resolved": [4], "auto_cpu_layers_fired": True},
    )
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    assert doc["completion"]["supplementary_condition"]["required"] is True
    assert doc["execution_status"] == "INCOMPLETE"
    assert doc["validity"] == "INVALID"
    assert doc["stopped_early_reason"] is not None
    assert "baseline_b1_kv_matched" in doc["stopped_early_reason"]
    kv_w1 = [
        json.loads(line)
        for line in (
            Path(doc["run_directory"]) / "baseline_b1_kv_matched" / "W1.jsonl"
        ).read_text().splitlines()
    ]
    assert len(kv_w1) == 12
    assert all(r["failed"] and "not executed" in r["error"] for r in kv_w1)
    # the primaries really completed before the supplementary abort
    assert doc["completion"]["failed_generations"] == 48
    assert ("reset", BASELINE_ARM_ID) in mocked_server["moe_ops"]
    assert ("reset", CANDIDATE_ARM_ID) in mocked_server["moe_ops"]


# --- resolved expert-cache slots: provenance, not a validity band ------------------------------


def test_resolved_cache_slots_outside_the_old_band_are_provenance_not_identity(
    tmp_path, mocked_server
):
    """No hidden numeric slot band: B1 resolving far outside the old +/-10% band
    still passes the identity gate (policy must be auto); the exact resolved slot
    count is recorded as provenance."""
    mocked_server["runtime_by_arm"][BASELINE_ARM_ID] = baseline_runtime_config(
        cache={"resolved_slots": 3000},  # far below 3774 - 10%
    )
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    assert doc["validity"] == "VALID"
    assert doc["execution_status"] == "COMPLETE"
    assert doc["baseline_identity_gate"]["passed"] is True
    assert "runtime.baseline_identity_drift" not in doc["campaign_invalidation_codes"]
    runtime = json.loads(
        (Path(doc["run_directory"]) / BASELINE_ARM_ID / "runtime.json").read_text()
    )
    assert runtime["validation"]["resolved"]["cache_resolved_slots"] == 3000
    assert "not a validity threshold" in runtime["validation"]["cache_slots_rule"]


def test_non_auto_cache_policy_is_still_a_b1_identity_failure(tmp_path, mocked_server):
    """Removing the numeric band does not remove the methodology's own rule: B1
    must run --moe-cache-auto; a fixed size is a different baseline."""
    mocked_server["runtime_by_arm"][BASELINE_ARM_ID] = baseline_runtime_config(
        cache={"policy_requested": "size"},
    )
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    assert doc["stopped_early_reason"] is not None
    assert "cache.policy_requested" in doc["stopped_early_reason"]
    assert [s["arm"] for s in mocked_server["started"]] == [BASELINE_ARM_ID]


def test_candidate_contract_mismatch_aborts_the_arm_before_any_generation(
    tmp_path, mocked_server
):
    """A resolved-arm mismatch found after /health and before warmups must not
    generate performance observations (updated P5 review expectation)."""
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
    # the session is INCOMPLETE: the candidate's planned generations never ran
    assert doc["execution_status"] == "INCOMPLETE"
    assert doc["stopped_early_reason"] is not None
    assert "placement" in doc["stopped_early_reason"]
    # NO candidate generation was collected: every planned candidate generation is
    # preserved as a not-executed failure, warmups included
    candidate_w1 = [
        json.loads(line)
        for line in (
            Path(doc["run_directory"]) / CANDIDATE_ARM_ID / "W1.jsonl"
        ).read_text().splitlines()
    ]
    assert len(candidate_w1) == 12
    assert all(r["failed"] and "not executed" in r["error"] for r in candidate_w1)
    # no measurement window was ever opened for the candidate
    assert ("reset", CANDIDATE_ARM_ID) not in mocked_server["moe_ops"]
    # the baseline (first arm in session 1) really did run to completion
    assert ("reset", BASELINE_ARM_ID) in mocked_server["moe_ops"]


def _wrong_remote(**overrides):
    block = {
        "enabled": True, "execution_mode": "overlap", "transport": "host_staged",
        # replaced with the currently frozen placement SHA at test time so a
        # single-field mutation produces exactly one contract finding
        "placement_sha256": "<frozen>", "primary": {"uuid": GPU0_UUID},
        "secondary": {"uuid": GPU1_UUID},
    }
    block.update(overrides)
    return block


# Every resolved-arm contract field the review names: a mismatch found after
# /health and before warmups must abort the candidate arm before its first
# generation — never benchmark the wrong arm and label it later.
_CANDIDATE_CONTRACT_MUTATIONS = [
    (
        "wrong placement SHA",
        {"inferswarm_remote_decode": _wrong_remote(placement_sha256="f" * 64)},
    ),
    (
        "wrong remote primary UUID",
        {"inferswarm_remote_decode": _wrong_remote(primary={"uuid": GPU1_UUID})},
    ),
    (
        "wrong remote secondary UUID",
        {"inferswarm_remote_decode": _wrong_remote(secondary={"uuid": GPU0_UUID})},
    ),
    (
        "wrong secondary-device UUID",
        {
            "inferswarm_secondary_device": {
                "configured": True,
                "requested_secondary_spec": GPU1_UUID,
                "validation_passed": True,
                "primary": {"uuid": GPU0_UUID, "visible_cuda_ordinal": 0},
                "secondary": {"uuid": GPU0_UUID, "visible_cuda_ordinal": 1},
                "peer_access": {
                    "primary_to_secondary": False,
                    "secondary_to_primary": False,
                },
                "transport_classification": "host_staged",
            }
        },
    ),
    ("wrong resident slots", {"inferswarm_resident_bank": {"resident_slots": 5000}}),
    (
        "wrong resident bank bytes",
        {
            "inferswarm_resident_bank": {
                "banks": [
                    {"name": "w1", "total_resident_bytes": 1 << 30},
                    {"name": "w2", "total_resident_bytes": 1 << 30},
                ]
            }
        },
    ),
    (
        "wrong transport",
        {"inferswarm_remote_decode": _wrong_remote(transport="p2p")},
    ),
    (
        "mode not overlap",
        {"inferswarm_remote_decode": _wrong_remote(execution_mode="serialized")},
    ),
    (
        "remote decode disabled",
        {"inferswarm_remote_decode": _wrong_remote(enabled=False)},
    ),
    (
        "graphs unexpectedly enabled",
        {"runtime": {"cuda_graph_max_bs": 1, "cuda_graph_capture_happened": True}},
    ),
    ("wrong backend", {"moe": {"backend_resolved": "fused"}}),
    ("wrong cpu-layer shape", {"moe": {"cpu_layers_resolved": [4]}}),
    ("wrong gpu0 cache size", {"cache": {"resolved_slots": 3000}}),
    ("resolved KV != 17075", {"runtime": {"num_pages": 12345}}),
]


@pytest.mark.parametrize(
    "label,overrides",
    _CANDIDATE_CONTRACT_MUTATIONS,
    ids=[label for label, _ in _CANDIDATE_CONTRACT_MUTATIONS],
)
def test_any_prewarmup_candidate_contract_mismatch_stops_measurement(
    tmp_path, mocked_server, label, overrides
):
    """Aborts the candidate arm before the first warmup; no candidate generation
    is recorded as a successful measurement; the session is INVALID/INCOMPLETE."""
    import copy

    kwargs = copy.deepcopy(overrides)
    remote = kwargs.get("inferswarm_remote_decode")
    if isinstance(remote, dict) and remote.get("placement_sha256") == "<frozen>":
        remote["placement_sha256"] = _FROZEN["placement_sha"]
    mocked_server["runtime_by_arm"][CANDIDATE_ARM_ID] = candidate_runtime_config(
        **kwargs
    )
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    assert doc["validity"] == "INVALID"
    assert "runtime.candidate_contract_mismatch" in doc["campaign_invalidation_codes"]
    assert doc["execution_status"] == "INCOMPLETE"
    assert doc["stopped_early_reason"] is not None
    # zero candidate performance: every planned candidate generation is preserved
    # as a not-executed failure (warmups included), and no measurement window was
    # ever opened for the candidate
    candidate_w1 = [
        json.loads(line)
        for line in (
            Path(doc["run_directory"]) / CANDIDATE_ARM_ID / "W1.jsonl"
        ).read_text().splitlines()
    ]
    assert len(candidate_w1) == 12
    assert all(r["failed"] and "not executed" in r["error"] for r in candidate_w1)
    assert ("reset", CANDIDATE_ARM_ID) not in mocked_server["moe_ops"]
    assert ("snapshot", CANDIDATE_ARM_ID) not in mocked_server["moe_ops"]


def test_an_unreadable_candidate_runtime_report_aborts_before_warmups(
    tmp_path, mocked_server
):
    mocked_server["runtime_by_arm"][CANDIDATE_ARM_ID] = None
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    assert doc["validity"] == "INVALID"
    assert doc["execution_status"] == "INCOMPLETE"
    assert doc["stopped_early_reason"] is not None
    candidate_w1 = [
        json.loads(line)
        for line in (
            Path(doc["run_directory"]) / CANDIDATE_ARM_ID / "W1.jsonl"
        ).read_text().splitlines()
    ]
    assert all(r["failed"] for r in candidate_w1)
    assert ("reset", CANDIDATE_ARM_ID) not in mocked_server["moe_ops"]


def test_candidate_contract_abort_in_session_two_preserves_the_whole_session_plan(
    tmp_path, mocked_server
):
    """Session 2 (candidate first): the abort stops the session; the baseline's
    planned generations are preserved as not-executed evidence too."""
    write_session_one_gate(tmp_path)
    mocked_server["runtime_by_arm"][CANDIDATE_ARM_ID] = candidate_runtime_config(
        inferswarm_remote_decode=_wrong_remote(placement_sha256="f" * 64),
    )
    doc = SessionExecution(
        definition=_definition(tmp_path),
        session_number=2,
        thermal_reset_attested="independently cooled reset observed",
    ).execute()
    assert doc["validity"] == "INVALID"
    assert doc["execution_status"] == "INCOMPLETE"
    # both arms' planned generations are preserved; nothing was measured at all
    assert doc["completion"]["failed_generations"] == 96
    assert mocked_server["moe_ops"] == []
    assert [s["arm"] for s in mocked_server["started"]] == [CANDIDATE_ARM_ID]


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
    assert requirement["condition"] == (
        "candidate_resolved_kv_capacity != baseline_resolved_kv_capacity"
    )
    assert requirement["pinned_kv_capacity_tokens"] == 17075


def test_supplementary_requirement_fires_when_kv_capacities_differ(tmp_path, mocked_server):
    mocked_server["runtime_by_arm"][BASELINE_ARM_ID] = baseline_runtime_config(
        runtime={"num_pages": 19000},
    )
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    requirement = doc["supplementary_arm_requirement"]
    assert requirement["required"] is True
    assert requirement["baseline_kv_tokens"] == 19000
    assert requirement["candidate_kv_tokens"] == 17075
    # the predeclared arm actually executes, after both primaries
    served = [s["arm"] for s in mocked_server["started"]]
    assert served == [BASELINE_ARM_ID, CANDIDATE_ARM_ID, "baseline_b1_kv_matched"]


def test_equal_capacities_execute_no_supplementary_generations(tmp_path, mocked_server):
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    condition = doc["completion"]["supplementary_condition"]
    assert condition["required"] is False
    assert condition["status"] == "NOT_REQUIRED_BY_KV_RULE"
    assert condition["required_supplementary_block_completed"] is None
    assert doc["completion"]["conditional_supplementary_generations"] is None
    # no kv-matched arm directory, no generation records for it
    assert not (Path(doc["run_directory"]) / "baseline_b1_kv_matched").exists()
    assert doc["completion"]["observed_generations"] == 96
    assert doc["execution_status"] == "COMPLETE"
    assert doc["validity"] == "VALID"


def test_unequal_capacities_run_the_exact_predeclared_supplementary_arm(
    tmp_path, mocked_server
):
    mocked_server["runtime_by_arm"][BASELINE_ARM_ID] = baseline_runtime_config(
        runtime={"num_pages": 19000},
    )
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    condition = doc["completion"]["supplementary_condition"]
    assert condition["required"] is True
    assert condition["status"] == "REQUIRED_BY_KV_RULE"
    assert condition["pinned_kv_capacity_tokens"] == 17075
    assert condition["required_supplementary_block_completed"] is True
    assert doc["completion"]["conditional_supplementary_generations"] == 48
    assert doc["completion"]["observed_generations"] == 96 + 48
    assert doc["completion"]["expected_primary_generations"] == 96
    # the supplementary server was started with the predeclared --num-tokens 17075
    kv_command = mocked_server["started"][-1]["command"]
    assert kv_command[kv_command.index("--num-tokens") + 1] == "17075"
    assert doc["execution_status"] == "COMPLETE"
    assert doc["validity"] == "VALID"


def test_required_but_not_run_supplementary_block_is_incomplete_and_invalid(
    tmp_path, mocked_server
):
    mocked_server["runtime_by_arm"][BASELINE_ARM_ID] = baseline_runtime_config(
        runtime={"num_pages": 19000},
    )
    # the GPU idle check refuses exactly the supplementary arm's start: only once
    # both primary servers have finished does GPU0 look occupied
    from unittest import mock

    def stale_after_primaries(uuid):
        return (6 << 30) if len(mocked_server["started"]) >= 2 else (8 << 20)

    with mock.patch.object(
        campaign_mod, "gpu_memory_used_bytes", stale_after_primaries
    ):
        doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    assert doc["completion"]["supplementary_condition"]["required"] is True
    assert (
        doc["completion"]["supplementary_condition"]["required_supplementary_block_completed"]
        is False
    )
    assert doc["execution_status"] == "INCOMPLETE"
    assert doc["validity"] == "INVALID"
    assert "supplementary.required_block_missing" in doc["campaign_invalidation_codes"]
    # the required block's generations are preserved as not-executed failures
    assert doc["completion"]["failed_generations"] == 48
    kv_w1 = Path(doc["run_directory"]) / "baseline_b1_kv_matched" / "W1.jsonl"
    assert kv_w1.exists()


def test_dev_smoke_forced_kv_arm_runs_unconditionally(tmp_path, mocked_server):
    definition = _definition(
        tmp_path,
        canonical=False,
        protocol=build_protocol(warmups=1, repetitions=1, classes=["W1"], dev_smoke=True),
        arms=[
            baseline_b1_arm(),
            candidate_v2_arm(),
            kv_matched_arm(17075, conditional=False),
        ],
    )
    doc = SessionExecution(definition=definition, session_number=1).execute()
    served = [s["arm"] for s in mocked_server["started"]]
    assert served == [BASELINE_ARM_ID, CANDIDATE_ARM_ID, "baseline_b1_kv_matched"]
    assert doc["completion"]["observed_generations"] == 3 * 2  # 3 arms x (1+1)
    assert doc["completion"]["expected_primary_generations"] is None


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
    # planned order includes the predeclared conditional arm; it did not execute
    assert order["arm_order"] == [BASELINE_ARM_ID, CANDIDATE_ARM_ID, "baseline_b1_kv_matched"]
    assert order["executed_arm_order"] == [BASELINE_ARM_ID, CANDIDATE_ARM_ID]
    assert order["primary_arm_order"] == [BASELINE_ARM_ID, CANDIDATE_ARM_ID]
    assert order["class_order"] == ["W1", "W2", "W3", "W4"]
    assert len(order["all_block_identities"]) == 8
    assert order["execution_indices_recorded"] == list(range(96))
    consistency = doc["provenance_consistency"]
    assert consistency["manifest_canonical"] is True
    assert consistency["placement_canonical"] is True
    assert doc["held_constant_validation"]["runtime.memory_ratio"]["equal"] is True
    assert doc["baseline_noise_floor_status"]["per_class"]["W1"]["within_5_percent_ceiling"] is True
    assert doc["baseline_identity_gate"]["passed"] is True
    assert doc["baseline_identity_gate"]["checked"] is True
    # every artifact file is hash-indexed; the embedded index is written inside
    # session-summary.json and therefore predates it (no self-referential hash),
    # while the returned-only full-directory index covers the summary as well
    assert "plan.json" in doc["artifact_sha256"]
    assert "session-summary.json" not in doc["artifact_sha256"]
    assert "session-summary.json" in doc["full_directory_sha256"]
    on_disk = json.loads(
        (Path(doc["run_directory"]) / "session-summary.json").read_text()
    )
    assert on_disk["artifact_sha256"] == doc["artifact_sha256"]


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
    write_session_one_gate(tmp_path, out_root=str(run2 / "runs"))
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
    # 96 expected primary + 48 possible conditional supplementary generations
    assert plan["generation_count"] == 144
    assert plan["primary_generation_count"] == 96
    assert plan["conditional_generation_count"] == 48
    assert plan["no_dynamic_shortening"] is True
    provenance = json.loads((root / "provenance.json").read_text())
    assert provenance["schema"] == "inferswarm.phase1.session-provenance/1"
    assert provenance["software"]["phase1_campaign_runner_version"]
    assert provenance["prerequisites"]["supplied"] is True
    assert provenance["prerequisites"]["verification"]["freetoken_runtime_commit"][
        "equals_current_head"
    ] is True
    assert provenance["placement"]["artifact_sha256"] == provenance["placement"]["frozen_sha256"]
    assert provenance["historical_phase0_baseline_commit"] == "2c3da952e47391bf392e0ece8ae4c67acbc91762"
    runtime = json.loads((root / CANDIDATE_ARM_ID / "runtime.json").read_text())
    assert runtime["schema"] == "inferswarm.phase1.arm-runtime/1"
    assert runtime["validation"]["checked"] is True
    summary = json.loads((root / "session-summary.json").read_text())
    assert summary["schema"] == "inferswarm.phase1.session-summary/1"


