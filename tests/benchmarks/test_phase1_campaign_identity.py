"""The campaign identity binding the session-1 gate to session 2.

Session 2 must not accept an arbitrary old COMPLETE/VALID session-1 summary:
the gate record must come from the expected session-1 artifact set AND record
EXACTLY the current campaign identity (one SHA-256 over the canonical-JSON
component set: FreeToken HEAD, InferSwarm methodology commit, campaign runner
version, model repository + exact revision, workload manifest SHA, placement
SHA, canonical protocol identity, primary arm definitions). Any differing
component refuses session 2 before any server starts, naming the component.
"""

from __future__ import annotations

import re

import pytest
from inferswarm_phase1.campaign import (
    CAMPAIGN_IDENTITY_SCHEMA,
    CampaignDefinition,
    CampaignRefused,
    CampaignSettings,
    SessionExecution,
    campaign_identity,
    campaign_identity_differences,
    session_one_gate_record,
    validation_document,
)
from inferswarm_phase1.campaign_arms import (
    baseline_b1_arm,
    candidate_v2_arm,
    predeclared_kv_matched_arm,
)
from inferswarm_phase1.campaign_protocol import build_protocol
from inferswarm_phase0.manifest import load_manifest

from .phase1_fakes import (
    FAKE_FREETOKEN_HEAD,
    INFERSWARM_SHA40,
    SHA40,
    default_campaign_identity,
    install_clean_environment,
    install_mocked_server,
    write_canonical_manifest,
    write_session_one_gate,
)

_FROZEN: dict = {}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    frozen = install_clean_environment(monkeypatch, tmp_path)
    _FROZEN.clear()
    _FROZEN.update(frozen)
    return frozen


@pytest.fixture
def mocked_server(monkeypatch):
    return install_mocked_server(monkeypatch)


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


def _definition(tmp_path, **settings_kw):
    return CampaignDefinition(
        arms=[baseline_b1_arm(), candidate_v2_arm(), predeclared_kv_matched_arm()],
        protocol=build_protocol(warmups=None, repetitions=None, classes=None, dev_smoke=False),
        settings=_settings(tmp_path, **settings_kw),
        canonical=True,
    )


def _refuse_session_two(tmp_path, **settings_kw) -> pytest.ExceptionInfo:
    with pytest.raises(CampaignRefused) as excinfo:
        SessionExecution(
            definition=_definition(tmp_path, **settings_kw),
            session_number=2,
            thermal_reset_attested="independently cooled reset observed",
        ).execute()
    return excinfo


# --- the fingerprint itself -------------------------------------------------------------------


def test_campaign_identity_is_a_deterministic_sha256_over_named_components(tmp_path):
    from inferswarm_phase1.campaign_arms import load_placement_reference

    manifest = load_manifest(_FROZEN["manifest"], canonical=True)
    placement = load_placement_reference(_FROZEN["placement"])
    first = campaign_identity(_definition(tmp_path), manifest, placement)
    second = campaign_identity(_definition(tmp_path), manifest, placement)
    assert first["schema"] == CAMPAIGN_IDENTITY_SCHEMA
    assert first["sha256"] == second["sha256"]
    assert _SHA256_RE.match(first["sha256"])
    components = first["components"]
    for key in (
        "freetoken_head",
        "inferswarm_methodology_commit",
        "campaign_runner_version",
        "model_repository",
        "model_revision",
        "workload_manifest_sha256",
        "placement_sha256",
        "protocol",
        "primary_arms",
    ):
        assert key in components
    assert components["freetoken_head"] == FAKE_FREETOKEN_HEAD
    assert components["inferswarm_methodology_commit"] == INFERSWARM_SHA40
    assert components["model_revision"] == SHA40
    assert components["workload_manifest_sha256"] == manifest.manifest_sha256
    assert components["placement_sha256"] == placement["sha256"]
    assert components["primary_arms"]["baseline_b1"]
    assert components["primary_arms"]["candidate_v2"]

    # a single changed component changes the fingerprint, and the difference
    # helper names exactly that component
    drifted = campaign_identity(
        _definition(tmp_path, model_revision="7" * 40), manifest, placement
    )
    assert drifted["sha256"] != first["sha256"]
    differences = campaign_identity_differences(
        drifted, first["components"]
    )
    assert [d["component"] for d in differences] == ["model_revision"]


def test_session_one_records_the_identity_in_provenance_summary_and_validation(
    tmp_path, mocked_server
):
    import json
    from pathlib import Path

    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    assert doc["validity"] == "VALID"
    identity = doc["campaign_identity"]
    assert identity["sha256"]
    summary = json.loads(
        (Path(doc["run_directory"]) / "session-summary.json").read_text()
    )
    assert summary["campaign_identity"]["sha256"] == identity["sha256"]
    validation = validation_document(_definition(tmp_path))
    assert validation["campaign_identity"]["sha256"] == identity["sha256"]


# --- session-2 refusals on any differing component --------------------------------------------


def test_session_two_refuses_when_session_one_ran_at_an_old_freetoken_head(
    tmp_path, mocked_server
):
    write_session_one_gate(
        tmp_path, identity=default_campaign_identity(tmp_path, repo_head="0" * 40)
    )
    excinfo = _refuse_session_two(tmp_path)
    message = str(excinfo.value)
    assert "different campaign identity" in message
    assert "freetoken_head" in message
    assert mocked_server["started"] == []  # refused before any server starts


def test_session_two_refuses_on_a_wrong_model_revision_identity(tmp_path, mocked_server):
    drifted = CampaignDefinition(
        arms=[baseline_b1_arm(), candidate_v2_arm(), predeclared_kv_matched_arm()],
        protocol=build_protocol(warmups=None, repetitions=None, classes=None, dev_smoke=False),
        settings=_settings(tmp_path, model_revision="7" * 40),
        canonical=True,
    )
    write_session_one_gate(
        tmp_path, identity=default_campaign_identity(tmp_path, definition=drifted)
    )
    excinfo = _refuse_session_two(tmp_path)
    assert "model_revision" in str(excinfo.value)
    assert mocked_server["started"] == []


def test_session_two_refuses_on_a_wrong_workload_manifest_sha_identity(
    tmp_path, mocked_server
):
    import json

    from inferswarm_phase0.manifest import sha256_text

    alt_dir = tmp_path / "alt"
    alt_dir.mkdir()
    alt_manifest = write_canonical_manifest(alt_dir)
    doc = json.loads(alt_manifest.read_text())
    doc["workloads"][0]["content"] = "different fixture prompt for W1"
    doc["workloads"][0]["content_sha256"] = sha256_text(
        "different fixture prompt for W1"
    )
    alt_manifest.write_text(json.dumps(doc, indent=2))
    drifted_manifest = load_manifest(str(alt_manifest), canonical=True)
    assert drifted_manifest.manifest_sha256 != load_manifest(
        _FROZEN["manifest"], canonical=True
    ).manifest_sha256
    write_session_one_gate(
        tmp_path,
        identity=default_campaign_identity(tmp_path, manifest=drifted_manifest),
    )
    excinfo = _refuse_session_two(tmp_path)
    assert "workload_manifest_sha256" in str(excinfo.value)
    assert mocked_server["started"] == []


def test_session_two_refuses_on_a_wrong_placement_sha_identity(tmp_path, mocked_server):
    write_session_one_gate(
        tmp_path,
        identity=default_campaign_identity(
            tmp_path, placement={"sha256": "e" * 64}
        ),
    )
    excinfo = _refuse_session_two(tmp_path)
    assert "placement_sha256" in str(excinfo.value)
    assert mocked_server["started"] == []


def test_session_two_refuses_on_a_wrong_runner_version_identity(tmp_path, mocked_server):
    write_session_one_gate(
        tmp_path, identity=default_campaign_identity(tmp_path, runner_version="0.2.0")
    )
    excinfo = _refuse_session_two(tmp_path)
    assert "campaign_runner_version" in str(excinfo.value)
    assert mocked_server["started"] == []


def test_session_two_refuses_a_gate_without_a_recorded_identity(tmp_path, mocked_server):
    import json

    gate = write_session_one_gate(tmp_path)
    doc = json.loads(gate.read_text())
    del doc["campaign_identity"]
    gate.write_text(json.dumps(doc))
    excinfo = _refuse_session_two(tmp_path)
    assert "campaign identity" in str(excinfo.value)
    assert mocked_server["started"] == []


# --- the gate must come from the expected artifact set ------------------------------------------


def test_session_two_refuses_a_gate_without_the_session_one_artifact_set(
    tmp_path, mocked_server
):
    write_session_one_gate(tmp_path, artifact_set=False)
    excinfo = _refuse_session_two(tmp_path)
    assert "expected artifact set" in str(excinfo.value)
    assert mocked_server["started"] == []


def test_session_two_refuses_when_a_session_one_artifact_was_tampered(
    tmp_path, mocked_server
):
    from pathlib import Path

    write_session_one_gate(tmp_path)
    plan = Path(_FROZEN_ROOT(tmp_path)) / "session-1" / "plan.json"
    plan.write_text('{"schema": "tampered"}')
    excinfo = _refuse_session_two(tmp_path)
    assert "expected artifact set" in str(excinfo.value)
    assert "plan.json" in str(excinfo.value)
    assert mocked_server["started"] == []


def _FROZEN_ROOT(tmp_path) -> str:
    return str(tmp_path / "runs")


# --- exact identity passes ----------------------------------------------------------------------


def test_exact_campaign_identity_passes_the_session_two_gate(tmp_path, mocked_server):
    write_session_one_gate(tmp_path)  # default identity == the current one
    doc = SessionExecution(
        definition=_definition(tmp_path),
        session_number=2,
        thermal_reset_attested="independently cooled reset observed",
    ).execute()
    assert [s["arm"] for s in mocked_server["started"]] == [
        "candidate_v2",
        "baseline_b1",
    ]
    assert doc["execution_status"] == "COMPLETE"
    assert doc["validity"] == "VALID"


def test_gate_record_reports_identity_equality_and_differences(tmp_path):
    from pathlib import Path

    write_session_one_gate(tmp_path)
    current = default_campaign_identity(tmp_path)
    record = session_one_gate_record(
        Path(tmp_path) / "runs", current_identity=current
    )
    assert record["ok"] is True
    assert record["identity"]["equal"] is True
    assert record["identity"]["differences"] == []
    assert record["artifact_set"]["verified"] is True

    drifted = default_campaign_identity(tmp_path, repo_head="0" * 40)
    record2 = session_one_gate_record(
        Path(tmp_path) / "runs", current_identity=drifted
    )
    assert record2["ok"] is False
    assert record2["identity"]["equal"] is False
    assert [
        d["component"] for d in record2["identity"]["differences"]
    ] == ["freetoken_head"]
