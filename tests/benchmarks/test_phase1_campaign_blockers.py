"""Focused tests for the three review blockers on FreeToken PR #12.

Blocker 1 — Session-2 baseline-drift semantics: Session 1's B1 resolution is the
campaign-build baseline identity gate (no candidate generation anywhere until it
passes); Session 2 runs the candidate first BY DESIGN and revalidates B1 when its
counterbalanced B1 arm runs; a later Session-2 drift makes the ENTIRE session
invalid with none of its candidate measurements eligible for analysis.

Blocker 2 — correctness prerequisites bound to the exact campaign checkout: the
canonical preflight refuses (before any server starts) a stale or malformed
runtime commit, a malformed evidence SHA, and bytes that disagree with a declared
artifact path; current-commit equality is mandatory.

Blocker 3 — the KV-matched supplementary protocol is predeclared: trigger and
pinned capacity fixed in every canonical plan before execution, the branch is
controlled by resolved KV capacities only (never a performance number), and the
arm can never become a primary comparator or enter cross-arm primary analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from inferswarm_phase1.campaign import (
    CampaignDefinition,
    CampaignRefused,
    CampaignSettings,
    SessionExecution,
    plan_document,
    validation_document,
)
from inferswarm_phase1.campaign_arms import (
    BASELINE_ARM_ID,
    CANDIDATE_ARM_ID,
    CANDIDATE_PINNED_KV_TOKENS,
    KV_MATCHED_ARM_ID,
    KV_RULE_CONDITION,
    baseline_b1_arm,
    candidate_v2_arm,
    predeclared_kv_matched_arm,
)
from inferswarm_phase1.campaign_cli import build_parser
from inferswarm_phase1.campaign_protocol import build_protocol
from inferswarm_phase1.campaign_validity import (
    SessionValidity,
    validate_candidate_runtime,
)

from .phase1_fakes import (
    FAKE_FREETOKEN_HEAD,
    INFERSWARM_SHA40,
    SHA40,
    baseline_runtime_config,
    candidate_runtime_config,
    install_clean_environment,
    install_mocked_server,
    write_prerequisites,
    write_session_one_gate,
)

_FROZEN: dict[str, str] = {}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    frozen = install_clean_environment(monkeypatch, tmp_path)
    _FROZEN.clear()
    _FROZEN.update(frozen)
    return frozen


@pytest.fixture
def mocked_server(monkeypatch):
    return install_mocked_server(monkeypatch)


def _definition(tmp_path, prerequisites_path=None):
    return CampaignDefinition(
        arms=[baseline_b1_arm(), candidate_v2_arm(), predeclared_kv_matched_arm()],
        protocol=build_protocol(warmups=None, repetitions=None, classes=None, dev_smoke=False),
        settings=CampaignSettings(
            model_path=str(tmp_path / "model"),
            manifest_path=_FROZEN["manifest"],
            model_revision=SHA40,
            placement_path=_FROZEN["placement"],
            inferswarm_commit=INFERSWARM_SHA40,
            out_root=tmp_path / "runs",
            prerequisites_path=prerequisites_path or _FROZEN["prerequisites"],
            echo_server_output=False,
        ),
        canonical=True,
    )


# --- blocker 1: session-2 baseline-drift semantics -----------------------------------------


def test_session_one_baseline_drift_prevents_any_candidate_generation(
    tmp_path, mocked_server
):
    """Session 1's B1 gate fails first: no candidate server is ever started."""
    mocked_server["runtime_by_arm"][BASELINE_ARM_ID] = baseline_runtime_config(
        nvfp4={"requested": "auto", "resolved": "marlin", "inert": False},
    )
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    assert [s["arm"] for s in mocked_server["started"]] == [BASELINE_ARM_ID]
    assert doc["stopped_early_reason"] is not None
    assert "session-1 campaign-build baseline identity gate" in doc["stopped_early_reason"]
    assert doc["validity"] == "INVALID"
    assert doc["baseline_identity_gate"]["passed"] is False
    # no candidate generation exists anywhere: the candidate's planned
    # generations are preserved as not-executed failures, never measured
    candidate_w1 = (Path(doc["run_directory"]) / CANDIDATE_ARM_ID / "W1.jsonl")
    records = [json.loads(l) for l in candidate_w1.read_text().splitlines()]
    assert len(records) == 12
    assert all(r["failed"] and "not executed" in r["error"] for r in records)


def test_session_two_candidate_executes_first_by_design(tmp_path, mocked_server):
    """The counterbalanced order places the candidate first in session 2; the
    predeclared gate record (not a preliminary B1 server) is what authorizes it."""
    write_session_one_gate(tmp_path)
    doc = SessionExecution(
        definition=_definition(tmp_path),
        session_number=2,
        thermal_reset_attested="independently cooled reset observed",
    ).execute()
    assert [s["arm"] for s in mocked_server["started"]] == [CANDIDATE_ARM_ID, BASELINE_ARM_ID]
    assert doc["execution_status"] == "COMPLETE"
    assert doc["validity"] == "VALID"
    assert doc["baseline_identity_gate"]["passed"] is True
    assert "revalidation" in doc["baseline_identity_gate"]["role"]


def test_later_session_two_baseline_drift_invalidates_the_entire_session(
    tmp_path, mocked_server
):
    """B1 drift discovered at session 2's SECOND arm: the candidate already ran."""
    write_session_one_gate(tmp_path)
    mocked_server["runtime_by_arm"][BASELINE_ARM_ID] = baseline_runtime_config(
        moe={"cpu_layers_resolved": [4, 7], "auto_cpu_layers_fired": True},
    )
    doc = SessionExecution(
        definition=_definition(tmp_path),
        session_number=2,
        thermal_reset_attested="independently cooled reset observed",
    ).execute()
    # the candidate really did run first — 48 real candidate generations exist
    assert [s["arm"] for s in mocked_server["started"]] == [CANDIDATE_ARM_ID, BASELINE_ARM_ID]
    candidate_w1 = (Path(doc["run_directory"]) / CANDIDATE_ARM_ID / "W1.jsonl")
    candidate_records = [json.loads(l) for l in candidate_w1.read_text().splitlines()]
    assert len(candidate_records) == 12
    assert all(not r.get("failed") for r in candidate_records)
    # ... and the ENTIRE session is invalid: none of it is eligible for analysis
    assert doc["validity"] == "INVALID"
    assert "runtime.baseline_identity_drift" in doc["campaign_invalidation_codes"]
    assert doc["baseline_identity_gate"]["passed"] is False
    disposition = doc["baseline_drift_disposition"]
    assert disposition["session_validity"] == "INVALID"
    assert "not eligible" in disposition["candidate_measurements"]
    assert "complete affected campaign" in disposition["required_remediation"]
    assert "spliced" in disposition["reuse_policy"]
    assert "session-2 revalidation" in doc["stopped_early_reason"]


def test_session_two_cannot_start_before_the_campaign_build_gate_passed(tmp_path):
    """No candidate measurement anywhere may precede session 1's B1 gate —
    enforced mechanically, not by operator discipline."""
    with pytest.raises(CampaignRefused, match="session 2 cannot start"):
        SessionExecution(
            definition=_definition(tmp_path),
            session_number=2,
            thermal_reset_attested="independently cooled reset observed",
        ).execute()


# --- blocker 2: prerequisites bound to the exact campaign checkout --------------------------


def test_stale_runtime_commit_refuses_the_canonical_preflight_before_any_server(
    tmp_path, mocked_server
):
    stale = write_prerequisites(tmp_path, commit="9" * 40)
    with pytest.raises(CampaignRefused, match="does not equal the current FreeToken HEAD"):
        SessionExecution(
            definition=_definition(tmp_path, prerequisites_path=str(stale)),
            session_number=1,
        ).execute()
    assert mocked_server["started"] == []  # refused before any server start


def test_malformed_runtime_commit_refuses_before_any_server(tmp_path, mocked_server):
    malformed = write_prerequisites(tmp_path, commit="not-a-sha")
    with pytest.raises(CampaignRefused, match="not a valid 40-hex commit SHA"):
        SessionExecution(
            definition=_definition(tmp_path, prerequisites_path=str(malformed)),
            session_number=1,
        ).execute()
    assert mocked_server["started"] == []


def test_malformed_artifact_sha_refuses_before_any_server(tmp_path, mocked_server):
    malformed = write_prerequisites(
        tmp_path, shas={"p2_p3_p4_requalification_artifact_sha256": "XYZ"}
    )
    with pytest.raises(CampaignRefused, match="not a lowercase normalized 64-hex SHA-256"):
        SessionExecution(
            definition=_definition(tmp_path, prerequisites_path=str(malformed)),
            session_number=1,
        ).execute()
    assert mocked_server["started"] == []


def test_exact_current_commit_and_valid_shas_pass_the_canonical_preflight(
    tmp_path, mocked_server
):
    exact = write_prerequisites(tmp_path, commit=FAKE_FREETOKEN_HEAD)
    doc = SessionExecution(
        definition=_definition(tmp_path, prerequisites_path=str(exact)),
        session_number=1,
    ).execute()
    assert doc["validity"] == "VALID"
    assert doc["execution_status"] == "COMPLETE"


# --- blocker 3: the predeclared KV-matched supplementary protocol ----------------------------


def test_the_plan_predeclares_the_supplementary_arm_fully_before_execution(tmp_path):
    from inferswarm_phase0.manifest import load_manifest

    definition = _definition(tmp_path)
    manifest = load_manifest(_FROZEN["manifest"], canonical=True)
    plan = plan_document(definition, manifest)
    for session in plan["sessions"]:
        declaration = session["conditional_supplementary_arms"][0]
        assert declaration["arm_id"] == KV_MATCHED_ARM_ID
        assert declaration["condition"] == KV_RULE_CONDITION
        assert declaration["trigger_fixed_before_execution"] is True
        assert declaration["pinned_kv_capacity_tokens"] == CANDIDATE_PINNED_KV_TOKENS == 17075
        assert declaration["possible_generations_per_session"] == 48
        assert declaration["position"] == "after both primary arms"
        assert "non-gating" in declaration["supplementary_status"]
        flags = declaration["exact_flags"]
        assert flags[flags.index("--num-tokens") + 1] == "17075"
        conditional = [g for g in session["generations"] if g["conditional"]]
        assert len(conditional) == 48
        assert session["primary_generation_count"] == 96


def test_candidate_runtime_contract_pins_the_resolved_kv_capacity():
    """The candidate's resolved KV capacity must equal its requested --num-tokens;
    that pin is what makes the predeclared supplementary capacity fully known."""
    state = SessionValidity(canonical_intent=True)
    record = validate_candidate_runtime(
        state,
        candidate_runtime_config(),
        arm_id=CANDIDATE_ARM_ID,
        gpu0_uuid="GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55",
        gpu1_uuid="GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176",
        placement_sha256=_FROZEN["placement_sha"],
        expected_gpu1_slots=5442,
        expected_gpu1_expert_bytes=9_662_902_272,
    )
    assert record["contract_findings"] == []
    drifted = candidate_runtime_config(runtime={"num_pages": 12345})
    state2 = SessionValidity(canonical_intent=True)
    record2 = validate_candidate_runtime(
        state2,
        drifted,
        arm_id=CANDIDATE_ARM_ID,
        gpu0_uuid="GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55",
        gpu1_uuid="GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176",
        placement_sha256=_FROZEN["placement_sha"],
        expected_gpu1_slots=5442,
        expected_gpu1_expert_bytes=9_662_902_272,
    )
    assert any("resolved KV capacity" in f for f in record2["contract_findings"])


def test_the_kv_branch_is_controlled_by_capacities_not_performance(tmp_path, mocked_server):
    """Equal capacities -> NOT_REQUIRED_BY_KV_RULE and no supplementary server,
    even though the mocked candidate is measurably 'winning' (rising tok/s)."""
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    condition = doc["completion"]["supplementary_condition"]
    assert condition["status"] == "NOT_REQUIRED_BY_KV_RULE"
    assert KV_MATCHED_ARM_ID not in [
        s["arm"] for s in mocked_server["started"]
    ]
    assert doc["supplementary_arm_requirement"]["condition"] == KV_RULE_CONDITION


def test_supplementary_metrics_never_enter_cross_arm_primary_analysis(
    tmp_path, mocked_server
):
    """With the supplementary arm executed (unequal capacities), the artifacts
    still contain per-arm values only — no cross-arm key touches it, and the
    primary counts exclude its generations."""
    mocked_server["runtime_by_arm"][BASELINE_ARM_ID] = baseline_runtime_config(
        runtime={"num_pages": 19000},
    )
    doc = SessionExecution(definition=_definition(tmp_path), session_number=1).execute()
    assert doc["completion"]["supplementary_condition"]["required"] is True
    assert doc["completion"]["observed_generations"] == 144
    assert doc["completion"]["expected_primary_generations"] == 96

    def all_keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from all_keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from all_keys(v)

    keys = set(all_keys(doc))
    for forbidden in (
        "ratio", "R_c", "R_agg", "speedup", "candidate_over_baseline",
    ):
        assert forbidden not in keys
    # per-arm summaries only; the supplementary arm summarizes itself per class
    summary = json.loads(
        (Path(doc["run_directory"]) / KV_MATCHED_ARM_ID / "summary.json").read_text()
    )
    assert set(summary["statistics"].keys()) <= {"W1", "W2", "W3", "W4"}
    assert summary["arm_role"] == "supplementary"
    assert not list(Path(doc["run_directory"]).glob("comparison*"))


def test_canonical_cli_refuses_manual_kv_matched_tokens():
    """The operator cannot guess/pass --kv-matched-tokens on a canonical run."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "plan",
            "--model", "m",
            "--manifest", "manifest.json",
            "--kv-matched-tokens", "17075",
        ]
    )
    from inferswarm_phase1.campaign_cli import _definition as cli_definition

    with pytest.raises(ValueError, match="dev-smoke/testing override only"):
        cli_definition(args)


def test_validation_proves_the_predeclaration(tmp_path):
    doc = validation_document(_definition(tmp_path))
    assert doc["canonical"] is True
    assert doc["supplementary_predeclaration"]["predeclared"] is True
    declaration = doc["supplementary_predeclaration"]["conditional_arms"][0]
    assert declaration["pinned_kv_capacity_tokens"] == 17075
    assert declaration["possible_generations_per_session"] == 48
