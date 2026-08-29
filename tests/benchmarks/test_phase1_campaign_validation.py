"""Dry-run validation and provenance preflight tests: everything fails closed.

Each test breaks exactly one frozen identity or provenance requirement and proves the
canonical validation refuses it. The frozen manifest/placement digests are pinned to
local fixtures (see ``phase1_fakes.freeze_frozen_identities``); the production values
are asserted separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from inferswarm_phase1 import campaign as campaign_mod
from inferswarm_phase1 import campaign_arms as arms_mod
from inferswarm_phase1.campaign import (
    CampaignDefinition,
    CampaignRefused,
    CampaignSettings,
    SessionExecution,
    validation_document,
)
from inferswarm_phase1.campaign_arms import (
    GPU0_UUID,
    GPU1_UUID,
    baseline_b1_arm,
    candidate_v2_arm,
)
from inferswarm_phase1.campaign_protocol import build_protocol

from .phase1_fakes import (
    INFERSWARM_SHA40,
    SHA40,
    install_clean_environment,
    write_canonical_manifest,
)

_FROZEN: dict = {}


def _settings(tmp_path, **kw) -> CampaignSettings:
    """Settings anchored to the autouse fixture's frozen manifest/placement pair.

    Tests needing a mutated artifact pass an explicit ``manifest_path``/``placement_path``.
    """
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


def _definition(tmp_path, *, canonical=True, protocol=None, settings=None, arms=None):
    return CampaignDefinition(
        arms=arms or [baseline_b1_arm(), candidate_v2_arm()],
        protocol=protocol
        or build_protocol(warmups=None, repetitions=None, classes=None, dev_smoke=not canonical),
        settings=settings or _settings(tmp_path),
        canonical=canonical,
    )


@pytest.fixture(autouse=True)
def clean_provenance(monkeypatch, tmp_path):
    """A clean tree, proven GPUs, and a host profile, so refusal tests break one thing."""
    frozen = install_clean_environment(monkeypatch, tmp_path)
    _FROZEN.clear()
    _FROZEN.update(frozen)
    return frozen




def test_canonical_validation_passes_and_proves_comparability(tmp_path):
    doc = validation_document(_definition(tmp_path))
    assert doc["canonical"] is True
    assert doc["preflight_refusals"] == []
    assert doc["held_equal_all"] is True
    assert doc["undeclared_differences"] == []
    assert doc["counts"]["counts_ok"] is True
    assert doc["counts"]["campaign"] == 192
    assert doc["session_ordering_ok"] is True
    assert doc["class_orders_ok"] is True
    assert doc["workload_identity"]["canonical"] is True


# --- deliberate held-constant mutations ---------------------------------------------------


def test_mutated_model_revision_is_rejected(tmp_path):
    settings = _settings(tmp_path, model_revision="main")
    doc = validation_document(_definition(tmp_path, settings=settings))
    assert any("not a 40-hex commit SHA" in r for r in doc["preflight_refusals"])
    assert doc["canonical"] is False


def test_missing_model_revision_is_rejected(tmp_path):
    settings = _settings(tmp_path, model_revision=None)
    doc = validation_document(_definition(tmp_path, settings=settings))
    assert any("--model-revision is required" in r for r in doc["preflight_refusals"])


def test_mutated_manifest_bytes_are_rejected(tmp_path):
    """A structurally valid manifest whose BYTES differ from the frozen artifact is
    refused by the digest pin (phase-0's own content check refuses cruder edits)."""
    (tmp_path / "alt").mkdir()
    path = write_canonical_manifest(tmp_path / "alt")
    doc = json.loads(path.read_text())
    from inferswarm_phase0.manifest import sha256_text

    doc["workloads"][0]["content"] = "tampered prompt bytes"
    doc["workloads"][0]["content_sha256"] = sha256_text("tampered prompt bytes")
    path.write_text(json.dumps(doc))
    settings = _settings(tmp_path, manifest_path=str(path))
    with pytest.raises(CampaignRefused, match="canonical workload manifest"):
        validation_document(_definition(tmp_path, settings=settings))


def test_altered_output_length_is_rejected(tmp_path):
    from inferswarm_phase0.manifest import ManifestError

    (tmp_path / "alt").mkdir()
    path = write_canonical_manifest(tmp_path / "alt")
    doc = json.loads(path.read_text())
    doc["workloads"][2]["output_tokens"] = 250  # W3 is frozen at 256
    path.write_text(json.dumps(doc))
    settings = _settings(tmp_path, manifest_path=str(path))
    with pytest.raises(ManifestError, match="output_tokens"):
        validation_document(_definition(tmp_path, settings=settings))


def test_wrong_gpu0_uuid_in_an_arm_is_rejected(tmp_path):
    import dataclasses

    from inferswarm_phase1.campaign_arms import GPU1_UUID

    drifted = dataclasses.replace(
        baseline_b1_arm(),
        config_flags=(
            "--gpu", GPU1_UUID,  # baseline aimed at the wrong card
            *[f for f in baseline_b1_arm().config_flags[2:]],
        ),
    )
    doc = validation_document(_definition(tmp_path, arms=[drifted, candidate_v2_arm()]))
    assert any(
        "must be the frozen physical UUID" in r for r in doc["preflight_refusals"]
    )
    assert doc["canonical"] is False


def test_mutated_sampling_is_rejected(tmp_path):
    """Sampling is frozen in the manifest; changing it changes the manifest digest."""
    (tmp_path / "alt").mkdir()
    path = write_canonical_manifest(tmp_path / "alt")
    doc = json.loads(path.read_text())
    doc["workloads"][0]["sampling"]["temperature"] = 0.7
    path.write_text(json.dumps(doc))
    settings = _settings(tmp_path, manifest_path=str(path))
    with pytest.raises(CampaignRefused, match="canonical workload manifest"):
        validation_document(_definition(tmp_path, settings=settings))


def test_memory_ratio_drift_between_arms_is_rejected(tmp_path):
    import dataclasses

    flags = list(candidate_v2_arm().config_flags)
    flags[flags.index("0.85")] = "0.9"
    drifted = dataclasses.replace(candidate_v2_arm(), config_flags=tuple(flags))
    doc = validation_document(_definition(tmp_path, arms=[baseline_b1_arm(), drifted]))
    assert any("--memory-ratio differs" in r for r in doc["preflight_refusals"])


def test_kv_reserve_drift_between_arms_is_rejected(tmp_path):
    import dataclasses

    flags = list(candidate_v2_arm().config_flags)
    flags[flags.index("17075")] = "16000"
    drifted = dataclasses.replace(candidate_v2_arm(), config_flags=tuple(flags))
    doc = validation_document(_definition(tmp_path, arms=[baseline_b1_arm(), drifted]))
    assert any("--kv-reserve-tokens differs" in r for r in doc["preflight_refusals"])


# --- provenance refusals ------------------------------------------------------------------


def test_dirty_tree_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign_mod.prov,
        "git_commit",
        lambda repo_dir: {
            "value": "a" * 40,
            "dirty": True,
            "dirty_paths": ["python/freetoken/engine/engine.py"],
        },
    )
    doc = validation_document(_definition(tmp_path))
    assert any("working tree is dirty" in r for r in doc["preflight_refusals"])
    assert doc["canonical"] is False


def test_missing_runner_version_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign_mod, "CAMPAIGN_RUNNER_VERSION", "")
    doc = validation_document(_definition(tmp_path))
    assert any("runner version is missing" in r for r in doc["preflight_refusals"])


def test_wrong_placement_sha_refusal(tmp_path, monkeypatch):
    # Freeze a different digest than the artifact on disk hashes to.
    monkeypatch.setattr(campaign_mod, "CANONICAL_PLACEMENT_SHA256", "e" * 64)
    monkeypatch.setattr(arms_mod, "CANONICAL_PLACEMENT_SHA256", "e" * 64)
    doc = validation_document(_definition(tmp_path))
    assert any("placement" in r.lower() for r in doc["preflight_refusals"])
    assert doc["canonical"] is False


def test_missing_placement_path_refusal(tmp_path):
    settings = _settings(tmp_path, placement_path=None)
    doc = validation_document(_definition(tmp_path, settings=settings))
    assert any("frozen placement artifact" in r for r in doc["preflight_refusals"])


def test_placement_disagreeing_with_model_revision_is_rejected(tmp_path):
    settings = _settings(tmp_path, model_revision="f" * 40)
    doc = validation_document(_definition(tmp_path, settings=settings))
    assert any("pins model revision" in r for r in doc["preflight_refusals"])


def test_unresolvable_gpu_refusal(tmp_path, monkeypatch):
    def resolve(selector):
        if str(selector) == GPU1_UUID:
            return ()  # NVML cannot see the secondary
        return (selector,)

    monkeypatch.setattr(campaign_mod.gpu_mod, "_resolve_uuids", resolve)
    doc = validation_document(_definition(tmp_path))
    assert any("GPU1 must resolve" in r for r in doc["preflight_refusals"])


def test_missing_correctness_prerequisites_refusal(tmp_path):
    settings = _settings(tmp_path, prerequisites_path=None)
    doc = validation_document(_definition(tmp_path, settings=settings))
    assert any("prerequisites" in r.lower() for r in doc["preflight_refusals"])
    incomplete = tmp_path / "partial.json"
    incomplete.write_text(json.dumps({"correctness_reference_v2_artifact_sha256": "a" * 64}))
    settings2 = _settings(tmp_path, prerequisites_path=str(incomplete))
    doc2 = validation_document(_definition(tmp_path, settings=settings2))
    assert any("candidate_c3_artifact_sha256" in r for r in doc2["preflight_refusals"])


def test_missing_inferswarm_commit_refusal(tmp_path):
    settings = _settings(tmp_path, inferswarm_commit=None)
    doc = validation_document(_definition(tmp_path, settings=settings))
    assert any("--inferswarm-commit is required" in r for r in doc["preflight_refusals"])


# --- frozen production identities ---------------------------------------------------------


def test_production_frozen_identities_are_the_published_values():
    """The frozen digests the production code pins are the InferSwarm-published ones."""
    source = Path(campaign_mod.__file__).read_text()
    assert 'CANONICAL_MANIFEST_ID = "phase0-v1-2026-08-27"' in source
    assert (
        "10f81e5418a71a68f387632de422c3337cc7ba0518111a8746ad856d0210b24a" in source
    )
    source_arms = Path(arms_mod.__file__).read_text()
    assert (
        "2f62bb84df40d4cc5649e940a39cb53d2975eadecbc320fb97d2b037d4e005f4"
        in source_arms
    )


def test_frozen_gpu_uuids_are_the_rig_identities():
    assert GPU0_UUID == "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
    assert GPU1_UUID == "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"


# --- session boundary ---------------------------------------------------------------------


def test_session_two_refuses_to_start_without_thermal_attestation(tmp_path):
    with pytest.raises(CampaignRefused, match="thermal"):
        SessionExecution(
            definition=_definition(tmp_path), session_number=2
        ).execute()


def test_dev_smoke_validation_is_stampedly_noncanonical(tmp_path):
    doc = validation_document(_definition(tmp_path, canonical=False))
    assert doc["canonical"] is False
    assert any("--dev-smoke" in b for b in doc["canonical_blockers"])
    assert doc["preflight_refusals"] == []
