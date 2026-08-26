"""Provenance capture: explicit nulls, refused guesses, and the canonical gate.

The rule under test is InferSwarm's benchmark contract: "If a field cannot be filled,
record why. A benchmark with silent holes in its provenance is not a benchmark."
"""

from __future__ import annotations

import pytest

from inferswarm_phase0 import provenance as prov


def test_unavailable_is_an_explicit_null_with_a_reason():
    field = prov.unavailable("nvidia-smi is not installed")
    assert field["value"] is None
    assert field["unavailable"]


def test_missing_required_reports_absent_and_explicitly_unavailable_fields():
    doc = {
        "software": {
            "freetoken_commit": {"value": "a" * 40},
            "inferswarm_commit": prov.unavailable("not supplied"),
        },
        "model": {"repository": "nvidia/Qwen3.6-35B-A3B-NVFP4"},
        "host": {"cpu_model": "Some CPU", "ram_total_bytes": 1},
        "gpu": {"gpus": [{"uuid": "GPU-1"}], "topology": "X"},
    }
    missing = prov.missing_required(doc)
    assert any("inferswarm_commit" in m for m in missing)
    assert any(m.startswith("model.revision") for m in missing)
    assert not any("freetoken_commit" in m for m in missing)


def test_a_complete_document_has_nothing_missing():
    doc = {
        "software": {
            "freetoken_commit": {"value": "a" * 40},
            "inferswarm_commit": {"value": "b" * 40},
        },
        "model": {"repository": "r", "revision": "c" * 40},
        "host": {"cpu_model": "cpu", "ram_total_bytes": 8},
        "gpu": {"gpus": [{}], "topology": "t"},
    }
    assert prov.missing_required(doc) == []


@pytest.mark.parametrize("revision", ["main", "refs/heads/main", "v1.0", "abc123", "", None])
def test_canonical_runs_reject_a_symbolic_or_short_revision(revision):
    with pytest.raises(ValueError, match="revision"):
        prov.validate_revision(revision, canonical=True)


def test_canonical_runs_accept_a_full_commit_sha():
    prov.validate_revision("0123456789abcdef0123456789abcdef01234567", canonical=True)


def test_non_canonical_runs_do_not_require_a_revision():
    prov.validate_revision(None, canonical=False)  # no raise


def test_model_provenance_flags_whether_the_revision_is_pinned():
    pinned = prov.model_provenance(
        prov.ModelPin("nvidia/Qwen3.6-35B-A3B-NVFP4", "a" * 40, None)
    )
    assert pinned["revision_is_pinned_sha"] is True
    loose = prov.model_provenance(prov.ModelPin("nvidia/Qwen3.6-35B-A3B-NVFP4", "main", None))
    assert loose["revision_is_pinned_sha"] is False


def test_snapshot_identity_is_read_from_a_huggingface_layout(tmp_path):
    snapshot = tmp_path / "models--nvidia--Qwen" / "snapshots" / ("d" * 40)
    snapshot.mkdir(parents=True)
    doc = prov.model_provenance(prov.ModelPin("repo", "d" * 40, str(snapshot)))
    assert doc["snapshot_identity"]["value"] == "d" * 40


def test_snapshot_identity_says_why_it_could_not_be_read(tmp_path):
    plain = tmp_path / "my-model"
    plain.mkdir()
    doc = prov.model_provenance(prov.ModelPin("repo", "d" * 40, str(plain)))
    assert doc["snapshot_identity"]["value"] is None
    assert "cannot be cross-checked" in doc["snapshot_identity"]["unavailable"]


@pytest.mark.parametrize(
    "name,sensitive",
    [
        ("HF_TOKEN", True),
        ("HUGGING_FACE_HUB_TOKEN", True),
        ("MY_API_KEY", True),
        ("SOME_SECRET_THING", True),
        ("AUTH_HEADER", True),
        ("FREETOKEN_INSTRUMENT_PREFILL", False),
        ("CUDA_VISIBLE_DEVICES", False),
    ],
)
def test_credential_shaped_env_names_are_excluded(name, sensitive):
    assert prov.is_sensitive_env(name) is sensitive


def test_relevant_env_captures_runtime_vars_and_drops_credentials(monkeypatch):
    monkeypatch.setenv("FREETOKEN_INSTRUMENT_PREFILL", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HF_TOKEN", "hunter2")
    monkeypatch.setenv("SOMETHING_ELSE", "ignored")
    env = prov.relevant_env()
    assert env["FREETOKEN_INSTRUMENT_PREFILL"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "HF_TOKEN" not in env
    assert "SOMETHING_ELSE" not in env
    assert "hunter2" not in " ".join(env.values())


def test_gpu_provenance_degrades_to_explicit_nulls(monkeypatch):
    monkeypatch.setattr(prov.shutil, "which", lambda _cmd: None)
    doc = prov.gpu_provenance("GPU-abc")
    assert doc["gpus"]["value"] is None and "nvidia-smi" in doc["gpus"]["unavailable"]
    assert doc["topology"]["value"] is None


def test_git_commit_reports_the_repo_it_could_not_read(tmp_path):
    field = prov.git_commit(tmp_path)
    assert field["value"] is None
    assert str(tmp_path) in field["unavailable"]
