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


# --- the InferSwarm commit ---------------------------------------------------------------

@pytest.mark.parametrize("commit", ["main", "phase0", "abc1234", "g" * 40, "", None])
def test_canonical_runs_reject_an_inferswarm_commit_that_is_not_a_full_sha(commit):
    """A non-empty string is not a provenance record: 'main' moves and a short SHA is
    ambiguous, and the benchmark contract requires the exact commit a result belongs to."""
    with pytest.raises(ValueError):
        prov.validate_inferswarm_commit(commit, canonical=True)


def test_canonical_runs_accept_a_full_inferswarm_sha():
    prov.validate_inferswarm_commit("b" * 40, canonical=True)
    # git prints lower-case, but a pasted upper-case SHA names the same commit
    prov.validate_inferswarm_commit("B" * 40, canonical=True)


def test_non_canonical_runs_do_not_require_an_inferswarm_commit():
    prov.validate_inferswarm_commit(None, canonical=False)


# --- reconciling the pin with the local checkpoint ----------------------------------------

def _snapshot(tmp_path, repo="nvidia--Qwen3.6-35B-A3B-NVFP4", sha="c" * 40):
    path = tmp_path / "hub" / f"models--{repo}" / "snapshots" / sha
    path.mkdir(parents=True)
    return path


def test_a_snapshot_sha_that_disagrees_with_the_revision_is_a_refusal(tmp_path):
    pin = prov.ModelPin("nvidia/Qwen3.6-35B-A3B-NVFP4", "d" * 40, str(_snapshot(tmp_path)))
    reason = prov.check_snapshot_revision(pin)
    assert "disagrees with --model-revision" in reason


def test_a_matching_snapshot_sha_is_accepted(tmp_path):
    pin = prov.ModelPin("nvidia/Qwen3.6-35B-A3B-NVFP4", "c" * 40, str(_snapshot(tmp_path)))
    assert prov.check_snapshot_revision(pin) is None


def test_the_repository_is_cross_checked_from_the_cache_directory(tmp_path):
    """models--<org>--<name> is one level above snapshots/, costs nothing to read, and
    downloads nothing -- so a same-shaped path from another model is caught."""
    path = _snapshot(tmp_path, repo="someone-else--OtherModel")
    pin = prov.ModelPin("nvidia/Qwen3.6-35B-A3B-NVFP4", "c" * 40, str(path))
    assert "disagrees with --model-repository" in prov.check_snapshot_revision(pin)
    identity = prov._snapshot_identity(str(path))
    assert identity["repository"] == "someone-else/OtherModel"


def test_a_non_snapshot_path_records_that_it_cannot_be_cross_checked(tmp_path):
    """An unverifiable path is not the same as a contradicted one; guessing either way
    would be worse than saying so."""
    local = tmp_path / "checkpoint"
    local.mkdir()
    pin = prov.ModelPin("nvidia/Qwen3.6-35B-A3B-NVFP4", "c" * 40, str(local))
    assert prov.check_snapshot_revision(pin) is None
    identity = prov._snapshot_identity(str(local))
    assert "cannot be cross-checked" in identity["unavailable"]


# --- a dirty FreeToken checkout ----------------------------------------------------------

def test_a_dirty_tree_is_refused_and_names_the_paths():
    """Recording the modified filenames does not make a run reproducible -- the contents
    are what changed, and they are nowhere in the artifact."""
    reason = prov.check_clean_working_tree(
        {"value": "a" * 40, "dirty": True,
         "dirty_paths": [" M python/freetoken/engine/engine.py", "?? scratch.py"]}
    )
    assert "cannot be reproduced from commit" in reason
    assert "engine.py" in reason and "scratch.py" in reason
    assert "--dev-smoke" in reason


def test_a_clean_tree_is_accepted():
    assert prov.check_clean_working_tree(
        {"value": "a" * 40, "dirty": False, "dirty_paths": []}
    ) is None


def test_an_unreadable_commit_block_is_not_treated_as_dirty():
    assert prov.check_clean_working_tree(prov.unavailable("not a git checkout")) is None
