"""Workload-manifest validation.

The expectations come from InferSwarm criteria section 9 (the four classes and their fixed
output lengths), section 3 rule 5 (ignore_eos), and FreeToken's own ``SamplingParams``
(no seed exists), not from what the validator happens to do.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from inferswarm_phase0.manifest import (
    CLASS_SPECS,
    REQUIRED_CLASSES,
    ManifestError,
    check_prompt_tokens,
    load_manifest,
    sha256_text,
)

GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": -1}


def _entry(class_id: str, content: str, **overrides):
    entry = {
        "class_id": class_id,
        "content": content,
        "content_sha256": sha256_text(content),
        "output_tokens": CLASS_SPECS[class_id].output_tokens,
        "ignore_eos": True,
        "sampling": dict(GREEDY),
        "seed": None,
        "chat_template_kwargs": {},
        "role": "user",
    }
    entry.update(overrides)
    return entry


def _write(tmp_path, entries, *, canonical=True, name="m.json"):
    doc = {
        "schema": "inferswarm.phase0.workload-manifest/1",
        "manifest_id": "test",
        "canonical": canonical,
        "workloads": entries,
    }
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return path


def _full_set(tmp_path, **kw):
    return _write(tmp_path, [_entry(c, f"prompt for {c}") for c in REQUIRED_CLASSES], **kw)


def test_a_complete_manifest_loads_and_pins_its_prompts(tmp_path):
    manifest = load_manifest(_full_set(tmp_path), canonical=True)
    assert [w.class_id for w in manifest.workloads] == list(REQUIRED_CLASSES)
    assert manifest.missing_classes() == []
    w2 = manifest.by_class()["W2"]
    assert w2.content_sha256 == hashlib.sha256(b"prompt for W2").hexdigest()
    assert w2.greedy is True
    # the run artifact records hashes and settings, never the prompt text
    record = manifest.record()
    assert all("prompt" not in entry for entry in record["workloads"])
    assert record["manifest_sha256"]


def test_canonical_run_refuses_an_incomplete_class_set(tmp_path):
    path = _write(tmp_path, [_entry("W2", "only one")])
    with pytest.raises(ManifestError, match=r"missing \['W1', 'W3', 'W4'\]"):
        load_manifest(path, canonical=True)
    # ... but a developer smoke test may run a subset
    manifest = load_manifest(path, canonical=False)
    assert manifest.missing_classes() == ["W1", "W3", "W4"]


def test_canonical_run_refuses_a_manifest_declared_non_canonical(tmp_path):
    path = _full_set(tmp_path, canonical=False)
    with pytest.raises(ManifestError, match="declares canonical=false"):
        load_manifest(path, canonical=True)


def test_content_hash_mismatch_is_fatal(tmp_path):
    entry = _entry("W2", "the frozen prompt")
    entry["content"] = "an edited prompt"
    path = _write(tmp_path, [entry])
    with pytest.raises(ManifestError, match="content_sha256 mismatch"):
        load_manifest(path, canonical=False)


def test_fixture_files_are_hashed_from_their_bytes(tmp_path):
    text = "fixture on disk\n"
    (tmp_path / "w2.txt").write_text(text)
    entry = _entry("W2", "unused")
    entry.pop("content")
    entry["fixture_path"] = "w2.txt"
    entry["content_sha256"] = sha256_text(text)
    manifest = load_manifest(_write(tmp_path, [entry]), canonical=False)
    assert manifest.by_class()["W2"].prompt == text


def test_a_missing_fixture_is_fatal(tmp_path):
    entry = _entry("W2", "unused")
    entry.pop("content")
    entry["fixture_path"] = "nope.txt"
    entry["content_sha256"] = "0" * 64
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(_write(tmp_path, [entry]), canonical=False)


def test_content_and_fixture_path_are_mutually_exclusive(tmp_path):
    entry = _entry("W2", "inline")
    entry["fixture_path"] = "w2.txt"
    with pytest.raises(ManifestError, match="exactly one"):
        load_manifest(_write(tmp_path, [entry]), canonical=False)


@pytest.mark.parametrize("class_id,expected", [(c, CLASS_SPECS[c].output_tokens) for c in REQUIRED_CLASSES])
def test_canonical_output_token_counts_are_enforced(tmp_path, class_id, expected):
    """W1/W2 512, W3 256, W4 128 (criteria section 9)."""
    entries = [_entry(c, f"p{c}") for c in REQUIRED_CLASSES]
    for entry in entries:
        if entry["class_id"] == class_id:
            entry["output_tokens"] = expected + 1
    with pytest.raises(ManifestError, match=f"canonical output_tokens is {expected}"):
        load_manifest(_write(tmp_path, entries), canonical=True)


def test_canonical_runs_require_ignore_eos(tmp_path):
    entries = [_entry(c, f"p{c}") for c in REQUIRED_CLASSES]
    entries[0]["ignore_eos"] = False
    with pytest.raises(ManifestError, match="ignore_eos=true"):
        load_manifest(_write(tmp_path, entries), canonical=True)


def test_a_seed_is_rejected_because_freetoken_has_none(tmp_path):
    entry = _entry("W2", "p")
    entry["seed"] = 1234
    with pytest.raises(ManifestError, match="exposes no seed parameter"):
        load_manifest(_write(tmp_path, [entry]), canonical=False)


def test_sampling_must_be_stated_in_full(tmp_path):
    entry = _entry("W2", "p")
    entry["sampling"] = {"temperature": 0.7}
    with pytest.raises(ManifestError, match=r"missing \['top_p', 'top_k'\]"):
        load_manifest(_write(tmp_path, [entry]), canonical=False)


def test_greedy_detection_mirrors_sampling_params(tmp_path):
    """SamplingParams.is_greedy: (temperature <= 0 or top_k == 1) and top_p == 1."""
    cases = {
        (0.0, 1.0, -1): True,
        (1.0, 1.0, 1): True,
        (1.0, 0.95, 64): False,
        (0.0, 0.95, -1): False,
    }
    for (temperature, top_p, top_k), expected in cases.items():
        entry = _entry("W2", f"p{temperature}{top_p}{top_k}")
        entry["sampling"] = {"temperature": temperature, "top_p": top_p, "top_k": top_k}
        manifest = load_manifest(
            _write(tmp_path, [entry], name=f"m{temperature}{top_p}{top_k}.json"), canonical=False
        )
        assert manifest.by_class()["W2"].greedy is expected


def test_duplicate_and_unknown_classes_are_rejected(tmp_path):
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(_write(tmp_path, [_entry("W2", "a"), _entry("W2", "b")]), canonical=False)
    bad = _entry("W2", "a")
    bad["class_id"] = "W9"
    with pytest.raises(ManifestError, match="not one of"):
        load_manifest(_write(tmp_path, [bad]), canonical=False)


def test_schema_version_is_checked(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"schema": "something/2", "manifest_id": "x", "workloads": []}))
    with pytest.raises(ManifestError, match="schema must be"):
        load_manifest(path, canonical=False)


def test_request_body_states_every_generation_field():
    from inferswarm_phase0.manifest import Workload

    w = Workload(
        class_id="W2", prompt="hi", content_sha256="x", output_tokens=512,
        sampling=dict(GREEDY), ignore_eos=True, greedy=True,
        chat_template_kwargs={"enable_thinking": True}, role="user", fixture_path=None,
    )
    body = w.request_body("qwen")
    assert body["max_tokens"] == 512
    assert body["ignore_eos"] is True
    assert body["temperature"] == 0.0 and body["top_p"] == 1.0 and body["top_k"] == -1
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize(
    "class_id,tokens,fits",
    [
        ("W1", 2000, True), ("W1", 2001, False),
        ("W2", 999, True), ("W2", 1500, False),
        ("W3", 16000, True), ("W3", 4000, False),
        ("W4", 128, True), ("W4", 900, False),
    ],
)
def test_prompt_token_bounds_are_observed_not_enforced(class_id, tokens, fits):
    """A prompt outside its class band is recorded as a deviation, not silently rewritten
    and not treated as a crash: the tokenizer decides the real count."""
    result = check_prompt_tokens(class_id, tokens)
    assert (result is None) is fits
