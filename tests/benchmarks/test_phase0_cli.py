"""CLI surface: the dry run must make the canonical/non-canonical distinction unmissable,
and a canonical invocation must refuse the things the criteria forbid."""

from __future__ import annotations

import json

import pytest

from inferswarm_phase0.cli import main
from inferswarm_phase0.manifest import CLASS_SPECS, REQUIRED_CLASSES, sha256_text

SHA40 = "a" * 40


def _manifest_file(tmp_path, *, canonical=True, classes=REQUIRED_CLASSES):
    entries = []
    for c in classes:
        content = f"prompt for {c}"
        entries.append({
            "class_id": c, "content": content, "content_sha256": sha256_text(content),
            "output_tokens": CLASS_SPECS[c].output_tokens, "ignore_eos": True,
            "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
            "seed": None, "chat_template_kwargs": {}, "role": "user",
        })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema": "inferswarm.phase0.workload-manifest/1",
        "manifest_id": "cli-test", "canonical": canonical, "workloads": entries,
    }))
    return str(path)


def _argv(tmp_path, *extra, canonical=True, classes=REQUIRED_CLASSES):
    return [
        "sweep",
        "--model", str(tmp_path / "model"),
        "--manifest", _manifest_file(tmp_path, canonical=canonical, classes=classes),
        "--model-revision", SHA40,
        "--inferswarm-commit", "b" * 40,
        "--gpu", "GPU-abc",
        "--out-root", str(tmp_path / "runs"),
        "--dry-run",
        *extra,
    ]


def test_canonical_dry_run_prints_the_full_plan(tmp_path, capsys):
    assert main(_argv(tmp_path)) == 0
    out = capsys.readouterr()
    doc = json.loads(out.out)
    assert doc["canonical"] is True
    assert doc["canonical_blockers"] == []
    assert [a["id"] for a in doc["arms"]] == ["B1", "B2", "B3", "B4", "B5"]
    # 5 arms x 4 classes x (2 warmup + 10 measured)
    assert doc["total_generations"] == 240
    assert doc["measured_generations"] == 200
    assert "CANONICAL Phase-0 protocol" in out.err


def test_dry_run_marks_a_smoke_test_unmistakably(tmp_path, capsys):
    argv = _argv(tmp_path, "--dev-smoke", "--repetitions", "2", "--warmups", "0",
                 "--arms", "B1", canonical=False, classes=("W2",))
    assert main(argv) == 0
    out = capsys.readouterr()
    doc = json.loads(out.out)
    assert doc["canonical"] is False
    assert "NON-CANONICAL developer smoke test" in out.err
    assert doc["total_generations"] == 2
    assert any("canonical=false" in b for b in doc["canonical_blockers"])


def test_dry_run_shows_the_exact_serve_command_per_arm(tmp_path, capsys):
    main(_argv(tmp_path))
    doc = json.loads(capsys.readouterr().out)
    by_id = {a["id"]: a for a in doc["arms"]}
    b5 = " ".join(by_id["B5"]["serve_command"])
    assert "--moe-backend cpu" in b5 and "--nvfp4-backend triton" in b5
    assert "--gpu GPU-abc" in b5
    assert by_id["B2"]["bench_bw_command"] is not None
    assert by_id["B1"]["bench_bw_command"] is None


def test_dry_run_records_the_execution_order(tmp_path, capsys):
    main(_argv(tmp_path))
    doc = json.loads(capsys.readouterr().out)
    order = doc["execution_order"]
    assert [s["execution_index"] for s in order] == list(range(len(order)))
    assert order[0]["phase"] == "warmup" and order[2]["phase"] == "measured"


def test_reverse_order_is_visible_in_the_plan(tmp_path, capsys):
    main(_argv(tmp_path, "--reverse-order", "--session-id", "session-2"))
    doc = json.loads(capsys.readouterr().out)
    assert doc["execution_order"][0]["arm_id"] == "B5"
    assert doc["protocol"]["session_id"] == "session-2"
    assert doc["protocol"]["order_reversed"] is True


def test_canonical_sweep_refuses_a_symbolic_model_revision(tmp_path, capsys):
    argv = _argv(tmp_path)
    argv[argv.index("--model-revision") + 1] = "main"
    assert main(argv) == 2
    assert "not a 40-hex commit SHA" in capsys.readouterr().err


def test_canonical_sweep_refuses_a_partial_arm_set(tmp_path, capsys):
    with pytest.raises(SystemExit, match="all of B1-B5"):
        main(_argv(tmp_path, "--arms", "B1,B2"))


def test_canonical_sweep_refuses_protocol_overrides(tmp_path, capsys):
    assert main(_argv(tmp_path, "--repetitions", "3")) == 2
    assert "--dev-smoke" in capsys.readouterr().err


def test_unknown_arm_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="unknown arm"):
        main(_argv(tmp_path, "--arms", "B9"))


def test_reference_subcommand_requires_a_resolved_backend_and_fixed_cache(tmp_path, capsys):
    base = [
        "reference", "--model", str(tmp_path / "model"),
        "--manifest", _manifest_file(tmp_path),
        "--model-revision", SHA40, "--inferswarm-commit", "b" * 40,
        "--out-root", str(tmp_path / "runs"), "--dry-run",
    ]
    assert main(base + ["--nvfp4-backend", "triton", "--moe-cache-size", "512"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["arms"][0]["id"] == "CORRECTNESS_REFERENCE"
    assert doc["arms"][0]["role"] == "correctness"
    assert "--moe-cpu-layers" in doc["arms"][0]["moe_flags"]
    # 'auto' is not a choice argparse accepts here at all
    with pytest.raises(SystemExit):
        main(base + ["--nvfp4-backend", "auto", "--moe-cache-size", "512"])


def test_hash_subcommand_prints_a_freezable_digest(tmp_path, capsys):
    fixture = tmp_path / "w1.txt"
    fixture.write_text("some prompt\n")
    assert main(["hash", str(fixture)]) == 0
    printed = capsys.readouterr().out.split()[0]
    assert printed == sha256_text("some prompt\n")
