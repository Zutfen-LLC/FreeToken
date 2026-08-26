"""CLI surface: the dry run must make the canonical/non-canonical distinction unmissable,
and a canonical invocation must refuse the things the criteria forbid."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from inferswarm_phase0 import CANONICAL_MODEL_REPOSITORY
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
    # The `ft bench bw` refresh is no longer a per-arm side effect: it is one session-level
    # prerequisite that runs before the traversal, because B2 AND B3 read the profile.
    assert by_id["B2"]["consumes_bench_bw"] is True
    assert by_id["B3"]["consumes_bench_bw"] is True
    assert by_id["B1"]["consumes_bench_bw"] is False
    prerequisite = doc["bench_bw_prerequisite"]
    assert prerequisite["runs_before_the_sweep"] is True
    assert prerequisite["consuming_arms"] == ["B2", "B3"]
    assert "--gpu" in prerequisite["command"]
    assert prerequisite["dtype"] == "nvfp4"
    assert prerequisite["command"][prerequisite["command"].index("--dtype") + 1] == "nvfp4"


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


def test_canonical_exact_model_repository_is_accepted(tmp_path, capsys):
    argv = _argv(tmp_path, "--model-repository", CANONICAL_MODEL_REPOSITORY)
    assert main(argv) == 0
    doc = json.loads(capsys.readouterr().out)
    assert not any("model repository" in reason for reason in doc["preflight_refusals"])


def test_canonical_alternate_model_repository_is_a_dry_run_refusal(tmp_path, capsys):
    argv = _argv(tmp_path, "--model-repository", "Qwen/Qwen3.6-35B-A3B-FP8")
    assert main(argv) == 0
    output = capsys.readouterr()
    doc = json.loads(output.out)
    assert any(CANONICAL_MODEL_REPOSITORY in reason for reason in doc["preflight_refusals"])
    assert "alternate models require --dev-smoke" in output.err


def test_canonical_alternate_model_repository_is_rejected_before_execution(tmp_path, capsys):
    argv = _argv(tmp_path, "--model-repository", "Qwen/Qwen3.6-35B-A3B-FP8")
    argv.remove("--dry-run")
    assert main(argv) == 2
    error = capsys.readouterr().err
    assert CANONICAL_MODEL_REPOSITORY in error
    assert "alternate models require --dev-smoke" in error


def test_alternate_model_repository_is_allowed_only_for_dev_smoke(tmp_path, capsys):
    argv = _argv(
        tmp_path,
        "--dev-smoke",
        "--model-repository",
        "Qwen/Qwen3.6-35B-A3B-FP8",
        canonical=False,
    )
    assert main(argv) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["canonical"] is False
    assert doc["preflight_refusals"] == []


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
        "--gpu", "GPU-abc",
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


def test_canonical_correctness_reference_requires_the_exact_model_repository(tmp_path, capsys):
    argv = [
        "reference", "--model", str(tmp_path / "model"),
        "--model-repository", "Qwen/Qwen3.6-35B-A3B-FP8",
        "--manifest", _manifest_file(tmp_path),
        "--model-revision", SHA40, "--inferswarm-commit", "b" * 40,
        "--gpu", "GPU-abc", "--out-root", str(tmp_path / "runs"), "--dry-run",
        "--nvfp4-backend", "triton", "--moe-cache-size", "512",
    ]
    assert main(argv) == 0
    output = capsys.readouterr()
    doc = json.loads(output.out)
    assert any(CANONICAL_MODEL_REPOSITORY in reason for reason in doc["preflight_refusals"])
    assert "alternate models require --dev-smoke" in output.err


def test_hash_subcommand_prints_a_freezable_digest(tmp_path, capsys):
    fixture = tmp_path / "w1.txt"
    fixture.write_text("some prompt\n")
    assert main(["hash", str(fixture)]) == 0
    printed = capsys.readouterr().out.split()[0]
    assert printed == sha256_text("some prompt\n")


# --- the canonical prerequisites the CLI must refuse outright -------------------------------

def test_canonical_sweep_refuses_no_bench_bw(tmp_path):
    """B2's fetch split AND B3's auto backend pick both read the profile, so skipping the
    refresh is refused rather than quietly downgraded."""
    with pytest.raises(SystemExit, match="--no-bench-bw is refused"):
        main(_argv(tmp_path, "--no-bench-bw"))


def test_canonical_sweep_refuses_an_alternate_bench_bw_dtype_even_on_dry_run(tmp_path):
    with pytest.raises(
        SystemExit,
        match="canonical Phase-0 sweep requires --bench-bw-dtype nvfp4",
    ):
        main(_argv(tmp_path, "--bench-bw-dtype", "bf16"))


def test_a_smoke_run_records_an_alternate_bench_bw_dtype(tmp_path, capsys):
    argv = _argv(
        tmp_path,
        "--dev-smoke",
        "--bench-bw-dtype",
        "bf16",
        canonical=False,
    )
    assert main(argv) == 0
    doc = json.loads(capsys.readouterr().out)
    prerequisite = doc["bench_bw_prerequisite"]
    assert doc["canonical"] is False
    assert prerequisite["dtype"] == "bf16"
    assert prerequisite["command"][prerequisite["command"].index("--dtype") + 1] == "bf16"


def test_a_smoke_run_may_skip_bench_bw_and_stays_non_canonical(tmp_path, capsys):
    argv = _argv(tmp_path, "--dev-smoke", "--no-bench-bw", canonical=False)
    assert main(argv) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["canonical"] is False
    assert doc["bench_bw_prerequisite"]["runs_before_the_sweep"] is False


def test_default_phase0_output_root_is_repository_ignored():
    repo_root = Path(__file__).resolve().parents[2]
    patterns = (repo_root / ".gitignore").read_text().splitlines()
    assert "/phase0-runs/" in patterns


def test_canonical_sweep_requires_a_gpu(tmp_path, capsys):
    argv = [a for a in _argv(tmp_path)]
    i = argv.index("--gpu")
    del argv[i:i + 2]
    assert main(argv) == 2
    assert "--gpu is required for a canonical run" in capsys.readouterr().err


def test_canonical_sweep_refuses_an_abbreviated_inferswarm_commit(tmp_path, capsys):
    argv = _argv(tmp_path)
    argv[argv.index("--inferswarm-commit") + 1] = "abc1234"
    assert main(argv) == 2
    assert "not a 40-hex commit SHA" in capsys.readouterr().err


def test_canonical_sweep_refuses_a_missing_inferswarm_commit(tmp_path, capsys):
    argv = _argv(tmp_path)
    i = argv.index("--inferswarm-commit")
    del argv[i:i + 2]
    assert main(argv) == 2
    assert "--inferswarm-commit is required" in capsys.readouterr().err


def test_the_dry_run_shows_the_reference_arm_is_forced_greedy(tmp_path, capsys):
    base = [
        "reference", "--model", str(tmp_path / "model"),
        "--manifest", _manifest_file(tmp_path),
        "--model-revision", SHA40, "--inferswarm-commit", "b" * 40,
        "--gpu", "GPU-abc",
        "--out-root", str(tmp_path / "runs"), "--dry-run",
        "--nvfp4-backend", "triton", "--moe-cache-size", "512",
    ]
    assert main(base) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["arms"][0]["request_sampling"] == {
        "temperature": 0.0, "top_p": 1.0, "top_k": -1
    }
