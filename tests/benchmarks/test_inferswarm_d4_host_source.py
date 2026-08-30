from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from inferswarm_d4.host_source import (
    fresh_arm_command,
    fresh_arm_env,
    safety_contract,
    snapshot_manifest,
    stage_read_only_tmpfs,
)


def test_stage_is_exact_and_read_only(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"revision":"frozen"}\n')
    (source / "model.safetensors").write_bytes(b"immutable-weight-bytes")
    staged = tmp_path / "staged"
    result = stage_read_only_tmpfs(source, staged)
    assert snapshot_manifest(source) == snapshot_manifest(staged)
    assert result.staged_bytes == sum(path.stat().st_size for path in source.iterdir())
    assert result.manifest_sha256 == hashlib.sha256(snapshot_manifest(staged)[0]).hexdigest()
    assert all(not path.stat().st_mode & 0o222 for path in [staged, *staged.iterdir()])


def test_only_model_source_crosses_fresh_arm_boundary(tmp_path: Path):
    source = type("Source", (), {"staged": tmp_path / "model", "manifest_sha256": "a" * 64})()
    original = ["python", "-m", "freetoken.cli", "serve", "--model", "/ssd/model", "--gpu", "UUID"]
    command = fresh_arm_command(original, source)
    assert original[5] == "/ssd/model"
    assert command[5] == str(tmp_path / "model")
    assert command[:5] == original[:5] and command[6:] == original[6:]
    assert fresh_arm_env(source, "arm-2") == {
        "FREETOKEN_D4_HOST_SOURCE_SHA256": "a" * 64,
        "FREETOKEN_D4_FRESH_ARM_NONCE": "arm-2",
    }
    with pytest.raises(ValueError):
        fresh_arm_env(source, "")


def test_contract_excludes_every_mutable_engine_resource():
    contract = safety_contract()
    assert contract["reused"] == ["immutable model snapshot files"]
    assert contract["parent_imports_torch"] is False
    assert contract["fork_after_cuda"] is False
    fresh = set(contract["fresh_per_arm"])
    assert {"Engine", "GPU0 MoE cache", "worker resident banks", "KV cache", "CUDA graphs",
            "CUDA streams/events", "D3/D4 counters", "device pointers", "generated request state"} <= fresh
