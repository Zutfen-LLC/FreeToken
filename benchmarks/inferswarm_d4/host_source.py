"""Benchmark-only immutable model snapshot reuse for fresh-engine D4 arms.

The parent deliberately does not import torch.  It stages ordinary files into tmpfs and
starts each arm as a brand-new process.  No Engine, CUDA object, or generated request can
cross this boundary.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ImmutableHostSource:
    source: Path
    staged: Path
    manifest_sha256: str
    staged_bytes: int


def _files(root: Path):
    return sorted(path for path in root.rglob("*") if path.is_file())


def snapshot_manifest(root: Path) -> tuple[bytes, int]:
    """Hash metadata plus every file byte; run once at staging, never per arm."""
    rows = []
    total = 0
    for path in _files(root):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
        size = path.stat().st_size
        total += size
        rows.append({"path": str(path.relative_to(root)), "size": size,
                     "sha256": digest.hexdigest()})
    return (json.dumps(rows, sort_keys=True, separators=(",", ":")).encode(), total)


def stage_read_only_tmpfs(source: str | Path, staged: str | Path) -> ImmutableHostSource:
    """Materialize one exact RAM-backed source before any child initializes CUDA."""
    source, staged = Path(source).resolve(), Path(staged).resolve()
    if not source.is_dir():
        raise ValueError(f"model snapshot is not a directory: {source}")
    if staged == source or source in staged.parents:
        raise ValueError("staged model source must be separate from the source snapshot")
    if staged.exists():
        shutil.rmtree(staged)
    staged.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-a", "--reflink=never", str(source), str(staged)], check=True)
    source_manifest, source_bytes = snapshot_manifest(source)
    staged_manifest, staged_bytes = snapshot_manifest(staged)
    if source_manifest != staged_manifest or source_bytes != staged_bytes:
        shutil.rmtree(staged)
        raise RuntimeError("tmpfs snapshot disagrees with immutable source")
    # Make accidental model-source mutation fail mechanically. The session parent owns this
    # copy and removes/replaces it before a future staging operation.
    for path in [*_files(staged), *sorted(p for p in staged.rglob("*") if p.is_dir()), staged]:
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    return ImmutableHostSource(source, staged, hashlib.sha256(staged_manifest).hexdigest(), staged_bytes)


def fresh_arm_command(command: Sequence[str], source: ImmutableHostSource) -> list[str]:
    """Replace the argument after --model; all other arm arguments remain frozen."""
    result = list(command)
    indices = [i for i, value in enumerate(result) if value == "--model"]
    if len(indices) != 1 or indices[0] + 1 >= len(result):
        raise ValueError("fresh arm command must contain exactly one --model value")
    result[indices[0] + 1] = str(source.staged)
    return result


def fresh_arm_env(source: ImmutableHostSource, arm_nonce: str) -> dict[str, str]:
    """Provenance only: the nonce cannot identify or restore mutable runtime state."""
    if not arm_nonce:
        raise ValueError("each fresh arm requires a non-empty nonce")
    return {"FREETOKEN_D4_HOST_SOURCE_SHA256": source.manifest_sha256,
            "FREETOKEN_D4_FRESH_ARM_NONCE": arm_nonce}


def safety_contract() -> dict[str, object]:
    """Machine-readable ownership boundary recorded in D4 evidence."""
    return {
        "reused": ["immutable model snapshot files"],
        "fresh_per_arm": ["process", "Engine", "CUDA allocator/runtime state", "GPU0 MoE cache",
                          "worker resident banks", "KV cache", "CUDA streams/events", "CUDA graphs",
                          "D3/D4 counters", "device pointers", "generated request state"],
        "parent_imports_torch": False,
        "fork_after_cuda": False,
        "child_creation": "new subprocess per arm",
        "staged_source_read_only": True,
    }
