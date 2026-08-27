"""Make the InferSwarm Phase-0 harness importable, and expose the shared fixtures.

It lives in ``benchmarks/inferswarm_phase0/`` rather than inside the installed
``freetoken`` package: it is downstream InferSwarm tooling, and the runtime wheel should
not grow an InferSwarm dependency for it. ``benchmarks/`` is therefore not on sys.path by
default, so put it there for these tests. The campaign builders live in ``fakes.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from .fakes import FAKE_UUID  # noqa: E402 -- must follow the sys.path insertion


@pytest.fixture
def resolved_gpu(monkeypatch):
    """A ``--gpu`` selector that resolves to a stable UUID on a host with no NVML."""
    from inferswarm_phase0 import gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, "_resolve_uuids", lambda selector: (FAKE_UUID,))
    monkeypatch.setattr(gpu_mod, "_smi_index_for", lambda uuid: 0)
    monkeypatch.setattr(
        gpu_mod,
        "engine_gpus",
        lambda origin: [{
            "index": 0,
            "uuid": FAKE_UUID,
            "name": "NVIDIA GeForce RTX 3060",
            "total_bytes": 12 << 30,
        }],
    )
    return FAKE_UUID
