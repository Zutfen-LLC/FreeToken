"""Make the InferSwarm Phase-0 harness importable.

It lives in ``benchmarks/inferswarm_phase0/`` rather than inside the installed
``freetoken`` package: it is downstream InferSwarm tooling, and the runtime wheel should
not grow an InferSwarm dependency for it. ``benchmarks/`` is therefore not on sys.path by
default, so put it there for these tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))
