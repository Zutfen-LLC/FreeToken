#!/usr/bin/env python3
"""InferSwarm Phase-0 baseline harness -- entry point.

    python benchmarks/phase0_baseline.py sweep --help
    python benchmarks/phase0_baseline.py reference --help
    python benchmarks/phase0_baseline.py profile --help
    python benchmarks/phase0_baseline.py routing --help

Canonical baseline issue: https://github.com/Zutfen-LLC/inferswarm/issues/2
Routing instrumentation support: https://github.com/Zutfen-LLC/inferswarm/issues/3
The harness itself lives in ``benchmarks/inferswarm_phase0/``; see ``benchmarks/README.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The harness is a package under benchmarks/, not part of the installed freetoken wheel:
# it is downstream InferSwarm tooling and deliberately does not widen the runtime package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inferswarm_phase0.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
