#!/usr/bin/env python3
"""InferSwarm Phase-1 campaign runner -- entry point (P5).

    python benchmarks/phase1_campaign.py plan --help
    python benchmarks/phase1_campaign.py validate --help
    python benchmarks/phase1_campaign.py run-session --help

Canonical campaign issues: https://github.com/Zutfen-LLC/inferswarm/issues/4 and #5.
The runner lives in ``benchmarks/inferswarm_phase1/``; see
``benchmarks/inferswarm_phase1/README.md``. ``plan`` and ``validate`` never start a
model; ``run-session`` is the P6 execution surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inferswarm_phase1.campaign_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
