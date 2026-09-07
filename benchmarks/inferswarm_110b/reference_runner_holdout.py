#!/usr/bin/env python3
"""Issue #110 v5 holdout reference entrypoint.

Accepts ONLY h109-* holdout cases and otherwise runs the EXACT frozen
calibration reference runner (``benchmarks/inferswarm_110/reference_runner.py``,
untouched) on the RTX 3090 reference path.

Mechanism (no source duplication of execution math): this wrapper swaps
the frozen runner module's admission callable for the h109-only one, then
delegates to the frozen ``main``. The runner's ``main`` resolves
``validate_campaign_case_ids`` from its own module globals, so the swap
redirects admission and NOTHING else: every model call, layer
partitioning, canonical-prefix construction, capture, FP32 consumer-row
retention, argmax/tie semantics, integrity check, and byte of
execution/math code is the frozen calibration producer's.

Equivalence is enforced by tests: the calibration runner files must be
byte-identical to their freeze (7e5c852) hashes, and this wrapper differs
from the calibration path ONLY in namespace admission (synthetic
fixtures; no holdout plaintext).
"""

from __future__ import annotations

import sys

import benchmarks.inferswarm_110.reference_runner as frozen_reference
from benchmarks.inferswarm_110b import validate_holdout_case_ids

if __name__ == "__main__":
    # Calibration admission replaced by h109-only admission BEFORE main()
    # runs (main resolves the admission callable from this module object).
    frozen_reference.validate_campaign_case_ids = validate_holdout_case_ids
    sys.exit(frozen_reference.main())
