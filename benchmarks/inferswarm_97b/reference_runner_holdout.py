#!/usr/bin/env python3
"""Issue #97 Phase B holdout reference entrypoint.

Accepts ONLY h95-* holdout cases and otherwise runs the EXACT frozen
Phase A reference runner (`benchmarks/inferswarm_97/reference_runner.py`,
untouched) on the RTX 3090 reference path.

Mechanism (no source duplication of execution math): this wrapper swaps
the frozen runner module's admission callable for the h95-only one, then
delegates to the frozen ``main``. The frozen module object carries the
swap, so its internal ``validate_campaign_case_ids`` reference resolves
to h95-only admission; every other function, constant, model call, and
byte of execution math is the frozen Phase A code.

Equivalence is enforced by tests: the Phase A runner files must be
byte-identical to their Phase A freeze hashes, and this wrapper differs
from the Phase A path ONLY in namespace admission (synthetic fixtures;
no secret material).
"""

from __future__ import annotations

import sys

import benchmarks.inferswarm_97.reference_runner as frozen_reference
from benchmarks.inferswarm_97b import validate_holdout_case_ids

if __name__ == "__main__":
    # Phase A admission replaced by h95-only admission BEFORE main() runs
    # (main resolves the admission callable from this module object).
    frozen_reference.validate_campaign_case_ids = validate_holdout_case_ids
    sys.exit(frozen_reference.main())
