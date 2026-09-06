#!/usr/bin/env python3
"""Issue #97 Phase B holdout chain entrypoint.

Accepts ONLY h95-* holdout cases and otherwise runs the EXACT frozen
Phase A chain runner (`benchmarks/inferswarm_97/chain_runner.py`,
untouched) teacher-forced against the h95 reference evidence.

Same mechanism as ``reference_runner_holdout``: swap the frozen runner
module's admission callable for the h95-only one, delegate to the frozen
``main``. No execution/model math is duplicated or altered.
"""

from __future__ import annotations

import sys

import benchmarks.inferswarm_97.chain_runner as frozen_chain
from benchmarks.inferswarm_97b import validate_holdout_case_ids

if __name__ == "__main__":
    frozen_chain.validate_campaign_case_ids = validate_holdout_case_ids
    sys.exit(frozen_chain.main())
