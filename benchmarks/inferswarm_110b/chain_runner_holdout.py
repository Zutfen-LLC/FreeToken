#!/usr/bin/env python3
"""Issue #110 v5 holdout chain entrypoint.

Accepts ONLY h109-* holdout cases and otherwise runs the EXACT frozen
calibration chain runner (``benchmarks/inferswarm_110/chain_runner.py``,
untouched) teacher-forced against the h109 reference evidence on the
accepted three-stage RTX 3060 chain.

Same mechanism as ``reference_runner_holdout``: swap the frozen runner
module's admission callable for the h109-only one, delegate to the frozen
``main``. No execution/model math is duplicated or altered.
"""

from __future__ import annotations

import sys

import benchmarks.inferswarm_110.chain_runner as frozen_chain
from benchmarks.inferswarm_110b import validate_holdout_case_ids

if __name__ == "__main__":
    frozen_chain.validate_campaign_case_ids = validate_holdout_case_ids
    sys.exit(frozen_chain.main())
