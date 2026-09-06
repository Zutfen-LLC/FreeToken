"""Issue #97 Phase B holdout-only admission (FreeToken, package init).

Executes the maintainer instruction issued 2026-09-06 after the
`V4_HOLDOUT_UNSEAL_READY_BLOCKED_BY_MISSING_EXECUTION_AUTHORITY` stop:

- the accepted Phase A calibration producer stays frozen at
  `57dfcb7289efac8f66de5b3abbe8de04f2580f75` (PR #31) and its
  c95-/p95- admission discipline is UNCHANGED;
- this package adds the separate, explicitly authorized h95 holdout
  path the frozen producer's docstring promised but never shipped;
- the Phase B producer (this commit) is a SEPARATE externally pinned
  SHA; nothing in `benchmarks/inferswarm_97/` changes.

Importing this module performs NO model work and imports no torch.
"""

from __future__ import annotations

from typing import Any, Sequence

# Re-exported frozen identities: the holdout runs under the SAME accepted
# v4 contract and subject as Phase A (no methodology/subject change).
from benchmarks.inferswarm_97 import (  # noqa: F401  (re-export)
    CONTRACT_ID,
    EXPECTED_CHECKPOINT_SHA256,
    MODEL_REVISION,
)

HOLDOUT_NAMESPACE_PREFIX = "h95-"

# Phase A calibration producer (immutable historical identity).
PHASE_A_PRODUCER_SHA256 = "57dfcb7289efac8f66de5b3abbe8de04f2580f75"

# Frozen holdout corpus identity: the 24 committed h95 cell case IDs from
# inferswarm@e1d3a16 docs/qualification/gemma4-12b-it-v4/manifests/
# sealed-holdout-commitment.json (public commitment, no plaintext).
FROZEN_HOLDOUT_CASE_IDS = (
    "h95-01-01-01", "h95-01-02-01", "h95-01-03-01", "h95-01-04-01",
    "h95-01-05-01", "h95-01-06-01",
    "h95-02-01-01", "h95-02-02-01", "h95-02-03-01", "h95-02-04-01",
    "h95-02-05-01", "h95-02-06-01",
    "h95-03-01-01", "h95-03-02-01", "h95-03-03-01", "h95-03-04-01",
    "h95-03-05-01", "h95-03-06-01",
    "h95-04-01-01", "h95-04-02-01", "h95-04-03-01", "h95-04-04-01",
    "h95-04-05-01", "h95-04-06-01",
)


def validate_holdout_case_ids(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """h95-only admission for the Phase B holdout entrypoints.

    Accepts EXACTLY the 24 frozen committed holdout case IDs. Rejects
    every c95-/p95- calibration case, every historical namespace, and
    any holdout-shaped ID outside the frozen commitment (unknown,
    renumbered, or substituted cells fail closed BEFORE any CUDA work).
    """
    checked = list(cases)
    seen: list[str] = []
    for case in checked:
        case_id = case.get("case_id") if isinstance(case, dict) else None
        if not isinstance(case_id, str) or not case_id.startswith(HOLDOUT_NAMESPACE_PREFIX):
            raise ValueError(
                "issue #97 Phase B accepts only h95- holdout case IDs, "
                f"got {case_id!r}"
            )
        if case_id not in FROZEN_HOLDOUT_CASE_IDS:
            raise ValueError(
                f"issue #97 Phase B: {case_id!r} is not one of the 24 "
                "committed h95 holdout cells"
            )
        if case_id in seen:
            raise ValueError(f"duplicate holdout case id: {case_id!r}")
        seen.append(case_id)
    return checked
