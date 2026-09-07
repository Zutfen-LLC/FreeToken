"""Issue #110 v5 holdout-only admission (FreeToken, package init).

Executes the maintainer instruction recorded on InferSwarm PR #114
(2026-09-07, ``V5_HOLDOUT_UNSEAL_AUTHORIZED``):

- the accepted calibration producer stays frozen at
  ``7e5c852163afd9aadfccc406be267e8d060e79ef`` (FreeToken PR #32) and its
  c109-/p109- admission discipline is UNCHANGED;
- this package adds the separate, explicitly authorized h109 holdout
  path that the frozen producer's admission gate defers to;
- the holdout producer (this commit) is a SEPARATE externally pinned
  SHA; nothing in ``benchmarks/inferswarm_110/`` changes.

Importing this module performs NO model work and imports no torch.
"""

from __future__ import annotations

from typing import Any, Sequence

# Re-exported frozen identities: the holdout runs under the SAME accepted
# v5 contract and subject as the calibration producer (no change).
from benchmarks.inferswarm_110 import (  # noqa: F401  (re-export)
    CONTRACT_ID,
    EXPECTED_CHECKPOINT_SHA256,
    METHODOLOGY_COMMIT,
    MODEL_REVISION,
)

HOLDOUT_NAMESPACE_PREFIX = "h109-"

# Calibration producer (immutable historical identity, PR #32 head at the
# maintainer authorization).
CALIBRATION_PRODUCER_SHA = "7e5c852163afd9aadfccc406be267e8d060e79ef"

# Frozen holdout corpus identity: the 24 committed h109 case IDs from
# inferswarm@fc62c4323f1f4dbea55f83a3bb17e3c00c46dad7
# docs/qualification/gemma4-12b-it-v5/manifests/sealed-holdout-commitment.json
# (public commitment, no plaintext).
FROZEN_HOLDOUT_CASE_IDS = (
    "h109-01-01-001", "h109-01-01-002",
    "h109-01-02-001", "h109-01-02-002",
    "h109-01-06-001", "h109-01-06-002",
    "h109-02-02-001",
    "h109-02-04-001",
    "h109-02-05-001", "h109-02-05-002",
    "h109-02-06-001", "h109-02-06-002",
    "h109-03-02-001",
    "h109-03-03-001",
    "h109-03-04-001", "h109-03-04-002",
    "h109-03-05-001", "h109-03-05-002",
    "h109-03-06-001", "h109-03-06-002", "h109-03-06-003",
    "h109-04-01-001",
    "h109-04-04-001", "h109-04-04-002",
)


def validate_holdout_case_ids(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """h109-only admission for the v5 holdout entrypoints.

    Accepts EXACTLY the 24 frozen committed holdout case IDs. Rejects
    every c109-/p109- calibration case, every historical namespace, and
    any holdout-shaped ID outside the frozen commitment (unknown,
    renumbered, or substituted cells fail closed BEFORE any CUDA work).
    """
    checked = list(cases)
    seen: list[str] = []
    for case in checked:
        case_id = case.get("case_id") if isinstance(case, dict) else None
        if not isinstance(case_id, str) or not case_id.startswith(HOLDOUT_NAMESPACE_PREFIX):
            raise ValueError(
                f"issue #110 holdout path accepts only h109- case IDs, got {case_id!r}"
            )
        if case_id not in FROZEN_HOLDOUT_CASE_IDS:
            raise ValueError(
                f"issue #110 holdout path: {case_id!r} is not one of the 24 "
                "committed h109 holdout cases"
            )
        if case_id in seen:
            raise ValueError(f"duplicate holdout case id: {case_id!r}")
        seen.append(case_id)
    return checked
