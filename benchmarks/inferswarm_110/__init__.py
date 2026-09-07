"""Issue #110 (InferSwarm): Gemma v5 physical calibration producer.

Executes the accepted issue #109 v5 mixture-population methodology
(inferswarm @ bc6f0ec657d025702d5928771bf8f51aa563a8be,
docs/qualification/gemma4-12b-it-v5/) on the frozen physical topology:

- reference: inferswarm04 RTX 3090 24 GiB, matched single-GPU FreeToken
  runtime (accepted R6 GemmaDenseStage, replay-prefill greedy);
- candidate: accepted three-stage RTX 3060 chain — inferswarm01 GPU-0
  (stage 1, layers [0,16)), inferswarm01 GPU-1 (stage 2, layers [16,32)),
  inferswarm03 (stage 3/last, layers [32,48) via the #76 R4 wire service).

The execution harness is the accepted #97 v4 producer path, renamed only to
bind the v5 evidence identity.  It adds ZERO model/execution math: every
execution function is imported byte-identical from
``benchmarks.inferswarm_97`` (itself the accepted #88 path).  The only v5
additions are identity bindings:

1. the v5 contract id and the accepted #109 methodology merge;
2. campaign case admission restricted to the public c109-*/p109-* corpora
   (the h109-* holdout has a separate, explicitly authorized path that does
   not exist yet and is never accepted here);
3. v5 evidence schema strings on the per-case/run summaries.

The v5 comparator differences from v4 — all-8-decision consumer-logit
reducers (capture_position_rule), p99 as mandatory telemetry — are enforced
by the accepted CPU-side InferSwarm tooling that consumes this evidence;
the harness already retains the full FP32 consumer row at ALL 8 decisions
on both arms, which is exactly the input that rule requires.

Execution-branch discipline (issue #110): this package and its tests freeze
as the physical implementation producer BEFORE the first model execution;
after that freeze no execution or model math may change during the campaign.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Sequence

# Accepted v5 methodology identity (inferswarm PR #112 merge).
METHODOLOGY_COMMIT = "bc6f0ec657d025702d5928771bf8f51aa563a8be"
V5_ISSUE = 110
CONTRACT_ID = "inferswarm.gemma4-mixture-population-qualification/1"

# Frozen subject (identical to v4; issue #109 preserves it unchanged).
EXPECTED_CHECKPOINT_SHA256 = (
    "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d"
)
MODEL_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"

# Semantic machinery is imported byte-identical from the accepted #97
# producer (zero new execution math); only the identity twins below are new.
from benchmarks.inferswarm_97 import (  # noqa: E402,F401 (re-exported identity)
    ARGMAX_TIE_BREAK_IDENTITY,
    DECISION_DOMAIN_CONSTRUCTION,
    DECISION_DOMAIN_K,
    assert_teacher_forcing,
    canonical_json_bytes,
    canonical_prefix,
    decision_domain_row,
    decision_row_evidence,
    executor_rule_proof,
    frozen_argmax_row,
    prefix_identity_proof,
    prefix_sha256,
    sha256_bytes,
    write_json_with_sha,
)

GENERATED_TOKENS = 8
CAPTURE_POSITIONS = (0, 1, 3, 7)
RUNTIME_CAPACITY_TOKENS = 64  # frozen single replay chunk bound

# v5 evidence schema identities.
REFERENCE_CASE_SCHEMA = "inferswarm.issue110.v5-reference-case/1"
REFERENCE_RUN_INDEX_SCHEMA = "inferswarm.issue110.v5-reference-run-index/1"
CHAIN_CASE_SCHEMA = "inferswarm.issue110.v5-chain-case/1"
CHAIN_RUN_INDEX_SCHEMA = "inferswarm.issue110.v5-chain-run-index/1"


def validate_campaign_case_ids(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reject holdout, historical, and unversioned cases before CUDA work.

    Issue #110 permits only the public c109 statistical corpus and p109
    stress pool through the physical runners.  The h109 holdout requires a
    separate, explicitly authorized path (which does not exist in this
    producer) and is never accepted here.
    """
    checked = list(cases)
    for case in checked:
        case_id = case.get("case_id") if isinstance(case, dict) else None
        if not isinstance(case_id, str) or not case_id.startswith(("c109-", "p109-")):
            raise ValueError(
                f"issue #110 calibration accepts only c109-/p109- case IDs, got {case_id!r}"
            )
    return checked


def frozen_subject_record(*, checkpoint_sha256: str, model_revision: str) -> dict[str, str]:
    """Validate and serialize the immutable v5 physical subject identity."""
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("issue #110 checkpoint SHA-256 does not match the frozen subject")
    if model_revision != MODEL_REVISION:
        raise ValueError("issue #110 model revision does not match the frozen subject")
    return {
        "contract_id": CONTRACT_ID,
        "methodology_commit": METHODOLOGY_COMMIT,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "model_revision": MODEL_REVISION,
    }


def require_producer_identity(repo: Path, expected_sha: str | None) -> dict[str, Any]:
    """Fail closed unless this clean tree is the externally frozen producer.

    ``expected_sha`` belongs in InferSwarm's retained execution authority,
    not in the producer source.  This check intentionally runs before
    callers import torch or construct a runtime.
    """
    if not isinstance(expected_sha, str) or len(expected_sha) != 40 or any(
        char not in "0123456789abcdef" for char in expected_sha
    ):
        raise ValueError("issue #110 requires a 40-character lowercase expected producer SHA")
    sha = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo),
         "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo),
         "status", "--porcelain"], text=True)
    if status:
        raise ValueError("issue #110 producer source is dirty")
    if sha != expected_sha:
        raise ValueError(
            f"issue #110 producer HEAD {sha!r} does not equal expected SHA {expected_sha!r}"
        )
    return {"commit": sha, "dirty": False, "expected_commit": expected_sha}


def build_chain_case_summary(
    *,
    case: dict[str, Any],
    reference_case: dict[str, Any],
    decision_rows: Sequence[dict[str, Any]],
    margins: Sequence[dict[str, Any]],
    nan_inf_total: int,
    capture_manifests: dict[str, Any],
    producer: dict[str, Any],
    tag: str,
    attempt_id: str,
    wall_seconds: float,
    capture_positions: Sequence[int] = CAPTURE_POSITIONS,
) -> dict[str, Any]:
    """Assemble one v5 candidate (chain) case summary binding exact identity.

    Identical structure to the accepted #97 builder; only the schema
    identity is v5.  ``decision_rows`` carry the ACTUAL candidate
    full-vocabulary winner per canonical-prefix decision under the frozen
    rule (with rule proofs), plus the reference prefix binding.
    """
    if reference_case["case_id"] != case["case_id"]:
        raise ValueError("reference/candidate case identity mismatch")
    if reference_case["case_sha256"] != case["case_sha256"]:
        raise ValueError("reference/candidate case hash mismatch")
    if len(decision_rows) != GENERATED_TOKENS:
        raise ValueError("candidate case must retain exactly 8 decision rows")
    indices = sorted(d["decision_index"] for d in decision_rows)
    if indices != list(range(GENERATED_TOKENS)):
        raise ValueError("decision rows must be emitted exactly once per index")
    for row, ref_row in zip(
        decision_rows,
        sorted(reference_case["decisions"], key=lambda d: d["decision_index"]),
    ):
        if row["prefix_sha256"] != ref_row["prefix_sha256"]:
            raise ValueError(
                f"decision {row['decision_index']}: candidate prefix does "
                "not match the reference canonical prefix"
            )
    return {
        "schema": CHAIN_CASE_SCHEMA,
        "contract_id": CONTRACT_ID,
        "attempt_id": attempt_id,
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "prompt_sha256": case["prompt_sha256"],
        "token_ids_sha256": case["token_ids_sha256"],
        "reference_forced_trajectory": list(
            reference_case["generated_token_ids"]),
        "step_margins": list(margins),
        "nan_inf_count": int(nan_inf_total),
        "argmax_tie_break": ARGMAX_TIE_BREAK_IDENTITY,
        "decisions": [dict(d) for d in decision_rows],
        "producer": producer,
        "role": "candidate-chain",
        "capture_positions": list(capture_positions),
        "capture_manifests": capture_manifests,
        "wall_seconds": wall_seconds,
    }
