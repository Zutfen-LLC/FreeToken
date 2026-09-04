"""Issue #88 (InferSwarm): Gemma v3 decision-stability physical producer.

Executes the frozen issue #86 v3 methodology
(inferswarm @ a8ec98a9fb9b673c93de5100d784ea772395efdb,
docs/qualification/gemma4-12b-it-v3/) on the frozen physical topology:

- reference: inferswarm04 RTX 3090 24 GiB, matched single-GPU FreeToken
  runtime (accepted R6 GemmaDenseStage, replay-prefill greedy);
- candidate: accepted three-stage RTX 3060 chain — inferswarm01 GPU-0
  (stage 1, layers [0,16)), inferswarm01 GPU-1 (stage 2, layers [16,32)),
  inferswarm03 (stage 3/last, layers [32,48) via the #76 R4 wire service).

The execution harness is the #76/#81 harness cherry-picked verbatim from
PR #29 (b1389e3^..9f06d81); this package adds ONLY the v3 semantic layer
required by issue #88 and ZERO new model/execution math:

1. the canonical frozen argmax/tie-break rule (ARGMAX_FIRST_MAX /
   lowest-token-id-among-exactly-equal-fp32-maxima) applied identically
   on both arms, with an executor proof-of-rule recorded per emitted
   token (lowest-index-among-equal-maxima check computed on-device
   against the FP32 row);
2. the frozen decision domain D(r) construction
   (reference-top-1024-with-cutoff-ties/1) computed from reference rows
   only, with canonical membership hashes;
3. candidate teacher-forcing against the exact canonical reference
   prefix at each of all 8 decisions, with mechanical prefix-identity
   proof before each execution;
4. retention of the candidate actual full-vocabulary FP32 winner per
   canonical-prefix decision row (diagnostic row hashes, never free-run);
5. evidence-sufficient rows for E_full (full 15-envelope capture set is
   unchanged from the #76 harness) and for decision_local_error over the
   frozen D(r).

Free-running post-branch tensors are diagnostic only (never calibration
or holdout evidence). The semantic adjudication itself (evaluate_decision,
threshold derivation, unseal preflight) lives in the accepted InferSwarm
CPU tooling and is NOT reimplemented here.

Execution-branch discipline (issue #88 Phase P): this package and its
tests freeze as the physical implementation producer BEFORE the first
model execution; after that freeze no execution or model math may change
during the campaign.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Sequence

# Frozen v3 methodology identity (inferswarm @ a8ec98a, PR #87).
V3_METHODOLOGY_COMMIT = "a8ec98a9fb9b673c93de5100d784ea772395efdb"
V3_ISSUE = 88
CONTRACT_ID = "inferswarm.gemma4-heterogeneous-numerical-equivalence/1"
V3_CONTRACT_ID = "inferswarm.issue88.v3-decision-stability/1"

# Frozen subject (issue #86 §"Qualification subject" == issue #88 §"Qualification subject").
EXPECTED_CHECKPOINT_SHA256 = (
    "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d"
)
MODEL_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"

GENERATED_TOKENS = 8
CAPTURE_POSITIONS = (0, 1, 3, 7)
RUNTIME_CAPACITY_TOKENS = 64  # frozen single replay chunk bound

# Frozen v3 semantic identities (byte-identical twins of
# scripts/issue86_v3_methodology.py constants; asserted by unit test
# against the committed inferswarm checkout when available).
ARGMAX_TIE_BREAK_IDENTITY = (
    "ARGMAX_FIRST_MAX/lowest-token-id-among-exactly-equal-fp32-maxima"
)
DECISION_DOMAIN_CONSTRUCTION = "reference-top-1024-with-cutoff-ties/1"
DECISION_DOMAIN_K = 1024


def producer_identity(repo: Path) -> dict[str, Any]:
    """Exact producer/device identity for applicability records."""
    sha = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo),
         "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo),
         "status", "--porcelain"], text=True)
    return {"commit": sha, "dirty": bool(status)}


def frozen_argmax_row(row: Sequence[float]) -> tuple[int, float]:
    """Frozen rule twin: lowest token id among exactly equal FP32 maxima.

    Returns (winner_index, best_value). Pure host-side reference
    implementation used to PROVE the executor's emitted token follows
    the frozen rule; the proof compares the executor token against this
    twin computed over the exact same FP32 row bytes.
    """
    if not row:
        raise ValueError("argmax requires a nonempty row")
    best_index = 0
    best_value = float(row[0])
    if not math.isfinite(best_value):
        raise ValueError("argmax requires finite logits")
    for index in range(1, len(row)):
        value = float(row[index])
        if not math.isfinite(value):
            raise ValueError("argmax requires finite logits")
        if value > best_value:
            best_value = value
            best_index = index
    return best_index, best_value


def executor_rule_proof(row: Sequence[float], emitted_token: int) -> dict[str, Any]:
    """Proof-of-rule for ONE emitted decision token.

    Proves the executor's emitted token equals the frozen-rule winner of
    the exact FP32 row, and that the winner is the LOWEST index among
    all exactly-equal FP32 maxima. Any violation is a rule failure, not
    a tolerance.
    """
    winner, best_value = frozen_argmax_row(row)
    equal_max_indices = [
        i for i, v in enumerate(row)
        if float(v) == best_value and math.isfinite(float(v))
    ]
    ok = int(emitted_token) == winner
    return {
        "emitted_token": int(emitted_token),
        "rule_winner_token": winner,
        "rule_ok": ok,
        "tie_count": len(equal_max_indices),
        "lowest_index_among_equal_maxima": (
            equal_max_indices[0] if equal_max_indices else winner
        ),
        "tie_break_identity": ARGMAX_TIE_BREAK_IDENTITY,
        "rule": "emitted==ARGMAX_FIRST_MAX(winner); winner is the lowest "
                "index among exactly-equal FP32 maxima",
    }


def decision_domain_row(row: Sequence[float], k: int = DECISION_DOMAIN_K) -> dict[str, Any]:
    """D(r) twin: every token with reference logit >= k-th-highest cutoff.

    Returns membership (ascending token ids), size, cutoff value, and
    the canonical membership sha256 (sorted-id canonical JSON) exactly
    as the accepted InferSwarm tooling hashes it. Reference-derived
    only; no candidate input.
    """
    import hashlib

    if k <= 0:
        raise ValueError("decision-domain K must be positive")
    values = [float(v) for v in row]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("reference logits must be nonempty and finite")
    cutoff = sorted(values, reverse=True)[min(k, len(values)) - 1]
    domain = tuple(i for i, v in enumerate(values) if v >= cutoff)
    if not domain:
        raise ValueError("decision domain construction produced an empty set")
    membership_bytes = (
        json.dumps(list(domain), ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")) + "\n"
    ).encode()
    return {
        "membership": list(domain),
        "domain_size": len(domain),
        "cutoff_hex": cutoff.hex(),
        "cutoff_rank": min(k, len(values)),
        "domain_membership_sha256": hashlib.sha256(membership_bytes).hexdigest(),
        "construction": DECISION_DOMAIN_CONSTRUCTION,
        "k": k,
    }


def prefix_sha256(prefix: Sequence[int]) -> str:
    """Canonical prefix hash used by EVERY arm and the assembler.

    sha256 over compact canonical JSON of the exact token-id list. The
    reference runner, the candidate runner, and the CPU assembler all
    call this ONE function, so prefix identity is byte-comparable
    across arms.
    """
    import hashlib

    payload = (
        json.dumps([int(t) for t in prefix], ensure_ascii=False,
                   separators=(",", ":")) + "\n"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def canonical_prefix(token_ids: Sequence[int], reference_generated: Sequence[int],
                     decision_index: int) -> list[int]:
    """The exact canonical prefix for decision ``decision_index`` (0-based).

    prompt tokens + reference-generated tokens [0, decision_index). The
    candidate must consume EXACTLY this prefix (teacher forcing); the
    identity proof hashes the prefix bytes.
    """
    import hashlib

    if not 0 <= decision_index < GENERATED_TOKENS:
        raise ValueError(f"decision index out of range: {decision_index}")
    if len(reference_generated) != GENERATED_TOKENS:
        raise ValueError("canonical prefix requires the complete 8-decision "
                         "reference trajectory")
    prefix = list(token_ids) + [
        int(t) for t in reference_generated[:decision_index]
    ]
    return prefix


def prefix_identity_proof(prefix: Sequence[int]) -> dict[str, Any]:
    """Mechanical prefix-identity proof (sha256 over canonical token JSON)."""
    return {
        "prefix_len": len(prefix),
        "prefix_sha256": prefix_sha256(prefix),
    }


def decision_row_evidence(
    *,
    decision_index: int,
    prefix: Sequence[int],
    domain_info: dict[str, Any],
    emitted_token: int,
    rule_proof: dict[str, Any],
) -> dict[str, Any]:
    """One canonical-prefix decision evidence row (harness-side).

    Binds prefix identity, domain membership, the emitted winner and its
    rule proof. The FP32 candidate row itself stays in the retained
    capture bundle; the row records its sha256 for later
    decision_local_error derivation on the CPU assembler.
    """
    if not rule_proof["rule_ok"]:
        raise ValueError(
            f"decision {decision_index}: emitted token violates the frozen "
            "argmax/tie-break rule"
        )
    proof = prefix_identity_proof(prefix)
    return {
        "decision_index": decision_index,
        "prefix_len": proof["prefix_len"],
        "prefix_sha256": proof["prefix_sha256"],
        "domain_membership_sha256": domain_info["domain_membership_sha256"],
        "domain_size": domain_info["domain_size"],
        "domain_cutoff_hex": domain_info["cutoff_hex"],
        "emitted_token": int(emitted_token),
        "emitted_rule": ARGMAX_TIE_BREAK_IDENTITY,
        "rule_proof": rule_proof,
    }


def assert_teacher_forcing(
    *, prefix: Sequence[int], reference_decision: dict[str, Any]
) -> None:
    """Prove prefix identity BEFORE candidate execution (fail closed).

    The candidate replay prefix for decision i must be byte-identical
    (same canonical sha256 and length) to the reference runner's recorded
    prefix for that decision.
    """
    expected_len = int(reference_decision["prefix_len"])
    expected_sha = reference_decision["prefix_sha256"]
    observed_len = len(prefix)
    observed_sha = prefix_sha256(prefix)
    if observed_len != expected_len or observed_sha != expected_sha:
        raise ValueError(
            f"teacher-forcing prefix identity failure: decision "
            f"{reference_decision.get('decision_index')}: len "
            f"{observed_len}!={expected_len} or sha "
            f"{observed_sha}!={expected_sha}"
        )


def build_reference_case_summary(
    *,
    case: dict[str, Any],
    generated: Sequence[int],
    margins: Sequence[dict[str, Any]],
    decision_rows: Sequence[dict[str, Any]],
    nan_inf_total: int,
    capture_manifest: dict[str, Any],
    producer: dict[str, Any],
    gpu_uuid: str,
    tag: str,
    attempt_id: str,
    wall_seconds: float,
    capture_positions: Sequence[int] = CAPTURE_POSITIONS,
) -> dict[str, Any]:
    """Assemble one v3 reference case summary binding exact identity.

    Pure (no torch): unit-tested for identity/provenance binding.
    """
    if len(generated) != GENERATED_TOKENS:
        raise ValueError("reference case must emit exactly 8 decisions")
    if len(decision_rows) != GENERATED_TOKENS:
        raise ValueError("reference case must retain exactly 8 decision rows")
    indices = sorted(d["decision_index"] for d in decision_rows)
    if indices != list(range(GENERATED_TOKENS)):
        raise ValueError("decision rows must be emitted exactly once per index")
    return {
        "schema": "inferswarm.issue88.v3-reference-case/1",
        "contract_id": V3_CONTRACT_ID,
        "attempt_id": attempt_id,
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "prompt_sha256": case["prompt_sha256"],
        "token_ids_sha256": case["token_ids_sha256"],
        "generated_token_ids": [int(t) for t in generated],
        "step_margins": list(margins),
        "min_top1_margin_hex": min(
            float.fromhex(m["margin_hex"]) for m in margins).hex(),
        "nan_inf_count": int(nan_inf_total),
        "decision_domain_construction": DECISION_DOMAIN_CONSTRUCTION,
        "argmax_tie_break": ARGMAX_TIE_BREAK_IDENTITY,
        "decisions": [dict(d) for d in decision_rows],
        "producer": producer,
        "gpu_uuid": gpu_uuid,
        "role": "reference-single",
        "capture_positions": list(capture_positions),
        "capture_manifest": capture_manifest,
        "wall_seconds": wall_seconds,
    }


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
    """Assemble one v3 candidate (chain) case summary.

    ``decision_rows`` carry the ACTUAL candidate full-vocabulary winner
    per canonical-prefix decision under the frozen rule (with rule
    proofs), plus the reference prefix binding. The candidate generated
    trajectory is NOT free-run: this summary records the reference
    trajectory it was forced against for audit.
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
        "schema": "inferswarm.issue88.v3-chain-case/1",
        "contract_id": V3_CONTRACT_ID,
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


def canonical_json_bytes(value: Any) -> bytes:
    """Byte-identical twin of the frozen inferswarm canonical JSON."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")) + "\n"
    ).encode()


def sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def write_json_with_sha(path: Path, value: Any) -> dict[str, Any]:
    """Write canonical JSON + record its sha256 (torch-free sidecar pair)."""
    payload = canonical_json_bytes(value)
    out = Path(path)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.write_bytes(payload)
    return {"path": str(out), "sha256": sha256_bytes(payload), "bytes": len(payload)}
