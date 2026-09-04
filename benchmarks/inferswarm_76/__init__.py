"""Issue #76 (InferSwarm) execution harness: shared case/capture utilities.

This package executes the frozen issue #74 numerical-equivalence methodology
(inferswarm @ f394dc9, docs/qualification/gemma4-12b-it-v1/) on the frozen
physical topology:

- matched single-GPU FreeToken reference: inferswarm04 RTX 3090 (24 GiB);
- frozen three-stage RTX 3060 distributed candidate: node inferswarm01
  (2x RTX 3060, stages 1-2) + inferswarm03 (RTX 3060, last stage via the
  accepted R4 wire service).

It adds NO new execution math. All model execution flows through the accepted
R6 runtime (benchmarks.inferswarm_r6.stage_runtime.GemmaDenseStage) with the
accepted replay-prefill greedy semantics. The only additions are:

1. case-driven corpora loading (exact frozen token IDs, never retokenized);
2. capture of the full 15-envelope checkpoint set at positions 0/1/3/7 by
   ARMING the existing #71 capture sink seams plus thin out-of-tree wrappers
   for the three checkpoints the R6/#71 seams do not emit
   (layer-0 o_proj input/output, global-layer-15 attention o_proj output,
   full final-row BF16 logits);
3. reference top-1 margin diagnostics per frozen selection input
   ``matched-reference-top1-margin``;
4. host-float64 reduction exactly per frozen REDUCER.md.

Execution-branch discipline (issue #76): this file and everything under
benchmarks/inferswarm_76/ plus its tests freeze as the physical implementation
producer BEFORE the first model execution. After that freeze no execution or
model math may change during the campaign.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

# Frozen contract identity (mirrors inferswarm@f394dc9 methodology.json).
CONTRACT_ID = "inferswarm.gemma4-heterogeneous-numerical-equivalence/1"
METHODOLOGY_COMMIT = "f394dc9fbf9979574324f2d037580659f1d63b39"
FREETOKEN_BASE = "d4d16089165917704a87f4e2f0c4a09969646f95"

EXPECTED_CHECKPOINT_SHA256 = (
    "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d"
)
MODEL_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"

GENERATED_TOKENS = 8
EXACT_TOKEN_POSITIONS = tuple(range(8))
CAPTURE_POSITIONS = (0, 1, 3, 7)
RUNTIME_CAPACITY_TOKENS = 64  # frozen single replay chunk bound

FAMILIES = (
    "local-bf16-backend-operation-output",
    "hidden-residual-stream",
    "final-normalized-hidden-state",
    "bf16-logits",
    "fp32-consumer-logits",
)
METRICS = (
    "max-absolute-difference",
    "rms-difference",
    "p99-absolute-error",
)
ENVELOPES = tuple(f"{family}:{metric}" for family in FAMILIES for metric in METRICS)

# checkpoint_id -> (family, semantic_dtype) exactly per frozen
# manifests/checkpoint-family-map.json at the methodology commit.
CHECKPOINT_FAMILY_MAP = {
    "embedding-output": ("local-bf16-backend-operation-output", "bfloat16"),
    "layer-0-o-proj-input": ("local-bf16-backend-operation-output", "bfloat16"),
    "layer-0-o-proj-output": ("local-bf16-backend-operation-output", "bfloat16"),
    "global-layer-15-attention-o-proj-output": (
        "local-bf16-backend-operation-output",
        "bfloat16",
    ),
    "post-global-layer-15-residual": ("hidden-residual-stream", "bfloat16"),
    "post-global-layer-31-residual": ("hidden-residual-stream", "bfloat16"),
    "post-global-layer-47-residual": ("hidden-residual-stream", "bfloat16"),
    "final-normalized-hidden-state": ("final-normalized-hidden-state", "bfloat16"),
    "full-final-row-bf16-logits": ("bf16-logits", "bfloat16"),
    "full-final-row-fp32-consumer-logits": ("fp32-consumer-logits", "float32"),
}

# Capture names emitted by the R6/#71 seams vs. the envelope checkpoint IDs.
# The #71 seams emit per stage; the harness maps stage-local records to the
# frozen global checkpoint IDs (single arm: all checkpoints on one device).
SEAM_TO_CHECKPOINT = {
    "single": {
        "embedding_output": "embedding-output",
        "layer0_o_proj_input": "layer-0-o-proj-input",
        "layer0_o_proj_output": "layer-0-o-proj-output",
        "layer15_attn_o_proj_output": "global-layer-15-attention-o-proj-output",
        "after_layer_15": "post-global-layer-15-residual",
        "after_layer_31": "post-global-layer-31-residual",
        "after_layer_47": "post-global-layer-47-residual",
        "final_norm": "final-normalized-hidden-state",
        "full_final_row_bf16_logits": "full-final-row-bf16-logits",
        "final_row_fp32": "full-final-row-fp32-consumer-logits",
    },
}
# Chain arm: which stage-role emits which frozen checkpoints.
CHAIN_STAGE_CHECKPOINTS = {
    "first": {
        "embedding_output": "embedding-output",
        "layer0_o_proj_input": "layer-0-o-proj-input",
        "layer0_o_proj_output": "layer-0-o-proj-output",
        "layer15_attn_o_proj_output": "global-layer-15-attention-o-proj-output",
        "after_layer_15": "post-global-layer-15-residual",
        "boundary_send_hidden": "post-global-layer-15-residual-boundary",
    },
    "middle": {
        "boundary_recv_hidden": "boundary1-recv",
        "after_layer_31": "post-global-layer-31-residual",
        "boundary_send_hidden": "post-global-layer-31-residual-boundary",
    },
    "last": {
        "boundary_recv_hidden": "boundary2-recv",
        "after_layer_47": "post-global-layer-47-residual",
        "final_norm": "final-normalized-hidden-state",
        "full_final_row_bf16_logits": "full-final-row-bf16-logits",
        "final_row_fp32": "full-final-row-fp32-consumer-logits",
    },
}
# Only checkpoints that feed envelopes (boundary probes are exact-layer only).
ENVELOPE_CHECKPOINT_IDS = frozenset(CHECKPOINT_FAMILY_MAP)


def canonical_json_bytes(value: Any) -> bytes:
    """Byte-identical twin of the frozen inferswarm canonical JSON."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_case_identity(case: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless a corpus row still hashes to its frozen identity."""
    identity = {
        "content_class": case["content_class"],
        "length_regime": list(case["length_regime"]),
        "prompt_text": case["prompt_text"],
        "token_ids": list(case["token_ids"]),
    }
    prompt_sha = sha256_bytes(case["prompt_text"].encode("utf-8"))
    ids_sha = sha256_bytes(canonical_json_bytes(list(case["token_ids"])))
    case_sha = sha256_bytes(canonical_json_bytes(identity))
    if prompt_sha != case["prompt_sha256"]:
        raise ValueError(f"{case['case_id']}: prompt hash mismatch")
    if ids_sha != case["token_ids_sha256"]:
        raise ValueError(f"{case['case_id']}: token-ids hash mismatch")
    if case_sha != case["case_sha256"]:
        raise ValueError(f"{case['case_id']}: case hash mismatch")
    if len(case["token_ids"]) != case["token_count"]:
        raise ValueError(f"{case['case_id']}: token count mismatch")
    return dict(case)


def load_corpus(path: Path, *, expected_count: int | None = None) -> list[dict]:
    corpus = json.loads(Path(path).read_text())
    cases = [verify_case_identity(row) for row in corpus["cases"]]
    if expected_count is not None and len(cases) != expected_count:
        raise ValueError(
            f"corpus {path}: expected {expected_count} cases, found {len(cases)}"
        )
    return cases


def nearest_rank_higher(values: Sequence[float], percentile: float = 0.99) -> float:
    """Frozen reducer tail rule: ascending sort, one-based ceil(0.99*N)."""
    if not values:
        raise ValueError("percentile domain must not be empty")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile out of range")
    finite = [float(v) for v in values]
    if not all(math.isfinite(v) and v >= 0.0 for v in finite):
        raise ValueError("absolute-error inputs must be finite and nonnegative")
    ordered = sorted(finite)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def tensor_metrics(
    reference: Sequence[float], candidate: Sequence[float]
) -> dict[str, float]:
    """Frozen host-float64 tensor metrics over the complete domain."""
    if len(reference) != len(candidate):
        raise ValueError("reference/candidate domain size mismatch")
    if not reference:
        raise ValueError("empty comparison domain")
    errors = []
    for r, c in zip(reference, candidate):
        e = abs(float(r) - float(c))
        if not math.isfinite(e):
            raise ValueError("nonfinite absolute error in comparison domain")
        errors.append(e)
    square_sum = math.fsum(e * e for e in errors)
    return {
        "max-absolute-difference": max(errors),
        "rms-difference": math.sqrt(square_sum / len(errors)),
        "p99-absolute-error": nearest_rank_higher(errors, 0.99),
    }


def conservative_case_family(
    checkpoint_metrics: Iterable[dict[str, float]],
) -> dict[str, float]:
    """Per-metric maximum across all declared checkpoints and positions."""
    result: dict[str, float] = {}
    for row in checkpoint_metrics:
        for metric in METRICS:
            value = float(row[metric])
            if metric not in result or value > result[metric]:
                result[metric] = value
    missing = [m for m in METRICS if m not in result]
    if missing:
        raise ValueError(f"case family reduction missing metrics: {missing}")
    return result


def envelopes_from_case_metrics(
    per_checkpoint: dict[str, dict[str, float]],
) -> dict[str, str]:
    """15 frozen envelope strings (exact hex binary64) for one case.

    Keys may be bare checkpoint ids or "<checkpoint_id>@<position>".
    """
    by_family: dict[str, list[dict[str, float]]] = {f: [] for f in FAMILIES}
    for key, metrics in per_checkpoint.items():
        checkpoint_id = key.split("@", 1)[0]
        family = CHECKPOINT_FAMILY_MAP[checkpoint_id][0]
        by_family[family].append(metrics)
    envelopes: dict[str, str] = {}
    for family, rows in by_family.items():
        if not rows:
            raise ValueError(f"no checkpoint evidence for family {family}")
        family_max = conservative_case_family(rows)
        for metric in METRICS:
            envelopes[f"{family}:{metric}"] = family_max[metric].hex()
    if len(envelopes) != 15:
        raise ValueError(f"expected 15 envelopes, built {len(envelopes)}")
    return envelopes


def hex_to_float(value: str) -> float:
    out = float.fromhex(value)
    if not math.isfinite(out):
        raise ValueError(f"nonfinite hex float {value!r}")
    return out
