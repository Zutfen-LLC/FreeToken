"""R6 final-row-only FP32 logits promotion: bit-exactness and structure.

Covers the SINGLE_GPU_CONTROL_AMENDMENT-002 optimization: the execution
paths (prefill/decode for roles single/last) must promote ONLY the final
logits row to FP32 instead of the whole [sequence, vocab] tensor, while
the full BF16 GEMM + softcap semantics stay byte-for-byte identical.

Numerical invariant (BF16->FP32 is elementwise and exact):

    legacy    = full_bf16_logits(final).float()[-1]
    optimized = full_bf16_logits(final)[-1].float()
    torch.equal(legacy, optimized) is True  (bit-identical, no tolerance)

CPU/torch-only: these tests exercise the helper methods on a stub stage
(the same object.__new__ pattern as the existing tied-embedding contract
test).  The REAL Gemma CUDA equivalence proof runs on inferswarm03 and is
retained as physical evidence, not as a unit test.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO = Path(__file__).resolve().parents[2]
_STAGE_PATH = _REPO / "benchmarks" / "inferswarm_r6" / "stage_runtime.py"
_SPEC = importlib.util.spec_from_file_location("r6_stage_runtime_finalrow", _STAGE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_STAGE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_STAGE)
GemmaDenseStage = _STAGE.GemmaDenseStage


def _stub_stage(
    vocab: int = 50,
    hidden: int = 8,
    cap: float | None = 30.0,
    dtype=torch.bfloat16,
    seed: int = 20260903,
):
    """Minimal GemmaDenseStage carrying only the lm-head helpers' state."""
    gen = torch.Generator().manual_seed(seed)
    stage = object.__new__(GemmaDenseStage)
    stage.block = SimpleNamespace(
        embed_tokens=SimpleNamespace(
            weight=torch.randn(vocab, hidden, dtype=dtype, generator=gen)
        )
    )
    stage.full_config = SimpleNamespace(final_logit_softcapping=cap)
    return stage


def _stub_final(rows: int, hidden: int = 8, seed: int = 7, dtype=torch.bfloat16):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, hidden, dtype=dtype, generator=gen)


def _sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


# --------------------------------------------------------------------------
# 1 + 2 + 4: bit-identity of the final row, softcap before selection, and
#            NaN/Inf observational identity (legacy vs optimized).
# --------------------------------------------------------------------------


def test_final_row_optimized_promotion_bit_identical_to_legacy():
    for rows, cap in [(1, None), (1, 30.0), (7, None), (7, 30.0), (32, 30.0)]:
        stage = _stub_stage(cap=cap)
        final = _stub_final(rows)
        legacy = stage.lm_head_logits(final)[-1].contiguous()
        optimized = stage.final_row_logits(final).contiguous()
        assert legacy.dtype == torch.float32 == optimized.dtype
        assert legacy.shape == optimized.shape == (stage.block.embed_tokens.weight.shape[0],)
        assert torch.equal(legacy, optimized), f"rows={rows} cap={cap}"
        assert _sha256(legacy) == _sha256(optimized)


def test_bit_identity_holds_for_nan_and_inf_inputs():
    # Softcap tanh saturates finite values; push actual NaN/Inf through the
    # GEMM to prove the observation domain (counts) is identical too.
    stage = _stub_stage(cap=30.0)
    final = _stub_final(4)
    final[1, :] = float("nan")
    final[2, 0] = float("inf")
    final[3, 0] = float("-inf")
    legacy = stage.lm_head_logits(final)
    full = stage.full_bf16_logits(final)
    optimized = stage.final_row_logits(final)
    assert torch.isnan(legacy).sum().item() == torch.isnan(optimized).sum().item()
    assert torch.isinf(legacy).sum().item() == torch.isinf(optimized).sum().item()
    assert torch.equal(legacy[-1].contiguous(), optimized.contiguous())
    # full-bf16 observation domain must also match the legacy promotion
    assert torch.isnan(full).sum().item() == torch.isnan(legacy).sum().item()


def test_softcap_applies_before_row_selection_in_both_paths():
    # If softcap were (wrongly) applied after row selection/dtype change,
    # cap-saturated rows would differ; identical outputs prove placement
    # is unchanged (tanh in BF16 before selection, exactly as legacy).
    stage = _stub_stage(cap=0.5)
    final = _stub_final(5) * 100.0  # force deep tanh saturation
    legacy = stage.lm_head_logits(final)
    optimized = stage.final_row_logits(final)
    assert torch.equal(legacy[-1].contiguous(), optimized.contiguous())
    # sanity: saturation actually engaged (|logits| bounded by cap)
    assert optimized.abs().max().item() <= 0.5


# --------------------------------------------------------------------------
# 3: argmax identity.
# --------------------------------------------------------------------------


def test_argmax_identical_legacy_vs_optimized():
    for seed in range(6):
        stage = _stub_stage(cap=30.0, seed=seed)
        final = _stub_final(9, seed=seed + 100)
        legacy = stage.lm_head_logits(final)
        optimized = stage.final_row_logits(final)
        assert int(legacy[-1].argmax().item()) == int(optimized.argmax().item())


# --------------------------------------------------------------------------
# 5 + 6: allocation shape — no [sequence, vocab] FP32 tensor, single-row
#        FP32 size is exactly vocab * 4.
# --------------------------------------------------------------------------


def test_optimized_path_never_promotes_full_2d_tensor_to_fp32():
    # Structural: the execution paths call final_row_logits, which selects
    # [-1] BEFORE .float(); lm_head_logits is the only full promotion and
    # no prefill/decode path routes through it anymore.
    source = _STAGE_PATH.read_text()
    # final_row_logits selects the row first, then converts
    m = re.search(
        r"def final_row_logits.*?return ([^\n]+)", source, re.DOTALL
    )
    assert m is not None
    assert "[-1].float()" in m.group(1).replace(" ", "")
    # no execution path calls the legacy whole-tensor promotion
    body = source.split("def lm_head_logits", 1)[1].split("def reset_session_state", 1)[0]
    exec_body = source.split("def prefill", 1)[1]
    assert "lm_head_logits" not in exec_body
    assert "lm_head_logits" not in source.split("def decode", 1)[1].split("def logical_state_records", 1)[0]


def test_single_row_fp32_size_is_exactly_vocab_times_four():
    stage = _stub_stage(vocab=262144, hidden=8)
    final = _stub_final(32, hidden=8)
    row = stage.final_row_logits(final)
    assert row.dim() == 1
    assert row.dtype == torch.float32
    assert row.numel() * row.element_size() == 262144 * 4 == 1_048_576
    # legacy whole-tensor promotion at the same shape is 32x larger
    legacy = stage.lm_head_logits(final)
    assert legacy.numel() * legacy.element_size() == 32 * 262144 * 4


def test_optimized_path_full_bf16_gemm_shape_unchanged():
    # The [sequence, vocab] BF16 GEMM is still executed in full: the full
    # tensor exists (BF16) and its final row equals the optimized result.
    stage = _stub_stage(vocab=64, hidden=8, cap=30.0)
    final = _stub_final(11)
    full = stage.full_bf16_logits(final)
    assert full.shape == (11, 64)
    assert full.dtype == torch.bfloat16
    weight = stage.block.embed_tokens.weight
    expected = final @ weight.t()
    cap = stage.full_config.final_logit_softcapping
    if cap is not None:
        expected = torch.tanh(expected / cap) * cap
    assert torch.equal(full, expected)
    assert torch.equal(full[-1].float(), stage.final_row_logits(final))


# --------------------------------------------------------------------------
# 8: decode one-row case unchanged.
# --------------------------------------------------------------------------


def test_decode_single_row_case_bit_identical():
    # rows=1: whole-tensor promotion and final-row promotion coincide in
    # ALLOCATION as well as value; both paths must still agree exactly.
    stage = _stub_stage(cap=30.0)
    final = _stub_final(1)
    legacy = stage.lm_head_logits(final)
    optimized = stage.final_row_logits(final)
    assert legacy.shape == optimized.shape == (legacy.shape[-1],)
    assert torch.equal(legacy.contiguous(), optimized.contiguous())


def test_decode_single_row_case_nan_inf_identical():
    stage = _stub_stage(cap=30.0)
    final = _stub_final(1)
    final[0, 2] = float("nan")
    legacy = stage.lm_head_logits(final)
    optimized = stage.final_row_logits(final)
    assert torch.isnan(legacy).sum().item() == torch.isnan(optimized).sum().item() == 1
    assert torch.equal(
        torch.nan_to_num(legacy, nan=0.0),
        torch.nan_to_num(optimized, nan=0.0),
    )


# --------------------------------------------------------------------------
# 9 + 10: frozen comparator/methodology constants and historical result
#         remain untouched by this change.
# --------------------------------------------------------------------------


def test_frozen_comparator_constants_unchanged():
    import json

    runner_src = (
        _REPO / "benchmarks" / "inferswarm_r6" / "single_gpu_control.py"
    ).read_text()
    assert "RUNTIME_CAPACITY_TOKENS = 64" in runner_src
    assert "FROZEN_THRESHOLD = 0.25" in runner_src
    assert "CAPTURE_STEPS = (0, 1, 7)" in runner_src
    result = json.loads(
        (_REPO / "docs" / "inferswarm_r6" / "result.json").read_text()
    )
    assert result["verdict"] == "R6_DENSE_ARCHITECTURE_FALSIFICATION_FAIL"


def test_historical_result_file_byte_unchanged_reference():
    # The historical result.json is pinned by content hash in the gate
    # contract tests; restate the pinned bytes hash so any accidental
    # modification of the historical verdict artifact fails here too.
    digest = hashlib.sha256(
        (_REPO / "docs" / "inferswarm_r6" / "result.json").read_bytes()
    ).hexdigest()
    assert digest == "b19e746c831484cb7ba07088d8959297e1d8641125215d6fe63f27c7255817e3"
