"""R6 in-place final-logit softcap: bit-exactness, structure, and OOM phase
evidence retention (SINGLE_GPU_CONTROL_AMENDMENT-003).

Covers the candidate in-place softcap transformation::

    logits.div_(cap); logits.tanh_(); logits.mul_(cap)

against the frozen legacy out-of-place expression::

    torch.tanh(logits / cap) * cap

Both are applied to the SAME full [sequence, vocab] BF16 logits tensor
produced by ONE full BF16 GEMM (never two separately executed GEMMs), from
byte-identical pre-softcap clones.  The full GEMM, full-matrix BF16
materialization, elementwise operation order, cap value, and final-row FP32
promotion are unchanged; only temporary-allocation lifetime changes.

The REAL Gemma CUDA equivalence proof runs on inferswarm03 and is retained
as physical evidence, not as a unit test.
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
_RUNNER_PATH = _REPO / "benchmarks" / "inferswarm_r6" / "single_gpu_control.py"
_SPEC = importlib.util.spec_from_file_location("r6_stage_runtime_softcap", _STAGE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_STAGE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_STAGE)
GemmaDenseStage = _STAGE.GemmaDenseStage
softcap_legacy = _STAGE.softcap_legacy
softcap_inplace = _STAGE.softcap_inplace
cuda_phase_probe = _STAGE.cuda_phase_probe


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
    stage._softcap_mode = "inplace"
    stage._phase_probe = None
    return stage


def _gemm(stage, rows: int, seed: int = 7) -> torch.Tensor:
    """The full BF16 [rows, vocab] GEMM — computed ONCE per comparison."""
    gen = torch.Generator().manual_seed(seed)
    final = torch.randn(rows, stage.block.embed_tokens.weight.shape[1],
                        dtype=torch.bfloat16, generator=gen)
    return final @ stage.block.embed_tokens.weight.t()


def _sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


# --------------------------------------------------------------------------
# 1 + 2 + 4 + 5: exact equality, value coverage, shape, dtype.
# --------------------------------------------------------------------------


def test_legacy_and_candidate_softcap_exactly_equal_on_bf16_matrices():
    for rows, cap, scale in [
        (1, 30.0, 1.0),
        (4, 30.0, 1.0),
        (26, 30.0, 1.0),
        (32, 30.0, 1.0),
        (33, 30.0, 1.0),
        (7, 0.5, 100.0),   # deep saturation
        (9, 0.5, 1e-4),    # near-zero (linear tanh regime)
    ]:
        stage = _stub_stage(cap=cap)
        pre = _gemm(stage, rows) * scale
        legacy = softcap_legacy(pre.clone(), cap)
        candidate = softcap_inplace(pre.clone(), cap)
        assert legacy.shape == candidate.shape == pre.shape
        assert legacy.dtype == candidate.dtype == torch.bfloat16
        assert torch.equal(legacy, candidate), f"rows={rows} cap={cap} scale={scale}"
        assert _sha256(legacy) == _sha256(candidate)


def test_equality_covers_positive_negative_zero_and_near_cap_values():
    cap = 30.0
    values = torch.tensor(
        [
            0.0, -0.0, 1.0, -1.0, 29.999, -29.999, 30.0, -30.0,
            1e-8, -1e-8, 1e4, -1e4, 123.456, -123.456,
        ],
        dtype=torch.bfloat16,
    ).reshape(1, -1)
    legacy = softcap_legacy(values.clone(), cap)
    candidate = softcap_inplace(values.clone(), cap)
    assert torch.equal(legacy, candidate)
    assert _sha256(legacy) == _sha256(candidate)
    # sanity: coverage actually engaged tanh's regimes
    assert (legacy.abs() <= cap).all()


def test_nan_inf_behavior_identical():
    cap = 30.0
    values = torch.tensor(
        [[0.0, float("nan"), float("inf"), float("-inf"), 1.0, -1.0]],
        dtype=torch.bfloat16,
    )
    legacy = softcap_legacy(values.clone(), cap)
    candidate = softcap_inplace(values.clone(), cap)
    assert torch.isnan(legacy).sum().item() == torch.isnan(candidate).sum().item()
    assert torch.isinf(legacy).sum().item() == torch.isinf(candidate).sum().item()
    # bit-identity under NaN via raw bytes (torch.equal says NaN != NaN)
    assert _sha256(legacy) == _sha256(candidate)


# --------------------------------------------------------------------------
# 6 + 7: final-row FP32 promotion and argmax identity through the stage
#        helpers (softcap mode seam).
# --------------------------------------------------------------------------


def test_final_row_fp32_bit_identical_across_softcap_modes():
    for rows in (1, 4, 33):
        stage = _stub_stage(cap=30.0)
        pre = _gemm(stage, rows)
        stage._softcap_mode = "legacy"
        legacy_row = stage.final_row_logits(_final_for(stage, rows))
        stage._softcap_mode = "inplace"
        inplace_row = stage.final_row_logits(_final_for(stage, rows))
        # same hidden input -> deterministic GEMM -> identical rows
        assert legacy_row.dtype == inplace_row.dtype == torch.float32
        assert _sha256(legacy_row.contiguous()) == _sha256(inplace_row.contiguous())
        assert int(legacy_row.argmax().item()) == int(inplace_row.argmax().item())


def _final_for(stage, rows: int, seed: int = 7):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(
        rows, stage.block.embed_tokens.weight.shape[1],
        dtype=torch.bfloat16, generator=gen,
    )


def test_argmax_identical_legacy_vs_inplace_on_same_preSoftcap_matrix():
    for seed in range(6):
        stage = _stub_stage(cap=30.0, seed=seed)
        pre = _gemm(stage, 9, seed=seed + 100)
        legacy = softcap_legacy(pre.clone(), 30.0)
        candidate = softcap_inplace(pre.clone(), 30.0)
        assert int(legacy[-1].argmax().item()) == int(candidate[-1].argmax().item())
        assert int(legacy.argmax(dim=-1).sum().item()) == int(
            candidate.argmax(dim=-1).sum().item()
        )


# --------------------------------------------------------------------------
# 8 + 9: structural — no pre-softcap slicing in the semantic path; the
#        optimized production path contains no out-of-place full-matrix
#        division/tanh/multiply chain.
# --------------------------------------------------------------------------


def test_no_semantic_path_slices_before_softcap():
    source = _STAGE_PATH.read_text()
    m = re.search(r"def full_bf16_logits.*?(?=\n    def )", source, re.DOTALL)
    assert m is not None
    body = m.group(0)
    # softcap applies to the full GEMM result; no row selection appears in
    # full_bf16_logits at all
    assert "logits = final @ weight.t()" in body
    assert "[-1]" not in body
    assert "softcap_inplace(logits, cap" in body
    assert "softcap_legacy(logits, cap" in body
    # final_row_logits still selects the row AFTER the full softcapped tensor
    m2 = re.search(r"def final_row_logits.*?return ([^\n]+)", source, re.DOTALL)
    assert m2 is not None
    assert "[-1].float()" in m2.group(1).replace(" ", "")


def test_optimized_production_path_has_no_out_of_place_softcap_chain():
    source = _STAGE_PATH.read_text()
    # the in-place production helper uses only in-place elementwise ops
    m = re.search(r"def softcap_inplace.*?(?=\ndef )", source, re.DOTALL)
    assert m is not None
    inplace_body = m.group(0)
    assert "logits.div_(cap)" in inplace_body
    assert "logits.tanh_()" in inplace_body
    assert "logits.mul_(cap)" in inplace_body
    assert "torch.tanh(" not in inplace_body
    # full_bf16_logits routes through softcap_inplace by default (no
    # torch.tanh(logits / cap) * cap chain in the production path)
    m2 = re.search(r"def full_bf16_logits.*?(?=\n    def )", source, re.DOTALL)
    assert m2 is not None
    assert "torch.tanh(logits / cap) * cap" not in m2.group(0)
    # the production default is the in-place mode
    assert '"inplace"' in source


def test_inplace_mode_is_the_production_default():
    stage = _stub_stage()
    assert stage._softcap_mode == "inplace"


def test_unknown_softcap_mode_fails_closed():
    stage = _stub_stage(cap=30.0)
    stage._softcap_mode = "bogus"
    import pytest

    with pytest.raises(ValueError, match="unknown softcap mode"):
        stage.full_bf16_logits(_final_for(stage, 3))


def test_legacy_mode_reproduces_frozen_expression_exactly():
    stage = _stub_stage(cap=30.0)
    stage._softcap_mode = "legacy"
    final = _final_for(stage, 5)
    out = stage.full_bf16_logits(final)
    expected = torch.tanh((final @ stage.block.embed_tokens.weight.t()) / 30.0) * 30.0
    assert torch.equal(out, expected)


# --------------------------------------------------------------------------
# 10: the existing final-row FP32 optimization remains intact (both modes).
# --------------------------------------------------------------------------


def test_final_row_fp32_optimization_intact_in_both_modes():
    for mode in ("legacy", "inplace"):
        stage = _stub_stage(cap=30.0)
        stage._softcap_mode = mode
        final = _final_for(stage, 32)
        row = stage.final_row_logits(final)
        full = stage.full_bf16_logits(_final_for(stage, 32))
        assert row.dtype == torch.float32
        assert row.shape == (stage.block.embed_tokens.weight.shape[0],)
        assert torch.equal(full[-1].float(), row)


# --------------------------------------------------------------------------
# 11: OOM diagnostic phase/step evidence retained on injected failure.
# --------------------------------------------------------------------------


def test_cuda_phase_probe_records_named_fields_without_cuda():
    record = cuda_phase_probe(
        "softcap_tanh_complete", step=7, replay_rows=32, generated_tokens=7
    )
    assert record["phase"] == "softcap_tanh_complete"
    assert record["step"] == 7
    assert record["replay_rows"] == 32
    assert record["generated_tokens"] == 7
    if not torch.cuda.is_available():
        assert record["cuda_available"] is False


def test_cuda_phase_probe_never_raises():
    # a probe against an invalid device must degrade to a recorded error,
    # not raise through the instrumented execution path
    record = cuda_phase_probe("step_begin", device="cuda:99")
    assert record["phase"] == "step_begin"
    if torch.cuda.is_available():
        assert "probe_error" in record


class _RecordingProbe:
    def __init__(self, fail_on=None):
        self.phases = []
        self.fail_on = fail_on or set()

    def __call__(self, phase):
        if phase in self.fail_on:
            raise RuntimeError("injected diagnostic failure")
        self.phases.append(phase)


def test_softcap_probe_failure_is_swallowed_and_semantics_unchanged():
    cap = 30.0
    pre = _gemm(_stub_stage(cap=cap), 8)
    probe = _RecordingProbe(fail_on={"softcap_div_complete"})
    out = softcap_inplace(pre.clone(), cap, probe=probe)
    expected = softcap_legacy(pre.clone(), cap)
    assert torch.equal(out, expected)  # broken diagnostic cannot alter results
    assert "softcap_div_complete" in probe.phases


def test_phase_evidence_survives_injected_oom_and_retains_step_metadata():
    """The failure-path retention contract: an injected OutOfMemoryError must
    leave the step/phase/rows/generated metadata readable afterwards."""
    phase_evidence = []

    def probe(phase):
        phase_evidence.append(
            cuda_phase_probe(phase, step=7, replay_rows=32, generated_tokens=7)
        )

    class _FailingStage:
        _softcap_mode = "inplace"

        def full_bf16_logits(self, final):
            pre = torch.randn(4, 16, dtype=torch.bfloat16)
            probe("lm_head_gemm_complete")
            probe("softcap_div_complete")
            raise torch.OutOfMemoryError(
                "CUDA out of memory. Tried to allocate 18.00 MiB."
            )

    stage = _FailingStage()
    try:
        stage.full_bf16_logits(None)
        raised = False
    except torch.OutOfMemoryError as exc:
        raised = True
        last = phase_evidence[-1]
        assert last["step"] == 7 and last["replay_rows"] == 32
        assert last["generated_tokens"] == 7
        assert last["phase"] == "softcap_div_complete"
        assert "18.00 MiB" in str(exc)
    assert raised
    assert len(phase_evidence) == 2  # evidence accumulated through failure


def test_runner_names_next_operation_after_last_completed_phase():
    spec = importlib.util.spec_from_file_location(
        "r6_runner_softcap", _RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    # the suspect 18 MiB allocation, if it is the softcap div temporary,
    # surfaces with last completed phase lm_head_gemm_complete -> next op
    # softcap division
    assert runner._next_phase_after("lm_head_gemm_complete") == "softcap_div_complete"
    assert runner._next_phase_after("softcap_div_complete") == "softcap_tanh_complete"
    assert runner._next_phase_after("softcap_mul_complete") == "final_row_fp32_complete"
    assert runner._next_phase_after("step_complete") == "next_step_replay"
    assert runner._next_phase_after("unknown_phase") is None
    assert tuple(runner._PHASE_ORDER) == (
        "step_begin",
        "batch_prepare_complete",
        "embedding_complete",
        "layers_complete",
        "final_norm_complete",
        "lm_head_gemm_complete",
        "softcap_div_complete",
        "softcap_tanh_complete",
        "softcap_mul_complete",
        "final_row_fp32_complete",
        "argmax_complete",
        "step_complete",
    )


# --------------------------------------------------------------------------
# 12: frozen constants / historical result remain unchanged.
# --------------------------------------------------------------------------


def test_frozen_runner_constants_and_historical_result_unchanged():
    import json

    runner_src = _RUNNER_PATH.read_text()
    assert "RUNTIME_CAPACITY_TOKENS = 64" in runner_src
    assert "FROZEN_THRESHOLD = 0.25" in runner_src
    assert "CAPTURE_STEPS = (0, 1, 7)" in runner_src
    result = json.loads(
        (_REPO / "docs" / "inferswarm_r6" / "result.json").read_text()
    )
    assert result["verdict"] == "R6_DENSE_ARCHITECTURE_FALSIFICATION_FAIL"
    digest = hashlib.sha256(
        (_REPO / "docs" / "inferswarm_r6" / "result.json").read_bytes()
    ).hexdigest()
    assert digest == "b19e746c831484cb7ba07088d8959297e1d8641125215d6fe63f27c7255817e3"
