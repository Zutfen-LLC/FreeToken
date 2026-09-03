# SINGLE_GPU_CONTROL_AMENDMENT-003 — in-place final-logit softcap

Chronology: frozen 2026-09-03, BEFORE any new RTX 3090 single-GPU-control
attempt. Implementation commits `ce51968c2fc97f71d09a1f23894de7343035b9cf` →
`d2f3c9eeb5871bf9ca951f59f1b053df186fc6bf` →
`7f1238fb25d9488c1df40afac72f8795b1e59fa4` (branch `inferswarm-r6`, parent
`b402ef5b2273a177b68687ff06191983c2c02ecc`, required ancestry base
`84ebd2b7ae56c60292f7b9c7ca256f41f64d8b11` — no rebase). This amendment is
committed BEFORE the next physical 3090 result exists.

## Parent methodology

`docs/inferswarm_r6/SINGLE_GPU_CONTROL_METHODOLOGY.md` and
`SINGLE_GPU_CONTROL_AMENDMENT-002.md` (both unchanged). This amendment
amends ONLY the temporary-allocation lifetime of the final-logit softcap.
The full BF16 GEMM, the full `[sequence, vocab]` BF16 logits
materialization, the elementwise softcap operation order, the cap value,
the final-row FP32 promotion (Amendment-002), the comparator threshold,
runtime capacity, model, representation, hardware, and the historical R6
verdict are all untouched.

## Retained 18 MiB OOM observation (exact)

The Amendment-002 requalification, produced by implementation
`b402ef5b2273a177b68687ff06191983c2c02ecc` (retained immutably at
`/srv/inferswarm/state/r6/single-gpu-control/requal-finalrow-fp32-b402ef5/`,
status `SINGLE_GPU_REFERENCE_CAPACITY_BLOCKED`), failed with:

```
CUDA out of memory. Tried to allocate 18.00 MiB.
GPU 0 has a total capacity of 23.56 GiB of which 16.31 MiB is free.
Of the allocated memory 23.07 GiB is allocated by PyTorch, and
134.37 MiB is reserved by PyTorch but unallocated.
cuda_allocated_bytes: 24,753,874,432
cuda_peak_bytes:     24,786,889,216
```

That run remains valid and immutable for `b402ef5` and is NOT reinterpreted
by this amendment.

## Motivating arithmetic (motivated, did NOT prove, the hypothesis)

Gemma vocabulary = 262,144. At the late replay step (33 rows):

```
33 x 262,144 x 2 B (BF16) = 17,301,504 bytes = 16.5 MiB
```

A CUDA allocator request of ~18 MiB for a 16.5 MiB tensor (allocator
rounding to 512-block granularity) is a strong match for an out-of-place
full-sequence BF16 softcap intermediate. The arithmetic alone could not
localize the failing allocation; physical measurement was required.

## Physical localization (inferswarm03, B0 RTX 3060 12 GiB)

Measured on the real canonical final-stage block `[32,48)` (tied
embedding/head, Triton backend, canonical checkpoint
`5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d`,
implementation `7f1238f`, clean detached tree), from one full BF16
`[rows, 262,144]` GEMM per probe with byte-identical pre-softcap clones:

| rows | legacy extra allocation | in-place extra allocation |
|------|--------------------------|---------------------------|
| 32   | +16,777,216 B (16 MiB)   | +0 B                      |
| 33   | +17,301,504 B (16.5 MiB) | +0 B                      |

The legacy out-of-place expression allocates an additional FULL
`[rows, vocab]` BF16 tensor for its intermediate result; at 33 rows that is
exactly 17,301,504 B — the precise byte class of the 18.00 MiB (allocator
granularity-rounded) request that failed on the 3090 with 16.31 MiB free.
The in-place transformation allocates nothing.

## Exact transformations

Legacy (frozen semantics, retained verbatim as `softcap_legacy`):

```python
logits = final @ weight.t()
if cap is not None:
    logits = torch.tanh(logits / cap) * cap
return logits
```

Candidate (in-place, `softcap_inplace`, production default via the
`_softcap_mode` seam; "legacy" remains selectable):

```python
logits = final @ weight.t()
if cap is not None:
    logits.div_(cap)
    logits.tanh_()
    logits.mul_(cap)
return logits
```

Identical elementwise operation order (divide, tanh, multiply), identical
cap (30.0), identical BF16 dtype, identical full-matrix shape. The ONLY
difference is temporary-allocation lifetime: no out-of-place full-matrix
division/tanh/multiply intermediates.

## Physical bit-equivalence proof

Evidence: `/srv/inferswarm/state/r6/softcap-equivalence-7f1238f/`
(`softcap-equivalence.json`, `loader-report.json`, `SHA256SUMS.txt`),
producer `7f1238fb25d9488c1df40afac72f8795b1e59fa4`, host inferswarm03,
torch 2.11.0+cu130, triton 3.6.0. Row probes 1, 4, 26, 32, 33; for each,
the real pre-softcap full BF16 logits matrix was produced by the real
owned decoder layers (Triton attention) + final norm + the single full
lm_head GEMM, cloned byte-identically, then transformed both ways:

```
rows  torch.equal  max_abs_diff  full sha256 equal  final-row FP32 sha256 equal  argmax equal  NaN/Inf
1     true         0.0           true               true                          true          0/0
4     true         0.0           true               true                          true          0/0
26    true         0.0           true               true                          true          0/0
32    true         0.0           true               true                          true          0/0
33    true         0.0           true               true                          true          0/0
```

Exact-equality gate passed on every probe: `torch.equal == true`,
`max_abs_diff == 0`, full-matrix SHA256 identical, final-row FP32 SHA256
identical, argmax identical, NaN count identical (0), Inf count identical
(0). Unit/regression coverage in
`tests/research/test_r6_inplace_softcap.py` (exact equality incl.
positive/negative/zero/near-cap values, NaN/Inf byte identity, argmax and
final-row FP32 identity across modes, structural no-slicing and
no-out-of-place-chain proofs, injected-failure phase-evidence retention).

## OOM localization instrumentation (prospective tooling, no semantic change)

`cuda_phase_probe()` records named-phase CUDA
allocated/reserved/free/peak snapshots (allocator bookkeeping only; no
forced synchronization; probes never alter semantics and swallow their own
failures). The single-GPU control runner records `phase_cuda_memory` per
replay step and, on `torch.OutOfMemoryError`, retains the last completed
phase, the named next operation at failure, allocator state, and
best-effort loader evidence. Canonical phase names:
`step_begin`, `batch_prepare_complete`, `embedding_complete`,
`layers_complete`, `final_norm_complete`, `lm_head_gemm_complete`,
`softcap_div_complete`, `softcap_tanh_complete`, `softcap_mul_complete`,
`final_row_fp32_complete`, `argmax_complete`, `step_complete`.

## Invariants preserved

* Full BF16 `[sequence, vocab]` GEMM: unchanged (`final @ weight.t()`).
* Full `[sequence, vocab]` BF16 materialization: unchanged.
* Elementwise softcap operation order and cap value: unchanged.
* Final-row FP32 promotion (Amendment-002): unchanged and re-verified.
* Runtime capacity: exactly 64 tokens.
* No CUDA allocator tuning (no `PYTORCH_CUDA_ALLOC_CONF`, no
  expandable_segments, no garbage-collection tuning).
* No CPU offload, no quantization, no attention-backend change, no replay
  reduction, no checkpoint/tokenizer/prompt change, no comparator or
  threshold change.
* Historical R6 result `R6_DENSE_ARCHITECTURE_FALSIFICATION_FAIL` and all
  retained evidence directories: byte-identical (verified by SHA256SUMS
  re-verification before this work).

## Freeze declaration

This amendment is frozen BEFORE any new inferswarm04 RTX 3090 result. The
authorized implementation producer for the next physical attempt is the
exact commit containing this amendment on `inferswarm-r6`; no moving
branch head may be run.
