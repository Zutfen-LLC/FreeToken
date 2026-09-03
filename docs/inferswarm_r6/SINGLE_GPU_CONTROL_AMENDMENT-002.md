# SINGLE_GPU_CONTROL_AMENDMENT-002 — final-row-only FP32 logits promotion

Chronology: frozen 2026-09-03, BEFORE the `requal-finalrow-fp32` RTX 3090
requalification run. Implementation commit `0475a1b260ed2bc798eda8004fdb9cd83f50aa29`
(branch `inferswarm-r6`, parent `9aa2713c782a6df152e8ab0f55f3e6203bc6afb6`,
required ancestry base `84ebd2b7ae56c60292f7b9c7ca256f41f64d8b11` — no rebase).

## Parent methodology

`docs/inferswarm_r6/SINGLE_GPU_CONTROL_METHODOLOGY.md` (unchanged). This
amendment amends ONLY the implementation's logits-materialization
lifecycle. Comparator, reference, threshold, runtime capacity, model,
representation, hardware, allocator configuration, and the historical R6
verdict are all untouched.

## Original implementation behavior

`GemmaDenseStage.lm_head_logits(final)` computed the full
`[sequence, vocab]` BF16 logits (tied embedding GEMM + softcap) and then
converted the ENTIRE tensor to FP32:

```python
logits = final @ weight.t()
if cap is not None:
    logits = torch.tanh(logits / cap) * cap
return logits.float()          # whole [seq, vocab] FP32 promotion
```

`prefill`/`decode` then consumed only `logits[-1]` (argmax, capture,
final row). The single-GPU control runner likewise consumed only
`logits[-1].detach().float().cpu()`.

## Retained 32 MiB OOM observation

The PATH-fixed requalification (retained immutably at
`/srv/inferswarm/state/r6/single-gpu-control/requal-9aa2713-pathfix/`,
status `SINGLE_GPU_REFERENCE_CAPACITY_BLOCKED`) failed during Triton
forward execution with:

```
CUDA OOM: tried to allocate 32.00 MiB
(GPU 0 total 23.56 GiB, 20.31 MiB free at failure)
```

That run remains valid for its frozen implementation `9aa2713` and is NOT
reinterpreted by this amendment.

## Why full-sequence FP32 promotion is implicated — exact arithmetic

The canonical replay re-executes full prefill each step with
`replay = prompt + generated` (prompt = 26 tokens). At late steps the
full-sequence logits tensor is:

```
step 6: rows = 26 + 6 = 32
32 rows x 262,144 vocab x 4 bytes (FP32) = 33,554,432 bytes = 32.00 MiB
```

This is exactly the failed allocation size. The BF16 GEMM result itself
(32 x 262,144 x 2 B = 16 MiB) succeeded; the subsequent whole-tensor
`.float()` promotion requested a NEW 32 MiB block and the device had
only 20.31 MiB free. Rows 0..30 of that FP32 tensor were never read by
any consumer.

## Exact code change

`benchmarks/inferswarm_r6/stage_runtime.py`:

- `full_bf16_logits(final)`: the unchanged frozen computation —
  `final @ weight.t()`, softcap `tanh(logits/cap)*cap` when configured,
  returned as BF16 `[seq, vocab]`. GEMM shape and dtype unchanged.
- `final_row_logits(final)`: `full_bf16_logits(final)[-1].float()` —
  the final row is selected BEFORE FP32 conversion.
- `lm_head_logits(final)`: retained unchanged (legacy whole-tensor
  promotion) for direct callers and the bit-exactness proof.

`prefill`/`decode` for roles `single`/`last` now call
`final_row_logits` and return the 1-D final-row FP32 logits directly.
`benchmarks/inferswarm_r6/single_gpu_control.py` drops its `[-1]`
indexing and records per-step CUDA allocated/peak/free evidence.
Distributed chain/wire services (`last_stage_service`, `two_stage`,
`stage_chain`) consume at most the token id and the (now 1-D) returned
row; no semantic change.

## What is NOT changed

- Full `[sequence, vocab]` BF16 logits GEMM: same shape, same dtype,
  same softcap placement (before row selection, exactly as before).
- Comparator methodology, Transformers reference, strict `< 0.25`
  threshold, capture steps 0/1/7, distributed comparator artifacts.
- `runtime_capacity_tokens = 64`; no sequence-length reduction.
- CUDA allocator settings; no `PYTORCH_CUDA_ALLOC_CONF` change.
- Checkpoint (`/srv/models/gemma-r6`, sha256
  `5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d`)
  or representation (BF16).
- Hardware target (RTX 3090, GPU-ecda1aaa-0c66-857b-8218-3d511dc75c03,
  BDF 00000000:01:00.0) and the frozen PREFLIGHT-INFERSWARM04 identity.
- The bounded-host loader and all host-memory evidence fields.
- The historical R6 verdict `R6_DENSE_ARCHITECTURE_FALSIFICATION_FAIL`
  and `docs/inferswarm_r6/result.json`.

## Numerical identity

BF16→FP32 conversion is elementwise and exact. Therefore for every
tensor `full_bf16_logits` produces:

```
legacy    = full_bf16_logits(final).float()[-1]   # whole-tensor promotion
optimized = full_bf16_logits(final)[-1].float()   # final-row promotion
torch.equal(legacy, optimized)                    # bit-identical
```

This is asserted by `tests/research/test_r6_final_row_fp32_promotion.py`
(CPU) and proven physically on inferswarm03 (real Gemma last-stage
realization incl. the actual tied head and softcap; see evidence
reference below): `torch.equal = true`, `max_abs_diff = 0`, identical
SHA256 raw-byte hashes, identical argmax, identical NaN/Inf counts.

Expected allocation change (capacity, not performance):

```
legacy FP32 promotion bytes:    rows x vocab x 4   (32 MiB at step 6)
optimized FP32 promotion bytes: 1 x vocab x 4      (1 MiB)
```

## Character

This is an allocation-lifetime/materialization correction: unused
earlier sequence rows are no longer retained/promoted at a higher
precision than their consumers require. It is NOT numerical tuning; the
final-row numerical computation is bit-identical.

## Provenance

- Implementation SHA: `0475a1b260ed2bc798eda8004fdb9cd83f50aa29`
- Parent: `9aa2713c782a6df152e8ab0f55f3e6203bc6afb6`
- Amendment frozen (committed) BEFORE the new RTX 3090 run
  `requal-finalrow-fp32-0475a1b`; physical equivalence evidence:
  inferswarm03 `/srv/inferswarm/state/r6/finalrow-equivalence-0475a1b/`
  (SHA-256 sidecar retained there; summary recorded in the completion
  report of the run this amendment authorizes).
