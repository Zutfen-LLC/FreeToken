# R6 Single-GPU FreeToken Control — Methodology (PRE-RESULT FREEZE)

This record freezes the post-failure diagnostic requested after the corrected
R6 result. It does not replace, amend, or reinterpret the historical
`R6_DENSE_ARCHITECTURE_FALSIFICATION_FAIL`: the frozen Transformers-versus-
distributed threshold remains strict `< 0.25`, and its observed aggregate
max absolute difference remains `0.515625`.

Freeze parent: `b0f49b960f7a025c6c24dbd54a3c5961f46c863a` on
`inferswarm-r6`; required ancestry base
`84ebd2b7ae56c60292f7b9c7ca256f41f64d8b11`. The exact implementation SHA
and a clean-tree assertion are recorded by the physical runner.

## Purpose and interpretations

The matched control is:

```text
Transformers Gemma reference
          |
          v
single-GPU FreeToken Gemma
          |
          v
3-stage distributed FreeToken Gemma
```

It holds FreeToken decoder math, BF16 checkpoint, Triton attention backend,
rename/fusion semantics, norm, tied output projection, logit softcap, prompt,
greedy sampling, and full replay-prefill semantics constant. Only physical
decomposition changes.

- Outcome A: Transformers versus single FreeToken is `>= 0.25`, while single
  versus distributed FreeToken is materially closer/equivalent. This supports
  a FreeToken-versus-Transformers numerical difference. Historical R6 remains
  FAIL; a matched-FreeToken reference may be proposed only prospectively.
- Outcome B: Transformers versus single FreeToken is `< 0.25`, while the
  distributed path remains outside the threshold. This identifies
  distribution/stage drift. R6 does not pass; the next experiment bisects
  embedding/layer 15, layer 31, layer 47/final norm, and final logits, then
  separates same-GPU decomposition from the LAN boundary.
- Outcome C: bounded materialization is correct but final weights plus the
  frozen minimal runtime do not fit. Report
  `SINGLE_GPU_REFERENCE_CAPACITY_BLOCKED` with allocation evidence; CPU
  offload is forbidden.

## Frozen model and topology

- Checkpoint: `google/gemma-4-12B-it` revision
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
- Native checkpoint SHA-256:
  `5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d`.
- Required text checkpoint bytes: exactly `23,814,700,640`.
- Representation: BF16, text-only; vision/audio state is not selected.
- Compute Unit: one `role = "single"` with
  `DenseBlockSpec(0, 48, True, True)`.
- Path: one embedding table, global decoder layers 0 through 47, final norm,
  and tied output projection using that same CUDA embedding tensor.
- Semantic boundaries: none. There is no stage activation transfer.
- Canonical physical host: `inferswarm04`, one RTX 3090 24 GiB. Hostname,
  UUID, PCI BDF, actual VRAM, and idle state must be captured and committed in
  a preflight amendment before model realization; no run is authorized from
  an assumed device identity.

## Frozen runtime and comparator

- Runtime capacity: exactly `64` tokens. The prompt is 26 tokens and the
  eight committed-token replays reach 33 input tokens, so no larger serving
  capacity is allocated.
- Prompt and sampling:
  `docs/inferswarm_r6/canonical-prompt.json`; exact recorded IDs; eight greedy
  tokens; temperature 0.
- Execution at every committed step: reset stage-local KV state, then prefill
  the full prompt plus already committed tokens in one chunk starting at
  position zero. This is the accepted R6 Coordinator replay path, not the
  known-broken incremental KV-extend path.
- Capture steps: 0, 1, and 7. Retain float32 full-vocabulary single-FreeToken
  rows, generated tokens, and NaN/Inf counts.
- Transformers versus single FreeToken: evaluate the retained Transformers
  top-32 domain with the unchanged strict `< 0.25` threshold. The retained
  reference does not contain a full-vocabulary row, so the domain limitation
  must be stated rather than invented away.
- Transformers versus distributed FreeToken: reproduce the already-retained
  values separately and unchanged.
- Single FreeToken versus distributed FreeToken: compare full-vocabulary
  float32 rows at steps 0, 1, and 7.

## Bounded-host materialization contract

Every source tensor is resolved to its exact shard from checkpoint metadata.
Only a single raw tensor (or an explicitly byte-bounded group) may be live.
The caller completes a synchronous final-device copy inside the tensor
context. On exit, the source tensor storage is invalidated, the safetensors
mapping closes, and only then is its exact file range advised
`POSIX_FADV_DONTNEED`. No system-wide cache flush or privileged kernel knob is
used.

Q/K/V and gate/up fusion allocates the final CUDA tensor first and copies each
raw source into its final row slice. A K=V layer reads K once and writes both
K and V slices. No CPU `torch.cat`, fused CPU tensor, long-lived model-state
dictionary, or complete host model is permitted.

The physical report must retain:

- selected and processed checkpoint bytes, largest raw tensor, cumulative
  bytes processed, peak simultaneously-live source bytes, peak live fusion
  source bytes, and current staging bytes;
- RSS before loading, RSS after sources close, final RSS, VmHWM, RssAnon,
  RssFile, VmSwap, host memory/swap snapshots, page-fault/swap deltas, and
  elapsed load time;
- CUDA allocation before loading, after weights, after 64-token runtime
  initialization, peak allocation, actual total/free memory, and final free
  margin;
- exact layer coverage, unexpected keys, fallback-iterator calls, embedding
  materialization count, and whether the complete model remains resident.

Passing loader evidence requires all required text bytes processed, layers
0..47 complete, no unplanned keys, no whole-checkpoint fallback, exactly one
tied table, `persistent_host_model_bytes == 0`,
`host_staging_current_bytes == 0`, no CPU-owned decoder layer, no swap-backed
model state, and live complete weights plus runtime on the RTX 3090.

Accelerate/Transformers device maps, CPU tails, unified-memory fallback,
automatic layer offload, quantization, and CPU model-weight offload are
prohibited. If capacity is insufficient after accidental duplication has been
excluded, the experiment stops as Outcome C.
