# InferSwarm #71 — R6 Distributed Drift Localization Methodology

Status: FROZEN BEFORE FIRST CANONICAL PHYSICAL CAPTURE.

This methodology governs the single-GPU-vs-distributed FreeToken numerical
drift localization subgate (#71). It does not modify, reinterpret, or reopen
the accepted historical R6 result:

`R6_DENSE_ARCHITECTURE_FALSIFICATION_FAIL`

## 1. Question and primary comparator

Central question: at what exact semantic execution point does the distributed
Gemma FreeToken path (D) first cease to be numerically identical to the
matched single-GPU FreeToken path (S), and is the difference inherited input
drift, stage-local semantics, backend execution, boundary transport, or a
generic InferSwarm seam?

Primary comparator: S vs D. Transformers is historical/contextual evidence
only in this campaign; the independently measured FreeToken-vs-Transformers
baseline (T−S) is thereby subtracted and is NOT re-litigated.

## 2. Exact frozen identities

- Source parent (exact branch point): `inferswarm-research@`
  `51e772ab88643df61888c8860c8e67e307190565` (merge of FreeToken PR #25).
  Fresh branch `inferswarm-71-localization`. The old `inferswarm-r6` branch
  is not a moving head for this work.
- Model: `google/gemma-4-12B-it`, revision
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, checkpoint SHA-256
  `5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d`,
  BF16 text path, Triton attention backend, in-place softcap, final-row FP32
  promotion, accepted bounded loader. No CPU offload.
- Canonical prompt: `docs/inferswarm_r6/canonical-prompt.json` (26 tokens).
- Canonical generated sequence (retained reference): 
  `[818, 6073, 529, 74413, 46515, 600, 2557, 532]`.
- Replay steps (capture points): 0, 1, 7.
- Chain plan: the accepted `docs/inferswarm_r6/chain-plan.json` blocks
  (0,16) / (16,32) / (32,48); runtime capacity 256 tokens. New runs use a
  NEW frozen-for-#71 plan produced by the same accepted code from this
  branch's producer SHA (producer supersession, recorded explicitly; the
  accepted R6 physical producer 44d6c94 remains the historical R6 producer).

## 3. Arms

### S arm (single GPU)

`benchmarks/inferswarm_r6/single_gpu_control.py` unchanged in semantics; run
on inferswarm04, GPU UUID `GPU-ecda1aaa-0c66-857b-8218-3d511dc75c03`
(RTX 3090), via a #71 capture runner that adds tensor checkpoints (below).

### D arm (distributed)

Canonical three-stage topology, coordinator inferswarm00 (external
Coordinator authority preserved; not collapsed):

- inferswarm01 GPU0 `GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099`: stage 1,
  embeddings + layers [0,16)
- inferswarm01 GPU1 `GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55`: stage 2,
  layers [16,32)
- inferswarm03 GPU0 `GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176`: stage 3,
  layers [32,48) + final norm + tied head

D topology launch follows the accepted R6 shape: last_stage_service on 03
(port 18485), chain runtime on 01 (stages 1–2 spawn, wire client to 03).
Diagnostic captures on the last stage use the accepted
`--allow-producer <exact-sha>` explicit evidence-arm override semantics
(recorded in every report); boundary-2 wire captures are taken at the
sender and receiver inside the same run.

### Arms comparability

S and D execute the identical layer code and identical weight-loading code
at the identical producer SHA, with identical inputs (token ids, positions,
zeroed KV per step, greedy). S runs on RTX 3090; D stages run on RTX 3060
(all sm_86). Any backend/kernel-selection difference arising from device
differences is part of the distributed deployment identity being localized,
and is further discriminated by the same-GPU diagnostic replays (§7).

## 4. Replay semantics (both arms, per retained comparator step k ∈ {0,1,7})

```
reset KV state
replay_rows = prompt + generated[0:k]
single prefill call over replay_rows at position 0
capture checkpoints (§5)
advance greedy token (argmax final-row FP32)
```

The D arm advances tokens through the chain exactly as the accepted R6
capture arm did: fresh full replay per comparator step, RESET all stages
between replays, single 64-row chunk, position 0, greedy argmax at the last
stage. The discarded speculative decode calls of the serving path are NOT
executed by this capture topology (its prefill-only replay matches the
accepted R6 D logit capture semantics, which is the D identity being
localized).

## 4b. Token-generation fidelity check (S vs D)

Because both arms must be compared on identical inputs at every comparator
step, the run first verifies D's greedy committed sequence under the capture
topology reproduces the canonical sequence exactly (8/8), and that S does
likewise (already proven in R6, re-verified in-run). A token divergence at
step j<7 localizes input divergence no later than step j and is retained as
localization evidence (the tensor campaign proceeds for the steps where
inputs still match).

## 5. Tensor serialization, hashing, statistics

- Capture form: `detach()` → `copy exact native tensor to CPU` (`.cpu()` on
  the live bf16 tensor; FP32 only where the checkpoint IS defined as FP32,
  i.e. final-row logits) → hash contiguous raw bytes (SHA-256) → store
  lossless (`.pt` via torch.save or raw bytes) → release the diagnostic
  copy. No dtype conversion before the checkpoint other than the documented
  host copy. No synchronizations added inside model math beyond the single
  `.cpu()` per checkpoint (which is synchronizing by nature and applied
  identically in both arms).
- Per captured tensor, retained metadata: semantic checkpoint ID, step,
  global layer identity where applicable, shape, dtype, logical position
  range, source role, device (at execution), byte count, deterministic
  SHA-256 after host capture, NaN count, Inf count.
- Per S-vs-D comparison: exact equality (torch.equal on CPU copies),
  max abs diff, mean abs diff, RMS diff, coordinate (row, hidden-dim) of
  max difference, S value and D value at that coordinate. Hidden-tensor
  maxima reported as row index + hidden dimension. For logits, also the
  common-domain (retained reference top-32) max-|Δ| per step for continuity
  with the historical record (descriptive only).
- Statistics are computed in float64 on the CPU copies by a pure-stdlib+
  torch analyzer script; retained artifacts carry the analyzer version.

## 6. Coarse semantic checkpoints (Phase 3)

Captured from BOTH arms at steps 0/1/7:

1. `embedding_output` — after `embed(input_ids)`, before layer 0.
2. `after_layer_15` — residual stream after global layer 15 (= stage-1
   output for D; intra-stack for S).
3. `after_layer_31` — after global layer 31 (= stage-2 output for D).
4. `after_layer_47` — after global layer 47.
5. `final_norm` — final-normalized hidden state.
6. `bf16_logits` — full [T, 262144] BF16 logits AFTER softcap (full-vocab
   FINAL-ROW only where memory requires: retain the full final-row BF16
   pre-FP32 row plus the final-row FP32; full-matrix retention only on the
   3090 where it fits, sequentially, if needed for finer head localization).
7. `final_row_fp32` — final-row FP32 logits (the frozen consumer row).

D captures at 2 and 3 are the tensors at the stage boundary (immediately
before send / after receive — see §8), so the semantic checkpoint and the
boundary proof share one capture.

## 7. Boundary sender/receiver proof (Phase 3, both boundaries)

Boundary 1 (stage1→stage2, multiprocessing pipe): capture the stage-1 tensor
immediately before serialization/send (device tensor before `.cpu()`) and
the stage-2 tensor immediately after receive/deserialization (device tensor
after `.to(cuda:0, bf16)`) before stage-2 layers execute. Machine-compare
shape, dtype, byte count, byte SHA-256, exact equality.

Boundary 2 (stage2→stage3, R4 wire): capture the stage-2 tensor immediately
before `wire_client._boundary` serialization (device tensor) AND the exact
payload bytes as transmitted (the `tobytes()` buffer); on 03 capture the
received payload bytes and the deserialized device tensor before stage-3
layers execute. Machine-compare transmitted vs received bytes (SHA-256,
length) and sender device tensor vs receiver device tensor (bytes, exact
equality). Retain transmitted byte count, wire format identity
(`inferswarm.r4.boundary-wire/1`), producer/consumer stage identity,
session/step attribution.

If sender and receiver are byte-identical at a boundary, transport
corruption is EXCLUDED for that boundary; a downstream difference is not a
network problem.

## 8. Interval selection and bisection (Phases 4–5)

Run §6/§7 first. Select the first divergent interval from the coarse map
(embedding / [0,15] / boundary1 / [16,31] / boundary2 / [32,47] / final
norm / head). Bisect ONLY that interval by layer outputs (after global
layer N for halving N), logarithmically, until the earliest layer
transition with nonzero D−S is identified. Do not instrument every layer
preemptively. Internal operator checkpoints inside the first divergent
layer (attention norm out, Q/K/V projection out, RoPE-applied Q/K,
attention out, post-attention residual, MLP norm, gate/up out, activation,
down proj, final residual) only as evidence demands.

## 9. Same-input diagnostic replay (Phase 6)

For the first divergent distributed stage: feed the EXACT S-captured input
tensor entering that stage (host-captured bytes → device) under the matched
step/position metadata, and compare the stage output against the
S stage-equivalent checkpoint. Separates inherited drift (stage output
matches S given S's input) from stage-local drift (same input, different
output). Explicit diagnostic arm; never silently replaces the canonical D
input; recorded as DIAGNOSTIC_SAME_INPUT_REPLAY.

Additionally a same-GPU-model diagnostic (3090-hosted single-role stage
running ONLY the divergent stage's layers with S-captured input) is
retained when the same-input replay shows stage-local drift, to separate
stage-local CONFIG causes from device/kernel causes (3090 vs 3060). This is
evidence-gathering, not a configuration change.

## 10. Configuration equivalence audit (Phase 7)

Machine-record, per stage and per first-divergent layer, S-vs-D: global
layer ID, local layer ID, attention group type (full vs swa), full/swa
config (kv heads, head_dim, sliding_window), k_eq_v, rope configuration
(base, rotary_dim, scaling), rope device, KV pool group/layer mapping
(kv_local_layer_ids + positional globals), max_seq_len/runtime capacity,
batch phase, positions, cached_len, token count, page table identity,
out_loc, dtype, softcap/norm parameters. Verified from the constructed
objects (not source text) on the compute nodes. Fail-closed: any semantic
property expected to depend on global layer identity that resolves
differently is a localization finding.

## 11. Backend/kernel equivalence evidence (Phase 8)

When same-input replay still differs inside a layer: record which Triton
kernel path each arm selects (decode vs extend vs naive paged attention;
tile sizes/launch geometry where observable), same/different attention
group semantics, Q/K/V dims, dtype, sequence length/positions, KV layout,
fused projection representation. Observe only; never switch kernels to
make results match.

## 12. Stop conditions

- Localization complete when the earliest responsible layer/operator/seam
  is identified with retained evidence (exit criteria §16).
- Any violation of the frozen identities (checkpoint SHA, producer drift,
  hardware mismatch, dirty tree) fails the run closed.
- Hardware OOM on the 3090 during capture: capture sequentially, releasing
  device-side diagnostic buffers; if still infeasible, retain the failure
  honestly and re-plan capture granularity (never weaken model config).
- NaN/Inf contamination of a captured tensor invalidates that capture
  (retained as an anomaly record, not interpreted).

## 12b. Non-invasive capture rule

Diagnostics must not materially alter execution semantics: no dtype
conversions before the actual checkpoint, no extra synchronizations inside
model math beyond the checkpoint host copy, no tensor cloning that changes
allocator pressure materially on the 3090 (checkpoint copies are released
immediately after host capture), no extra model copies, no recomputing
layers merely for capture outside the explicitly isolated diagnostic arm.

## 13. Evidence format

Per capture: manifest row {producer SHA, host, GPU UUID, role, step,
checkpoint, global layer, shape, dtype, byte count, SHA256, NaN/Inf}. A
`localization-summary.json` with per-(step, checkpoint) rows: S hash, D
hash, exact_equal, max/mean/RMS absdiff, max coordinate (row, hidden), S/D
values at max. Boundary records additionally carry sender and receiver
hashes + transmitted byte count + wire identity. Raw tensors retained where
size-reasonable; lossless compression only with the original raw hash
retained and decompression verified byte-exact.

## 14. Fail-closed interpretation

No tolerance is invented after seeing data. Exact equality at a checkpoint
means no drift has appeared yet. The first nonzero residual marks the first
observed divergent checkpoint; magnitude progression is descriptive. Small
differences are never declared equivalence. Historical R6 comparator,
threshold, and reference are untouched; no R6 PASS is claimed or implied;
#65 remains open.

## 15. Seam classification (at completion)

`MODEL_STRATEGY_LOCAL` | `BACKEND_EXECUTION_LOCAL` | `BOUNDARY_TRANSPORT` |
`GENERIC_INFERSWARM_SEAM` | `INHERITED_UPSTREAM_DRIFT` — per the issue's
taxonomy, with the evidence trail that forces the classification.

## 16. Exit criteria

1. Last checkpoint where S == D (byte-exact).
2. First checkpoint where S != D.
3. Both boundaries' sender/receiver byte identity.
4. Whether the first divergent stage differs under identical S input.
5. Earliest layer/operator/configuration seam responsible.
6. Seam classification per §15.
7. Concrete next-correction proposal. Stop before fixing unless trivial.

## 17. Regressions required before physical execution

New: tensor-capture tests, boundary sender/receiver identity tests,
configuration-report tests, same-input replay tests. Existing: R6
single/distributed suites affected by instrumentation; R5B/#67 attribution/
fencing tests if touched; generic planner tests if touched. The generic
planner is ideally untouched; any change is explained.

## 18. Non-goals

No historical R6 comparator/threshold/reference change; no R6 PASS claim;
no performance work; no planner redesign; no representation change; no CPU
offload; no Coordinator collapse; no fixing the T−S baseline here; no
conflation of T−S with D−S.

## 19. Hardware identities (frozen)

- inferswarm04 S arm: RTX 3090 `GPU-ecda1aaa-0c66-857b-8218-3d511dc75c03`.
- inferswarm01 stage 1: RTX 3060 `GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099`.
- inferswarm01 stage 2: RTX 3060 `GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55`.
- inferswarm03 stage 3: RTX 3060 `GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176`.
- Coordinator: inferswarm00 (CPU-only; external authority preserved).
- Stack (all compute nodes): driver 610.57.04, CUDA 13.1.2, torch
  2.11.0+cu130, triton 3.6.0, python 3.13.5 (verified per-run in preflight).
