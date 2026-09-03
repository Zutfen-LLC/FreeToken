# R6 Dense Architecture Falsification — Methodology (FROZEN)

Frozen BEFORE canonical R6 results, per inferswarm issue #65.
Producer at freeze: 95b408b (branch inferswarm-r6, base 84ebd2b7ae56c60292f7b9c7ca256f41f64d8b11).

## Objective

Falsify the InferSwarm abstractions against dense Gemma 4 12B
(google/gemma-4-12B-it @ 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7,
text-only, native BF16 single safetensors,
sha256 5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d).

## Model census (measured)

- Text tower: 48 decoder layers, hidden 3840, intermediate 15360,
  vocab 262144, tied embeddings (no lm_head tensor), 40 SWA layers
  (8 kv heads x 256, window 1024) + 8 full-attention layers
  (1 kv head x 512, k_eq_v), logit softcap 30.0.
- Text state = 21.72 GiB layers + 2.01 GiB tied embedding + 7 KiB norm.
  Vision/audio towers (~2 GiB) excluded: never materialized for text-only.
- Representation rule: BF16 does not fit one 12 GiB 3060 (11.63 GiB
  usable); a 2-stage split (~12.8 GiB/stage) MEASURED_INFEASIBLE
  (hardware OOM during selective load, producer 2c8e381). Canonical
  candidate: 3-stage contiguous chain
  [0,16)+embed / [16,32) / [32,48)+norm+tied-head, max 9.26 GiB/stage.

## Canonical topology (frozen)

- inferswarm00: CPU-only Coordinator (xc serving waist, unchanged seam).
- inferswarm01 GPU0 (GPU-1fc28f83...): stage 1 (first).
- inferswarm01 GPU1 (GPU-d5c05739...): stage 2 (middle).
- inferswarm03 GPU0 (GPU-e1f2f90c...): stage 3 (last) via the accepted
  R4 boundary wire (single-plane bf16 hidden, row width 3840,
  plane-major-contiguous, 32-row prefill chunks = 245,760 B).
- Tied embedding declared shared state, materialized on first+last only.

## Correctness contract (frozen BEFORE distributed results)

- Comparator: unpartitioned transformers 5.16.1 reference (eager
  attention; 2-GPU + CPU-tail device map on inferswarm01 — no single
  24 GiB-class device exists on the fabric; 3090rig offline).
- Canonical prompt: docs/inferswarm_r6/canonical-prompt.json
  (26 token ids, chat-template path identical to the accepted #67 waist).
- PRIMARY: exact greedy generated-token-id equality, 8 tokens.
- SECONDARY: max |logit_ref − logit_dist| over the union top-32 of
  steps 0/1/7 < 0.25 (float32).
- NaN/Inf in any distributed logit fails the run.
- Comparator adjudication domain: COMMITTED tokens only (the accepted
  epoch controller discards speculative step-1 outputs; see
  anomaly-incremental-decode.md — incremental KV decode diverges while
  replay-prefill is exact).

## Passing conditions (from issue #65)

R6_DENSE_ARCHITECTURE_FALSIFICATION_PASS requires: census complete;
3-stage plan through the generic planner (opaque units); no model-family
branching in generic code; selective materialization with
unexplained_persistent_host_mirror_bytes == 0; the canonical distributed
arm serving through the external-Coordinator ordinary path with exact
committed-token equality vs the reference; fencing/attribution valid;
residency invariants clean; assumption audit complete (incl. the #67
seam); regressions green.

## Non-claims

No performance claim vs Qwen3.6; no production API; no multimodal; no
steady-state dense decode claim (open anomaly, retained).
