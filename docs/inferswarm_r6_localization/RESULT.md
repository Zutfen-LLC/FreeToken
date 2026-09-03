# InferSwarm #71 — Localization Result (D−S)

Status: LOCALIZATION COMPLETE — maintainer adjudication requested.
No fix implemented (per issue instruction: localize → review → freeze fix →
requalify). Historical R6 remains `R6_DENSE_ARCHITECTURE_FALSIFICATION_FAIL`;
#65 remains open; no threshold or reference was changed; no R6 PASS claimed.

## Physical producers

- Methodology freeze: `81a328a` (METHODOLOGY.md committed before any capture)
- Canonical coarse physical captures (S + D arms): `2032a13c9cd57bd8471e04708be1d3f742f13cc7`
- Bisection rounds 1–3: `2032a13` (capture-layer selection only, same code)
- Device diagnostic + operator probe: `47055cc` / `d42965b` / `6930ae1` / `761fd19`
- Model: google/gemma-4-12B-it @ 707f0a3b, checkpoint sha256
  `5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d`
  (re-verified sha256sum on inferswarm04 during the campaign).

## Hardware (all verified by ready-file / result JSON GPU UUID)

- S arm: inferswarm04 RTX 3090 `GPU-ecda1aaa-0c66-857b-8218-3d511dc75c03`
- D stage 1: inferswarm01 GPU0 RTX 3060 `GPU-1fc28f83-…`
- D stage 2: inferswarm01 GPU1 RTX 3060 `GPU-d5c05739-…`
- D stage 3: inferswarm03 GPU0 RTX 3060 `GPU-e1f2f90c-…`

## Exit criteria answers (issue §Exit criteria)

1. **Last S==D checkpoint:** `embedding_output` — byte-exact SHA-256
   equality at steps 0, 1, and 7.
2. **First S≠D checkpoint:** after global layer 0 — the FIRST decoder
   layer's output already differs (max |Δ| 0.125–0.25 bf16, ~38–55% of
   elements differ by 1–2 ULP). Bisection: after_15 ≠ → after_3 ≠ →
   after_1 ≠ → after_0 ≠ → (embedding ==) ⇒ layer 0.
3. **Boundaries:** BOTH byte-identical. Boundary 1 (pipe) and boundary 2
   (R4 wire): sender SHA == receiver SHA, exact tensor equality, and the
   raw transmitted payload bytes hash-identical on 03. Transport EXCLUDED.
4. **Same-input stage replay:** stage-1's exact distributed code path fed
   the same embedding input on the 3090 reproduces S BYTE-EXACT through
   layer 15 (and on the 3060 reproduces D stage 1 byte-exact). The
   distributed code is bit-faithful on a given device; the difference is
   the DEVICE, not the stage code or its input.
5. **Earliest seam:** inside layer 0, ops 1–5 (input_layernorm, fused qkv
   GEMM, q/k/v norms, RoPE, SWA Triton attention incl. store_kv) are all
   byte-exact across devices. The FIRST divergent operator is
   **op6: the attention output projection GEMM (o_proj,
   [26,2048]@[2048,3840] bf16 matmul)** — 41% of elements differ by 1–2
   ULP. Everything downstream inherits it.
6. **Classification: `BACKEND_EXECUTION_LOCAL`.** Same code, same merged
   weights (byte-identical embedding input proves the loader), same
   stage-local config (machine audit: zero mismatches over all 48 layers —
   attention group kind, kv heads, head_dim, sliding window, k_eq_v, rope
   base/dim/scaling, sm_scale), byte-exact transport — but the cuBLAS bf16
   GEMM accumulates in a different order on RTX 3060 vs RTX 3090 (same
   sm_86 ISA, different SM count → different kernel/tile selection).
   NOT model-strategy-local, NOT transport, NOT generic InferSwarm, NOT
   inherited-input drift.
7. **Concrete correction proposal (prospective, NOT implemented):** if
   bit-exactness of D vs S is required, the deployment contract must pin
   device-class-homogeneous compute (all stages on the same GPU model as
   the reference), or the correctness comparator must be re-frozen
   PROSPECTIVELY to a tolerance that accounts for cross-device bf16 GEMM
   accumulation-order differences (expected magnitude: ~1–2 ULP per GEMM,
   compounding through 48 layers — consistent with the observed
   final-logit |Δ| ≤ 0.5 aggregate). The second option is a methodology
   decision for the maintainer; per doctrine it may NOT be back-fitted to
   the historical R6 result.

## Why the historical comparator saw what it saw

The R6 aggregate D−S = 0.46875 (step-1 max at vocab 1156 = 0.46875) is the
compounding of ~48 layers × (attention+MLP GEMMs) of 1–2-ULP bf16
accumulation-order differences originating at layer-0 o_proj on the 3060
vs 3090 — plus the T−S baseline (~0.03 at the same coordinate) which this
campaign did not re-litigate. The distributed architecture itself
(boundaries, stage-local config, coordinator) is numerically clean.

## Evidence index

- `evidence/coarse/` — S+D manifests, results, boundary2 rx log,
  localization-summary.json (21 comparison rows, boundary proofs)
- `evidence/bisect-1/` (after 3/7/11), `evidence/bisect-2/` (after 1/2),
  `evidence/bisect-3/` (after 0) — per-round manifests + cmp JSONs
- `evidence/device-diag/` — stage-1 code run alone on 3090 and on 3060:
  byte-exact vs S and vs D-stage1 respectively (cmp-diag.json)
- `evidence/op-probe/` — layer-0 op-by-op probe both devices
  (cmp-probe.json: ops 1–5 exact, op6 o_proj first divergent)
- `evidence/config-audit/` — zero-mismatch per-layer config equivalence
  audit + the #71 producer-bound chain plan
- Root capture bundles retained for the diag/probe arms (small);
  full-logit bundles are hash-recorded in their manifests (bundle SHA-256
  in each manifest-*.json) and were 50 MB each — retained on the nodes at
  /home/zutfen/r6loc-71 during the campaign, not committed wholesale.

## Regressions

- New suite `tests/research/test_r6_localization.py`: 17 passed (CPU-pure,
  runs on the torch-free coordinator).
- Existing torch-free R6 suites re-run locally: 45 passed
  (compose_result, common_domain_adjudication, single_gpu_control_gate).
- Instrumentation is no-op without an armed sink (source-contract tests
  enforce; uninstrumented execution statement-sequence identical).

## Non-claims

- No claim that distributed execution is "wrong": greedy tokens are 8/8
  exact in both arms; the drift is sub-ULP-class bf16 accumulation order.
- The T−S baseline is not re-examined; Transformers is contextual only.
- No performance, planner, representation, or Coordinator changes. The
  generic planner was NOT touched.
