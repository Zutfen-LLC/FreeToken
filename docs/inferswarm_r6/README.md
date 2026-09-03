# R6 — Dense Architecture Falsification (Gemma 4 12B) — Evidence

inferswarm issue #65.  Canonical producer:
44d6c94e4fd2ee967451cc959f930883ca3f4a25 (branch `inferswarm-r6`, canonical
base 84ebd2b7ae56c60292f7b9c7ca256f41f64d8b11).  Methodology: original
freeze in `METHODOLOGY.md` (95b408b) plus the dated, pre-canonical
`METHODOLOGY-AMENDMENT-001.md` (32→64-row prefill geometry, frozen at
ff561e5).  The secondary-comparator evidence arm ran under a distinct,
explicitly-recorded producer (see `lifecycle/secondary-comparator.json`).

## What ran

Canonical topology (frozen in `environment.json`):

- inferswarm00 — CPU-only external Coordinator (`benchmarks/inferswarm_r6/coordinator.py`), torch-free by construction.
- inferswarm01 GPU0 — stage 1 `[0,16)` + embeddings (9.26 GiB selective).
- inferswarm01 GPU1 — stage 2 `[16,32)` (7.28 GiB selective).
- inferswarm03 GPU0 — stage 3 `[32,48)` + final norm + tied lm_head (7.28 GiB owned + 2.01 GiB declared shared embedding selective), reached over the accepted R4 boundary wire.

Ordinary OpenAI-compatible chat request → Coordinator tokenizer → generic
planner (frozen snapshot + strategy problem + controlled evidence override)
→ frozen Execution Plan + Coordinator-owned epoch/realization authority →
node agent (`benchmarks/inferswarm_r6/node_agent.py`) realizes the dense
chain → committed-token stream with fencing/accounting on 00.

## Headline results (machine-verified by compose_result.py)

- Canonical request: **exact 8/8 greedy token equality** vs the
  unpartitioned transformers reference (`reference-generation.json`):
  `[818, 6073, 529, 74413, 46515, 600, 2557, 532]`.
- Stage accounting: all three stages' observed fetched bytes equal their
  frozen planned bytes (stage 3: owned 7,278,946,848 + shared 2,013,265,920
  = 9,292,212,768, exact); zero unexpected checkpoint keys; zero
  whole-shard sentinel calls; zero host staging residue and zero
  unexplained persistent host mirror on all three stages.
- Fencing arm: duplicate already-committed position AND retired-epoch
  injections both rejected; output still 8/8 exact.
- Graceful SIGTERM: epoch state RECLAIMED; postflight VRAM idle; no
  orphan processes.
- Coordinator purity: "PyTorch was not found" on 00 is the proof token.
- Boundary geometry: single plane (`planes: 1`, row width 3840, bf16);
  Qwen's 2-plane boundary was a first-model artifact (R6 finding).

**Secondary logit comparator — HONEST FAIL (retained).**  The frozen
secondary criterion (max |logit_ref − logit_dist| < 0.25 over the declared
top-32 domain at steps 0/1/7) was physically captured through the canonical
distributed dense path (evidence arm, explicit-override producer) and
measured: per-step max 0.25 / 0.50 / 0.515625; aggregate **0.515625 ≥ 0.25
→ FAIL**.  NaN/Inf count 0.  The threshold was frozen before canonical
results and was NOT loosened; the corrected result composer reports
`R6_DENSE_ARCHITECTURE_FALSIFICATION_FAIL` (27/28 checks; the comparator
threshold is the single failing check).  The maintainer adjudicates whether
this numeric-drift observation blocks the gate or joins the retained
anomalies as a non-claim; the repository does not decide this silently.

## Assumption audit (issue #65 falsification classes)

| Seam | Classification | Evidence |
|---|---|---|
| r1/r2/r3 planner core, r5b epoch controller | GENERIC_AND_REUSED_UNCHANGED | zero model references; drove the dense plan unchanged |
| #67 xc wire + remote-realization authority | GENERIC_AND_REUSED_UNCHANGED | agent/coordinator consumed Coordinator-authorized epoch identity verbatim |
| N0 Qwen-pinned census/loader | FIRST_MODEL_ARTIFACT_REFACTORED | replaced by `research/r6_dense_census.py` (model-agnostic) |
| Qwen 2-plane hidden+residual boundary | FIRST_MODEL_ARTIFACT_REFACTORED | dense Gemma uses a single-plane hidden boundary (`planes: 1` everywhere, regression-tested) |
| Hard-coded 2-block A→B split | GENERIC_BUT_REQUIRED_EXTENSION | N-stage chain (`stage_chain.py`), granularity census-driven (2-stage MEASURED_INFEASIBLE, retained) |
| Boundary geometry (amended 64-row chunks, 491,520 B) | GENERIC_BUT_REQUIRED_EXTENSION | within the unchanged 1 MiB r4_wire budget; see METHODOLOGY-AMENDMENT-001 |
| Selective materialization | GENERIC_AND_REUSED_UNCHANGED (pattern) | meta-build + streamed per-tensor loads; tied-embedding declared shared state on first/last only |
| Secondary comparator | MEASURED, FAILED THRESHOLD | `lifecycle/secondary-comparator.json` + full-vocab float32 retention |

No foundational doctrine invariant was falsified: the strategy-constrains /
generic-planner-chooses boundary, frozen plans, epoch authority, fencing,
and the external Coordinator all held under a dense, expert-free model.

## Anomaly retained

`anomaly-incremental-decode.md`: the KV-extend path (incremental decode;
multi-chunk prefill continuation) diverges, A/B-proven, while replay
prefill is exact.  Non-blocking for this gate: the accepted epoch
controller commits only replay step-0 tokens.  Mitigated by 64-row
single-chunk replays (METHODOLOGY-AMENDMENT-001); open engineering debt,
explicitly a non-claim.  The measured secondary-comparator drift
(aggregate 0.5156) plausibly shares numerics with this class of
boundary/KV drift; retained as evidence, not adjudicated here.

## Files

- `METHODOLOGY.md` — frozen before canonical results (comparator contract included; original 32-row geometry, superseded by the amendment).
- `METHODOLOGY-AMENDMENT-001.md` — dated amendment to 64 rows/491,520 B, frozen (ff561e5) BEFORE the successful canonical rerun.
- `canonical-prompt.json` — 26 frozen prompt token IDs.
- `environment.json`, `serving-evidence.json`, `chain-plan.json` — frozen inputs (sha-sidecar'd on the hosts).
- `lifecycle/serving-report.json` — both requests, epoch lineage, fencing record, RECLAIMED state.
- `lifecycle/last-stage-final-report.json` — canonical stage-3 materialization/accounting (recovered from the run, producer+digest bound).
- `lifecycle/secondary-comparator.json` — frozen secondary comparator evaluation (FAIL, honest).
- `lifecycle/distributed-logits-0-1-7.f32.bin` — full-vocab float32 distributed logits, steps 0/1/7.
- `lifecycle/secondary-comparator-capture-run.json` — evidence-arm chain-side provenance.
- `reference-generation.json` — unpartitioned comparator output (top-32 logits retained).
- `compose_result.py` — fail-closed machine re-derivation of every condition → `result.json`.
- `result.json` — 27/28 checks pass; verdict FAIL on the comparator threshold (honest).

## Explicit non-claims

No performance claim vs Qwen3.6 (TTFT ~11 s on this chain is honest and
unoptimized); no production API/protocol; no multimodal; no steady-state
dense decode claim (see anomaly); no secondary-comparator pass (measured
FAIL retained); no public strategy/planner API freeze.

## Regression summary

See `regressions.md` (01 research+benchmark suites; 00 torch-free
control-plane; focused R6 gate-contract + composer negative-control
suites).
