# R6 — Dense Architecture Falsification (Gemma 4 12B) — Evidence

inferswarm issue #65.  Producer: 44d6c94e4fd2ee967451cc959f930883ca3f4a25
(branch `inferswarm-r6`, canonical base 84ebd2b7ae56c60292f7b9c7ca256f41f64d8b11).

## What ran

Canonical topology (frozen in `environment.json`):

- inferswarm00 — CPU-only external Coordinator (`benchmarks/inferswarm_r6/coordinator.py`), torch-free by construction.
- inferswarm01 GPU0 — stage 1 `[0,16)` + embeddings (9.26 GiB selective).
- inferswarm01 GPU1 — stage 2 `[16,32)` (7.28 GiB selective).
- inferswarm03 GPU0 — stage 3 `[32,48)` + final norm + tied lm_head (9.29 GiB selective), reached over the accepted R4 boundary wire.

Ordinary OpenAI-compatible chat request → Coordinator tokenizer → generic
planner (frozen snapshot + strategy problem + controlled evidence override)
→ frozen Execution Plan + Coordinator-owned epoch/realization authority →
node agent (`benchmarks/inferswarm_r6/node_agent.py`) realizes the dense
chain → committed-token stream with fencing/accounting on 00.

## Headline results

- Canonical request: **exact 8/8 greedy token equality** vs the
  unpartitioned transformers reference (`reference-generation.json`):
  `[818, 6073, 529, 74413, 46515, 600, 2557, 532]`.
- Fencing arm (`inferswarm_fencing_arm_after_step: 1`): duplicate
  already-committed position AND retired-epoch injections both rejected;
  output still 8/8 exact.
- Graceful SIGTERM: epoch state RECLAIMED; postflight VRAM idle on all
  participants, no orphan processes.
- Coordinator purity: "PyTorch was not found" on 00 is the proof token.

## Assumption audit (issue #65 falsification classes)

| Seam | Classification | Evidence |
|---|---|---|
| r1/r2/r3 planner core, r5b epoch controller | GENERIC_AND_REUSED_UNCHANGED | zero model references; drove the dense plan unchanged |
| #67 xc wire + remote-realization authority | GENERIC_AND_REUSED_UNCHANGED | agent/coordinator consumed Coordinator-authorized epoch identity verbatim |
| N0 Qwen-pinned census/loader | FIRST_MODEL_ARTIFACT_REFACTORED | replaced by `research/r6_dense_census.py` (model-agnostic) |
| Qwen 2-plane hidden+residual boundary | FIRST_MODEL_ARTIFACT_REFACTORED | dense Gemma uses a single-plane hidden boundary |
| Hard-coded 2-block A→B split | GENERIC_BUT_REQUIRED_EXTENSION | N-stage chain (`stage_chain.py`), granularity census-driven (2-stage MEASURED_INFEASIBLE, retained) |
| Boundary geometry (64-row chunks, 491,520 B) | GENERIC_BUT_REQUIRED_EXTENSION | within the unchanged 1 MiB r4_wire budget |
| Selective materialization | GENERIC_AND_REUSED_UNCHANGED (pattern) | meta-build + streamed per-tensor loads; tied-embedding declared shared state on first/last only |

No foundational doctrine invariant was falsified: the strategy-constrains /
generic-planner-chooses boundary, frozen plans, epoch authority, fencing,
and the external Coordinator all held under a dense, expert-free model.

## Anomaly retained

`anomaly-incremental-decode.md`: the KV-extend path (incremental decode;
multi-chunk prefill continuation) diverges, A/B-proven, while replay
prefill is exact.  Non-blocking for this gate: the accepted epoch
controller commits only replay step-0 tokens.  Mitigated by 64-row
single-chunk replays; open engineering debt, explicitly a non-claim.

## Files

- `METHODOLOGY.md` — frozen before canonical results (comparator contract included).
- `canonical-prompt.json` — 26 frozen prompt token IDs.
- `environment.json`, `serving-evidence.json`, `chain-plan.json` — frozen inputs (sha-sidecar'd on the hosts).
- `lifecycle/serving-report.json` — both requests, epoch lineage, fencing record, RECLAIMED state.
- `reference-generation.json` — unpartitioned comparator output (top-32 logits retained).
- `compose_result.py` — machine re-derivation of every passing condition → `result.json`.

## Explicit non-claims

No performance claim vs Qwen3.6 (TTFT ~11 s on this chain is honest and
unoptimized); no production API/protocol; no multimodal; no steady-state
dense decode claim (see anomaly); no public strategy/planner API freeze.

## Regression summary

See `regressions.md` (01 research+benchmark suites; 00 torch-free
control-plane).
