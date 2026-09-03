# R6 Methodology Amendment 001 — Prefill chunk geometry 32 → 64 rows

Amended: 2026-09-03.  This amendment is part of the frozen R6 methodology;
it was frozen at implementation commit **ff561e5**
(ff561e5f029e807fbfd7373ea4aa8d5164e0350c, branch `inferswarm-r6`,
committed 2026-09-03 00:17:39 -0400) — BEFORE the successful canonical
run (producer 44d6c94e4fd2ee967451cc959f930883ca3f4a25, whose serving
evidence is retained in `lifecycle/serving-report.json`).  The original
frozen methodology (`METHODOLOGY.md`) is retained unmodified; this file
amends it and nothing is erased from the chronology.

## Original frozen geometry (METHODOLOGY.md, frozen at 95b408b)

- Prefill boundary chunks: **32 rows** = 245,760 bytes per boundary
  transfer (32 × 3840 row width × bf16), within the unchanged 1 MiB
  r4_wire payload budget.

## Observation that falsified the 32-row canonical-method assumption

The first canonical external-Coordinator attempt (run producer 9d400ec,
2026-09-03; superseded, not retained as canonical) produced 7/8 exact
greedy tokens; the 8th diverged exactly when prompt(26) + committed(7) =
33 rows first exceeded the 32-row prefill chunk, forcing the replay
prefill onto chunk 2 — i.e. onto the KV-extend path that
`anomaly-incremental-decode.md` had already A/B-proven defective
(incremental KV append diverges; full replay-prefill is exact).

## Why this invalidated the 32-row canonical-method assumption

The accepted epoch controller commits only replay-prefill step-0 tokens,
so canonical correctness requires every canonical replay
(26-token prompt + up to 8 committed tokens = up to 34 rows) to stay
within a SINGLE prefill chunk.  At 32 rows, canonical-scale replays
necessarily cross onto the known-broken KV-extend path — the 32-row
geometry is incompatible with the canonical replay method itself, not
merely unlucky.

## Exact amendment

- Prefill boundary chunks: **64 rows** = 491,520 bytes per boundary
  transfer (64 × 3840 × bf16).
- `strategy.PREFILL_CHUNK`: 32 → 64 (ff561e5).
- `stage_chain` prefill chunk: 32 → 64 (ff561e5).
- `last_stage_service.MAX_TOKEN_COUNT`: 32 → 64 (44d6c94 — the last-stage
  wire contract must accept the amended chunk before the canonical run
  can execute; this commit IS the canonical producer).

## Why this is a correctness-preserving strategy-owned geometry
## correction, not threshold tuning

- The amended value moves the chunking boundary of the SAME boundary
  transfer path; it changes no comparator, no threshold, no pass
  criterion, and no acceptance condition.
- The wire payload budget (1 MiB, `r4_wire.PAYLOAD_BUDGET`) was NOT
  weakened: 491,520 B < 1,048,576 B with margin.
- The correctness comparator (exact 8-token greedy equality; secondary
  top-32 logit absdiff < 0.25; NaN/Inf fail) is unchanged from the
  original freeze and is evaluated against the SAME reference.
- The geometry is strategy-owned (dense Gemma adapter constants), not
  generic-planner state; the generic planner consumed the frozen shapes
  unchanged.
- The alternative — weakening the comparator domain to exclude the 8th
  token — would have been threshold tuning; raising the chunk so the
  canonical replay stays on the proven-exact path is a geometry fix.

## Frozen-implementation commit

ff561e5f029e807fbfd7373ea4aa8d5164e0350c ("R6: 64-row prefill chunks
keep canonical replays single-chunk (KV-extend path broken)",
2026-09-03 00:17:39 -0400).  The complete amended chain (including the
last-stage wire acceptance of 64-row chunks) is the canonical producer
44d6c94e4fd2ee967451cc959f930883ca3f4a25 (2026-09-03 00:31:43 -0400).

## Evidence the amendment preceded the successful canonical rerun

- ff561e5 committed 2026-09-03 00:17:39 -0400; canonical producer
  44d6c94 committed 2026-09-03 00:31:43 -0400; the canonical run's
  serving report and last-stage final report are dated after that
  (last-stage final report mtime 2026-09-03 00:50 on inferswarm03;
  canonical prompt/session execution follows the producer commit).
- Ancestry: ff561e5 is an ancestor of 44d6c94 on `inferswarm-r6`
  (ff561e5 → abc4a5b → 44d6c94).
- The superseded first canonical attempt (producer 9d400ec, 00:04:22)
  PRE-dates the amendment and is documented in
  `anomaly-incremental-decode.md` — the chronology of
  attempt → falsification → amendment → successful rerun is preserved
  in git history and is independently auditable.
