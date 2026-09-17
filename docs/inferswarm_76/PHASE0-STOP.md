# Issue #76 campaign status — Phase 0 stop: reference-margin definition vs zero-margin ties

Execution issue: Zutfen-LLC/inferswarm#76
Methodology: inferswarm@f394dc9 (docs/qualification/gemma4-12b-it-v1), FROZEN
Physical producer: FreeToken `29e04d0` (branch `inferswarm-76-gemma-numerical-qualification`,
base `d4d1608`, clean trees on all nodes)

## What completed validly

1. Harness frozen and committed BEFORE any model execution:
   - `benchmarks/inferswarm_76/` — reference_runner (single-arm 3090),
     chain_runner + stage_entry + wire_client (stages 1-2 node-01, remote
     last stage node-03), RowPruningSink (#71-compatible capture with
     host-side final-row pruning of the full BF16 logits matrix),
     o_proj checkpoint wrappers, pure host-float64 reducer (frozen
     REDUCER.md identity), 17 torch-free unit/source-contract tests.
   - No execution/model math changed: all model execution flows through the
     accepted R6 GemmaDenseStage replay-prefill greedy semantics.
2. Preflight identity recorded on 01/03/04 (torch 2.11.0+cu130, CUDA 13.0,
   driver 610.57.04, triton 3.6.0, checkpoint sha 5a84cb31...ff18d verified
   on all three nodes, tokenizer cc8d3a0c... verified).
   NOTE (recorded honestly): native-extension .so hashes differ between
   nodes (same source, build-path-embedded binaries).
3. Non-canonical smokes passed on both arms, and the end-to-end reducer
   produced all 15 envelopes with clean integrity on a sentinel case
   (chain tokens exactly matched the reference on that case).
4. Phase 0 reference-only run COMPLETED on inferswarm04 (RTX 3090,
   GPU-ecda1aaa): all 48 frozen stress-pool cases, 8 greedy tokens each,
   zero NaN/Inf, 40 capture records per case, producer clean.
   Evidence: /srv/inferswarm/state/i76/phase0-reference on inferswarm04
   (3.6 GiB; tar sha256 3b2eb3e1d3de4bcef67e12b202be4b9da759baec786b0e5b5a3e25887b79a869
   retained at coordinator ~zutfen and /tmp/i76 on the orchestrator host).

## Stop condition (Phase 0)

The issue-#74/#76 text requires "the positive top-1 margin using the frozen
definition" but NO numeric margin definition is frozen in any repository
artifact (methodology.json, ADR 0010, the normative supplement, the stress
pool, the selection commitment, or the selector program — searched
exhaustively; the selector only constrains sort order and positivity).

The harness producer (29e04d0, committed before the reference run) pinned:

    positive_top1_margin(case) = min over the 8 greedy steps of
                                 fp32(top1 - top2) on the matched-reference
                                 final-row logits

Under this definition, 5 of the 48 frozen pool cases have margin EXACTLY 0
(bit-identical fp32 top-1/top-2 logits for DISTINCT token ids at one greedy
step — degenerate low-entropy continuations, ties are genuine):

    p74-02-01-02 step 2; p74-03-05-02 step 5; p74-04-05-01 step 2;
    p74-02-06-02 step 3; p74-03-06-01 step 3

The frozen selector (`scripts/select_issue74_margin_stress.py`) rejects any
nonpositive margin ("stress selection requires finite positive top-1
margins"). Therefore reference stress selection cannot complete validly
under the pre-registered definition → per issue #76: "If reference stress
selection cannot be completed validly, stop before candidate execution."

NO candidate (3060 chain) model execution has occurred. No threshold
derivation. The holdout remains sealed (ciphertext sha256
23311c5514b2561c66a2ecd0c9cfa25c3f4f91b83b67353aada8355f48e25c59 verified
unchanged).

## Positivity table over the retained reference margins

Definition (per case, over the 8 step margins) — nonpositive count of 48:

    min over all 8 steps      5  (the pre-registered definition; STOP)
    min over capture 0,1,3,7  2
    per single step k         0 for k in {0,1,4,6,7}; 2 for k in {2,3}; 1 for k=5
    max over 8 steps          0 (min value 0.875)
    mean over 8 steps         0 (min value 0.484375)
    median over 8 steps       0 (min value 0.375)

Full per-case table: margin-table.json (sha256
7016136b042af4e420cdb8a8b6483f2d331d8a260880a70091c9317ca66f7bb0) retained
at the coordinator (/srv/inferswarm/state/i76/) and in this branch's
evidence staging.

## Why this stops rather than re-pins

The reference margins have now been observed. Any NEW margin definition
chosen at this point is chosen with knowledge of the reference results, and
the selected eight cases could be influenced by that choice. Issue #76
forbids reinterpreting the methodology while results are being collected,
and the selector's commitment binds the selection algorithm's hash. A
definition change is a maintainer decision (prospective re-freeze of the
selection input contract, ideally with a fresh stress pool or an explicit
tie-handling amendment), not an execution-side fix.

## Also requiring maintainer input before Phase H

The holdout custodian private key was not found on any reachable host
(inferswarm00/01/03/04, orchestrator; searched /root /home /srv /tmp
/var/tmp). If custody is genuinely unavailable the campaign terminates at
`HOLDOUT_CUSTODY_BLOCKED` even after a successful calibration. This does
not block Phases A-G.

## Recommended decision options (for adjudication, not self-executed)

A. Amend the methodology (v2) to define the margin on a single committed
   step (e.g. step 0) or as max/mean over steps — chosen now by the
   MAINTAINER as a prospective rule, then re-run Phase 0 selection only
   (the reference evidence is reusable; margins are already retained).
B. Keep min-over-8 but amend the selector/pool to handle exact ties
   (exclude tie cases as "no positive margin exists" and select from the
   remaining 43) — requires a new selection-commitment hash and version.
C. Treat zero-margin ties as a corpus defect; regenerate the stress pool
   under a new seed with a prospective tie-exclusion filter (new pool hash,
   new commitment; calibration corpus untouched).

Under any option the already-retained reference run remains valid evidence
(it consumed only frozen pool token IDs and recorded all margins).
