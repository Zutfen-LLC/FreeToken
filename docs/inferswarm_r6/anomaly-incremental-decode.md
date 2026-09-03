# R6 anomaly record: incremental decode divergence over the dense N-stage seam

Producer: d53a701b111156b496e9417e6d20d2d4afb8bb27 (chain code)
Diagnosed: 2026-09-02, inferswarm01+03

## Observation

Incremental (KV-cache) decode over the 3-stage dense chain diverges from
the unpartitioned transformers reference from the first decode step, while
the prefill path is exactly correct:

- chain prefill step-0 token: 818 == reference step-0: 818
- chain incremental decode step-1 token: 11146 (then repeats 11146)
- reference step-1 token: 6073
- chain FULL-REPLAY of prompt+[818] step-1 token: 6073 == reference

Experiment (r6_diag.py, /tmp/r6_diag.log on 01): VERDICT INCREMENTAL_DECODE_BROKEN.

## Impact on the canonical serving path

None, by construction of the accepted epoch semantics: the R5B/#67 epoch
controller commits ONLY the step-0 token of each
``generate(prompt + committed, max_new_tokens=2)`` call; step 1 is
"explicitly speculative and discarded before replay"
(python/freetoken/research/r5b_epochs.py, serve_tokens). Every committed
token therefore emerges from a fresh full prefill replay, which this chain
proves numerically correct against the reference. The dense strategy's
recovery contract (KV reconstructible via replay-prefill) is exactly the
accepted R5B replay semantic, so the canonical arm never depends on the
defective incremental path.

## Classification

ANOMALY_RETAINED / NON_BLOCKING_FOR_CANONICAL_ARM. The incremental decode
defect remains open engineering debt for any future dense steady-state
decode work (suspect: single-token decode attention metadata or KV append
interplay with the triton SWA/full hybrid pool over renumbered stage-local
layers). Not papered over: the canonical evidence records that speculative
step-1 outputs of the distributed arm are not comparator-valid.


## Update (canonical run 1, producer 9d400ec, 2026-09-03)

The first canonical external-Coordinator request produced 7/8 exact
tokens vs the reference; the 8th diverged precisely when
prompt(26)+committed(7)=33 first exceeded the 32-row prefill chunk —
i.e. the same defective KV-extend path (chunk 2+ of a replay prefill
carries cached_len>0).  Mitigation (strategy-owned, within frozen
boundary budget): PREFILL_CHUNK raised to 64 rows (491,520 B < 1 MiB
wire budget) so canonical-scale replays stay single-chunk.  The
KV-extend defect itself remains open engineering debt; the anomaly
record and this update are retained unmodified in history.


## Update (gate correction pass, 2026-09-03)

The frozen SECONDARY comparator (max |logit_ref − logit_dist| < 0.25 over
the reference top-32 domain at steps 0/1/7) was physically captured
through the canonical distributed dense path during the gate-correction
pass (evidence arm, explicit-override producer — the frozen producer
44d6c94 retained no distributed logits; see
lifecycle/secondary-comparator.json).  Result: per-step max absdiff
0.25 / 0.50 / 0.515625; aggregate 0.515625 → the frozen threshold is
EXCEEDED.  NaN/Inf count 0.  Primary 8/8 greedy token equality
re-confirmed on the same replays.  The threshold was not loosened; the
corrected result composer reports the gate verdict as FAIL on this
single check and the maintainer adjudicates whether the drift joins
this anomaly record as a non-claim.  Plausible shared numerics: the
distributed chain computes attention through the triton SWA/full hybrid
with bf16 boundaries while the reference uses eager fp32-path
attention; exact token argmax survives, tail logits drift by up to
~0.5.
