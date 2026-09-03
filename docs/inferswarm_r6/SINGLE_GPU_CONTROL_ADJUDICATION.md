# R6 Single-GPU Control — Offline Common-Domain Residual Adjudication

Status: POST-RESULT ADJUDICATION (descriptive analysis; not a methodology
amendment, not a gate).

This document re-derives, purely offline from already-retained evidence, the
three R6 logit comparisons over one common frozen comparison domain — the
exact Transformers reference top-32 vocabulary indices per declared step —
and inspects the residual structure so the maintainer can adjudicate the
previously unfrozen second condition of Outcome A.

Explicit statements:

* No new physical execution occurred. No GemmaDenseStage launch, no CUDA
  initialization, no Transformers run, no distributed or single-GPU FreeToken
  rerun, no logits regeneration.
* No comparator threshold changed. The frozen 0.25 secondary threshold is
  untouched; no new threshold is defined for the single→distributed
  comparison (descriptive ratios below are explicitly not pass criteria).
* No historical R6 result changed. `docs/inferswarm_r6/result.json` is
  byte-identical to its accepted form (sha256
  `b19e746c831484cb7ba07088d8959297e1d8641125215d6fe63f27c7255817e3`,
  verified before and after this analysis).
* All numbers below are re-derived from already-retained logits binaries and
  the retained reference artifact.

## Canonical identities

| Role | Identity |
|---|---|
| Historical R6 physical producer | `44d6c94e4fd2ee967451cc959f930883ca3f4a25` |
| Historical distributed evidence-arm producer | `3018acf721e897355e362f965f510471bf88d64c` |
| Single-GPU diagnostic producer (this analysis head) | `b11f7739b1377fa6109e8e3154b1188e11a4ffec` |

Analysis source: FreeToken branch `inferswarm-r6` at
`b11f7739b1377fa6109e8e3154b1188e11a4ffec` (clean tree), analysis script
`analysis/r6_common_domain_adjudication.py` (pure-stdlib, CPU-only, no
torch/numpy import), tests
`tests/research/test_r6_common_domain_adjudication.py` (16 passed on the
torch-free coordinator).

## Retained inputs (verified)

| Input | Path | SHA256 |
|---|---|---|
| Transformers reference (top-32 indices/values, steps 0/1/7) | `docs/inferswarm_r6/reference-generation.json` | `361192cd7972fd3b6b5561af6f9978b35f67bb4dacf0beaf9a60445d77bd2016` |
| Distributed FreeToken full-vocab logits | `docs/inferswarm_r6/lifecycle/distributed-logits-0-1-7.f32.bin` | `7b30239b2d1a7e940e5e5c7946ea23ceb828f41fb32ea026ea8ded68801b9aeb` |
| Single-GPU FreeToken full-vocab logits (Amendment-003 run, copied byte-identical from inferswarm04) | `docs/inferswarm_r6/lifecycle/single-gpu-logits-0-1-7.f32.bin` | `d94e0cb572408e586ceb7d79ed3458fa738eceac17d52c7efc3fbbd30d0d71ce` |

Binary metadata (both logits files): float32 little-endian, 3,145,728 bytes
= 3 rows × 262,144 vocab entries, rows ordered steps 0, 1, 7 — matching the
retained `secondary-comparator.json` full_vocab_binary contract and the
Amendment-003 `result.json` captured_logits record (producer
`b11f773…`, status `SINGLE_GPU_CONTROL_COMPLETE`, NaN/Inf count 0).

Integrity verification performed before analysis:

* inferswarm04 `/srv/inferswarm/state/r6/single-gpu-control/requal-inplace-softcap-b11f773/SHA256SUMS.txt`
  verifies clean (result.json, runner.log, logits bin);
* the three earlier evidence dirs (`canonical-9aa2713`,
  `requal-9aa2713-pathfix`, `requal-finalrow-fp32-b402ef5`) verify clean;
* historical `result.json` sha256 unchanged before and after.

## Common domain

For each step s ∈ {0, 1, 7}, the 32 vocabulary indices are taken ONLY from
`reference-generation.json` `step_top32_logits[s].top_indices`. T = reference
values at those indices; S = single-GPU row entries; D = distributed row
entries, at the SAME indices. No substitution (S top-32, D top-32, union
top-k, full vocabulary, generated-token-only) was used.

Common indices by step (frozen reference order, descending reference value):

* step 0: 818, 147505, 93480, 109090, 44281, 159416, 25694, 11146, 1018,
  32899, 1437, 10450, 669, 116499, 23244, 92903, 2205, 69073, 14947, 902,
  91005, 132473, 236829, 22515, 140505, 126515, 79629, 19892, 41866,
  236769, 115218, 165684
* step 1: 6073, 21530, 18188, 2870, 73686, 105405, 506, 119331, 5213,
  236789, 9734, 44623, 2803, 68191, 78104, 4532, 8520, 621, 3413, 507,
  116355, 1156, 3670, 62633, 4518, 7963, 15841, 568, 17748, 16813, 4185,
  1791
* step 7: 532, 236764, 1402, 1079, 238229, 3884, 237842, 237206, 941,
  236746, 17996, 578, 1212, 236772, 236744, 236762, 568, 990, 1008, 624,
  7519, 525, 237032, 22136, 236786, 653, 60730, 688, 90931, 77891, 236774,
  169403

Full per-index residual table (96 rows): machine-readable companion
`docs/inferswarm_r6/lifecycle/common-domain-residuals.csv` and
`docs/inferswarm_r6/lifecycle/single-vs-distributed-common-domain.json`.

## Reproduction gate (hard precondition)

The re-derivation reproduced the retained historical values EXACTLY before
any interpretation (the analysis script asserts equality and aborts
otherwise):

```
T→S  step 0/1/7: 0.234375 / 0.34375  / 0.34375   (retained: identical)
T→D  step 0/1/7: 0.25      / 0.50    / 0.515625 (retained: identical)
```

New same-domain measurement:

```
S→D  step 0/1/7: 0.25      / 0.46875 / 0.375    aggregate 0.46875
```

(The earlier full-vocabulary S→D figures 0.4375/0.5/0.5 were superseded by
this common-domain measurement for adjudication purposes; both remain
retained. The common-domain S→D maxima are smaller or equal at every step,
consistent with the full-vocabulary maxima being attained outside the
reference top-32 at steps 0 and 1.)

## Per-step statistics

All values exact doubles over the 32 frozen entries (see JSON for full
precision).

### step 0

| metric | T→S (S−T) | T→D (D−T) | S→D (D−S) |
|---|---|---|---|
| max abs | 0.234375 | 0.25 | 0.25 |
| argmax index | 1437 | 109090 | 11146 |
| T / S / D at argmax | 3.890625 / 4.125 / 4.03125 | 9.3125 / 9.1875 / 9.0625 | 4.5 / 4.5625 / 4.3125 |
| signed residual at max | +0.234375 | −0.25 | −0.25 |
| mean signed | +0.0357 | −0.0037 | −0.0394 |
| mean abs | 0.0904 | 0.0733 | 0.0741 |
| RMS | 0.1121 | 0.1030 | 0.0951 |
| median abs | 0.09375 | 0.0546875 | 0.0625 |
| p90 abs | 0.1709 | 0.1844 | 0.1396 |
| min signed | −0.21875 | −0.25 | −0.25 |
| max signed | +0.234375 | +0.21875 | +0.140625 |

### step 1

| metric | T→S | T→D | S→D |
|---|---|---|---|
| max abs | 0.34375 | 0.5 | 0.46875 |
| argmax index | 68191 | 1156 | 1156 |
| T / S / D at argmax | −4.0 / −4.34375 / −4.3125 | −5.09375 / −5.125 / −5.59375 | −5.09375 / −5.125 / −5.59375 |
| signed residual at max | −0.34375 | −0.5 | −0.46875 |
| mean signed | +0.0399 | −0.1173 | −0.1572 |
| mean abs | 0.1019 | 0.1290 | 0.1621 |
| RMS | 0.1258 | 0.1710 | 0.1903 |
| median abs | 0.0801 | 0.1016 | 0.1484 |
| p90 abs | 0.1859 | 0.2652 | 0.3063 |
| min signed | −0.34375 | −0.5 | −0.46875 |
| max signed | +0.21875 | +0.125 | +0.03125 |

### step 7

| metric | T→S | T→D | S→D |
|---|---|---|---|
| max abs | 0.34375 | 0.515625 | 0.375 |
| argmax index | 688 | 688 | 236746 |
| T / S / D at argmax | −1.2265625 / −0.8828125 / −0.7109375 | same | 1.484375 / 1.34375 / 1.71875 |
| signed residual at max | +0.34375 | +0.515625 | +0.375 |
| mean signed | +0.0542 | +0.0351 | −0.0190 |
| mean abs | 0.0887 | 0.1333 | 0.1011 |
| RMS | 0.1167 | 0.1672 | 0.1305 |
| median abs | 0.0906 | 0.1543 | 0.0796 |
| p90 abs | 0.1555 | 0.2219 | 0.1859 |
| min signed | −0.15625 | −0.1875 | −0.296875 |
| max signed | +0.34375 | +0.515625 | +0.375 |

## Residual-direction analysis: A = S−T vs B = D−T

| step | Pearson | cosine | dot | same sign | opposite sign | \|D−T\|>\|S−T\| | \|D−T\|<\|S−T\| |
|---|---|---|---|---|---|---|---|
| 0 | 0.658 | 0.612 | 0.2260 | 0.625 | 0.375 | 0.375 | 0.531 |
| 1 | 0.614 | 0.206 | 0.1416 | 0.3125 | 0.6875 | 0.500 | 0.500 |
| 7 | 0.614 | 0.629 | 0.3928 | 0.750 | 0.250 | 0.719 | 0.281 |

Decomposition identity `D−T = (S−T) + (D−S)` machine-checked at every index
in exact float32 arithmetic: maximum reconstruction error **0.0** at all
three steps (all residuals are exact multiples of the float32 ulp lattice at
these magnitudes, so the identity holds exactly).

## Relative contribution ratios (descriptive only — NOT a pass criterion)

`|D−S| / |D−T|` per step (zero denominators counted, never divided; steps
0/1 had 4 zero-|D−T| entries each, step 7 had none):

| step | median | mean | max | q1 | q3 | p90 |
|---|---|---|---|---|---|---|
| 0 | 1.000 | 2.141 | 10.75 | 0.455 | 2.125 | 5.60 |
| 1 | 1.323 | 1.743 | 6.00 | 0.805 | 2.417 | 4.00 |
| 7 | 0.827 | 1.430 | 9.00 | 0.444 | 1.028 | 1.88 |

`|S−T| / |D−T|` per step:

| step | median | mean | max |
|---|---|---|---|
| 0 | 1.029 | 2.265 | 9.75 |
| 1 | 0.559 | 1.133 | 5.00 |
| 7 | 0.633 | 1.399 | 8.51 |

Reading: the distribution-only residual |D−S| is of the SAME ORDER as the
total distributed error |D−T| (medians 0.83–1.32), and at the median index
the single-GPU baseline contributes only ~0.56–1.03 of |D−T|. The
distribution residual is not a small perturbation on top of a dominant
shared baseline.

## Generated-token positions

All three greedy tokens are inside the frozen reference top-32 at every
declared step, and all three executors agree exactly on argmax (8/8 token
equality re-confirmed):

| step | token | T | S | D | S−T | D−T | D−S |
|---|---|---|---|---|---|---|---|
| 0 | 818 | 19.25 | 19.25 | 19.375 | 0.0 | +0.125 | +0.125 |
| 1 | 6073 | 20.0 | 20.0 | 19.875 | 0.0 | −0.125 | −0.125 |
| 7 | 532 | 23.5 | 23.5 | 23.625 | 0.0 | +0.125 | +0.125 |

At the greedy positions the single-GPU executor is bit-exact against
Transformers and the distributed executor differs by only ±0.125 — the large
residuals live elsewhere in the domain.

## Rank-order diagnostics (within the 32-index set only)

| step | top-1 (T/S/D) | Spearman T~S | Spearman T~D | Spearman S~D |
|---|---|---|---|---|
| 0 | 818 / 818 / 818 | 0.99899 | 0.99881 | 0.99890 |
| 1 | 6073 / 6073 / 6073 | 0.99541 | 0.98413 | 0.98312 |
| 7 | 532 / 532 / 532 | 0.99423 | 0.99404 | 0.99267 |

Ordering within the domain is nearly preserved across all three executors
(no inference about global top-k agreement is drawn from this subset).

## Answers to the decision questions (measurements only)

**Q1 — Is same-domain S→D large relative to T→S?** Yes. S→D maxima
(0.25 / 0.46875 / 0.375, aggregate 0.46875) exceed the corresponding T→S
maxima (0.234375 / 0.34375 / 0.34375, aggregate 0.34375) at every step. The
distribution-only residual is not small relative to the baseline residual;
it is comparable-to-larger.

**Q2 — Are S−T and D−T strongly aligned?** Moderately at best, and
inconsistently. Pearson is 0.61–0.66 at all steps, but cosine collapses to
0.21 at step 1; sign agreement ranges from 0.31 (step 1, mostly OPPOSITE
sign) to 0.75 (step 7). There is shared structure, but not the strong,
persistent directional alignment that "D−T ≈ S−T" would require.

**Q3 — Amplify, cancel, or new direction?** All three behaviors appear,
step-dependent: at step 7 distribution mostly amplifies the existing
baseline residual (72% of indices have |D−T| > |S−T|, same direction at
index 688 where T→D = 0.515625 = T→S 0.34375 + D−S 0.171875); at step 1 it
introduces differently directed error (69% opposite sign, and the historical
max at index 1156 is almost entirely distribution-owned: S−T = −0.03125 vs
D−S = −0.46875); at step 0 it is mixed (53% of indices have |D−T| <
|S−T|, i.e. partial cancellation, yet the S→D max 0.25 at index 11146 is
fully distribution-owned with S−T only +0.0625).

**Q4 — At historical max-difference indices, how much of D−T is baseline vs
additional?**

* step 0, index 109090 (T→D max 0.25): baseline S−T = −0.125 (half),
  additional D−S = −0.125 (half).
* step 1, index 1156 (T→D max 0.5): baseline S−T = −0.03125 (6%),
  additional D−S = −0.46875 (94%) — almost entirely distribution-owned.
* step 7, index 688 (T→D max 0.515625): baseline S−T = +0.34375 (67%),
  additional D−S = +0.171875 (33%) — mostly baseline, same direction.

**Q5 — Which qualitative claim does the evidence support?** The evidence
does NOT support "distributed ≈ single-GPU baseline + comparatively small
residual": the distribution-only residual is the same order of magnitude as
the baseline residual (median |D−S|/|D−T| 0.83–1.32), exceeds T→S at every
step, is weakly/not aligned in direction at step 1 (cosine 0.21, 69%
opposite sign), and at the step-1 historical max it supplies ~94% of the
error. The evidence instead supports: FreeToken has a baseline numerical
discrepancy from Transformers (single-GPU T→S aggregate 0.34375 ≥ frozen
0.25, an honest violation of the original comparator) **and** distributed
execution introduces additional material numerical drift of its own.

## Maintainer disposition labels (evidence presentation — NOT machine-selected)

`OUTCOME_A_SUPPORTED` would require the distributed residual relative to
single to be comparatively small and aligned. Measured: comparable-to-larger
(agg 0.46875 vs 0.34375), alignment moderate-to-weak and step-dependent
(cosine 0.21–0.63; sign agreement 0.31–0.75), and the largest historical
T→D error at step 1 is ~94% distribution-owned. The evidence does not fit
this label.

`MIXED_BASELINE_AND_DISTRIBUTION_DRIFT` fits the measurements: the
undistributed FreeToken executor already violates the original Transformers
comparator (0.34375 vs 0.25), and distribution adds a material additional
residual rather than merely reproducing the single-GPU baseline. The
original R6 secondary-comparator failure therefore cannot be explained away
as merely FreeToken-vs-Transformers baseline behavior.

**Qualitative maintainer recommendation (not a frozen machine gate):
`MIXED_BASELINE_AND_DISTRIBUTION_DRIFT`.**

This recommendation introduces no new threshold; every supporting number
above is a direct measurement over the frozen common domain.

## Artifacts

* `docs/inferswarm_r6/lifecycle/single-vs-distributed-common-domain.json` —
  machine-readable companion (sha256 recorded in the repo commit;
  determinism: two independent runs produced byte-identical outputs).
* `docs/inferswarm_r6/lifecycle/common-domain-residuals.csv` — 96-row
  per-index residual table.
* `docs/inferswarm_r6/lifecycle/single-gpu-logits-0-1-7.f32.bin` —
  byte-identical copy of the retained inferswarm04 Amendment-003 capture
  (sha256 `d94e0cb5…d71ce`), retained in-repo so the adjudication is
  reproducible offline from the repo alone.
* `analysis/r6_common_domain_adjudication.py` + unit tests — the
  re-derivation tooling.

Historical `docs/inferswarm_r6/result.json` was NOT modified.
