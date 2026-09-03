# R6 Regression Summary

Canonical producer: 44d6c94 (physical run); gate-correction code head:
59b2e52 (planes/kv-identity/capture-arm + fail-closed reader + composer
rewrite + tests + docs).  The correction commits touch the R6 dense
adapter modules, the torch-free census reader, the evidence composer,
and docs only; no generic planner/epoch/wire semantic was altered
(r4_wire framing untouched).

## inferswarm01 (compute node, torch venv, head 59b2e52)

- tests/research (benchmarks.* import style, not slow / not
  needs_weights, r5a preflight excluded): **265 passed, 0 failed**
  (includes the 17 focused R6 gate-contract tests and the 10 composer
  negative-control/positive-fixture tests; 1 safetensors-reader test
  included here — the venv has safetensors).
- tests/research/test_r5a_preflight.py (legacy top-level import style,
  own interpreter): **4 passed, 0 failed** (historical dual-import
  split, documented in prior gates).
- tests/benchmarks (not slow / not needs_weights): **563 passed,
  0 failed**.

## inferswarm00 (CPU-only coordinator, torch-free venv, head 59b2e52)

- Control-plane selection (xc wire/coordinator/realization
  authority/result identity/seam, planner, r1) + focused R6
  gate-contract + composer suites: **122 passed, 0 failed**
  (composer negative controls exercise the retained canonical evidence
  in a tmp copy; coordinator purity unbroken).

## Gate-correction specific runs

- Corrected composer against retained canonical evidence:
  **27/28 checks pass**; the single failing check is
  `secondary_logit_comparator_threshold` (measured aggregate absdiff
  0.515625 vs frozen 0.25; honest FAIL verdict retained in result.json).
- Composer negative controls (missing stage-3 report, fetched-byte
  mismatch, host mirror, unexpected key, missing comparator, threshold
  violation, NaN/Inf, producer substitution): all fail the composer as
  designed; positive fixture reproduces 28/28 PASS with distinct
  provenance identities.

No thresholds were adjusted after observing any result.  The one test
failure observed during this pass (selective reader single-shard
fail-closed gap) was fixed in code (59b2e52), not waived.
