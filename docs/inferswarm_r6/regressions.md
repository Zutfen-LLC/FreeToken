# R6 Regression Summary

Producer: 44d6c94 (canonical-run producer); suites re-run at e2f54e9
(same code under test for all r6 modules; e2f54e9 adds only docs +
a repo-root conftest that does not alter any tested module).

## inferswarm01 (compute node, torch venv)

- tests/benchmarks: 563 passed, 0 failed (not slow / not needs_weights).
- tests/research: 238 passed, 0 failed (same markers), plus
  tests/research/test_r5a_preflight.py: 4 passed — run separately because
  that module still uses the legacy top-level `inferswarm_r5a` import
  (PYTHONPATH=python:benchmarks), which is mutually exclusive in one
  interpreter with the `benchmarks.inferswarm_*` package imports used by
  the other research tests (documented historical pitfall; both styles
  pass in their own invocation).

## inferswarm00 (CPU-only coordinator, torch-free venv)

- tests/research control-plane selection (xc wire/coordinator/realization
  authority, planner, r1): 72 passed, 0 failed.
- Out of scope on this host by design: modules importing torch
  (n0_model_block, r2 local-split, r4/r5a/r5b runtime-adjacent tests)
  — those ran on 01 above.

## Gemma/FreeToken model tests

- tests/models/test_gemma4_* included in the 01 runs above (part of
  tests/benchmarks suite collection set); no Gemma test failures.

No thresholds were adjusted after observing any result.
