# R6 Regression Summary

Canonical producer: 44d6c94 (physical run); gate-correction code head:
5aaa35c (planes/kv-identity/capture-arm + fail-closed reader + composer
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
  gate-contract + composer suites: **121 passed, 1 skipped, 0 failed**
  (skip: selective-reader tensor read needs safetensors+torch —
  exercised on 01 above; 00 is torch-free by design)
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

## Post-failure single-GPU control extension — review fixes (head c06762f + this pass)

Code-review findings against `single_gpu_control.py`/`stage_runtime.py`
(the bounded-host single-GPU diagnostic added at c06762f) corrected before
any canonical `inferswarm04` control is run:

- Fixed a fatal real-runtime bug: `_StageModules.load_state_dict` defensively
  copied its input (`state = dict(state)`); since `BaseOP.load_state_dict`
  pops keys from whatever dict it is given, the copy drained instead of the
  caller's dict, so `_selective_load`'s `if loaded: raise` would have fired
  on every real (non-empty) load. `_StageModules` is now module-level
  (directly importable) and mutates the caller's dict in place, as intended.
- Outcome A/B interpretation is now gated on validity: exact 8/8
  committed-token equality and zero NaN/Inf are required before any
  interpretation runs; a violation is reported as
  `SINGLE_GPU_CONTROL_INVALID_NUMERICS` rather than silently mis-selecting
  an outcome.
- Outcome A is no longer machine-declared from Transformers-vs-single drift
  alone (the frozen methodology requires a second, unfrozen condition too).
  The runner now reports `OUTCOME_A_CANDIDATE_REQUIRES_MAINTAINER_ADJUDICATION`
  plus the full raw single-vs-distributed comparison, deferring the second
  condition to explicit adjudication instead of inventing a threshold after
  the fact.
- `persistent_host_model_bytes`, `cpu_owned_decoder_layers`,
  `cpu_weight_offload`, `resident_only`, and
  `unexplained_persistent_host_mirror_bytes` are now mechanically derived
  from the realized module `state_dict()`'s tensor devices
  (`_host_resident_bytes`/`_cpu_owned_decoder_layers`), not asserted
  constants.
- Hardware preflight is now mechanically enforced: `MemTotal < 23,814,700,640`
  is machine-checked from `/proc/meminfo`, and a
  `docs/inferswarm_r6/PREFLIGHT-INFERSWARM04.json` frozen-identity amendment
  (`gpu_name`/`gpu_uuid`/`pci_bus_id`/`memory_total_mib`) is required to
  exist and match the queried GPU before any weight is touched. That file
  intentionally does not exist yet (inferswarm04 is not provisioned), so the
  preflight fails closed by design until it is committed.
- Runtime failures after model realization now retain a terminal
  status/type/message plus best-effort partial loader evidence
  (`SINGLE_GPU_CONTROL_FAILED`) instead of leaving the report at `RUNNING`.

New tests:

- `tests/research/test_r6_single_gpu_control_gate.py` (torch-free; runs on
  the torch-free coordinator profile): **14 passed** — preflight helpers,
  fail-closed frozen-identity matching, `InvalidNumericalEvidenceError`,
  `_best_effort_partial_report`, and structural regressions pinning the
  removed placeholder classification and the validity-gate ordering.
- `tests/research/test_r6_single_gpu_control.py`: 5 new torch-gated tests
  (`_StageModules.load_state_dict` full end-to-end consumption/leftover-key
  rejection using `BaseOP`-backed stand-ins, and
  `_host_resident_bytes`/`_cpu_owned_decoder_layers` residency-derivation
  cases). Not executed in this pass (no torch in this environment); needs a
  compute-node run (01/03) before the `inferswarm04` canonical control, per
  the review's validation-gap finding.

No canonical `inferswarm04` run was performed in this pass — this is a
code/evidence correction only, matching "do not run the canonical control
yet" from the review.
