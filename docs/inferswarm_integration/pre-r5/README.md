# Pre-R5 integration record

This record establishes `inferswarm-research` as the durable FreeToken
implementation line for R5A. The requested `inferswarm/research` spelling was
not used because the clone already had a legacy local branch named
`inferswarm`, and Git cannot store both refs. The equivalent hyphenated branch
avoids renaming or deleting that legacy ref.

This is a new implementation context. It does not replace, regenerate, or
extend the accepted R4 evidence claim.

## Frozen ancestry

- FreeToken `main`: `a05c26543f2d9a8cc2168fe789cdd4c92273378e`
- accepted R4 preservation head: `b2d72a36e79624028e74a2e7256f03546d4b8b5b`
- merge base: `9ef3651309fe4058672f2cc92069238dea06be1b`
- integration merge: `e1c68e91b59a09e5ddb113828f43f71bbe127898`
- merge parents, in order: R4 preservation head, then FreeToken `main`

The integration was an explicit `--no-ff` merge with no rebase or force
update. Both frozen heads are ancestors of the merge. The accepted producer
`e97f60b7b0120a72a7cf9926cf6a5c558782c9b2` and corrected evidence commit
`d5735c6b5075e835e7e8118922c44a7b0cf7439b` remain reachable through the first
parent. `docs/inferswarm_r4/` has no tree diff from the preservation head.

## Conflicts and adaptations

Two files conflicted:

- `python/freetoken/engine/engine.py`: retained InferSwarm runtime reporting
  (`auto_cpu_layer_ids`) and adopted upstream QSA dispatch, QSA dtype checks,
  host-table-aware pin-budget accounting, and the newer linear-attention
  capability model. Upstream deliberately removed `hybrid_linear_ok`; retaining
  the old field access caused five upstream tests to fail, so the integration
  explicitly follows the newer model in which linear layers bypass the
  attention backend.
- `python/freetoken/moe/offload_cache.py`: adopted upstream's padded FP8 scale
  bank byte calculation because it matches the new physical bank layout.

One pre-existing test import was corrected from
`benchmarks.inferswarm_phase0` to `inferswarm_phase0`, matching the path that
`tests/benchmarks/conftest.py` installs. This lets the full benchmark regression
directory collect without changing runtime behavior.

## Environment

Validation ran on 2026-09-01 on `inferswarm01` and, for the bounded physical
smoke, `inferswarm03`:

- Linux `6.12.105+deb13-amd64`
- Python 3.13.5; pytest 9.1.1
- torch 2.11.0+cu130; CUDA runtime 13.0; driver 610.57.04
- transformers 5.16.1; Triton 3.6.0
- two RTX 3060 GPUs on Node A; the accepted RTX 3060 Block-B identity on Node B

Native `_pinned_tensor` and `_cpu_moe` extensions were rebuilt in place from
the integrated source before the failure rerun and physical smoke.

## Validation

Focused R1-R4 research regressions:

```text
PYTHONPATH=.:python /home/zutfen/FreeToken/.venv/bin/python -m pytest -q \
  tests/research -m 'not slow and not needs_weights'
182 passed in 8.20s
```

InferSwarm benchmark regressions:

```text
PYTHONPATH=.:python /home/zutfen/FreeToken/.venv/bin/python -m pytest -q \
  tests/benchmarks -m 'not slow and not needs_weights'
551 passed in 18.61s
```

Applicable FreeToken engine/model/runtime tests were run without model
downloads. The initial broad run produced 1,468 passes, 12 skips, 51 marker
deselections, and 56 failures. After the merge adaptation and native-extension
rebuild, targeted reruns cleared 52 failures. Four upstream Qwen4 experimental
exact-equality tests remain (one PLE snapshot and three chunked-QSA cases); all
four reproduce unchanged on the exact frozen `main` SHA on this RTX 3060/CUDA
environment. They are therefore retained as upstream baseline deviations, not
classified as integration regressions.

The merge touched execution and model paths, so a bounded physical R4
diagnostic smoke was required. A temporary plan for the integration merge ran
W2 across `inferswarm01` and `inferswarm03` without changing accepted R4 files:

- generated tokens exact: true
- selected float32 logit hashes exact: true
- all boundary checksums match: true
- 32 boundaries; 696,320 semantic activation bytes
- Node B fallbacks, graph recaptures, host expert fetches, resident source
  accesses, unexplained persistent host mirror bytes, and steady model-state
  movement: all zero
- Node B staging-process `VmSwap`: 0 KiB; no swap reliance

The bounded smoke summary is in `smoke-summary.json`. Its temporary full
artifacts had SHA-256 values:

- plan: `337f41bf7b9b0d747cff12ef108f946746875684776f6c1c9be05b64db058f07`
- Node-A diagnostic result: `2397fe38ef005feba804b22b4b5d69ac0c7488f09f91f4b767703afa409f4ec6`
- Node-B final report: `96ac11e5c1a2da21d79cd8c0115244350bf60687a387334ecb1d15c7c833faef`

The older R4 performance and capacity results remain historical and specific
to their accepted producer/runtime context. This integration smoke checks
correctness and invariants only; it does not transfer or renew the old
performance claim.

