# Regression qualification

All commands used the merged tree and `PYTHONPATH=.:python:benchmarks` where
the benchmark package import required it.

| Scope | Result | Elapsed |
| --- | --- | ---: |
| focused changed surfaces | 26 passed | retained as focused precheck |
| `tests/research` | 199 passed | 9.06 s |
| InferSwarm benchmark tests | 563 passed | 18.88 s |
| server tests | 583 passed, 1 warning | 12.67 s |
| applicable broad upstream set, excluding `slow` and `needs_weights` | 1,214 passed, 31 skipped, 58 deselected, 6 failed | 655.31 s |

The first research invocation omitted `benchmarks` from `PYTHONPATH` and
failed during collection. Rerunning with the repository's benchmark import
root produced 199/199; this was an invocation artifact, not a product failure.

## Upstream-baseline reproduction

The six broad-run failures were investigated on a separately rebuilt, clean,
detached `main@6eca2d7d2b8576c7ad0ba62853df9f618cba929f` worktree:

- `tests/kernels/test_pinned_tensor.py::test_host_device_ptr_is_identity_under_uva`
  failed identically on the integration and upstream trees;
- one Qwen4 PLE exact-equality case failed identically;
- three Qwen4 QSA chunk/exact-equality cases failed identically;
- the `swiglu_clamp` failure followed the pinned-memory CUDA failure in the
  combined run, but passed in isolation on both trees.

Thus five failures are upstream/environment baseline behavior and the sixth is
poisoned-CUDA-state collateral. The relevant GLM-5.3 config, CPU
`swiglu_clamp`, Qwen4 PLE disk, and N0 model-block focused tests passed after
the fresh build. No threshold was weakened and no test was removed.

