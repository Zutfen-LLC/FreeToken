# Native and generated-artifact rebuild record

Qualification did not reuse compiled artifacts from another source tree.

On both `inferswarm01` and `inferswarm03`, from exact clean producer
`0f44f86f91db3c7a82f6d380c074a3731535f3a9`, the extension build was run from
the integrated worktree with bounded parallelism (`MAX_JOBS=4`) using
`python setup.py build_ext --inplace` in the qualification environment.

Rebuilt outputs:

- `freetoken._pinned_tensor`;
- `freetoken._cpu_moe`;
- `freetoken._ple_store` (new in the integrated upstream delta).

The clean `main@6eca2d7...` baseline used for failure classification was also
rebuilt independently before reproduction. No canonical test or physical
claim was based on a stale `_cpu_moe`, `_pinned_tensor`, `_ple_store`, or
kernel-cache artifact from another source revision.

The integrated kernel and model code remained source-driven at runtime; no
additional checked-in generated artifact required regeneration.

