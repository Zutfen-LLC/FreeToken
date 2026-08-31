# InferSwarm #48 resident-only decode evidence

This fixture starts from the canonical N0 selective block loader, establishes any
required prefill while host sources still exist, fills a complete block-scoped NVFP4
GPU expert cache, validates exact residency, detaches every cache/source owner, and
runs at least two real decode computations against the frozen same-backend reference.

It proves only steady-state fully resident decode for the pinned Qwen3.6/NVFP4
selective-block path. Post-detach prefill is intentionally rejected and is not claimed.
It does not implement N1 split execution, networking, planning, or a public strategy API.

Run one block per process on hardware with enough memory for that block's complete
expert cache:

```bash
PYTHONPATH=python python -m benchmarks.inferswarm_p48.run_resident_block \
  --model /path/to/Qwen3.6-35B-A3B-NVFP4 \
  --revision 491c2f1ea524c639598bf8fa787a93fed5a6fbce \
  --plan /path/to/n0-plan.json \
  --fixture /path/to/n0-reference-fixtures.pt \
  --block a \
  --repetitions 4 \
  --out docs/inferswarm_p48/block-a.json
```

Repeat with `--block b`. The command exits nonzero on correctness deviation, live
host-bank weak references, a source-access/rematerialization sentinel, or nonzero
post-replay host source bytes. Reports use the versioned
`inferswarm.p48.accelerator-residency/1` schema and receive a sibling SHA-256 file.

No hardware result is committed by the implementation alone. A report may be treated
as physical #48 evidence only when the command completed on the pinned model revision
and its top-level `passed` field is true.
