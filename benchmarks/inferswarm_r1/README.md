# InferSwarm R1 frozen-plan realization

This research runner loads one already-frozen plan, constructs the explicitly known
physical environment, calls the internal plan realizer, runs repeated correctness
decode, and writes intended-versus-observed evidence. Lifecycle choices—including
release of host staging after validated accelerator residency—belong to the realizer
and pinned adapter, not the runner.

Freeze version 1 before retaining a physical run:

```bash
PYTHONPATH=python python -m benchmarks.inferswarm_r1.freeze_plan \
  --n0-plan /path/to/n0-block-plan.json \
  --out docs/inferswarm_r1/frozen-plan.json
```

Then realize that exact file:

```bash
PYTHONPATH=python python -m benchmarks.inferswarm_r1.run_frozen_plan \
  --plan docs/inferswarm_r1/frozen-plan.json \
  --model /path/to/Qwen3.6-35B-A3B-NVFP4 \
  --fixture /path/to/n0-reference-fixtures.pt \
  --out docs/inferswarm_r1/result.json
```

The manifest and adapter are temporary research structures. They are not a public
`ExecutionPlan` schema or model-strategy ABI and do not perform automatic placement.
