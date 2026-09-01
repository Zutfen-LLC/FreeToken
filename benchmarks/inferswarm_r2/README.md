# InferSwarm R2 local split execution

This research namespace realizes exactly one frozen, local, two-block plan. It is
doctrine-shaped and API-unfrozen: it is not a planner, worker framework, network
protocol, or public strategy ABI.

The coordinator spawns one process per planned Compute Unit. Each process validates
the same plan digest and stable GPU UUID, calls the R1 realizer for only its assigned
state, captures one block-local decode graph, and owns only its block-local mutable
state. A registered pinned-host shared buffer carries the adapter-defined boundary;
small versioned control messages use multiprocessing pipes.

Typical retained sequence:

```bash
python -m benchmarks.inferswarm_r2.preflight_transport ...
python -m benchmarks.inferswarm_r2.freeze_plan ...
python -m benchmarks.inferswarm_r2.run_correctness ...
python -m benchmarks.inferswarm_r2.run_benchmark ...
python -m benchmarks.inferswarm_r2.compose_result \
  --evidence-dir docs/inferswarm_r2 --out docs/inferswarm_r2/result.json
```

`run_correctness` intentionally exits 2 when any canonical correctness gate fails,
after writing the checksummed evidence. `run_benchmark` separates diagnostics from
timing and retains one warmup plus every timed repetition.

The retired N1 experiment informed only process isolation, the hidden/residual pair
boundary, session reset discipline, and useful checkpoint locations. R2 does not
reuse N1 evidence, does not copy its socket bulk-data path, and is independently
implemented and rerun from accepted R1 commit
`6a242a34083c3080aa6d8f92625a6be4a0d124db`.
