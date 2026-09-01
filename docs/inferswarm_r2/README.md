# InferSwarm R2 physical evidence

This directory contains retained physical evidence for
[InferSwarm #51](https://github.com/Zutfen-LLC/inferswarm/issues/51), produced from
the exact accepted R1 merge commit
`6a242a34083c3080aa6d8f92625a6be4a0d124db` on branch
`poc/r2-local-split-execution`.

## Review status

`R2_LOCAL_SPLIT_EXECUTION_PASS` is **not declared**. The execution proof produced
the exact 32-token sequence for W1-W4, but W1, W3, and W4 selected logits exceeded
the existing canonical `rtol=2e-3, atol=2e-3` threshold. Maximum absolute deviation
was 1.25; NaN and Inf counts were zero. The committed result therefore says
`R2_LOCAL_SPLIT_EXECUTION_BLOCKED_CORRECTNESS` pending diagnosis and review.

The separate measured placement assessment is `PERFORMANCE_NEGATIVE`: median split
decode throughput was 0.9122 times the matched baseline across workload medians.
This statement applies only to this exact plan, hardware, software, and topology.
The resident split materially reduced TTFT relative to the single-GPU offload
baseline, especially for long prompts.

## Frozen plan and resource roles

Plan digest:
`sha256:6128dd6705d6d692df3d5fc11cc130dba5c010cfff40c0e4c5ec7c19e1b78ff0`.

- GPU-A, UUID `GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099`, is required because it
  owns embeddings/input, layers `[0,19)`, their routed state, their mutable state,
  and the Block A decode graph.
- GPU-B, UUID `GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55`, is required because it
  owns layers `[19,40)`, their routed and mutable state, final normalization/LM head,
  and the Block B decode graph. Together the two 12 GiB VRAM resources hold the
  complete resident state that the matched single-GPU baseline must offload.
- System RAM provides bounded checkpoint staging and one 524,288-byte registered
  pinned-host activation buffer. Both expert staging banks are released after final
  residency. The baseline uses RAM as its explicitly configured offload source.
- Checkpoint backing is the immutable local model revision source.
- The GPU-A → RAM → GPU-B link carries only the strategy-defined two-plane bf16
  activation boundary plus tiny control/token results. It carries 8,192 bytes per
  decode step and up to 524,288 bytes per 64-row prefill chunk. Steady-state model
  state movement is zero.

Both selected RTX 3060 devices are on the same host/NUMA node across a `PHB` path.
Peer access was unavailable, so transport preflight selected ordinary supported
registered pinned-host staging. Under-load PCIe state was Gen3 x16 on both devices;
idle discovery records may show the devices' power-managed Gen1 state.

## Evidence index

- `capacity-preflight.json`: component capacity calculation and deterministic N0
  split retention.
- `transport-preflight.json`: stable resource identity, topology, peer probe, and
  exact-size transfer measurements.
- `frozen-plan.json`: versioned candidate plan; the sibling SHA file protects it.
- `baseline-config.json`: hashed matched baseline configuration.
- `correctness.json`: diagnostic W1-W4, boundary checksums, session isolation,
  graph counters, ownership, residency, and the correctness blocker.
- `benchmark.json`: raw warmup/five-repetition A/B timings and summaries.
- `result.json`: compact review-facing synthesis of all retained artifacts.

Each retained JSON has a sibling `.sha256`. Startup/materialization is reported
separately from inference. No retired N1 result is included as R2 physical evidence.
