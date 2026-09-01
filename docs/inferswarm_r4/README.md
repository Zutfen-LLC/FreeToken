# InferSwarm R4 — first measured physical two-node boundary over ordinary 1 GbE

**Result: `R4_MULTI_NODE_BOUNDARY_PASS` is claimed, pending gate review.**
**1-GbE arm disposition: `R4_1GBE_PRIMITIVE_CAPACITY_VIABLE`.**

InferSwarm issue #57 is the authoritative specification. This directory retains
the complete canonical evidence. The implementation producer for all physical
evidence is `f8e743bf05f1766336f17fb8243c531633efce65` (branch
`poc/r4-multi-node-boundary-1gbe`, based exactly on the accepted R3 merge
`2ac72d547b2a24a3672d1b83268865db5490084d`).

## What was proven

The accepted R2 contiguous Qwen split ([0,19) / [19,40)) executed with both
blocks resident, captured, and backend-native on **two physical nodes** —
Block A on `inferswarm01` (GPU `GPU-1fc28f83`@02:00.0), Block B on
`inferswarm03` (CPU-attached x16 GPU `GPU-e1f2f90c`@01:00.0) — joined by one
persistent ordinary-TCP connection over negotiated **1 GbE full duplex,
MTU 1500, direct LAN** (10.0.0.141 eno1 ↔ 10.0.0.219 enp5s0). The only
architectural variable changed from R2 is node/network locality.

### Correctness (diagnostic arm, W2 + W4, frozen R2-v2 methodology)

- generated token sequences byte-exact vs the frozen reference, both workloads;
- selected logits (steps 0/1/15/31) float32-hash-identical to
  `docs/inferswarm_r2/reference-v2-session-a.json` (the frozen comparator is
  hash identity; deviations are 0.0 by construction on identity);
- NaN/Inf counts 0;
- producer == receiver sha256 on every boundary;
- exact boundary geometry: 8,192 B decode / 524,288 B 64-row prefill.

### Runtime/residency invariants (both nodes, both arms)

`fallbacks=0 recaptures=0 host_expert_fetches=0 resident_source_accesses=0
unexplained_persistent_host_mirror_bytes=0 steady_model_state_movement_bytes=0`;
one capture, 62 decode replays per node per arm; #53 RELEASE lifecycle
observed on Node B: staged 9,545,711,616 B routed host source, released to 0,
`resident_only=true`, `pswpin/pswpout=0` throughout.

### Network measurements (raw/)

- iperf3: A→B 933.9 Mb/s, B→A 941.5 Mb/s, bidirectional ~930 Mb/s, 0 retransmits;
- ping (100 @ 50 ms): min/avg/max = 0.131/0.378/0.511 ms, 0% loss;
- transport-only framed request/response (200 reps, TCP_NODELAY=1):
  8,192 B p50 0.528 ms (124.1 Mb/s effective), 524,288 B p50 9.67 ms
  (433.7 Mb/s effective).

### 1-GbE capacity disposition

Sustained application wire demand (transport-only, largest boundary payload)
433.7 Mb/s ≤ 80% × 933.9 Mb/s = 747.1 Mb/s, zero retransmits and no
backpressure pathology observed → `R4_1GBE_PRIMITIVE_CAPACITY_VIABLE`.
Socket `send()` durations are retained per boundary but explicitly NOT used
as wire-rate evidence (kernel-buffer acceptance, not wire transmission).
Clean-arm decode: W2 32.85 tok/s, W4 33.38 tok/s; ITL p50 ≈ 29.7–29.8 ms;
network round trip ≈ 0.53 ms of each decode step wall (~1.8%).

## Artifacts

- `r4-frozen-plan.json` — R4 plan derived from the accepted R2 plan
  (digest `sha256:2caf58f2e7a01da757ec11e2578c7575cd25e30486d20a6a3f8e486d58f25b3a`);
- `planner-authorization.json` — generic R3 planner classifies the network
  candidate `FEASIBLE_UNRANKED`; execution is evidence-collection-only
  authorization, not an automatic planner preference;
- `node-a-hardware.json` / `node-b-hardware.json` — retained hardware/network
  freezes incl. mechanical 1-GbE link proof;
- `arm-diagnostic.json` / `arm-clean.json` — per-arm sessions, boundary
  records (Block A/B compute ns, D2H/H2D ns, send ns, coordinator wall ns),
  wire accounting, runtime reports;
- `transport-microbenchmark.json`; `result.json` (machine-readable final);
- `raw/` — iperf3 JSON, ping, Node B ready/final reports and logs, node
  identity proofs (same producer SHA, clean trees).

All top-level JSON artifacts carry `.sha256` sidecars.

## Deviations and honest boundaries

- **Node B physical RAM**: 16 GiB installed (4×4 GiB); `MemTotal` reports
  15.48 GiB after firmware reservations. `MemAvailable` ≥ 12 GiB was measured
  immediately before realization (13.1–13.8 GiB observed) and swap activity
  was zero during the pinned/registered staging lifecycle (`pswpin/pswpout=0`).
- The R2-v2 comparator is float32 **hash identity**; full logit values are
  retained reference evidence, never transferred on the wire (diagnostic
  responses carry only bounded hash/count records).
- flashinfer on inferswarm03 was aligned 0.6.18 → 0.6.17 to match the frozen
  reference environment on inferswarm01. No other environment drift.
- `sgl_kernel` remains installed on Node B only (optional import; absent on
  Node A/the reference host; not on the R4 execution path).

## Explicitly NOT established

- No production distributed-serving, concurrency, or aggregate-throughput claim;
- no automatic planner preference for the network candidate (it remains
  FEASIBLE_UNRANKED until post-R4 evidence is ingested);
- no R5 end-to-end economics; no failover/elasticity; no reconnect recovery;
- the 2.5-GbE comparison arm was NOT measured (optional; not required);
- one-way cross-host latency is never derived from unsynchronized clocks.
