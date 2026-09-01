# InferSwarm R4 — two-node 1-GbE boundary (corrected evidence)

Canonical evidence for InferSwarm issue #57, regenerated after gate-review
methodology corrections. **The previous evidence head `9a26fd2` (producer
`f8e743b`) is INVALIDATED by those corrections** and is retained only for
provenance (historical copy on inferswarm01: `~/r4-historical/`).

## Identity

- base: accepted R3 merge `2ac72d547b2a24a3672d1b83268865db5490084d`
- implementation producer (executed identically on both nodes, clean trees):
  `e97f60b7b0120a72a7cf9926cf6a5c558782c9b2`
  (follow-up tooling-only commit `1110296` touches no experiment semantics)
- Node A: inferswarm01, Block A, GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099 @ 00000000:02:00.0
- Node B: inferswarm03, Block B, GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176 @ 00000000:01:00.0
- model: nvidia/Qwen3.6-35B-A3B-NVFP4 @ 491c2f1ea524c639598bf8fa787a93fed5a6fbce,
  17/17 files sha256-identical across nodes (preflight-gate.json)

## Fail-closed preflight (new; mechanically enforced)

`preflight-gate.json` proves, before any heavyweight realization:
producer SHA + clean tree on BOTH nodes (safe.directory-neutral, any stderr
in identity collection aborts), frozen GPU UUID+BDF per node, VRAM headroom
for both blocks (A: 10.86 GiB required / 12 GiB; B: 11.17 GiB required),
Block-B host RAM from DMI, checkpoint per-file hash equality, and the
canonical link (1000 Mb/s full-duplex MTU-1500 direct route on eno1/enp5s0).
The gate aborted the campaign during deployment on dirty trees — twice —
which is the intended behavior. Arm runners and the Node-B service
additionally refuse producer drift at execution time; the Node-B service
re-proves MemAvailable >= 12 GiB immediately before realization.

## Corrected hardware facts (gate-review finding 3)

- inferswarm03 CPU: i3-10100F, **4 physical cores / 8 logical CPUs**
  (previous profile wrongly recorded 8/1)
- inferswarm01 CPU: E5-2683 v3, 14 physical / 28 logical
- Node B RAM: **16 GiB physically installed (DMI)**, Linux MemTotal
  15.48 GiB (firmware reservation — not a failure), MemAvailable
  14.15 GiB at preflight, >= 14 GiB re-proven immediately before each
  Block-B realization
- BDF keys normalized (`pci_bus_id`); frozen plan retains selected BDFs

## Corrected 1-GbE capacity disposition (gate-review finding 1)

The 80% rule now compares ACTUAL clean-arm workload wire demand against
80% of the lower measured sustainable TCP throughput (iperf A->B 933.9 /
B->A 941.1 Mb/s, 0 retransmits; limit 747.12 Mb/s). The transport-only
microbenchmark is retained separately as transport service capability and
is never used as workload demand.

Actual demand (from arm-clean.json wire accounting + measured cadence):

| workload | decode A->B | decode B->A | prefill A->B | prefill B->A |
|---------|------------|------------|--------------|--------------|
| W2      | 2.15 Mb/s  | 0.075 Mb/s | 2.15 Mb/s    | 0.004 Mb/s   |
| W4      | 2.20 Mb/s  | 0.077 Mb/s | 2.95 Mb/s    | 0.004 Mb/s   |

Peak demand 2.95 Mb/s (A->B) / 0.077 Mb/s (B->A) << 747.12 Mb/s →
**R4_1GBE_PRIMITIVE_CAPACITY_VIABLE**. (The prior claim used the
microbenchmark's 433.7 Mb/s achieved throughput as "demand" — wrong basis,
same label; both are now honestly derived.)

Clean decode: W2 31.30 tok/s, W4 32.05 tok/s (inter-token p50 31.9/31.1 ms);
network service is ~1.7% of decode step wall.

## Correctness and invariants (unchanged semantics, regenerated)

W2/W4 byte-exact generated tokens; selected logits (steps 0/1/15/31)
float32-hash-identical to the frozen R2-v2 reference; producer==receiver
checksum every boundary; 0 NaN/Inf. Both nodes, both arms: fallbacks=0,
recaptures=0, host expert fetches=0, resident source accesses=0,
unexplained mirror bytes=0, steady model-state movement=0. Node B staged
and released exactly 9,545,711,616 B (#53 RELEASE); staging process
VmSwap = 0 kB — no swap reliance (system-wide pswpin/pswpout deltas are
informational: kernel reclaim of unrelated cold pages is not staging swap
reliance).

## Tests (test-summary.json, ALL_TESTS_PASSED)

Focused R4: 65 passed (32 gate + 33 wire). Predecessor research suite:
182 passed. Producer `e97f60b`, both suites on inferswarm01.

## Canonical artifacts

MANIFEST.sha256 covers: README, result.json, arm-diagnostic.json,
arm-clean.json, r4-frozen-plan.json, planner-authorization.json,
preflight-gate.json, node-a/b-hardware.json,
transport-microbenchmark.json, test-summary.json (+ .sha256 sidecars;
raw/ holds noncanonical context: identity proofs, checkpoint manifest,
characterization, ping, iperf, node-B ready/final reports, pytest logs).

## Honest deviations this run

- arm console logs (raw/arm-*.log) not retained: the campaign driver's
  remote-launch ssh pattern hangs, so arms were launched with identical
  detached commands; the canonical arm JSONs are unaffected
- iperf server is one-shot per direction (restarted between directions)
- 2.5-GbE comparison arm still not measured (optional); candidate remains
  planner-FEASIBLE_UNRANKED; no production/concurrency/elasticity claims
