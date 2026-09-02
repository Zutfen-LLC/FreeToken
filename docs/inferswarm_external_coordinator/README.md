# InferSwarm external-Coordinator evidence (issue #67)

Result: `EXTERNAL_COORDINATOR_SEPARATION_PASS`

This directory retains the proof that the Coordinator is genuinely a
replaceable control-plane role: request ingress, session identity, generic
planning, frozen Execution Plan authority, epoch coordination, and
committed-output accounting all executed on the CPU-only VM `inferswarm00`,
while every correctness-bearing model materialization and inference
execution happened on remote compute Nodes.

## Canonical physical topology

- Coordinator: `inferswarm00` (KVM VM, 4 vCPU, no NVIDIA driver, no CUDA, no
  torch, no triton, no FreeToken native extensions, no model weights —
  32 MB of checkpoint tokenizer/config metadata only).
- Compute Nodes: `inferswarm01` (Block A resident on GPU A0) and
  `inferswarm03` (Block B resident on GPU B0), the accepted RTX 3060 proving
  pair, joined by the accepted R4 persistent boundary (10.0.0.141 ↔
  10.0.0.219:18485, ordinary LAN path).
- Research control wire: ordinary TCP `inferswarm00 → inferswarm01:18486`,
  length-framed JSON (`inferswarm.external-coordinator.realization-wire/1`),
  fail-closed, identity-bound (protocol/scope/session/epoch/plan/position),
  sha256 result checksums. Not a public API.

## Canonical run

One ordinary OpenAI-compatible `/v1/chat/completions` request entered on
`inferswarm00` (54 prompt tokens, max_tokens 8, greedy, thinking enabled).
The generic planner physically executed on `inferswarm00` over the frozen
resource snapshot and requalified ranking evidence; the selected
resident-two-node-two-slot Execution Plan was frozen before realization;
realization and all eight decode operations executed remotely.

Committed output: `[9764, 393, 45, 283, 220, 24, 22, 853]` — exactly the
accepted R5A/R5B reference — with single-epoch/single-plan attribution,
clean reconciliation (`matched: true`, zero mismatches), and
`usage = {prompt 54, completion 8, total 62}`.

Fencing negative arm: after the first real commit, a duplicate-position
result and a stale-epoch result were routed through the same Coordinator
acceptance path. Both were mechanically rejected
(`NON_NEXT_COMMIT_POSITION`, `RETIRED_OR_SUPERSEDED_EPOCH`) with the
committed ledger unchanged.

Shutdown: SIGTERM exercised the graceful path — the epoch reached
`RECLAIMED`, the remote node-agent runtime closed with a recorded
reclamation (`remote-node-agent-epoch-close`), the final runtime report was
retained, and both compute Nodes' GPUs returned to idle (0 compute
processes, 1 MiB used) with every proof-specific process stopped.

## Contents

- `preflight/` — fresh R5A physical freeze (environment digest
  `sha256:01644ecc…`, participant plan `sha256:8035d4c4…`,
  ALL_PREFLIGHT_CHECKS_PASSED).
- `r5b-env/` — epoch-layer frozen environment, participant plans, hardware
  and identity records.
- `planning/` — requalified ranking evidence (9 records, dependency-scoped
  to the unchanged accepted R2/R4 runtime, per the accepted R5B pattern).
- `lifecycle/` — the final serving report (sessions, epoch record, fencing
  injections, reclamation, final runtime report) and the client-side HTTP
  observable.
- `result-check.json` — machine-readable gate summary over the retained
  artifacts.

Physical producer SHA: `df8a429e9110…` (branch
`inferswarm-external-coordinator`, descended from the exact accepted
Pre-R6 head `8cfcda4c`). Native extensions were rebuilt in-tree on both
compute Nodes for this producer (gitignored build artifacts; trees clean).

## Explicit non-claims

No Coordinator HA/election/consensus/term fencing/zero-downtime failover;
no public Node-agent API or production wire protocol; no production daemon;
no TLS/PKI design beyond the bounded trusted-LAN research proof; no Gemma 4;
no R6 falsification claim; no new Qwen placement or performance claims; no
optimized remote-control latency claim.

Accepted R0–R5B and Pre-R6 evidence directories were not modified.
