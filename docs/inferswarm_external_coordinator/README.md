# InferSwarm external-Coordinator evidence (issue #67)

Result: `EXTERNAL_COORDINATOR_SEPARATION_PASS` (re-earned under activation-
derived remote epoch authority; see Supersession below)

This directory retains the proof that the Coordinator is genuinely a
replaceable control-plane role: request ingress, session identity, generic
planning, frozen Execution Plan authority, epoch coordination, and
committed-output accounting all executed on the CPU-only VM `inferswarm00`,
while every correctness-bearing model materialization and inference
execution happened on remote compute Nodes.

## Supersession of the first physical producer (IMPORTANT)

The first physical producer `df8a429e9110…` **did** prove physical
Coordinator separation (CPU-only control plane, remote realization/execution,
exact reference tokens, fencing, reclamation). It was **superseded** because
its remote wire epoch identity was **plan-derived** rather than
**activation-derived**: both the coordinator realizer and the Node agent
independently derived `remote-realization:<plan-digest-prefix>`, which made
remote epoch authority a function of the Execution Plan digest instead of
Coordinator-owned epoch/generation authority. The historical evidence of
that producer remains retained in git history (commit `861a404`, producer
`df8a429`); it must not be cited as the final canonical proof. The corrected
producer is `84e531971d6c…` (this directory).

## Epoch/generation authorization design (the corrected seam)

- Before heavyweight realization, `EpochServingController` allocates a
  realization-authorization record: prospective authoritative epoch id
  (`research-generation-{N}:<plan-digest-prefix>`), prospective generation,
  plan digest, and a unique realization-attempt identity
  (`realization-{attempt}-{time_ns}`).
- The authorization is passed to the realizer (optional
  `realization_authorization` keyword; legacy single-argument realizers
  unchanged, so all accepted R5B semantics/tests are preserved, including
  the generation-0 initial activation and contiguous activation
  generations).
- A successful realization's activated `Epoch` uses exactly the authorized
  epoch id/generation.
- If realization fails, that attempt identity is dead and can never be
  accepted later; a later attempt of the same plan/generation slot receives
  a fresh unique realization identity (attempt uniqueness lives in the
  nonce, so contiguous generations never imply authorization reuse).
- Every `REALIZE`/`GENERATE`/`REPORT`/`CLOSE` request/response on the
  research wire now binds the actual Coordinator-authorized epoch id,
  generation, and realization id (plus session/position/operation identity
  where applicable). Plan-derived epoch namespaces are rejected at the wire
  and at the Node agent.
- The Node agent consumes the Coordinator-provided epoch/generation; it does
  not derive epoch identity from the plan digest. Closed, failed, or
  torn-down authorizations are refused forever by the agent's retired-
  authorization registry.
- Coordinator response validation verifies the full authoritative
  epoch/generation/realization/plan identity on every exchange.

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
  fail-closed, identity-bound
  (protocol/scope/session/epoch/generation/realization/plan/position),
  sha256 result checksums. Not a public API.

## Canonical run (producer 84e5319)

One ordinary OpenAI-compatible `/v1/chat/completions` request entered on
`inferswarm00` (54 prompt tokens, max_tokens 8, greedy, thinking enabled).
The generic planner physically executed on `inferswarm00` over the frozen
resource snapshot and requalified ranking evidence; the selected
resident-two-node-two-slot Execution Plan was frozen before realization;
realization and all eight decode operations executed remotely under the
Coordinator-authorized activation identity.

- Wire epoch identity (Coordinator-authorized, activation-derived):
  `research-generation-0:cbba220611a7` — carried by every remote
  REALIZE/GENERATE/REPORT/CLOSE exchange, never derived from the plan digest.
- Realization authorization:
  `realization-1-18d1975ba81bedba` (prospective epoch id, generation 0,
  plan digest `sha256:cbba2206…`, allocated before realization and adopted
  verbatim by the activated epoch).
- Committed output: `[9764, 393, 45, 283, 220, 24, 22, 853]` — exactly the
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

## Regression of record

`tests/research/test_xc_realization_authority.py` proves: same Execution
Plan digest does not imply same remote epoch authority. Generation-0
activation of plan digest P → supersession → later activation of the SAME
digest P under a distinct generation/realization authorization; distinct
wire epoch/generation identity; an old remote result from the earlier
authorization is rejected; committed session/output/accounting unchanged.
Retained alongside: duplicate-position rejection, stale-controller-epoch
rejection, mid-frame/disconnect fail-closed tests.

## Contents

- `preflight/` — fresh R5A physical freeze against `84e5319` (environment
  digest `sha256:e7a18da3…`, participant plan `sha256:c339627e…`,
  ALL_PREFLIGHT_CHECKS_PASSED).
- `r5b-env/` — epoch-layer frozen environment (`sha256:ddaf7dd6…`),
  participant plans, hardware and identity records.
- `planning/` — requalified ranking evidence (9 records, dependency-scoped
  to the unchanged accepted R2/R4 runtime, per the accepted R5B pattern).
- `lifecycle/` — the final serving report (sessions, epoch record with
  realization authorization, fencing injections, reclamation, final runtime
  report) and the client-side HTTP observable.
- `regressions.json` — machine-readable regression summary (research 235,
  benchmarks 563, server 583, xc focus 42, coordinator CPU-only 36; zero
  failures).

Physical producer SHA: `84e531971d6c02530fd0d541da3478df7510f263` (branch
`inferswarm-external-coordinator`, descended from the exact accepted
Pre-R6 head `8cfcda4c`, superseding `df8a429e9110…` per above). Native
extensions were verified present in-tree on both compute Nodes (unchanged
by this producer, which touches only pure-Python control-plane modules).

## Explicit non-claims

No Coordinator HA/election/consensus/term fencing/zero-downtime failover;
no public Node-agent API or production wire protocol; no production daemon;
no TLS/PKI design beyond the bounded trusted-LAN research proof; no Gemma 4;
no R6 falsification claim; no new Qwen placement or performance claims; no
optimized remote-control latency claim.

Accepted R0–R5B and Pre-R6 evidence directories were not modified.
