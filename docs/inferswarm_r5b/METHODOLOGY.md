# InferSwarm R5B frozen methodology

Status: predeclared implementation and physical methodology for InferSwarm
issue #62. This file does not declare a result.

## Lineage and exclusions

- Exact accepted FreeToken R5A merge head:
  `d9f45a9ef7b5f89800f96c54397202a7d43beb52`.
- Accepted R5A physical producer:
  `60ea7bd9841a636a26bfe7f140dba04b0a562f03`.
- Accepted R5A evidence head:
  `136fc6385afaa0864e289746484b211f3a1fcdd8`.
- R5B physical producer: the clean committed implementation SHA supplied
  identically to the fail-closed preflight on both Nodes.
- Newer upstream-tracking FreeToken `main`, GLM-5.3, and materially different
  model work are excluded from this lineage.

Accepted R4/R5A and pre-R5 evidence directories are immutable. R5B writes only
under `docs/inferswarm_r5b/`.

## Architecture and authority

The ordinary `/v1/chat/completions -> GenSpec -> TokenizeMsg` serving waist
dispatches to a research-internal, model-neutral epoch controller. Each
activation wraps one complete immutable R5A-style frozen plan and independently
reconciled accepted R2 or R4 runtime in a distinct generation. Epoch identity is
research-internal and is not a proposed public field or encoding.

For the one transition-bearing session, exactly the active epoch owns mutable
runtime and output-commit authority. A replacement may be planned and its
immutable prerequisites validated while the old epoch remains valid, but
preparation conveys no authority. Activation order is: select a strategy-safe
boundary, settle old work, realize/catch up the replacement, switch the single
authority/routing pointer, activate the replacement, retire the old epoch, and
reclaim it. Late results are accepted only when epoch, plan, session, and next
commit position all match current authority.

Both accepted resident runtimes require Node-A GPU0. Full R2/R4 overlap is
therefore frozen as physically infeasible for this campaign. R5B will retain
immutable preparation beside the valid old epoch, then report an honest
strategy-safe interrupted/cold cutover. It will not claim make-before-break
runtime materialization or zero downtime.

## Qwen strategy transition contract

Only the pinned `nvidia/Qwen3.6-35B-A3B-NVFP4` revision
`491c2f1ea524c639598bf8fa787a93fed5a6fbce` is in scope.

For this strategy, the safe boundary is a committed generated-token boundary.
Mutable block-local KV and linear-recurrent state is reconstructible by
prefilling the exact original prompt token IDs followed by the exact committed
generated token IDs. The host ledger retains those IDs, committed position,
active epoch identity, plan digest, and deterministic sampling values. Replay
output is suppressed. The first externally usable post-transition token is the
next uncommitted token. This token-boundary/replay rule is not generalized to
other models or strategies.

Each accepted runtime invocation produces the next commit candidate plus one
speculative token because the accepted R2/R4 measurement record requires one
decode interval. Only step zero is eligible to commit; the speculative step is
discarded and never emitted or used as recovery history.

Continuation is legal only with exact trusted history, the pinned model and
representation/backend contract, and greedy deterministic sampling. If those
inputs are missing or untrusted, the session fails closed. Mutable device state
from a failed epoch is never copied merely for convenience.

## Frozen workload and correctness

The physical lifecycle uses canonical W2 prompt bytes with SHA-256
`a4f2fdc66c946d8f9097d34fe8d173c7dbb9d647401e8f6bc9b79a0158d26e5d`,
greedy sampling (`temperature=0`, `top_p=1`, `top_k=-1`), `ignore_eos=true`,
and eight generated tokens. The accepted reference prefix is:

`[9764, 393, 45, 283, 220, 24, 22, 853]`

The complete prompt token sequence must equal the accepted 54-token R5A W2
reference. The transition result must match the frozen output prefix exactly,
with zero duplicate, missing, reordered, or mixed-epoch token. Retain selected
logits and boundary checksums where the active accepted backend exposes them,
and require zero NaN/Inf, exact intended/observed realization, zero silent plan
substitution/in-epoch fallback/unauthorized source fetch/unexplained host
mirror/unplanned steady-state model movement.

## Resource events and physical lifecycle

Resource changes arrive over a mode-0600 local Unix datagram socket with an
HMAC-authenticated JSON research envelope. This is a one-shot event seam, not a
polling scheduler or public protocol. The receiver re-observes GPU A1 and
validates UUID, BDF, VRAM, integrity eligibility, and representation/backend
compatibility. An event updates and freezes the resource graph; the generic R3
planner still chooses the candidate.

1. Start with GPU A1's execution participant genuinely not started. GPU A0 and
   Node-B GPU B0 participate in the expected automatically selected two-Node R4
   E0. Hidden A1 runtime use is forbidden.
2. After the first committed token, authenticate A1 availability, replan, and
   if policy authorizes it, cold-cut over at that token boundary to a new
   same-Node R2 epoch.
3. After the third committed token, first ensure the surviving Node-B
   participant is ready, then terminate the required local Block-B process.
   The old epoch becomes physically non-executable. Freeze a graph excluding
   A1, plan all surviving candidates, reconstruct through the third committed
   token, and resume under a new epoch.
4. After the fifth committed token, return/revalidate A1 and perform another
   planner/policy-authorized optimization epoch.

At the first activation, inject one delayed correctness-bearing result copied
from actual old-epoch serving work after the replacement owns authority. It must
be rejected without changing the committed ledger, client output, or token
accounting, and the audit counter must increment.

## Predeclared transition economics

The frozen policy uses the accepted context-specific R5A median TTFT evidence:

- same-Node resident: `373.6170495 ms`;
- two-Node resident: `1877.4567285 ms`;
- source-backed control: `2630.5871575 ms`.

It also uses accepted realization-cost estimates `68.131071624 s` (same-Node)
and `47.826231041 s` (two-Node), 64 expected remaining requests, stability
confidence at least `0.9`, and a `90 s` interruption budget. Optimization is
authorized only when per-request objective gain times the frozen horizon
exceeds preparation cost and stability passes. Required-resource recovery is
authorized by correctness/feasibility before performance. Thresholds are not
changed after results. The retained policy artifact is authoritative.

R5A numbers are carried into the new producer context only through an explicit
dependency-scoped evidence derivative after unchanged R2/R4 runtime regressions
pass. The derivative preserves original measurement identity/value, labels that
the number was not remeasured, and updates applicability context; it does not
rewrite accepted R5A claims.

## Negative and preparation arms

A deterministic controller integration test forces replacement realization to
fail during overlap preparation while the old epoch remains executable. It must
retain old authority and correct output, activate no partial replacement, and
record the failure.

A deterministic strategy/controller test removes the trusted prompt/committed
history source during required-resource loss. It must emit a terminal session
failure and must not continue, guess, restart with changed semantics, reuse
stale state, or report recovery. Manufacturing physical data corruption would
add no authority information and is not required.

## Measurements and evidence

Retain planner, preparation/materialization, replay, authority cutover,
interruption, first-post-transition-token, post-transition service, maximum
client-visible gap, RAM/VRAM/paging/staging, network preparation/recovery
traffic, reclamation, retry/failure, and late-rejection records. Coordinator
monotonic time owns epoch phases; client monotonic time owns visible gaps. Wire,
staging, protocol, replay, and total interruption remain distinct. Every numeric
claim is labeled under `BENCHMARKING.md`.

Each epoch validates the current correctness-bearing inputs independently.
Canonical execution rejects producer/tree, model/checkpoint, Node/GPU/BDF,
headroom, backend/representation, network, RAM/swap/paging, initial availability,
plan, or policy/methodology drift.

## Passing rule and non-claims

`R5B_PLAN_EPOCH_RECOVERY_PASS` may be written only if every issue #62 passing
condition is substantiated by retained raw evidence and the hashed manifest.
Otherwise the branch remains a reviewable blocked result with the exact gate.

R5B makes no production HA/load-balancing/zero-downtime claim, freezes no public
planner/strategy/epoch/wire/daemon/control-plane API, and makes no 2.5-GbE,
heterogeneous-vendor, GLM-5.3, or R6 claim.
