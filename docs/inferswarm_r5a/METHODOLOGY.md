# InferSwarm R5A frozen methodology

Status: predeclared implementation/physical methodology for InferSwarm issue
60. This file does not declare a result.

## Frozen lineage and scope

- required base: `584c2ae77ff37b932f4da6cd2b1652b0696066a9`
- accepted Pre-R5 integration parent: `e1c68e91b59a09e5ddb113828f43f71bbe127898`
- accepted R4 preservation head: `b2d72a36e79624028e74a2e7256f03546d4b8b5b`
- physical producer: the clean committed R5A implementation SHA supplied to
  the fail-closed preflight and recorded identically on both Nodes

The gate is static. It does not test or claim live membership changes, plan
epoch transitions, scale-up/down, recovery, failover, adaptive demand, 2.5-GbE,
production load, final APIs, or GLM-5.3 support.

## Serving seam

The retained distributed arm uses the existing FreeToken
`/v1/chat/completions` route. The unchanged protocol adapter constructs a
protocol-neutral `GenSpec`, and its existing submission waist creates a
`TokenizeMsg`. In R5A research mode that ordinary message is dispatched to one
planner-frozen static runtime. The benchmark driver only sends HTTP requests
and reads evidence; it does not enumerate candidates, select placement, invoke
Block A/B, or call the boundary coordinator.

R5A mode is explicitly separate from historical InferSwarm flags. Without the
R5A configuration, the existing FreeToken worker/ZMQ serving path is unchanged.

## Planning and objective

The pinned Qwen Model Execution Strategy exposes these legal shapes where the
fresh resource snapshot makes them technically expressible:

1. ordinary source-backed execution on the selected Node-A Compute Unit;
2. accepted same-Node two-slot resident split;
3. accepted two-Node two-slot resident split.

The generic planner sees opaque slots, resources, paths, capacities, policy,
normalized evidence, and the declared objective. It does not contain Qwen,
TCP, hostnames, R4, or network-split conditions.

The declared selection objective is **minimize the median measured TTFT across
the matched W2/W4 static serving workload**, with one warmup and five measured
requests per class. TTFT is chosen because it is a user-visible service metric
and the accepted local lineage showed a real source-backed-versus-resident
tradeoff. It is not a rule that the two-Node candidate must win. Decode and
complete-request results are retained independently and may favor another
candidate.

Correctness, technical feasibility, integrity, hard operator policy, and any
applicable admission constraint precede ranking. Unknown or rejected evidence
does not become a fabricated score. Ties use the generic planner's stable
candidate-id ordering.

Accepted R4 application-demand/capacity evidence is always presented through
the generic evidence catalog. Its required model, producer/runtime, network,
and workload context is retained. If the fresh R5A context is not exact, the
planner must record the mismatch and no influence. The value must not be copied
into a model/network-specific conditional. Fresh R5A service evidence is a new
record and does not regenerate or amend R4.

Before matched measurements exist, a legal feasible two-Node candidate may be
run only as `CONTROLLED_EVIDENCE_COLLECTION_OVERRIDE`. The automatic planner
result (including abstention) remains preserved. After matched candidate
evidence exists, the planner is rerun and the selected plan is frozen before
heavyweight realization.

## Frozen plan and fail-closed realization

The immutable R5A plan retains its digest, strategy/model identities,
participants, selected Compute Units, representations/backends, state
placement and authority, semantic boundaries, evidence audit, objective,
policy, exclusions, lower-ranked feasible candidates, unused resources, and
expected resource accounting. The selected R4-derived participant plan digest
is bound inside it.

Immediately before realization, the resource-snapshot digest is rechecked.
Observed participants, Compute Units, representations, backend choices, state
placement/authority, and boundaries must exactly reconcile. Plan or participant
digest mismatch, source/environment drift, or runtime substitution aborts.

## Physical freeze

Before canonical model realization, both Nodes must mechanically pass:

- identical expected producer SHA and clean trees;
- exact hostname/Node identity;
- selected GPU UUID and BDF;
- VRAM capacity plus 512-MiB reservation/headroom treatment;
- runtime/package identity;
- exact model repository/revision and per-file checkpoint hashes on both Nodes;
- negotiated 1000-Mb/s full-duplex MTU-1500 direct route on the frozen
  interfaces;
- Block-B physical RAM and immediate MemAvailable gate;
- swap/paging capture;
- representation/backend compatibility.

The freeze derives a new R5A participant plan from the accepted R2 geometry.
It does not modify `docs/inferswarm_r4/` or claim the old producer was rerun.

## Arms and sampling

All service arms use greedy decoding, `ignore_eos=true`, exactly 32 generated
tokens, W2 then W4, the pinned checkpoint revision, and the frozen workload
manifest.

- Diagnostic two-Node HTTP arm: W2/W4 once, selected logit steps
  `0,1,15,31`, boundary checksums, and full invariant collection.
- Clean two-Node HTTP arm: one W2/W4 warmup followed by five measured
  repetitions per class; diagnostic transfer hashing is disabled.
- Ordinary local single-resource HTTP control: the same warmup/repetition and
  HTTP methodology under the legal source-backed shape.
- Accepted local split control: rerun only if current compatibility/preflight
  remains valid; otherwise retain an explicit exclusion rather than adapting
  the old plan.
- Planner-selected HTTP arm: run from the automatic frozen decision after the
  matched evidence catalog is built. If it is byte-identical to an already-run
  arm/configuration, reference that arm rather than manufacture duplicate work.
- Bounded concurrency arm: exactly two outstanding requests (one W2 and one
  W4), zero warmups, one repetition. The accepted static R4 connection executes
  one boundary stream at a time; the serving surface may have both requests
  outstanding and must retain distinct session identities, FIFO plan authority,
  and zero cross-request state corruption. This is not a throughput/load claim.

## Correctness

The diagnostic contract reuses the accepted R2-v2/R4 comparator without
post-result changes:

- exact prompt and 32-token generated sequences;
- exact float32 selected-logit hashes and argmax at steps `0,1,15,31`
  (the retained `rtol=2e-3, atol=2e-3` comparator is also recorded);
- zero NaN and Inf;
- exact producer/consumer boundary checksums;
- distinct request/session identity and state reset;
- exact intended/observed placement and authority;
- zero fallback, graph recapture, unauthorized host expert fetch, resident
  source access, unexplained persistent host mirror, and steady-state model
  state movement;
- exact planned staging release and process-scoped Block-B `VmSwap=0`;
- no silent plan substitution.

Correctness thresholds are not changed after results.

## Measurements and labels

Coordinator monotonic time owns distributed phase measurements. HTTP client
monotonic time owns HTTP TTFT/request-wall observations. No cross-host one-way
clock arithmetic is used. Retain:

- correctness per request;
- HTTP and coordinator TTFT;
- prefill wall and throughput;
- decode tok/s;
- complete inter-token latency arrays plus p50/p95;
- complete request wall;
- semantic/application wire bytes and framing/control bytes;
- measured staging/transport/protocol residual, explicitly not mislabelled as
  pure wire time;
- per-Node startup/realization;
- component RAM/VRAM and transient/persistent lifecycle accounting;
- paging/swap evidence;
- capture/replay/recapture/fallback/source-access counters;
- planner decision, evidence audit, and plan provenance.

Every numeric claim is labelled `MEASURED`, `CALCULATED`, `ESTIMATED`, or
`SPECULATIVE` under `BENCHMARKING.md`. Medians are calculated from the retained
five measured values; warmups never enter them. A faster microbenchmark or
capacity headroom does not become an end-to-end claim.

## Passing rule

`R5A_STATIC_MULTI_NODE_SERVING_PASS` may be written only if every Issue #60
passing condition is supported by retained machine-readable evidence. A
correct implementation with an unavailable physical prerequisite remains a
reviewable blocked result, not a PASS.

