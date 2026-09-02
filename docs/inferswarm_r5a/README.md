# InferSwarm R5A retained result

Disposition: `R5A_STATIC_MULTI_NODE_SERVING_PASS`.

R5A proves that an ordinary FreeToken `/v1/chat/completions` request can enter the existing host serving path, cross the existing `TokenizeMsg` waist, invoke generic evidence-aware planning, freeze an explainable static plan before heavyweight realization, realize that plan on `inferswarm01` and `inferswarm03`, execute through the accepted R4 backend-native boundary, and stream a correct response.

## Provenance

- Required base: `584c2ae77ff37b932f4da6cd2b1652b0696066a9`.
- Physical implementation producer: `c82a79206d9223cc9d4e94b92e780a4cab71fda7` (clean on both nodes).
- Deliberately excluded newer `main`: `a2538a428baa4c6d823c76efe96cb3bc0cbd1f86`.
- Frozen environment: `sha256:f929cdc3567e24b3210f7dc9dbcc90692105e292aec01832b48ee62c56fa7d0e`.
- R4-derived participant plan: `sha256:6b2f967feabe6febf968eb925ba54ded29e73fb506ac48ee838367734f9a7369`.
- Automatic R5A plan: `sha256:04dcf8699c627782aed5eddf8b169a6e79cbe98e3828bec753c3a4a1fe72cb62`.

The raw frozen environment, full 14.8 MB participant plan, request records, planner decision, execution plan, per-node realization reports, and summaries are retained under `evidence/`; they are not represented only by hashes.

## Planner result

The declared objective was to minimize median TTFT on the matched W2/W4 generate-32 workload. Current matched evidence ranked the two-node resident candidate first at 1,904.6 ms and the local source-backed control second at 2,624.8 ms. The legal same-node resident split remained feasible but unranked because no current matched evidence applied. The unused second GPU on `inferswarm01` is explained as unnecessary for the highest-ranked candidate.

The accepted R4 2.947 Mb/s demand/capacity record was normalized into the generic evidence catalog with its original provenance. Its runtime context did not exactly match the R5A producer, so it was explicitly rejected and did not influence admission or ranking. R5A ranking instead used freshly measured, exact-context serving evidence.

The first two-node arm was a controlled evidence-collection override and remains labeled as such. After matched local and network evidence existed, the fresh automatic planner run selected the two-node candidate without an override.

## Correctness and residency

The unchanged R2/R4 comparator passed W2 and W4 with exact prompt tokens, exact generated tokens, selected logits within the frozen threshold, matching boundary checksums, and zero NaN/Inf. Intended and observed R5A realization matched exactly. Both nodes reported zero fallback, recapture, host expert fetch, resident-source access, steady-state model-state movement, and unexplained persistent host mirror bytes.

Node A released 8,636,596,224 staging bytes and retained 10,878,279,168 CUDA-allocated bytes; Node B released 9,545,711,616 staging bytes and retained 11,170,319,360 CUDA-allocated bytes. Neither staging process relied on swap.

## Service measurements

The clean matched arm used one warmup plus five measured repetitions for each of W2 and W4. Across the ten measured requests, medians were:

- TTFT: 1,904.6 ms.
- prefill wall: 1,904.6 ms; prefill throughput: 45.87 tok/s.
- decode: 29.46 tok/s.
- per-request median inter-token p50/p95: 33.64/35.55 ms.
- complete request wall: 3,023.1 ms.
- measured network staging/protocol residual: 2,170.8 ms, 72.14% of request wall.

Across all 12 clean requests, including the two warmups, application wire accounting was 11,797,047 bytes A→B and 122,535 bytes B→A, of which 11,649,024 bytes were semantic payload A→B. Node A and B materialization took 45.60 s and 33.59 s respectively; coordinator-observed realization wall was 47.69 s. Numeric summaries are derived only from the predeclared measured repetitions and are labeled `MEASURED` in the JSON artifacts.

## Bounded concurrency

The automatic planner-selected arm admitted W2 and W4 concurrently with a frozen bound of two. It observed two outstanding requests, two distinct session IDs, exact generated sequences, and the same immutable plan digest on both controller request records. Execution was serialized at the accepted single-connection R4 boundary; this proves request/session isolation and static-plan behavior, not production load capacity.

## Regression result

With the R5A worktree pinned first on `PYTHONPATH`, 190 research, 554 benchmark, and 578 server tests passed. An initial invalid invocation imported the unrelated `/home/zutfen/FreeToken` editable worktree; the traceback-based classification and corrected rerun are retained in `test-summary.json`.

## Proof boundary

This is a static two-node research gate. It makes no production throughput, elasticity, live-membership, epoch-transition, failover, availability, 2.5-GbE, heterogeneous-accelerator, final API/control-plane, or GLM-5.3 claim. Accepted R4 evidence was not regenerated or modified.
