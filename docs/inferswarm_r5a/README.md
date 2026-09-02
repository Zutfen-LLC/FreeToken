# InferSwarm R5A retained result

Disposition: `R5A_STATIC_MULTI_NODE_SERVING_PASS`.

This correction closes the remaining comparison gate in InferSwarm #60. The ordinary FreeToken `/v1/chat/completions -> TokenizeMsg -> R5A controller -> frozen plan -> realization` path can now realize either accepted resident strategy: the R2 same-node split or the R4 two-node boundary. The generic planner received current matched evidence for every legal candidate and truthfully selected the same-node resident split under the declared median-TTFT objective.

## Provenance

- Required base: `584c2ae77ff37b932f4da6cd2b1652b0696066a9`.
- Clean physical implementation producer on both nodes: `60ea7bd9841a636a26bfe7f140dba04b0a562f03`.
- Deliberately excluded newer `main`: `a2538a428baa4c6d823c76efe96cb3bc0cbd1f86`.
- Frozen environment: `sha256:bedf54e6602800f5b179e7ae7eb9eec816cdf1c49181a7ecc4ca904247b9fce2`.
- R4-derived two-node participant plan: `sha256:bc306ddb4076f4c7ed72dbb9effef9ef08ba402b20b0ec35b609c92feb94a584`.
- Accepted R2 local participant plan: `sha256:6128dd6705d692df3d5fc11cc130dba5c010cfff40c0e4c5ec7c19e1b78ff0`.
- Automatic R5A execution plan: `sha256:11c9e21c51c2bdd037ec8644f656a7333ac7878e1ee31ff71c99220a6e2aea17`.

The full environment, participant plan, request records, planner explanations, execution plans, realization reports, diagnostics, and metric inputs are retained under `evidence/`; required raw evidence is not represented only by hashes. The superseded `c82a792` evidence was replaced and remains recoverable from Git ancestry. Accepted R4 evidence was not changed.

## Frozen local resources

The additional Node-A RTX 3060 was frozen and fail-closed with UUID `GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55`, BDF `00000000:03:00.0`, 12,884,901,888 bytes total VRAM, 11,170,278,912 required bytes, and a 536,870,912-byte reservation. The primary Node-A GPU remained `GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099` at BDF `00000000:02:00.0`. Preflight also froze clean producer state, model revision and hashes, runtime versions, representation/backend compatibility, RAM, paging, and the Node-B/network inputs.

## Matched placement economics and planner result

All three legal candidates used the same W2/W4 generate-32 workload, one warmup plus five measured repetitions per class, deterministic settings, ordinary HTTP entry point, and measurement definitions.

| Candidate | Median TTFT | Median request wall | Median decode |
|---|---:|---:|---:|
| same-node resident R2 split | 373.617 ms | 844.386 ms | 65.766 tok/s |
| two-node resident R4 split | 1,877.457 ms | 3,067.985 ms | 29.223 tok/s |
| source-backed single-resource control | 2,630.587 ms | 3,305.816 ms | 46.000 tok/s |

The two-node placement added 1,503.840 ms median TTFT and 2,223.599 ms median request wall relative to the resident same-node split; its TTFT was 5.025 times the same-node value. The compute-subtracted residual remains a combined staging/transport/protocol residual: 46.457 ms for same-node and 2,248.427 ms for two-node. It is not labeled pure network time.

The frozen nine-record serving-evidence derivative gave each legal candidate an exact-context TTFT record. The generic planner ranked same-node resident first, two-node resident second, and source-backed single-resource third. The automatic arm selected same-node without an override and explained the unused Node-B GPU. Accepted R4 demand/capacity evidence was presented with provenance but rejected because its runtime context did not exactly match this producer; it did not influence admission or ranking.

## Correctness, residency, and service evidence

Fresh diagnostic W2/W4 arms for both resident placements passed the accepted R2/R4 comparator: exact prompt and generated tokens, selected logits within the frozen threshold, matching boundary checksums, and zero NaN/Inf. Clean arms retained exact token equality and zero NaN/Inf. Intended and observed realization reconciled without mismatch or plan substitution.

Both resident placements reported zero fallback, graph recapture, host expert fetch, resident-source access, steady-state model-state movement, and unexplained persistent host mirror bytes. Both blocks released their 8,636,596,224-byte and 9,545,711,616-byte staging sources and had zero process-scoped swap reliance. Local-resident block VRAM allocations were 10,861,239,808 and 11,170,319,360 bytes; the two-node arm retained 10,886,798,848 and 11,170,319,360 bytes. Clean graph replay counts were 372 per block.

The same-node arm additionally retained median prefill wall 373.616 ms, prefill throughput 227.357 tok/s, per-request inter-token p50/p95 15.127/15.183 ms, and registered-host activation/control accounting. The two-node arm retained median prefill wall 1,877.456 ms, prefill throughput 47.122 tok/s, inter-token p50/p95 33.798/35.538 ms, and application wire totals of 11,797,047 bytes A-to-B and 122,529 bytes B-to-A across all 12 clean requests, including 11,649,024 semantic payload bytes.

Node-B treats the coordinator's terminal socket close after final report capture as a fail-closed zero-byte `disconnect mid-frame` and exits nonzero. This is the accepted R4 teardown behavior, occurs after complete retained request/final accounting, and is not relabeled as a serving fallback.

## Bounded concurrency and regressions

The automatic planner-selected arm admitted W2 and W4 with a frozen bound of two. It observed two outstanding requests, distinct session IDs, exact generated sequences, clean invariants, and the same immutable plan digest. Execution remained serialized through the accepted static runtime; this is an isolation proof, not a production load claim.

Final regressions from the physical producer passed: 193 research, 559 benchmark, and 578 server tests. The server suite emitted one unrelated `StarletteDeprecationWarning`.

## Proof boundary

This is a static research gate. It makes no production throughput, elasticity, live-membership, epoch-transition, failover, availability, 2.5-GbE, heterogeneous-accelerator, final API/control-plane, or GLM-5.3 claim.
