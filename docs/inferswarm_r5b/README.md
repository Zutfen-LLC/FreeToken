# InferSwarm R5B evidence

Result: `R5B_PLAN_EPOCH_RECOVERY_PASS`

This directory retains the execution-plan epoch, live resource-change, and
truthful recovery proof tracked by InferSwarm issue #62. The canonical physical
producer is `7dd945a67c04198ec2d9afe782a39c90e8141f5e`, descended from exact
accepted R5A head `d9f45a9ef7b5f89800f96c54397202a7d43beb52` with no current-`main`
content incorporated.

The one ordinary HTTP session committed the frozen eight-token reference
exactly across four immutable generations:

1. E0 two-node resident;
2. E1 same-node resident after GPU1 availability;
3. E2 two-node resident after real termination of E1's required GPU1 worker;
4. E3 same-node resident after GPU1 return.

The planner chose each placement from its frozen resource graph and applicable
evidence. The emitted sequence was
`[9764, 393, 45, 283, 220, 24, 22, 853]`, with exact epoch attribution, no
duplicate/missing/reordered token, zero retained-logit NaN/Inf, and one injected
late E0 result rejected without changing the commit ledger or client output.

The cold-cutover measurements were:

| Trigger | Replan | Replacement realization | Total authority cutover | Client-visible gap | Replay + first candidate prefill |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scale up | 28.865 ms | 67.867 s | 69.670 s | 72.193 s | 0.908 s |
| GPU1 loss/recovery | 67.243 ms | 53.326 s | 54.823 s | 57.113 s | 1.694 s |
| Scale back up | 28.729 ms | 67.595 s | 69.553 s | 72.064 s | 0.912 s |

These are observed monotonic wall times, not pure network latency. Full runtime,
wire, staging, paging, reconciliation, authority, plan, event, and reclamation
records remain in the subdirectories rather than only as hashes.

`METHODOLOGY.md` and `transition-body.json` were frozen before the canonical
physical results. `result.json` is the inspectable gate summary;
`lifecycle/serving-report-after-request.json` proves E3 authority after the
response; `lifecycle/serving-report.json` proves all-epoch reclamation after
server shutdown; `test-summary.json` and `regressions/` retain the regression
and negative-arm results; `ANOMALIES.json` classifies the failed development
runs that preceded the canonical producer.

Regressions passed: research 199, InferSwarm benchmarks 563, server 581, and
four named R5B negative/fencing cases. The sole warning was the existing
Starlette/httpx test-client deprecation warning.

This result does not claim zero downtime or make-before-break runtime overlap,
and it does not freeze a production scheduler, public epoch field, daemon,
wire protocol, strategy API, or control plane. It makes no GLM-5.3 or R6 claim.

Accepted `docs/inferswarm_r4/`, `docs/inferswarm_r5a/`, and pre-R5 evidence were
not modified by R5B.
