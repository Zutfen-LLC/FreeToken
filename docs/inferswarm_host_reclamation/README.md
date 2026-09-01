# Pre-R3 host-staging lifecycle proof

Canonical issue: [InferSwarm #53](https://github.com/Zutfen-LLC/inferswarm/issues/53)

Exact FreeToken base: `8627f441c880398389042ce8c0a604f6c4321dfa`

Disposition: `HOST_STAGING_RECLAMATION_PASS`

This is a capacity-accounting prerequisite, not R3.

## Ownership finding

The retained pages were the selective routed source banks themselves. Their default
allocation path was anonymous `mmap`, populated from the checkpoint, followed by
`cudaHostRegister`. The loader returned tensor views, while `host_banks.py` also retained
every mmap in a module-global `_LIVE_BUFFERS` list for the worker process lifetime.

That process-lifetime rule was deliberate for ordinary offload: attached host source
banks let `OffloadMoeCache.rebuild` resize/recreate accelerator slot caches without an
SSD reread. It was not, however, a usable post-R2 iteration cache. R2 severed all tensor
and source-bank registries, leaving only inaccessible mmap objects in `_LIVE_BUFFERS`.
The observed R2 retention was therefore an accidental lifetime overreach of a useful
ordinary-offload policy, not evidence of a CUDA or PyTorch allocator leak.

The bounded primitive evidence confirms that `cudaHostAlloc`/`cudaFreeHost`,
`cudaMallocHost`/`cudaFreeHost`, registered mmap with explicit unregister/release, and a
pageable control all return at least 99.98% of their 512 MiB process RSS on this host.

## Explicit lifecycle

The internal generic host-materialization owner now records a pointer-free allocation
identity, requested/allocation bytes, mechanism, owner class, tensor/storage state,
registration state, lifecycle state, and release invocation count.

- `retain_reusable_source` moves detached source tensors into an explicitly optional,
  reclaimable host cache. It is not required host residency and is not an unexplained
  mirror.
- `release_after_final_residency` requires validated full accelerator residency, rejects
  live source Tensor aliases, synchronizes CUDA, unregisters owned registrations, and
  discards/closes the actual mapping. Double release and source-dependent operations
  fail closed.

Ordinary offload never enters either final-residency transition and retains its attached
source behavior unchanged. The small R2 boundary transport mapping is outside model
staging accounting and remains live until worker shutdown.

## Physical result

The two workers were synchronized before and after release so system `MemAvailable`
represented the combined transition. Per-worker reclamation is attributed from process
RSS; the combined conservative result is the smaller of summed process reduction and
system-available increase.

| Participant | Staging bytes | RSS before | RSS after | Reclaimed bytes | Fraction |
|---|---:|---:|---:|---:|---:|
| Block A | 8,636,596,224 | 10,403,823,616 | 1,769,586,688 | 8,634,236,928 | 99.973% |
| Block B | 9,545,711,616 | 11,326,976,000 | 1,783,685,120 | 9,543,290,880 | 99.975% |
| Combined conservative | 18,182,307,840 | — | — | 18,177,527,808 | 99.974% |

Coordinated system `MemAvailable` increased from 85,691,170,816 to 104,683,438,080
bytes, an 18,992,267,264-byte increase. Both processes remained alive for resident W2/W4
execution. Their final non-staging `RssShmem` was 37,773,312 bytes each, including the
retained boundary transport/runtime mappings.

## Resident correctness after RELEASE

W2 and W4 each generated 32 greedy tokens against the accepted R2 v2 canonical
reference. Generated tokens and all selected logits were byte-exact; NaN/Inf counts were
zero and every boundary hash matched. Each worker reported one graph capture, 62 replays,
zero recaptures, zero host expert fetches, zero resident source accesses, zero fallbacks,
one initial population, and zero steady-state model-state movement.

Matched RETAIN execution also remained byte-exact. It explicitly retained
8,636,596,224 bytes for Block A and 9,545,711,616 bytes for Block B, all classified as
optional/reclaimable and none as required persistent host bytes.

No supported operation currently destroys final accelerator residency and rematerializes
it from RETAIN state. Consequently no post-finalization rematerialization speedup is
claimed or timed. The existing proven SSD-avoidance operation is ordinary pre-finalization
`OffloadMoeCache.rebuild`; adding a research rematerialization workflow is outside this
proof.

## Artifacts

- `allocation-primitives.json`: isolated 512 MiB primitive behavior.
- `retention-baseline.json`: R2 observation, owner, lifetime, and history classification.
- `retain-mode.json`: matched optional-cache execution.
- `release-mode.json`: physical reclamation and W2/W4 resident proof.
- `result.json`: concise disposition and future accounting principle.

Every JSON artifact has a sibling SHA-256 sidecar. No run dropped global page cache,
used swap as reclamation, moved resident weights back to RAM, or killed a worker to pass
the physical gate.
