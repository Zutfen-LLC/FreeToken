# InferSwarm #48 physical result

Status: `P48_ACCELERATOR_RESIDENCY_PASS` for the measured Block A path.

The run used `nvidia/Qwen3.6-35B-A3B-NVFP4` revision
`491c2f1ea524c639598bf8fa787a93fed5a6fbce`, the canonical N0 FreeToken base
`4c60ff522a95cf147456a4333271ee05b505fc58`, and global layers 0–18 on an
NVIDIA GeForce RTX 3060 (12,485,525,504 bytes, compute capability 8.6).

Measured/calculated result:

- MEASURED fully resident NVFP4 accelerator expert banks: 8,636,596,224 bytes.
- MEASURED transient host expert banks before detach: 8,636,596,224 bytes.
- MEASURED live host expert banks after detach: 0 bytes.
- MEASURED live host expert banks after four decode executions: 0 bytes.
- MEASURED persistent host tensors: 327,680 bytes, all owned by the frozen
  same-backend correctness reference fixture.
- CALCULATED conservative transient staging/overlap upper bound: 18,931,555,660
  bytes. This adds the complete transient bank to cumulative selectively fetched
  tensor bytes and therefore intentionally overstates simultaneous live staging.
- MEASURED weak-reference release: all 114 source-bank Tensor objects unreachable.
- MEASURED correctness: pre-detach prefill and all four post-detach decode runs were
  exact against the same-backend N0 fixture; maximum absolute/relative deviation was
  0, with zero NaN and Inf values.
- MEASURED negative sentinels: zero whole-shard loader, legacy bank constructor,
  selective bank constructor, internal rematerializer, and resident source-access
  attempts after detach.
- Unexplained persistent host mirror bytes: 0.

The process retained a large `RssShmem` value after detach. That is not classified as
a live model mirror: component ownership is zero and every old source Tensor weak
reference is dead. It is consistent with raw registered pages retained by the pinned
memory allocator. RSS/HWM remains supporting physical evidence in the JSON report.

This result proves steady-state fully resident decode for one N0-derived selective
Qwen3.6/NVFP4 accelerator path without an equivalent persistent host expert bank. It
does not prove post-detach prefill, local split execution, networking, multi-node
execution, generalized planning, a public strategy API, or non-NVIDIA support.

Machine-readable evidence: [`block-a.json`](block-a.json), with integrity record in
[`block-a.json.sha256`](block-a.json.sha256).
