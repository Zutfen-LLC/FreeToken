# InferSwarm R1 frozen-plan realization evidence

The committed `frozen-plan.json` is the predeclared Block A research plan. Its
embedded canonical payload digest is
`sha256:430f84e406f51d037ab8ae2844865ff4d6b60d17a1d43d54180d509682ca2281`;
the sibling SHA-256 file independently covers the exact serialized artifact.

The plan is doctrine-shaped but intentionally not a public API. It names stable
resource, logical-state, materialization, execution, authority, and memory-role
concepts while keeping the pinned model mechanics behind a research adapter.

Status: `R1_FROZEN_PLAN_REALIZATION_PASS` for the measured Block A path.

The retained run realized that exact frozen file on `inferswarm01`, using NVIDIA
GeForce RTX 3060 resource `GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099`
(12,485,525,504 VRAM bytes, compute capability 8.6), Torch 2.11.0+cu130, and
CUDA 13.0. The pinned model/revision was
`nvidia/Qwen3.6-35B-A3B-NVFP4` at
`491c2f1ea524c639598bf8fa787a93fed5a6fbce`.

Measured/calculated result:

- validation accepted 10,391,381,656 required persistent VRAM bytes before model
  materialization;
- all required materializations were realized with their exact planned resource,
  representation, and byte count;
- 8,636,596,224 bytes of host routed-state staging were released after the planned
  accelerator materialization was validated;
- all 114 staging Tensor weak references were dead after release and repeated decode;
- persistent optional and unplanned persistent model-state bytes were both zero;
- unexplained persistent host mirror bytes were zero;
- the only persistent host evidence Tensor bytes were 327,680 bytes belonging to the
  same-backend correctness fixture, not a live model-state mirror;
- prefill and four steady-state decode runs were exact (maximum absolute/relative
  deviation zero, no NaN or Inf);
- mechanical reconciliation reported no missing, unplanned, or mismatched state;
- post-release loader/rematerialization and resident source-access sentinels stayed
  at zero.

The process checkpoints retain VmRSS, VmHWM, RssAnon, RssFile, RssShmem, VmSwap,
CUDA allocated/peak, and page/swap counters. Component ownership—not RSS alone—is
authoritative for live Logical State Unit accounting because the pinned allocator may
retain released raw pages.

The physical fixture used ordinary backend-native resident-only decode and the frozen
device mapping proven by #48. It did not capture CUDA Graph decode; graph lifecycle
compatibility is structurally preserved and regression-tested, not physically
benchmarked by this result.

This proves one predeclared doctrine-shaped plan caused the pinned runtime to realize
and audit the intended arrangement correctly. It does not prove automatic planning,
candidate selection, R2 split execution, multi-GPU/network/multi-node execution,
elasticity, a public planner/strategy API, production discovery, or model/vendor
generality.

Machine-readable evidence: [`result.json`](result.json), with integrity record in
[`result.json.sha256`](result.json.sha256).
