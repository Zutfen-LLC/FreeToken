"""Single-expert NVFP4 execution latency -- the one Phase-0 hardware number no existing
FreeToken benchmark provides.

What already exists, and why none of it answers this question:

* ``ft bench bw`` (``freetoken.moe.benchbw``) measures **bandwidth**: STREAM-style CPU DRAM,
  a linear pinned<->device copy, the real CPU MoE GEMV, and the real PCIe gather. It never
  times the GPU NVFP4 expert GEMM.
* ``benchmarks/bench_offload_cache_copy.py`` measures **copy** cost (``ensure_experts`` +
  ``copy_missing``), not compute.
* ``benchmarks/bench_decode_moe.py`` measures **end-to-end** decode through the server.
* ``tests/moe/test_nvfp4_backends.py`` checks numerics, not time.

So this module times the production decode kernel itself
(``fused_experts_decode_nvfp4_marlin`` over native ModelOpt banks -- the "Marlin-style" name
is the wide-load dequant, not the vLLM Marlin backend) at M=1 over a GPU-resident slot cache.

**Two measurements, never confused with each other.**

``single_expert``
    ``top_k = 1``: one routed expert, one call, timed directly. This is the actual latency of
    executing one expert, which is what "single-expert execution latency" means.

``grouped_topk`` (optional)
    ``top_k = k``: the whole routed-expert step of a real decode token, timed as one call and
    reported as one call. It is **not** divided by ``k``. Expert work inside a grouped call
    executes concurrently, so ``step_ms / k`` is an amortized throughput-like quantity, not a
    latency -- dividing it and calling the result per-expert latency would fabricate a number
    the hardware never produced. The grouped figure is labelled as a grouped/batched
    diagnostic and stands on its own.

**Diagnostic only.** InferSwarm's benchmark contract is explicit that a faster expert
microbenchmark is not evidence that inference improved, and neither number here is ever
combined arithmetically into an end-to-end claim. It exists to explain an end-to-end result,
not to stand in for one.
"""

from __future__ import annotations

from typing import Any, Dict

from . import provenance as prov
from .gpu import GpuBindError, bind_torch_device

SINGLE_EXPERT_MEASUREMENT = (
    "CUDA-event elapsed time of fused_experts_decode_nvfp4_marlin at M=1 with top_k=1 -- ONE "
    "routed expert per call -- over a GPU-resident NVFP4 slot cache, averaged over "
    "repetitions after warmup. Expert weights are already resident: this is compute, with NO "
    "PCIe transfer in the interval."
)

GROUPED_MEASUREMENT = (
    "CUDA-event elapsed time of ONE grouped routed-expert step: fused_experts_decode_nvfp4_"
    "marlin at M=1 with the model's real top_k, over a GPU-resident NVFP4 slot cache. This is "
    "the latency of the whole step, reported as such. It is deliberately NOT divided by "
    "top_k: the experts in a grouped call execute concurrently, so step_ms / top_k is an "
    "amortized throughput-like quantity and not single-expert latency."
)


def _synthetic_banks(torch, device, slots: int, hidden: int, inter: int) -> Dict[str, Any]:
    """GPU-resident native-layout NVFP4 banks, in the shapes the decode kernel reads.

    Byte *values* are arbitrary -- every e2m1 nibble decodes to a finite value, so the
    kernel does the same work whatever the bits say. Scales are set to a mid-range e4m3
    value rather than left uninitialized so no product lands in the denormal range.
    """
    g = torch.Generator(device="cpu").manual_seed(0)

    def packed(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g).to(device)

    def scale(*shape):
        return torch.full(shape, 1.0, dtype=torch.float32).to(torch.float8_e4m3fn).to(device)

    return {
        "gate_up_packed": packed(slots, 2 * inter, hidden // 2),
        "gate_up_scale": scale(slots, 2 * inter, hidden // 16),
        "gate_up_global": torch.full((slots, 2 * inter), 1.0, dtype=torch.float16, device=device),
        "down_packed": packed(slots, hidden, inter // 2),
        "down_scale": scale(slots, hidden, inter // 16),
        "down_global": torch.full((slots, hidden), 1.0, dtype=torch.float16, device=device),
    }


def per_expert_weight_bytes(hidden: int, intermediate: int) -> int:
    """Resident weight bytes one expert's decode step touches, from the native bank layout.

    gate_up ``[2I, H/2]`` packed + ``[2I, H/16]`` scale + ``[2I]`` global, down ``[H, I/2]`` +
    ``[H, I/16]`` + ``[H]``.
    """
    return (
        2 * intermediate * (hidden // 2 + hidden // 16 + 2)
        + hidden * (intermediate // 2 + intermediate // 16 + 2)
    )


def validate_geometry(hidden: int, intermediate: int, top_k: int, slots: int) -> str | None:
    """The reason this geometry cannot be benched, or None. Pure: testable without a GPU."""
    for value, name in ((hidden, "hidden"), (intermediate, "intermediate")):
        if value <= 0:
            return f"{name} must be positive, got {value}"
        if value % 16:
            return f"{name} must be a multiple of 16 for the NVFP4 bank layout, got {value}"
    if top_k < 1:
        return f"top_k must be at least 1, got {top_k}"
    if slots < top_k:
        return f"cache_slots {slots} cannot hold top_k {top_k} distinct routed experts"
    return None


def _time_kernel(torch, perf_cuda, fn, *, warmup: int, repetitions: int, device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    return perf_cuda(fn, repetitions=repetitions, cuda_graph_repetitions=None)


def measure_single_expert_nvfp4(
    *,
    hidden: int = 2048,
    intermediate: int = 512,
    top_k: int = 8,
    slots: int | None = None,
    repetitions: int = 200,
    warmup: int = 20,
    gpu: str | None = None,
    include_grouped: bool = True,
) -> Dict[str, Any]:
    """True single-expert NVFP4 decode-GEMV latency, plus an optional grouped-step diagnostic.

    Defaults are Qwen3.6-35B-A3B's routed-expert shape (hidden 2048, moe_intermediate 512,
    top_k 8), which is also ``ft bench bw``'s ``qwen3.6-moe`` workload. ``top_k`` sizes the
    *grouped* diagnostic only; the single-expert measurement is always ``top_k = 1``.

    Returns explicit nulls with reasons when CUDA or the Triton kernels are unavailable, and
    refuses outright when the process cannot be bound to the requested GPU -- a hardware
    number attributed to the wrong card is worse than no number.
    """
    try:
        import torch
    except ImportError as e:
        return prov.unavailable(f"torch is not importable: {e!r}")
    if not torch.cuda.is_available():
        return prov.unavailable("no CUDA device available")

    slots = slots or max(top_k, 32)
    bad = validate_geometry(hidden, intermediate, top_k, slots)
    if bad:
        return prov.unavailable(bad)

    try:
        device, identity, verification = bind_torch_device(gpu)
    except GpuBindError as e:
        return prov.unavailable(str(e))

    try:
        from freetoken.benchmark.perf import perf_cuda
        from freetoken.moe.fused_nvfp4 import fused_experts_decode_nvfp4_marlin
    except ImportError as e:
        return prov.unavailable(f"FreeToken kernels are not importable: {e!r}")

    banks = _synthetic_banks(torch, device, slots, hidden, intermediate)
    hidden_states = (torch.randn(1, hidden, dtype=torch.bfloat16, device=device) / 4)

    def step_for(k: int):
        # Distinct slots, as a decode step's top-k routing would produce.
        topk_ids = torch.arange(k, dtype=torch.int32, device=device).view(1, k) % slots
        topk_weights = torch.full((1, k), 1.0 / k, dtype=torch.float32, device=device)

        def step():
            return fused_experts_decode_nvfp4_marlin(
                hidden_states,
                banks["gate_up_packed"], banks["gate_up_scale"], banks["gate_up_global"],
                banks["down_packed"], banks["down_scale"], banks["down_global"],
                topk_weights, topk_ids, "silu", False,
            )

        return step

    expert_bytes = per_expert_weight_bytes(hidden, intermediate)
    common: Dict[str, Any] = {
        "label": "MEASURED",
        "diagnostic_only": True,
        "diagnostic_note": (
            "Per InferSwarm BENCHMARKING.md this is a microbenchmark: it diagnoses, it does "
            "not conclude, and it is never mixed into an end-to-end throughput claim."
        ),
        "kernel": "freetoken.moe.fused_nvfp4.fused_experts_decode_nvfp4_marlin",
        "device": {
            "name": identity.get("name"),
            "uuid": identity.get("uuid"),
            "cuda_index": identity.get("index"),
        },
        "gpu_verification": verification,
        "compute_capability": ".".join(str(x) for x in torch.cuda.get_device_capability(device)),
        "geometry": {
            "hidden": hidden,
            "moe_intermediate": intermediate,
            "cache_slots": slots,
            "batch_tokens": 1,
        },
        "per_expert_weight_bytes": expert_bytes,
        "warmup": warmup,
        "repetitions": repetitions,
    }

    try:
        single_ms = _time_kernel(
            torch, perf_cuda, step_for(1), warmup=warmup, repetitions=repetitions, device=device
        )
    except Exception as e:  # noqa: BLE001 -- a kernel that cannot run is a recorded fact
        return {**common, "single_expert": prov.unavailable(f"kernel execution failed: {e!r}")}

    out: Dict[str, Any] = {
        **common,
        "single_expert": {
            "measurement": SINGLE_EXPERT_MEASUREMENT,
            "top_k": 1,
            "latency_ms": single_ms,
            "weight_read_gbs": (expert_bytes / (single_ms / 1e3) / 1e9) if single_ms > 0 else None,
        },
    }

    if not include_grouped or top_k <= 1:
        out["grouped_topk"] = prov.unavailable(
            "not requested" if not include_grouped
            else f"top_k={top_k} is not a grouped step; the single-expert figure already covers it"
        )
        return out

    try:
        grouped_ms = _time_kernel(
            torch, perf_cuda, step_for(top_k), warmup=warmup, repetitions=repetitions,
            device=device,
        )
    except Exception as e:  # noqa: BLE001
        out["grouped_topk"] = prov.unavailable(f"kernel execution failed: {e!r}")
        return out

    out["grouped_topk"] = {
        "measurement": GROUPED_MEASUREMENT,
        "diagnostic_kind": "grouped/batched routed-expert step latency",
        "top_k": top_k,
        "step_ms": grouped_ms,
        # Named to make the prohibition explicit where a reader would otherwise reach for
        # the division themselves.
        "per_expert_ms_deliberately_absent": (
            "experts inside a grouped call execute concurrently, so step_ms / top_k is an "
            "amortized quantity, not single-expert latency; see single_expert above, which "
            "measures top_k=1 directly"
        ),
        "grouped_weight_read_gbs": (
            (expert_bytes * top_k) / (grouped_ms / 1e3) / 1e9 if grouped_ms > 0 else None
        ),
    }
    return out
