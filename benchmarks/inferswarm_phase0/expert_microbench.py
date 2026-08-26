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
(``fused_experts_decode_nvfp4_marlin`` over native ModelOpt banks -- the "Marlin-style"
name is the wide-load dequant, not the vLLM Marlin backend) at M=1, over a GPU-resident
slot cache, and divides by ``top_k`` to state a per-expert figure.

**Diagnostic only.** InferSwarm's benchmark contract is explicit that a faster expert
microbenchmark is not evidence that inference improved, and this number must never be
combined arithmetically into an end-to-end claim. It exists to explain an end-to-end
result, not to stand in for one.
"""

from __future__ import annotations

from typing import Any, Dict

from . import provenance as prov

MEASUREMENT = (
    "CUDA-event elapsed time of fused_experts_decode_nvfp4_marlin at M=1 over a "
    "GPU-resident NVFP4 slot cache, averaged over repetitions after warmup. Expert weights "
    "are already resident: this is compute, with NO PCIe transfer in the interval."
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


def measure_single_expert_nvfp4(
    *,
    hidden: int = 2048,
    intermediate: int = 512,
    top_k: int = 8,
    slots: int | None = None,
    repetitions: int = 200,
    warmup: int = 20,
    gpu: str | None = None,
) -> Dict[str, Any]:
    """Per-expert NVFP4 decode-GEMV latency at the Phase-1 geometry.

    Defaults are Qwen3.6-35B-A3B's routed-expert shape (hidden 2048, moe_intermediate 512,
    top_k 8), which is also ``ft bench bw``'s ``qwen3.6-moe`` workload. Returns explicit
    nulls with reasons when CUDA or the Triton kernels are unavailable.
    """
    try:
        import torch
    except ImportError as e:
        return prov.unavailable(f"torch is not importable: {e!r}")
    if not torch.cuda.is_available():
        return prov.unavailable("no CUDA device available")
    for bad, name in ((hidden % 16, "hidden"), (intermediate % 16, "intermediate")):
        if bad:
            return prov.unavailable(f"{name} must be a multiple of 16 for the NVFP4 bank layout")

    device = torch.device("cuda")
    slots = slots or max(top_k, 32)
    try:
        from freetoken.benchmark.perf import perf_cuda
        from freetoken.moe.fused_nvfp4 import fused_experts_decode_nvfp4_marlin
    except ImportError as e:
        return prov.unavailable(f"FreeToken kernels are not importable: {e!r}")

    banks = _synthetic_banks(torch, device, slots, hidden, intermediate)
    hidden_states = (torch.randn(1, hidden, dtype=torch.bfloat16, device=device) / 4)
    # Distinct slots, as a decode step's top-k routing would produce.
    topk_ids = torch.arange(top_k, dtype=torch.int32, device=device).view(1, top_k) % slots
    topk_weights = torch.full((1, top_k), 1.0 / top_k, dtype=torch.float32, device=device)

    def step():
        return fused_experts_decode_nvfp4_marlin(
            hidden_states,
            banks["gate_up_packed"], banks["gate_up_scale"], banks["gate_up_global"],
            banks["down_packed"], banks["down_scale"], banks["down_global"],
            topk_weights, topk_ids, "silu", False,
        )

    try:
        for _ in range(warmup):
            step()
        torch.cuda.synchronize(device)
        ms = perf_cuda(step, repetitions=repetitions, cuda_graph_repetitions=None)
    except Exception as e:  # noqa: BLE001 -- a kernel that cannot run is a recorded fact
        return prov.unavailable(f"kernel execution failed: {e!r}")

    # Resident weight bytes touched per decode step, from the native NVFP4 bank layout:
    # gate_up [2I, H/2] packed + [2I, H/16] scale + [2I] global, down [H, I/2] + [H, I/16] + [H].
    per_expert_bytes = (
        2 * intermediate * (hidden // 2 + hidden // 16 + 2)
        + hidden * (intermediate // 2 + intermediate // 16 + 2)
    )
    return {
        "label": "MEASURED",
        "diagnostic_only": True,
        "diagnostic_note": (
            "Per InferSwarm BENCHMARKING.md this is a microbenchmark: it diagnoses, it does "
            "not conclude, and it is never mixed into an end-to-end throughput claim."
        ),
        "measurement": MEASUREMENT,
        "kernel": "freetoken.moe.fused_nvfp4.fused_experts_decode_nvfp4_marlin",
        "geometry": {
            "hidden": hidden, "moe_intermediate": intermediate, "top_k": top_k,
            "cache_slots": slots, "batch_tokens": 1,
        },
        "device": torch.cuda.get_device_name(device),
        "compute_capability": ".".join(str(x) for x in torch.cuda.get_device_capability(device)),
        "gpu_selector": gpu or prov.unavailable("--gpu not supplied; CUDA device 0 was used"),
        "repetitions": repetitions,
        "step_ms": ms,
        "per_expert_ms": ms / top_k if top_k else None,
        "per_expert_weight_bytes": per_expert_bytes,
        "effective_weight_read_gbs": (
            (per_expert_bytes * top_k) / (ms / 1e3) / 1e9 if ms > 0 else None
        ),
    }
