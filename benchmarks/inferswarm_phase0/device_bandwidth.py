"""Device (VRAM) memory bandwidth of the selected GPU -- the hardware number issue #2 asks for.

Issue #2's hardware profile lists "memory bandwidth". ``ft bench bw`` does **not** measure
it: its ceilings are host DRAM (a STREAM-style CPU read) and the PCIe link (pinned host <->
device copies), plus the production CPU-MoE GEMV and PCIe expert gather. Those are the right
numbers for the offload/hybrid tuning decision and the wrong number for "how fast is this
card's VRAM". Nothing else in ``benchmarks/`` measures device-resident bandwidth either
(``bench_offload_cache_copy.py`` times ``ensure_experts``/``copy_missing``, i.e. the PCIe
path again), so this is added rather than duplicated.

The narrowest trustworthy thing that answers the question:

* one large **device-resident** buffer copied to another device-resident buffer
  (``dst.copy_(src)``, a pure D2D copy on the selected card);
* a working set far larger than L2 (default 512 MiB per buffer, so 1 GiB touched per
  repetition) -- a cache-resident buffer would report an L2 number and call it VRAM;
* byte accounting stated explicitly and not fudged: a copy **reads** ``nbytes`` and
  **writes** ``nbytes``, so ``2 * nbytes`` moves per repetition. Both conventions are
  reported, named, so a reader can use either without having to reverse-engineer which one
  this is;
* CUDA-event timing around the copy only, warmup first, and **every repetition kept** --
  the distribution is the result, not a single opaque number.

**Diagnostic / hardware-profile only.** Per InferSwarm's benchmark contract a microbenchmark
diagnoses, it never concludes: this number is never combined arithmetically into an
end-to-end inference claim.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List

from . import provenance as prov
from .gpu import GpuBindError, bind_torch_device

METHOD = (
    "CUDA-event elapsed time around a device-to-device torch.Tensor.copy_ between two "
    "device-resident buffers of `buffer_bytes` each, on the bound GPU. Bytes moved per "
    "repetition = 2 x buffer_bytes (one read + one write). Warmup repetitions are discarded; "
    "every measured repetition is reported individually."
)

# 512 MiB per buffer: comfortably beyond any consumer L2 (the RTX 3060's is 3 MiB), so the
# copy is served from VRAM rather than from cache, while leaving room on a 12 GB card.
DEFAULT_BUFFER_BYTES = 512 << 20


def _summary(values: List[float]) -> Dict[str, Any]:
    ordered = sorted(values)
    n = len(ordered)
    mean = statistics.fmean(ordered)
    stdev = statistics.stdev(ordered) if n > 1 else 0.0
    return {
        "n": n,
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
        "mean": mean,
        "stdev": stdev,
        "cv_percent": (stdev / mean * 100.0) if mean else None,
    }


def measure_device_memory_bandwidth(
    *,
    gpu: str | None = None,
    buffer_bytes: int = DEFAULT_BUFFER_BYTES,
    repetitions: int = 30,
    warmup: int = 5,
) -> Dict[str, Any]:
    """D2D copy bandwidth of the selected GPU, as a distribution.

    Returns an explicit null with a reason when torch or CUDA is unavailable, and refuses
    (rather than mislabels) when the process cannot be bound to the requested card.
    """
    try:
        import torch
    except ImportError as e:
        return prov.unavailable(f"torch is not importable: {e!r}")
    if not torch.cuda.is_available():
        return prov.unavailable("no CUDA device available")
    if buffer_bytes <= 0 or repetitions < 1 or warmup < 0:
        return prov.unavailable(
            f"invalid geometry: buffer_bytes={buffer_bytes} repetitions={repetitions} "
            f"warmup={warmup}"
        )

    try:
        device, identity, verification = bind_torch_device(gpu)
    except GpuBindError as e:
        return prov.unavailable(str(e))

    elements = buffer_bytes // 2  # float16: 2 bytes per element
    try:
        src = torch.empty(elements, dtype=torch.float16, device=device)
        dst = torch.empty(elements, dtype=torch.float16, device=device)
        src.fill_(1.0)
    except RuntimeError as e:  # OOM on a card too small for this working set
        return prov.unavailable(f"could not allocate 2 x {buffer_bytes} bytes on {device}: {e!r}")

    moved_bytes = 2 * (elements * src.element_size())
    per_rep_ms: List[float] = []
    try:
        for _ in range(warmup):
            dst.copy_(src)
        torch.cuda.synchronize(device)
        for _ in range(repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dst.copy_(src)
            end.record()
            end.synchronize()
            per_rep_ms.append(float(start.elapsed_time(end)))
    except Exception as e:  # noqa: BLE001 -- a copy that cannot run is a recorded fact
        return prov.unavailable(f"device copy failed: {e!r}")
    finally:
        del src, dst
        torch.cuda.empty_cache()

    if not per_rep_ms or min(per_rep_ms) <= 0.0:
        return prov.unavailable(
            f"non-positive CUDA-event timings ({per_rep_ms[:5]}); nothing to divide"
        )

    read_write_gbs = [moved_bytes / (ms / 1e3) / 1e9 for ms in per_rep_ms]
    read_only_gbs = [g / 2.0 for g in read_write_gbs]
    return {
        "label": "MEASURED",
        "diagnostic_only": True,
        "diagnostic_note": (
            "Hardware profile only. Per InferSwarm BENCHMARKING.md a microbenchmark "
            "diagnoses and never concludes: this bandwidth is never extrapolated into an "
            "end-to-end inference claim, and it is not what `ft bench bw` measures (that is "
            "host DRAM and the PCIe link)."
        ),
        "method": METHOD,
        "kernel": "torch.Tensor.copy_ (device-to-device)",
        "device": {
            "name": identity.get("name"),
            "uuid": identity.get("uuid"),
            "cuda_index": identity.get("index"),
            "total_bytes": identity.get("total_bytes"),
        },
        "gpu_verification": verification,
        "geometry": {
            "buffer_bytes": int(elements * 2),
            "dtype": "float16",
            "elements": int(elements),
            "bytes_moved_per_repetition": int(moved_bytes),
            "byte_accounting": "read + write (2 x buffer_bytes)",
        },
        "warmup": warmup,
        "repetitions": repetitions,
        # Raw repetitions, not just a summary: criteria section 10's rule that every
        # repetition is preserved is the house style, and a single opaque number hides
        # exactly the variance a hardware profile is supposed to expose.
        "per_repetition_ms": per_rep_ms,
        "per_repetition_gbs_read_write": read_write_gbs,
        "per_repetition_gbs_read_only": read_only_gbs,
        "summary_ms": _summary(per_rep_ms),
        "summary_gbs_read_write": _summary(read_write_gbs),
        "summary_gbs_read_only": _summary(read_only_gbs),
    }
