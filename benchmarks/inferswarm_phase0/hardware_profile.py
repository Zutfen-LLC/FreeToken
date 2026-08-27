"""Reproducible hardware profile for the Phase-0 RTX 3060 (ROADMAP Phase 0, issue #2).

What it captures, and where each number comes from -- no number here is invented, and every
one that cannot be read becomes an explicit null with a reason:

* **GPU / VRAM identity, driver, compute capability, PCIe link generation and width**
  (current *and* max, so a card sitting in a downgraded slot is visible) -- ``nvidia-smi``.
  The ``--gpu`` selector is resolved to a stable UUID first (``gpu.resolve_gpu``) and every
  child measurement is given that UUID, so the bench, the microbenchmarks and the sweep
  provably touch one physical card rather than three selectors that ought to agree.
* **Topology** -- ``nvidia-smi topo -m`` and ``topo -p2p r``.
* **Host DRAM and PCIe bandwidth for the real gather** -- FreeToken's own ``ft bench bw``
  (``freetoken.moe.benchbw``): a STREAM-style CPU DRAM read, linear pinned<->device copies,
  the production CPU MoE GEMV, and the production ``OffloadMoeCache.copy_missing`` gather.
  This module does not re-implement any of them.
* **Device (VRAM) memory bandwidth** (``--device-bandwidth``) -- issue #2 asks for the
  card's memory bandwidth, and ``ft bench bw`` does not measure it: its ceilings are host
  DRAM and the PCIe link. See ``device_bandwidth.py`` for the method and the byte accounting.
* **CPU / RAM / OS / driver / runtime** -- the same provenance capture the run artifact uses.
* Optionally, **single-expert NVFP4 execution latency** (``--expert-microbench``), measured
  at ``top_k = 1`` so it is a latency and not an amortized share of a grouped call, plus a
  separately-named grouped top-k step diagnostic.

The microbenchmarks are **diagnostic only**. Per InferSwarm's benchmark contract, a faster
expert or transfer microbenchmark is never evidence that inference improved, and its numbers
are never combined arithmetically into an end-to-end claim.
"""

from __future__ import annotations

import sys
from typing import Any, Dict

from . import HARNESS_VERSION
from . import bench_bw as bench_bw_mod
from . import provenance as prov
from .gpu import GpuSelection, resolve_gpu


def _bench_bw(selection: GpuSelection, dtype: str) -> Dict[str, Any]:
    """Run ``ft bench bw`` on the resolved card and pin the profile it wrote.

    ``ft bench bw`` writes one JSON profile per GPU (default under
    ``$XDG_CACHE_HOME/freetoken/benchbw/<gpu-uuid>.json``) -- that file is the artifact the
    engine itself reads, so it is the right thing to record alongside a baseline, together
    with its sha256 and a check that it names this card.
    """
    return bench_bw_mod.run_bench_bw(
        python_executable=sys.executable, selection=selection, dtype=dtype
    ).record


def capture_profile(
    *,
    gpu: str | None = None,
    run_bench_bw: bool = True,
    dtype: str = "nvfp4",
    expert_microbench: bool = False,
    hidden: int = 2048,
    intermediate: int = 512,
    top_k: int = 8,
    include_grouped: bool = True,
    device_bandwidth: bool = False,
    device_bandwidth_bytes: int = 512 << 20,
    device_bandwidth_reps: int = 30,
) -> Dict[str, Any]:
    selection = resolve_gpu(gpu)
    doc: Dict[str, Any] = {
        "schema": "inferswarm.phase0.hardware-profile/2",
        "harness_version": HARNESS_VERSION,
        "captured_at": prov.utc_now_iso(),
        "label": "MEASURED",
        "label_note": (
            "Hardware facts observed on this machine. The microbenchmark blocks, when "
            "present, are DIAGNOSTIC: per InferSwarm BENCHMARKING.md they never constitute "
            "evidence about end-to-end inference and are never combined into one."
        ),
        "software": prov.software_provenance(None, HARNESS_VERSION),
        "host": prov.host_provenance(),
        "gpu": prov.gpu_provenance(gpu, selection.resolved_uuid),
        "gpu_selection": selection.record(),
    }
    doc["bandwidth_host_and_pcie"] = (
        _bench_bw(selection, dtype)
        if run_bench_bw
        else prov.unavailable("`ft bench bw` skipped (--no-bench-bw)")
    )
    # Kept under the old key too: it is what earlier profiles called this block, and a
    # consumer that reads `bandwidth` should not silently get nothing.
    doc["bandwidth"] = doc["bandwidth_host_and_pcie"]
    doc["bandwidth_note"] = (
        "`ft bench bw` measures HOST DRAM and the PCIe link (plus the production CPU MoE "
        "GEMV and PCIe expert gather). It is not device/VRAM bandwidth -- that is the "
        "device_memory_bandwidth block."
    )

    if device_bandwidth:
        from .device_bandwidth import measure_device_memory_bandwidth

        doc["device_memory_bandwidth"] = measure_device_memory_bandwidth(
            gpu=selection.resolved_uuid or gpu,
            buffer_bytes=device_bandwidth_bytes,
            repetitions=device_bandwidth_reps,
        )
    else:
        doc["device_memory_bandwidth"] = prov.unavailable("not requested (--device-bandwidth)")

    if expert_microbench:
        from .expert_microbench import measure_single_expert_nvfp4

        doc["expert_microbenchmark"] = measure_single_expert_nvfp4(
            hidden=hidden,
            intermediate=intermediate,
            top_k=top_k,
            gpu=selection.resolved_uuid or gpu,
            include_grouped=include_grouped,
        )
    else:
        doc["expert_microbenchmark"] = prov.unavailable(
            "not requested (--expert-microbench)"
        )
    return doc
