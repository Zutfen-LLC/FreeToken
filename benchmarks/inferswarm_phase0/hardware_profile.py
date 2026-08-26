"""Reproducible hardware profile for the Phase-0 RTX 3060 (ROADMAP Phase 0, issue #2).

What it captures, and where each number comes from -- no number here is invented, and every
one that cannot be read becomes an explicit null with a reason:

* **GPU / VRAM identity, driver, compute capability, PCIe link generation and width**
  (current *and* max, so a card sitting in a downgraded slot is visible) -- ``nvidia-smi``.
* **Topology** -- ``nvidia-smi topo -m`` and ``topo -p2p r``.
* **Device memory bandwidth** -- FreeToken's own ``ft bench bw`` linear pinned<->device and
  STREAM-style CPU DRAM measurements (``freetoken.moe.benchbw``), which is the trustworthy
  existing path; this module does not re-implement a bandwidth kernel.
* **PCIe / host-RAM bandwidth for the real gather** -- also ``ft bench bw``: it drives the
  production ``OffloadMoeCache.copy_missing``, which is the transfer the offload backend
  actually performs.
* **CPU / RAM / OS / driver / runtime** -- the same provenance capture the run artifact uses.
* Optionally, **single-expert NVFP4 execution latency** (``--expert-microbench``), which no
  existing FreeToken benchmark provides: ``ft bench bw`` measures *bandwidth* of the CPU MoE
  GEMV and the PCIe gather, and ``bench_offload_cache_copy.py`` measures *copy* cost.

The microbenchmark is **diagnostic only**. Per InferSwarm's benchmark contract, a faster
expert or transfer microbenchmark is never evidence that inference improved, and its numbers
are never combined arithmetically into an end-to-end claim.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any, Dict

from . import HARNESS_VERSION
from . import provenance as prov


def _bench_bw(gpu: str | None, dtype: str) -> Dict[str, Any]:
    """Run ``ft bench bw`` and return its report plus the profile path it wrote.

    ``ft bench bw`` writes one JSON profile per GPU (default under
    ``$XDG_CACHE_HOME/freetoken/benchbw/<gpu-uuid>.json``) -- that file is the artifact the
    engine itself reads, so it is the right thing to record alongside a baseline.
    """
    cmd = [sys.executable, "-m", "freetoken.cli", "bench", "bw", "--dtype", dtype]
    if gpu:
        cmd += ["--gpu", gpu]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        return {"command": cmd, "ok": False, "error": repr(e)}
    result: Dict[str, Any] = {
        "command": cmd,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-20000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    try:
        from freetoken.moe.benchbw import default_out_path

        result["profile_path"] = default_out_path(gpu if gpu and gpu.startswith("GPU-") else None)
    except Exception as e:  # noqa: BLE001 -- best effort; the stdout report still stands
        result["profile_path"] = prov.unavailable(f"could not resolve the profile path: {e!r}")
    return result


def capture_profile(
    *,
    gpu: str | None = None,
    run_bench_bw: bool = True,
    dtype: str = "nvfp4",
    expert_microbench: bool = False,
    hidden: int = 2048,
    intermediate: int = 512,
    top_k: int = 8,
) -> Dict[str, Any]:
    doc: Dict[str, Any] = {
        "schema": "inferswarm.phase0.hardware-profile/1",
        "harness_version": HARNESS_VERSION,
        "captured_at": prov.utc_now_iso(),
        "label": "MEASURED",
        "label_note": (
            "Hardware facts observed on this machine. The microbenchmark block, when "
            "present, is DIAGNOSTIC: per InferSwarm BENCHMARKING.md it never constitutes "
            "evidence about end-to-end inference and is never combined into one."
        ),
        "software": prov.software_provenance(None, HARNESS_VERSION),
        "host": prov.host_provenance(),
        "gpu": prov.gpu_provenance(gpu),
    }
    doc["bandwidth"] = (
        _bench_bw(gpu, dtype)
        if run_bench_bw
        else prov.unavailable("`ft bench bw` skipped (--no-bench-bw)")
    )
    if expert_microbench:
        from .expert_microbench import measure_single_expert_nvfp4

        doc["expert_microbenchmark"] = measure_single_expert_nvfp4(
            hidden=hidden, intermediate=intermediate, top_k=top_k, gpu=gpu
        )
    else:
        doc["expert_microbenchmark"] = prov.unavailable(
            "not requested (--expert-microbench)"
        )
    return doc
