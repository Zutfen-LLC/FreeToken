"""Physical host-materialization diagnostics for the pre-R3 research gate.

This is deliberately internal evidence tooling, not a public planner schema.
"""

from __future__ import annotations

import os
from pathlib import Path

_STATUS_FIELDS = (
    "VmRSS", "RssAnon", "RssFile", "RssShmem", "VmHWM", "VmLck", "VmSwap"
)
_SMAPS_FIELDS = (
    "Rss", "Pss", "Private_Clean", "Private_Dirty", "Shared_Clean",
    "Shared_Dirty", "Anonymous", "Locked", "Swap", "SwapPss",
)
_MEMINFO_FIELDS = (
    "MemAvailable", "MemFree", "Cached", "Shmem", "Mlocked", "SwapFree"
)


def _read_kib(path: Path, wanted: tuple[str, ...]) -> dict[str, int | None]:
    values: dict[str, int | None] = {key: None for key in wanted}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return values
    for line in lines:
        fields = line.split()
        key = fields[0].rstrip(":") if fields else ""
        if key in values and len(fields) >= 2:
            values[key] = int(fields[1]) * 1024
    return values


def snapshot_host_memory(pid: int | None = None) -> dict[str, object]:
    """Capture process and system memory without privilege or cache mutation."""
    pid = os.getpid() if pid is None else pid
    proc = Path("/proc") / str(pid)
    return {
        "pid": pid,
        "worker_alive": proc.exists(),
        "process_status_bytes": _read_kib(proc / "status", _STATUS_FIELDS),
        "process_smaps_rollup_bytes": _read_kib(
            proc / "smaps_rollup", _SMAPS_FIELDS
        ),
        "system_meminfo_bytes": _read_kib(Path("/proc/meminfo"), _MEMINFO_FIELDS),
    }


def evaluate_physical_reclamation(
    *,
    staging_bytes: int,
    before: dict[str, object],
    after: dict[str, object],
    live_source_tensor_bytes: int,
    live_source_object_count: int,
    live_storage_owner_count: int,
    worker_alive: bool,
    minimum_fraction: float = 0.80,
) -> dict[str, object]:
    """Require independent process and system signals for the physical gate."""
    if staging_bytes <= 0:
        raise ValueError("staging_bytes must be positive")
    before_status = before["process_status_bytes"]
    after_status = after["process_status_bytes"]
    before_mem = before["system_meminfo_bytes"]
    after_mem = after["system_meminfo_bytes"]
    required = (
        before_status.get("VmRSS"), after_status.get("VmRSS"),
        before_mem.get("MemAvailable"), after_mem.get("MemAvailable"),
    )
    accounting_valid = all(value is not None for value in required)
    if accounting_valid:
        process_reclaimed = max(0, required[0] - required[1])
        system_available_increase = max(0, required[3] - required[2])
        reclaimed = min(process_reclaimed, system_available_increase)
    else:
        process_reclaimed = None
        system_available_increase = None
        reclaimed = None
    fraction = None if reclaimed is None else reclaimed / staging_bytes
    counters_clear = (
        live_source_tensor_bytes == 0
        and live_source_object_count == 0
        and live_storage_owner_count == 0
    )
    passed = bool(
        accounting_valid
        and fraction is not None
        and fraction >= minimum_fraction
        and counters_clear
        and worker_alive
    )
    return {
        "accounting_valid": accounting_valid,
        "measurement_basis": "min(process VmRSS reduction, system MemAvailable increase)",
        "staging_bytes": staging_bytes,
        "process_reclaimed_bytes": process_reclaimed,
        "system_available_increase_bytes": system_available_increase,
        "reclaimed_bytes": reclaimed,
        "reclaimed_fraction_of_staging": fraction,
        "minimum_fraction": minimum_fraction,
        "live_source_tensor_bytes": live_source_tensor_bytes,
        "live_source_object_count": live_source_object_count,
        "live_storage_owner_count": live_storage_owner_count,
        "worker_alive": worker_alive,
        "tensor_counters_sufficient_without_physical_signals": False,
        "passed": passed,
    }
