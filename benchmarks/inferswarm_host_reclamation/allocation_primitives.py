"""Measure physical destruction behavior of bounded host allocation primitives."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import weakref
from pathlib import Path

from freetoken.research.host_reclamation import snapshot_host_memory
from freetoken.research.n0_model_block import write_json_with_sha


def _child(primitive: str, size: int) -> dict:
    import torch
    from freetoken.kernel.pinned import alloc_pinned_tensor, create_pinned_tensor_like

    before = snapshot_host_memory()
    owner = None
    if primitive == "alloc_pinned_tensor":
        tensor = alloc_pinned_tensor(size, dtype=torch.uint8)
        mechanism = "cudaHostAlloc"
    elif primitive == "create_pinned_tensor_like":
        tensor = create_pinned_tensor_like(torch.empty(size, dtype=torch.uint8))
        mechanism = "cudaMallocHost"
    elif primitive == "pin_after_fill":
        from freetoken.moe.host_banks import HostBank

        owner = HostBank((size,), torch.uint8, backing="mmap")
        tensor = owner.tensor
        mechanism = "anonymous mmap + cudaHostRegister"
    elif primitive == "ordinary_pageable":
        tensor = torch.empty(size, dtype=torch.uint8)
        mechanism = "ordinary CPU allocation"
    else:
        raise ValueError(primitive)
    tensor.fill_(0x5A)
    if owner is not None:
        owner.pin()
    torch.cuda.synchronize()
    time.sleep(0.5)
    populated = snapshot_host_memory()
    reference = weakref.ref(tensor)
    if owner is not None:
        owner.cede_tensor_ownership()
    del tensor
    gc.collect()  # Explicitly allowed only for this diagnostic.
    if reference() is not None:
        raise RuntimeError("primitive tensor remained referenced after diagnostic GC")
    release_invoked_bytes = owner.release_physical() if owner is not None else size
    torch.cuda.synchronize()
    time.sleep(1.0)
    released = snapshot_host_memory()
    rss_before = populated["process_status_bytes"]["VmRSS"]
    rss_after = released["process_status_bytes"]["VmRSS"]
    shmem_before = populated["process_status_bytes"]["RssShmem"]
    shmem_after = released["process_status_bytes"]["RssShmem"]
    return {
        "primitive": primitive,
        "allocation_mechanism": mechanism,
        "requested_bytes": size,
        "release_invoked_bytes": release_invoked_bytes,
        "snapshots": {
            "before": before,
            "populated": populated,
            "after_reference_drop_gc": released,
        },
        "process_rss_reclaimed_bytes": rss_before - rss_after,
        "process_rss_reclaimed_fraction": (rss_before - rss_after) / size,
        "process_rss_shmem_reclaimed_bytes": shmem_before - shmem_after,
        "system_memavailable_delta_bytes": (
            released["system_meminfo_bytes"]["MemAvailable"]
            - populated["system_meminfo_bytes"]["MemAvailable"]
        ),
        "process_remained_alive": True,
        "owner": owner.diagnostics() if owner is not None else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-mib", type=int, default=512)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--child", choices=(
        "alloc_pinned_tensor", "create_pinned_tensor_like", "pin_after_fill",
        "ordinary_pageable",
    ))
    args = parser.parse_args(argv)
    size = args.size_mib << 20
    if args.child:
        print(json.dumps(_child(args.child, size)))
        return 0
    if args.out is None:
        parser.error("--out is required outside child mode")
    rows = []
    for primitive in (
        "alloc_pinned_tensor", "create_pinned_tensor_like", "pin_after_fill",
        "ordinary_pageable",
    ):
        env = dict(os.environ)
        command = [
            sys.executable, "-m",
            "benchmarks.inferswarm_host_reclamation.allocation_primitives",
            "--size-mib", str(args.size_mib), "--child", primitive,
        ]
        rows.append(json.loads(subprocess.check_output(command, text=True, env=env)))
    payload = {
        "schema": "inferswarm.pre-r3.allocation-primitives/1",
        "bounded_size_bytes": size,
        "python_gc_scope": "allocation primitive diagnostic only",
        "global_page_cache_drop": False,
        "results": rows,
        "all_process_rss_reclaimed_at_least_95_percent": all(
            row["process_rss_reclaimed_fraction"] >= 0.95 for row in rows
        ),
    }
    write_json_with_sha(args.out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
