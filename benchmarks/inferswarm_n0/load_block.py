from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import torch

from freetoken.research.n0_model_block import (
    ModelBlockSpec,
    load_selective_qwen35_block,
    write_json_with_sha,
)


def _proc_status() -> dict:
    values = {}
    wanted = {"VmPeak", "VmSize", "VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem"}
    for line in Path("/proc/self/status").read_text().splitlines():
        key = line.split(":", 1)[0]
        if key in wanted:
            values[key + "_kib"] = int(line.split()[1])
    return values


def _vmstat() -> dict:
    wanted = {"pswpin", "pswpout", "pgmajfault"}
    return {
        fields[0]: int(fields[1])
        for line in Path("/proc/vmstat").read_text().splitlines()
        if (fields := line.split())[0] in wanted
    }


def _mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable absent")


def _io() -> dict:
    return {
        key: int(value)
        for line in Path("/proc/self/io").read_text().splitlines()
        for key, value in [line.split(":", 1)]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--block", choices=("a", "b"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    block_plan = plan[f"block_{args.block}"]
    raw_spec = block_plan["spec"]
    spec = ModelBlockSpec(**raw_spec)
    allowed = frozenset(block_plan["allowed_tensor_keys"])

    # Sentinels make the physical run fail immediately if either forbidden authority is used.
    import safetensors.torch
    from freetoken.models import nvfp4_banks

    safetensors.torch.load_file = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("whole-shard load_file used")
    )
    nvfp4_banks.load_nvfp4_expert_source_banks = lambda *a, **k: (
        (_ for _ in ()).throw(AssertionError("legacy full expert-bank constructor used"))
    )

    before_status = _proc_status()
    before_vm = _vmstat()
    before_io = _io()
    before_available = _mem_available()
    gpu_before = torch.cuda.memory_allocated(0), torch.cuda.max_memory_allocated(0)
    started = time.monotonic()
    result = load_selective_qwen35_block(
        args.model, spec, allowed, device=torch.device("cuda:0")
    )
    torch.cuda.synchronize(0)
    elapsed = time.monotonic() - started
    after_status = _proc_status()
    after_vm = _vmstat()
    after_io = _io()
    after_available = _mem_available()
    expert_bank_bytes = sum(
        tensor.numel() * tensor.element_size()
        for layers in result.expert_banks.values()
        for tensor in layers
    )
    fetched_set = set(result.fetched_keys)
    payload = {
        "schema": "inferswarm.n0.block-load/1",
        "block": args.block.upper(),
        "free_token_sha": os.popen("git rev-parse HEAD").read().strip(),
        "model_repository": plan["model_repository"],
        "revision": plan["revision"],
        "spec": raw_spec,
        "allowed_tensor_key_count": len(allowed),
        "allowed_tensor_keys": sorted(allowed),
        "fetched_tensor_key_count": len(result.fetched_keys),
        "unique_fetched_tensor_key_count": len(fetched_set),
        "fetched_tensor_keys": result.fetched_keys,
        "unexpected_fetched_keys": sorted(fetched_set - allowed),
        "allowed_but_not_fetched_keys": sorted(allowed - fetched_set),
        "checkpoint_bytes_logically_owned": block_plan["owned_checkpoint_bytes"],
        "checkpoint_bytes_fetched": result.fetched_bytes,
        "expert_bank_final_host_bytes": expert_bank_bytes,
        "global_layer_ids": list(result.global_layer_ids),
        "full_expert_bank_constructor_used": False,
        "whole_shard_helper_used": False,
        "elapsed_load_seconds": elapsed,
        "process_before": before_status,
        "process_after": after_status,
        "ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "process_io_delta": {key: after_io[key] - before_io[key] for key in after_io},
        "vmstat_delta": {key: after_vm[key] - before_vm[key] for key in after_vm},
        "mem_available_before_bytes": before_available,
        "mem_available_after_bytes": after_available,
        "cuda_memory_before_bytes": gpu_before[0],
        "cuda_peak_before_bytes": gpu_before[1],
        "cuda_memory_after_bytes": torch.cuda.memory_allocated(0),
        "cuda_peak_after_bytes": torch.cuda.max_memory_allocated(0),
    }
    write_json_with_sha(args.out, payload)
    print(json.dumps({
        key: payload[key] for key in (
            "block", "allowed_tensor_key_count", "unique_fetched_tensor_key_count",
            "checkpoint_bytes_logically_owned", "checkpoint_bytes_fetched",
            "expert_bank_final_host_bytes", "unexpected_fetched_keys", "process_after",
            "ru_maxrss_kib", "vmstat_delta", "cuda_memory_after_bytes",
            "cuda_peak_after_bytes", "elapsed_load_seconds",
        )
    }, indent=2))
    if args.hold_seconds:
        time.sleep(args.hold_seconds)


if __name__ == "__main__":
    main()
