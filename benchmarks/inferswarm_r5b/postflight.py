"""Capture bounded two-node state after canonical R5B shutdown."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha


def _run(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def _local_state() -> dict:
    return {
        "gpu_compute_processes_csv": _run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        ),
        "gpu_memory_csv": _run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,pci.bus_id,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ),
        "meminfo": Path("/proc/meminfo").read_text(),
        "vmstat": Path("/proc/vmstat").read_text(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-node-b", required=True)
    parser.add_argument("--physical-producer", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    remote_program = r'''import json, pathlib, subprocess
def run(command): return subprocess.check_output(command, text=True).strip()
print(json.dumps({
 "gpu_compute_processes_csv": run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"]),
 "gpu_memory_csv": run(["nvidia-smi", "--query-gpu=uuid,pci.bus_id,memory.used,memory.total", "--format=csv,noheader,nounits"]),
 "meminfo": pathlib.Path("/proc/meminfo").read_text(),
 "vmstat": pathlib.Path("/proc/vmstat").read_text(),
}))'''
    remote = json.loads(
        _run(
            shlex.split(args.ssh_node_b)
            + [f"python3 -c {shlex.quote(remote_program)}"]
        )
    )
    local = _local_state()
    result = {
        "schema": "inferswarm.r5b.post-shutdown-resource-state/1",
        "physical_producer_sha": args.physical_producer,
        "observed_at_unix_ns": time.time_ns(),
        "node_a": local,
        "node_b": remote,
        "all_gpu_execution_processes_absent": not local["gpu_compute_processes_csv"]
        and not remote["gpu_compute_processes_csv"],
    }
    write_json_with_sha(args.out, result)
    print(json.dumps({"all_gpu_execution_processes_absent": result["all_gpu_execution_processes_absent"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
