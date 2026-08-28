"""Reusable two-device identity/topology probe for InferSwarm Phase-1 P1."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


def _run_nvidia_smi(args: list[str]) -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {
            "value": None,
            "unavailable": "nvidia-smi is not installed or not on PATH",
        }
    try:
        result = subprocess.run(
            [executable, *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"value": None, "unavailable": f"nvidia-smi failed: {exc!r}"}
    return {"value": result.stdout.rstrip(), "unavailable": None}


def _external_topology() -> dict[str, Any]:
    query_fields = (
        "uuid,name,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,"
        "pcie.link.width.current,pcie.link.width.max,driver_version"
    )
    gpu_query = _run_nvidia_smi(
        [f"--query-gpu={query_fields}", "--format=csv,noheader,nounits"]
    )
    structured_query: dict[str, Any]
    if gpu_query["value"] is None:
        structured_query = gpu_query
    else:
        fields = query_fields.split(",")
        rows = []
        for line in str(gpu_query["value"]).splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) != len(fields):
                return {
                    "nvidia_smi_topo_m": _run_nvidia_smi(["topo", "-m"]),
                    "nvidia_smi_topo_p2p_read": _run_nvidia_smi(["topo", "-p2p", "r"]),
                    "gpus": {
                        "value": None,
                        "unavailable": (
                            "could not parse nvidia-smi GPU query: expected "
                            f"{len(fields)} columns, got {len(values)}"
                        ),
                        "raw_csv": gpu_query["value"],
                    },
                }
            rows.append(dict(zip(fields, values, strict=True)))
        structured_query = {
            "value": rows,
            "unavailable": None,
            "raw_csv": gpu_query["value"],
        }
    return {
        "nvidia_smi_topo_m": _run_nvidia_smi(["topo", "-m"]),
        "nvidia_smi_topo_p2p_read": _run_nvidia_smi(["topo", "-p2p", "r"]),
        "gpus": structured_query,
    }


def build_probe_document(primary_spec: str, secondary_spec: str) -> dict[str, Any]:
    import torch
    from freetoken.gpu_select import (
        bind_assigned_gpu,
        parse_gpu_spec,
        resolve_gpu_uuids,
        set_assigned_gpu,
    )
    from freetoken.moe.inferswarm_secondary import probe_secondary_device

    primary = parse_gpu_spec(primary_spec)
    secondary = parse_gpu_spec(secondary_spec)
    if len(primary) != 1 or len(secondary) != 1:
        raise ValueError(
            "primary and secondary selectors must each name exactly one GPU"
        )

    primary_resolved = resolve_gpu_uuids(primary)
    secondary_resolved = resolve_gpu_uuids(secondary)
    set_assigned_gpu(primary_resolved[0] if primary_resolved else primary[0])
    primary_device = bind_assigned_gpu()
    info = probe_secondary_device(
        secondary[0],
        resolved_uuid=secondary_resolved[0] if secondary_resolved else None,
        primary_visible_ordinal=primary_device.index,
        primary_resolved_uuid=primary_resolved[0] if primary_resolved else None,
        torch_module=torch,
    )
    return {
        "schema": "inferswarm.phase1.secondary-device-probe/1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "torch_cuda_is_available": bool(torch.cuda.is_available()),
            "torch_cuda_device_count": int(torch.cuda.device_count()),
            "cudnn": (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None
            ),
        },
        "selection": {
            "primary_requested_spec": primary[0],
            "primary_resolved_uuid_nvml": (
                primary_resolved[0] if primary_resolved else None
            ),
            "secondary_requested_spec": secondary[0],
            "secondary_resolved_uuid_nvml": (
                secondary_resolved[0] if secondary_resolved else None
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "cuda_probe": info.as_dict(),
        "external_topology": _external_topology(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe two explicitly selected CUDA devices for InferSwarm Phase-1. "
            "This records capability only and allocates no expert bank."
        )
    )
    parser.add_argument("--primary-gpu", required=True)
    parser.add_argument("--secondary-gpu", required=True)
    parser.add_argument(
        "--output",
        help="write JSON to this path instead of stdout (parent directory must exist)",
    )
    args = parser.parse_args(argv)
    try:
        document = build_probe_document(args.primary_gpu, args.secondary_gpu)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output:
        # Diagnostic output is an explicit user-requested artifact, not a library side effect.
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
