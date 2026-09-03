"""#71 localization: semantic tensor capture (both arms, shared format).

Non-invasive by construction: the stage calls ``_emit_checkpoint`` at the
frozen semantic points; when no sink is installed the call is a no-op, so
uninstrumented execution is bit-identical to the pre-#71 code path (the
``if`` on a None attribute adds no tensor ops).

Capture rule (METHODOLOGY §5/§12b): ``detach`` → single ``.cpu()`` host copy
of the EXACT native tensor (no dtype conversion) → contiguous raw bytes →
SHA-256 → hand the host copy to the sink → release.  The host copy is the
only allocation, and it is freed by the sink after persisting.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_SCHEMA = "inferswarm.r6_localization.capture/1"


def tensor_sha256(host_tensor: torch.Tensor) -> str:
    raw = host_tensor.detach().contiguous()
    return hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest()


def describe_host_tensor(host_tensor: torch.Tensor) -> dict[str, Any]:
    t = host_tensor.detach().contiguous()
    nan = int(torch.isnan(t).sum().item()) if t.dtype.is_floating_point else 0
    inf = int(torch.isinf(t).sum().item()) if t.dtype.is_floating_point else 0
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "byte_count": t.numel() * t.element_size(),
        "sha256": tensor_sha256(t),
        "nan_count": nan,
        "inf_count": inf,
    }


class CaptureRecord:
    __slots__ = ("meta", "tensor")

    def __init__(self, meta: dict[str, Any], tensor: torch.Tensor):
        self.meta = meta
        self.tensor = tensor


class CaptureSink:
    """Collects checkpoint captures in memory; persists on ``save``.

    One sink per process (stage).  Keeps the host copy alive only until
    ``save`` writes the artifact bundle; callers may also drop tensors
    immediately (``keep_tensors=False``) when only hashes/metadata matter
    for the coarse pass — raw tensors are retained for the boundary proof
    and the same-input replay inputs regardless.
    """

    def __init__(self, *, role: str, host: str | None = None,
                 gpu_uuid: str | None = None, keep_tensors: bool = True):
        self.role = role
        self.host = host or socket.gethostname()
        self.gpu_uuid = gpu_uuid
        self.keep_tensors = keep_tensors
        self.records: list[CaptureRecord] = []

    def emit(self, *, checkpoint: str, step: int | None, global_layer: int | None,
             position_range: list[int] | None, source_device: str,
             tensor: torch.Tensor, extra: dict[str, Any] | None = None) -> None:
        host_copy = tensor.detach().cpu()
        meta = {
            "schema": CHECKPOINT_SCHEMA,
            "producer_sha": _producer_sha(),
            "captured_at_unix_ns": time.time_ns(),
            "host": self.host,
            "gpu_uuid": self.gpu_uuid,
            "role": self.role,
            "checkpoint": checkpoint,
            "step": step,
            "global_layer": global_layer,
            "position_range": position_range,
            "source_device": source_device,
        }
        meta.update(describe_host_tensor(host_copy))
        if extra:
            meta.update(extra)
        self.records.append(
            CaptureRecord(meta, host_copy if self.keep_tensors else None)
        )

    def save(self, out_dir: str | os.PathLike[str], tag: str) -> dict[str, Any]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        bundle_path = out / f"captures-{self.role}-{tag}.pt"
        manifest_path = out / f"manifest-{self.role}-{tag}.json"
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "role": self.role,
            "host": self.host,
            "gpu_uuid": self.gpu_uuid,
            "records": [r.meta for r in self.records],
            "tensors": [
                None if r.tensor is None else r.tensor.clone()
                for r in self.records
            ],
        }
        torch.save(payload, bundle_path)
        digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        manifest = {
            "schema": "inferswarm.r6_localization.manifest/1",
            "bundle": bundle_path.name,
            "bundle_sha256": digest,
            "bundle_bytes": bundle_path.stat().st_size,
            "record_count": len(self.records),
            "records": [r.meta for r in self.records],
            "producer_sha": _producer_sha(),
            "host": self.host,
            "platform": platform.platform(),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        return manifest


_PRODUCER_CACHE: str | None = None


def _producer_sha() -> str | None:
    global _PRODUCER_CACHE
    if _PRODUCER_CACHE is None:
        import subprocess

        try:
            repo = Path(__file__).resolve().parents[2]
            _PRODUCER_CACHE = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:
            _PRODUCER_CACHE = "unknown"
    return _PRODUCER_CACHE


__all__ = [
    "CaptureSink",
    "CHECKPOINT_SCHEMA",
    "describe_host_tensor",
    "tensor_sha256",
]
