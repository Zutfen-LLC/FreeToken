"""InferSwarm Phase-1 POC secondary-device discovery and provenance.

This is intentionally a device substrate, not a worker or execution abstraction.  It
validates one explicitly selected secondary CUDA device, records its relationship to the
already-bound primary, and leaves the primary current.  It does not allocate expert banks,
move model data, or participate in model execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from freetoken.gpu_select import (
    format_gpu_uuid,
    is_gpu_uuid,
    parse_gpu_spec,
    visible_gpu_for_uuid,
)

DIRECT_PEER_CAPABLE = "direct_peer_capable"
HOST_STAGED_REQUIRED = "host_staged_required"


@dataclass(frozen=True)
class CudaDeviceIdentity:
    uuid: str | None
    visible_ordinal: int
    name: str
    total_vram_bytes: int
    free_vram_bytes_at_probe: int
    compute_capability_major: int
    compute_capability_minor: int

    def as_dict(self) -> dict[str, Any]:
        major = self.compute_capability_major
        minor = self.compute_capability_minor
        return {
            "uuid": self.uuid,
            "visible_cuda_ordinal": self.visible_ordinal,
            "name": self.name,
            "total_vram_bytes": self.total_vram_bytes,
            "free_vram_bytes_at_probe": self.free_vram_bytes_at_probe,
            "compute_capability": {
                "major": major,
                "minor": minor,
                "label": f"sm_{major}{minor}",
            },
        }


@dataclass(frozen=True)
class InferSwarmSecondaryDevice:
    requested_spec: str
    primary: CudaDeviceIdentity
    secondary: CudaDeviceIdentity
    can_access_peer_primary_to_secondary: bool
    can_access_peer_secondary_to_primary: bool
    transport_classification: str
    primary_current_after_probe: bool
    validation_passed: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "configured": True,
            "requested_secondary_spec": self.requested_spec,
            "validation_passed": self.validation_passed,
            "primary": self.primary.as_dict(),
            "secondary": self.secondary.as_dict(),
            "peer_access": {
                "primary_to_secondary": self.can_access_peer_primary_to_secondary,
                "secondary_to_primary": self.can_access_peer_secondary_to_primary,
            },
            "transport_classification": self.transport_classification,
            "primary_current_after_probe": self.primary_current_after_probe,
            "note": (
                "Capability classification only; no cross-device model execution is "
                "implemented by the Phase-1 P1 substrate."
            ),
        }


def absent_secondary_device_report() -> dict[str, Any]:
    """Stable explicit-absence shape for the existing runtime report."""
    return {
        "configured": False,
        "requested_secondary_spec": None,
        "validation_passed": None,
        "primary": None,
        "secondary": None,
        "peer_access": None,
        "transport_classification": None,
        "primary_current_after_probe": None,
        "note": "--inferswarm-secondary-gpu was not supplied",
    }


def _device_identity(torch_module, visible_ordinal: int) -> CudaDeviceIdentity:
    cuda = torch_module.cuda
    # Use a device context even on APIs that accept an ordinal.  This makes the inspection
    # boundary explicit and lets the finally block below prove that the primary is retained.
    with cuda.device(visible_ordinal):
        props = cuda.get_device_properties(visible_ordinal)
        free_bytes, _total_from_allocator = cuda.mem_get_info()
    return CudaDeviceIdentity(
        uuid=format_gpu_uuid(getattr(props, "uuid", None)),
        visible_ordinal=visible_ordinal,
        name=str(props.name),
        total_vram_bytes=int(props.total_memory),
        free_vram_bytes_at_probe=int(free_bytes),
        compute_capability_major=int(props.major),
        compute_capability_minor=int(props.minor),
    )


def _secondary_visible_ordinal(
    requested_spec: str,
    resolved_uuid: str | None,
    torch_module,
) -> int:
    spec = parse_gpu_spec(requested_spec)
    if len(spec) != 1:
        raise ValueError("--inferswarm-secondary-gpu takes exactly one GPU")
    selected = resolved_uuid or spec[0]
    if is_gpu_uuid(selected):
        try:
            return visible_gpu_for_uuid(selected, torch_module=torch_module)
        except RuntimeError as exc:
            raise ValueError(f"secondary GPU {selected}: {exc}") from exc
    return int(selected)


def probe_secondary_device(
    requested_spec: str,
    *,
    resolved_uuid: str | None,
    primary_visible_ordinal: int,
    primary_resolved_uuid: str | None = None,
    torch_module=None,
) -> InferSwarmSecondaryDevice:
    """Validate and describe one secondary while retaining the bound primary.

    ``resolved_uuid`` is the parent's NVML result when available.  When NVML is absent,
    CUDA-visible UUIDs (or a numeric visible ordinal) complete the same resolution here.
    Every failure restores the primary before propagating.
    """
    if torch_module is None:
        import torch as torch_module

    cuda = torch_module.cuda
    count = int(cuda.device_count())
    if count < 2:
        raise ValueError(
            "--inferswarm-secondary-gpu requires at least two CUDA devices visible to "
            f"this process; only {count} is available"
        )
    if not 0 <= primary_visible_ordinal < count:
        raise ValueError(
            f"primary CUDA ordinal {primary_visible_ordinal} is invalid; "
            f"only {count} device(s) are visible"
        )

    try:
        secondary_ordinal = _secondary_visible_ordinal(
            requested_spec, resolved_uuid, torch_module
        )
        if not 0 <= secondary_ordinal < count:
            raise ValueError(
                f"secondary CUDA ordinal {secondary_ordinal} is invalid; "
                f"only {count} device(s) are visible"
            )
        if secondary_ordinal == primary_visible_ordinal:
            raise ValueError(
                "--inferswarm-secondary-gpu resolves to the same physical GPU as the "
                f"primary (visible CUDA ordinal {primary_visible_ordinal})"
            )

        primary = _device_identity(torch_module, primary_visible_ordinal)
        secondary = _device_identity(torch_module, secondary_ordinal)
        secondary_uuid = secondary.uuid or resolved_uuid
        primary_uuid = primary.uuid or primary_resolved_uuid
        if primary.uuid is None and primary_uuid is not None:
            primary = CudaDeviceIdentity(
                uuid=primary_uuid,
                visible_ordinal=primary.visible_ordinal,
                name=primary.name,
                total_vram_bytes=primary.total_vram_bytes,
                free_vram_bytes_at_probe=primary.free_vram_bytes_at_probe,
                compute_capability_major=primary.compute_capability_major,
                compute_capability_minor=primary.compute_capability_minor,
            )
        if (
            primary_uuid is not None
            and secondary_uuid is not None
            and primary_uuid.upper() == secondary_uuid.upper()
        ):
            raise ValueError(
                "--inferswarm-secondary-gpu resolves to the same physical GPU as the "
                f"primary ({primary_uuid})"
            )
        if secondary.uuid is None and secondary_uuid is not None:
            secondary = CudaDeviceIdentity(
                uuid=secondary_uuid,
                visible_ordinal=secondary.visible_ordinal,
                name=secondary.name,
                total_vram_bytes=secondary.total_vram_bytes,
                free_vram_bytes_at_probe=secondary.free_vram_bytes_at_probe,
                compute_capability_major=secondary.compute_capability_major,
                compute_capability_minor=secondary.compute_capability_minor,
            )

        primary_to_secondary = bool(
            cuda.can_device_access_peer(primary_visible_ordinal, secondary_ordinal)
        )
        secondary_to_primary = bool(
            cuda.can_device_access_peer(secondary_ordinal, primary_visible_ordinal)
        )
        transport = (
            DIRECT_PEER_CAPABLE
            if primary_to_secondary and secondary_to_primary
            else HOST_STAGED_REQUIRED
        )
    finally:
        # Probe failures are startup failures, but even their diagnostic path must not leak a
        # changed current device into error handling or a caller that catches the exception.
        cuda.set_device(primary_visible_ordinal)

    primary_retained = int(cuda.current_device()) == primary_visible_ordinal
    if not primary_retained:
        raise RuntimeError(
            "secondary-device probe failed to restore the primary CUDA device "
            f"{primary_visible_ordinal}"
        )
    return InferSwarmSecondaryDevice(
        requested_spec=requested_spec,
        primary=primary,
        secondary=secondary,
        can_access_peer_primary_to_secondary=primary_to_secondary,
        can_access_peer_secondary_to_primary=secondary_to_primary,
        transport_classification=transport,
        primary_current_after_probe=primary_retained,
    )
