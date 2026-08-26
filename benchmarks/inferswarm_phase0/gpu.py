"""Physical-GPU provenance: resolve ``--gpu`` to a stable UUID, then *prove* the run used it.

Criteria section 2.1 fixes Phase 0 to **one** RTX 3060 -- the same physical card the Phase-1
candidate later uses as GPU 0. A list of visible GPUs is not proof of which one ran the
benchmark, and an nvidia-smi index is not a stable identity (indices move between boots, and
``CUDA_DEVICE_ORDER`` / ``CUDA_VISIBLE_DEVICES`` renumber them again inside a process).

So this module does three things and refuses to guess at any of them:

1. **Resolve** the ``--gpu`` selector to a full ``GPU-...`` UUID using FreeToken's own
   selector (``freetoken.gpu_select.resolve_gpu_uuids``) -- the same code ``ft serve`` and
   ``ft bench bw`` use, so a numeric index resolves exactly the way the server will resolve
   it. No second selector policy is implemented here.
2. **Record** both the resolved UUID and the physical (nvidia-smi) index it came from.
3. **Verify** the UUID the running engine reports back (``/v1/stats`` ``gpus``, which the
   scheduler fills from ``gpu_identity`` on its own bound device) against the resolved one.
   A mismatch means the measurement did not happen on the declared card, which invalidates
   the canonical campaign rather than being recorded as a note.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from . import provenance as prov
from . import validity as V

# A marketed 12 GB RTX 3060 normally reports 12,288 MiB through nvidia-smi/CUDA.  Driver
# reservations and decimal-vs-binary reporting can move the observed total slightly, so
# canonical Phase 0 accepts the inclusive 11--13 GiB class rather than exact byte equality.
# The upper bound is intentional: this experiment is not "at least 12 GB", and must reject
# 16/24 GB cards just as decisively as an 8 GB RTX 3060.
PHASE0_VRAM_MIN_BYTES = 11 << 30
PHASE0_VRAM_MAX_BYTES = 13 << 30

_PHASE0_MODEL = re.compile(
    r"^(?:NVIDIA\s+)?(?:GeForce\s+)?RTX\s+3060$", re.IGNORECASE
)
_MEMORY_VALUE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]i?B)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Phase0GpuIssue:
    """A stable canonical-hardware refusal/invalidation plus its observed evidence."""

    code: str
    message: str
    observed_name: Any
    observed_total_memory: Any

    def record(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "observed_name": self.observed_name,
            "observed_total_memory": self.observed_total_memory,
        }


def _memory_total_bytes(identity: Dict[str, Any]) -> int | None:
    raw = identity.get("total_bytes")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw
    raw = identity.get("memory.total")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        # nvidia-smi names this field in MiB even when a mocked/provider row gives a number.
        return int(raw * (1 << 20))
    if not isinstance(raw, str):
        return None
    match = _MEMORY_VALUE.fullmatch(raw)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    factors = {
        "kb": 10**3, "kib": 1 << 10,
        "mb": 10**6, "mib": 1 << 20,
        "gb": 10**9, "gib": 1 << 30,
        "tb": 10**12, "tib": 1 << 40,
    }
    return int(value * factors[unit])


def validate_phase0_gpu(identity: Dict[str, Any]) -> Phase0GpuIssue | None:
    """Prove an observed identity is exactly an RTX 3060 in the 12-GB VRAM class."""
    name = identity.get("name")
    raw_memory = identity.get("total_bytes", identity.get("memory.total"))
    normalized_name = " ".join(str(name).split()) if name is not None else ""
    if not _PHASE0_MODEL.fullmatch(normalized_name):
        return Phase0GpuIssue(
            V.GPU_UNSUPPORTED_PHASE0_MODEL,
            "canonical Phase 0 requires GPU model NVIDIA GeForce RTX 3060; "
            f"observed name={name!r}",
            name,
            raw_memory,
        )
    total_bytes = _memory_total_bytes(identity)
    if total_bytes is None or not PHASE0_VRAM_MIN_BYTES <= total_bytes <= PHASE0_VRAM_MAX_BYTES:
        return Phase0GpuIssue(
            V.GPU_UNSUPPORTED_PHASE0_VRAM,
            "canonical Phase 0 requires an RTX 3060 in the 12-GB VRAM class "
            f"({PHASE0_VRAM_MIN_BYTES}..{PHASE0_VRAM_MAX_BYTES} bytes inclusive); "
            f"observed total={raw_memory!r}, parsed_bytes={total_bytes!r}",
            name,
            raw_memory,
        )
    return None


def phase0_gpu_validation_record(identity: Dict[str, Any] | None) -> Dict[str, Any]:
    """Artifact-ready hardware-class proof without dropping the observed identity."""
    if not isinstance(identity, dict):
        return {
            "valid": False,
            "code": V.GPU_PHASE0_IDENTITY_UNPROVEN,
            "message": "the selected GPU identity was unavailable, so its Phase-0 hardware class cannot be proven",
            "observed_identity": identity,
        }
    issue = validate_phase0_gpu(identity)
    return {
        "valid": issue is None,
        "code": issue.code if issue else None,
        "message": issue.message if issue else None,
        "observed_identity": dict(identity),
        "accepted_vram_bytes": {
            "minimum_inclusive": PHASE0_VRAM_MIN_BYTES,
            "maximum_inclusive": PHASE0_VRAM_MAX_BYTES,
        },
    }


def _resolve_uuids(selector: str) -> "tuple[str, ...] | None":
    """Indirection point for tests: FreeToken's own ``--gpu`` resolution, NVML-backed."""
    from freetoken.gpu_select import resolve_gpu_uuids

    return resolve_gpu_uuids([selector])


def _is_uuid(selector: str) -> bool:
    from freetoken.gpu_select import is_gpu_uuid

    return is_gpu_uuid(selector)


@dataclass(frozen=True)
class GpuSelection:
    """The physical GPU this campaign declared, resolved as far as this host allows."""

    requested: str | None
    resolved_uuid: str | None
    physical_index: int | None
    unavailable: str | None = None

    @property
    def proven(self) -> bool:
        return bool(self.resolved_uuid)

    def record(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "requested_is_uuid": bool(self.requested and _safe_is_uuid(self.requested)),
            "resolved_uuid": self.resolved_uuid,
            "physical_index": self.physical_index,
            "unavailable": self.unavailable,
            "note": (
                "Resolved through freetoken.gpu_select, the same selector ft serve and "
                "ft bench bw use. A numeric index is resolved to its stable UUID here and "
                "both are recorded; the UUID is the identity, the index is provenance."
            ),
        }


def _safe_is_uuid(selector: str) -> bool:
    try:
        return _is_uuid(selector)
    except Exception:  # noqa: BLE001 -- identity formatting must never break a record
        return False


def _smi_index_for(uuid: str) -> int | None:
    """nvidia-smi's own index for ``uuid``, so a numeric selector stays correlatable."""
    out = prov._run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"])
    if not out:
        return None
    for line in out.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) == 2 and cells[1].upper() == uuid.upper():
            try:
                return int(cells[0])
            except ValueError:
                return None
    return None


def resolve_gpu(selector: str | None) -> GpuSelection:
    """``--gpu`` -> a stable UUID plus its physical index, or an explicit reason it could not.

    Never raises: an unresolvable selector is a recorded fact that the canonical gate then
    refuses on, so the reason survives into the artifact instead of into a traceback.
    """
    if not selector:
        return GpuSelection(
            requested=None,
            resolved_uuid=None,
            physical_index=None,
            unavailable=(
                "--gpu was not supplied; the runtime chose a card and this run cannot name "
                "which physical GPU it measured (criteria section 2.1 fixes Phase 0 to one)"
            ),
        )
    try:
        resolved = _resolve_uuids(selector)
    except (ValueError, RuntimeError) as e:
        return GpuSelection(selector, None, None, f"--gpu {selector!r} did not resolve: {e}")
    except ImportError as e:  # pragma: no cover -- freetoken is always importable in practice
        return GpuSelection(selector, None, None, f"freetoken.gpu_select unavailable: {e!r}")
    if not resolved:
        return GpuSelection(
            selector,
            None,
            None,
            (
                "NVML is unavailable on this host, so --gpu could not be resolved to a "
                "stable GPU UUID; the selector cannot be proven to name a physical card"
            ),
        )
    uuid = resolved[0]
    return GpuSelection(selector, uuid, _smi_index_for(uuid), None)


def engine_gpus(origin: str) -> List[Dict[str, Any]]:
    """The GPU identities the *running engine* reported for itself, via ``/v1/stats``.

    The scheduler fills these from ``gpu_identity(self.device.index)`` on the device it
    actually bound, so this is the engine's own answer to "which card am I on", not a
    restatement of the flag.
    """
    from .client import get_json

    try:
        stats = get_json(f"{origin}/v1/stats", timeout=15)
    except (OSError, ValueError):
        return []
    gpus = stats.get("gpus")
    return list(gpus) if isinstance(gpus, list) else []


def verify_engine_gpu(
    selection: GpuSelection, reported: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare the engine's own GPU identity against the resolved selection.

    ``matches`` is tri-state on purpose: ``True`` proven, ``False`` proven wrong, ``None``
    not provable here (an older server that ships no ``gpus``, or a host without NVML).
    Only ``True`` may satisfy a canonical run.
    """
    reported_uuids = [
        str(g.get("uuid")) for g in reported if isinstance(g, dict) and g.get("uuid")
    ]
    block: Dict[str, Any] = {
        "requested": selection.requested,
        "resolved_uuid": selection.resolved_uuid,
        "engine_reported": list(reported),
        "engine_reported_uuids": reported_uuids,
    }
    if selection.resolved_uuid is None:
        block["matches"] = None
        block["unavailable"] = selection.unavailable or "no resolved GPU UUID to compare against"
        return block
    if not reported_uuids:
        block["matches"] = None
        block["unavailable"] = (
            "the server reported no GPU identity (/v1/stats gpus was empty); the physical "
            "card that ran this arm cannot be proven"
        )
        return block
    want = selection.resolved_uuid.upper()
    matched = next(
        (
            g for g in reported
            if isinstance(g, dict) and str(g.get("uuid", "")).upper() == want
        ),
        None,
    )
    block["matches"] = matched is not None
    if not block["matches"]:
        block["mismatch"] = (
            f"engine ran on {reported_uuids}, not the requested/resolved {selection.resolved_uuid}"
        )
    else:
        block["matched_identity"] = dict(matched)
        block["phase0_hardware"] = phase0_gpu_validation_record(matched)
    return block


class GpuBindError(RuntimeError):
    """The process could not be bound to the requested physical GPU.

    Raised rather than degraded: a hardware measurement that silently benchmarks device 0
    while labelling the result as another card is worse than no measurement at all.
    """


def bind_torch_device(selector: str | None):
    """Bind *this process* to ``selector`` and return ``(device, identity, verification)``.

    Uses FreeToken's own binding path -- ``assign_gpu`` resolves the ``--gpu`` value through
    NVML and publishes it, ``bind_assigned_gpu`` matches that UUID against CUDA's own device
    list and calls ``torch.cuda.set_device``. That is what makes the result right under any
    ``CUDA_DEVICE_ORDER`` / ``CUDA_VISIBLE_DEVICES``, and it is why this does not construct
    ``torch.device("cuda")`` and hope.

    ``identity`` is read back from the *bound* device (``gpu_identity``), so the recorded
    UUID is the card that will execute the kernels, not the card that was asked for.
    """
    import torch  # noqa: F401 -- imported for its side effect of initializing CUDA below
    from freetoken.gpu_select import assign_gpu, bind_assigned_gpu, gpu_identity

    selection = resolve_gpu(selector)
    try:
        assign_gpu(selector)
        device = bind_assigned_gpu()
    except (ValueError, RuntimeError) as e:
        raise GpuBindError(f"could not bind to --gpu {selector!r}: {e}") from e
    identity = gpu_identity(device.index)
    verification: Dict[str, Any] = {
        "requested": selector,
        "resolved_uuid": selection.resolved_uuid,
        "bound_cuda_index": int(device.index),
        "bound_uuid": identity.get("uuid"),
        "bound_name": identity.get("name"),
    }
    want = selection.resolved_uuid
    got = identity.get("uuid")
    if want and got:
        verification["matches"] = str(got).upper() == str(want).upper()
    elif selector is None:
        verification["matches"] = None
        verification["unavailable"] = (
            "--gpu was not supplied; the default CUDA ordinal was bound and recorded, but "
            "no requested identity exists to check it against"
        )
    else:
        verification["matches"] = None
        verification["unavailable"] = (
            "the requested selector could not be resolved to a stable UUID on this host "
            f"({selection.unavailable or 'reason unknown'}); the bound device is recorded "
            "but cannot be proven to be the requested card"
        )
    if verification.get("matches") is False:
        raise GpuBindError(
            f"--gpu {selector!r} resolved to {want}, but this process bound {got} "
            f"(CUDA index {device.index}). Refusing to attribute a measurement to a card "
            "it did not run on."
        )
    return device, identity, verification
