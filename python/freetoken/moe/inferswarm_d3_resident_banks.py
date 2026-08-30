"""D3-only dual resident storage; deliberately no route or execution surface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .inferswarm_d3_placement import D3Placement
from .inferswarm_resident_bank import SecondaryResidentExpertBank, load_secondary_resident_bank
from .inferswarm_secondary import probe_secondary_device

WORKER_A_UUID = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
WORKER_B_UUID = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
PRIMARY_UUID = "GPU-ecda1aaa-0c66-857b-8218-3d511dc75c03"


@dataclass(frozen=True, slots=True)
class D3ResidentExpertBanks:
    worker_a: SecondaryResidentExpertBank | None
    worker_b: SecondaryResidentExpertBank | None

    @property
    def total_native_expert_bytes(self) -> int:
        return sum(bank.report.placement.remote_resident_bytes for bank in (self.worker_a, self.worker_b) if bank is not None)


def probe_d3_workers(*, active_workers: tuple[str, ...] = ("a", "b"), worker_a_spec: str | None, worker_b_spec: str | None, worker_a_uuid: str | None,
                     worker_b_uuid: str | None, primary_visible_ordinal: int,
                     primary_resolved_uuid: str | None, torch_module=None):
    """Resolve the two named physical workers and restore GPU0 after both probes."""
    selectors = {"a": (worker_a_spec, worker_a_uuid, WORKER_A_UUID), "b": (worker_b_spec, worker_b_uuid, WORKER_B_UUID)}
    if not active_workers or any(label not in selectors for label in active_workers):
        raise ValueError("D3 active worker shape is invalid")
    if any((uuid or spec) is None or (uuid or spec).upper() != expected.upper() for label in active_workers for spec, uuid, expected in (selectors[label],)):
        raise ValueError("D3 worker selectors must resolve to the frozen worker-A and worker-B physical UUIDs")
    if primary_resolved_uuid is not None and primary_resolved_uuid.upper() != PRIMARY_UUID.upper():
        raise ValueError("D3 primary must resolve to the frozen GPU0 physical UUID")
    probed = {}
    for label in active_workers:
        spec, uuid, _ = selectors[label]
        probed[label] = probe_secondary_device(spec, resolved_uuid=uuid, primary_visible_ordinal=primary_visible_ordinal, primary_resolved_uuid=primary_resolved_uuid, torch_module=torch_module)
    a, b = probed.get("a"), probed.get("b")
    uuids = (a or b).primary.uuid, *(worker.secondary.uuid for worker in probed.values())
    if any(value is None for value in uuids) or len({value.upper() for value in uuids}) != 1 + len(active_workers):
        raise ValueError("D3 primary, worker A, and worker B must be three distinct physical CUDA devices")
    if any(worker.secondary.uuid.upper() != selectors[label][2].upper() for label, worker in probed.items()) or (a or b).primary.uuid.upper() != PRIMARY_UUID.upper():
        raise ValueError("D3 CUDA device UUID verification disagrees with the frozen physical assignment")
    return a, b


def load_d3_resident_banks(placement: D3Placement, banks: Any, model_config: Any,
                           worker_a_device: Any | None, worker_b_device: Any | None, *, active_workers: tuple[str, ...] = ("a", "b"),
                           primary_visible_ordinal: int, torch_module=None,
                           cuda_module=None, resident_devices=None, chunk_rows: int = 32) -> D3ResidentExpertBanks:
    """Materialize both disjoint 3,000-row banks, exactly verifying each independently."""
    if set(placement.worker_a.flat_ids_in_rank_order) & set(placement.worker_b.flat_ids_in_rank_order):
        raise ValueError("D3 resident workers must have disjoint identities")
    if "a" in active_workers and "b" in active_workers and worker_a_device.secondary.visible_ordinal == worker_b_device.secondary.visible_ordinal:
        raise ValueError("D3 resident workers must use distinct CUDA devices")
    devices = resident_devices or (None, None)
    common = dict(primary_visible_ordinal=primary_visible_ordinal, torch_module=torch_module or __import__("torch"), cuda_module=cuda_module, chunk_rows=chunk_rows)
    # Only active workers materialize storage.  Each loader restores GPU0.
    a = load_secondary_resident_bank(placement.worker_a, banks, model_config, worker_a_device, resident_device=devices[0], **common) if "a" in active_workers else None
    b = load_secondary_resident_bank(placement.worker_b, banks, model_config, worker_b_device, resident_device=devices[1], **common) if "b" in active_workers else None
    if any(bank.report.placement.remote_resident_bytes != 5_326_848_000 for bank in (a, b) if bank is not None):
        raise RuntimeError("D3 worker resident byte contract disagreement")
    return D3ResidentExpertBanks(a, b)
