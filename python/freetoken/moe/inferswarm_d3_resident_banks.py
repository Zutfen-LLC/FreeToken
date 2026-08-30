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
    worker_a: SecondaryResidentExpertBank
    worker_b: SecondaryResidentExpertBank

    @property
    def total_native_expert_bytes(self) -> int:
        return self.worker_a.report.placement.remote_resident_bytes + self.worker_b.report.placement.remote_resident_bytes


def probe_d3_workers(*, worker_a_spec: str, worker_b_spec: str, worker_a_uuid: str | None,
                     worker_b_uuid: str | None, primary_visible_ordinal: int,
                     primary_resolved_uuid: str | None, torch_module=None):
    """Resolve the two named physical workers and restore GPU0 after both probes."""
    if (worker_a_uuid or worker_a_spec).upper() != WORKER_A_UUID or (worker_b_uuid or worker_b_spec).upper() != WORKER_B_UUID:
        raise ValueError("D3 worker selectors must resolve to the frozen worker-A and worker-B physical UUIDs")
    if primary_resolved_uuid is not None and primary_resolved_uuid.upper() != PRIMARY_UUID:
        raise ValueError("D3 primary must resolve to the frozen GPU0 physical UUID")
    a = probe_secondary_device(worker_a_spec, resolved_uuid=worker_a_uuid,
                               primary_visible_ordinal=primary_visible_ordinal,
                               primary_resolved_uuid=primary_resolved_uuid, torch_module=torch_module)
    b = probe_secondary_device(worker_b_spec, resolved_uuid=worker_b_uuid,
                               primary_visible_ordinal=primary_visible_ordinal,
                               primary_resolved_uuid=primary_resolved_uuid, torch_module=torch_module)
    uuids = (a.primary.uuid, a.secondary.uuid, b.secondary.uuid)
    if any(value is None for value in uuids) or len({value.upper() for value in uuids}) != 3:
        raise ValueError("D3 primary, worker A, and worker B must be three distinct physical CUDA devices")
    if a.secondary.uuid.upper() != WORKER_A_UUID or b.secondary.uuid.upper() != WORKER_B_UUID or a.primary.uuid.upper() != PRIMARY_UUID:
        raise ValueError("D3 CUDA device UUID verification disagrees with the frozen physical assignment")
    return a, b


def load_d3_resident_banks(placement: D3Placement, banks: Any, model_config: Any,
                           worker_a_device: Any, worker_b_device: Any, *,
                           primary_visible_ordinal: int, torch_module=None,
                           cuda_module=None, resident_devices=None, chunk_rows: int = 32) -> D3ResidentExpertBanks:
    """Materialize both disjoint 3,000-row banks, exactly verifying each independently."""
    if set(placement.worker_a.flat_ids_in_rank_order) & set(placement.worker_b.flat_ids_in_rank_order):
        raise ValueError("D3 resident workers must have disjoint identities")
    if worker_a_device.secondary.visible_ordinal == worker_b_device.secondary.visible_ordinal:
        raise ValueError("D3 resident workers must use distinct CUDA devices")
    devices = resident_devices or (None, None)
    common = dict(primary_visible_ordinal=primary_visible_ordinal, torch_module=torch_module or __import__("torch"), cuda_module=cuda_module, chunk_rows=chunk_rows)
    # Sequential startup is intentional: a failure of A prevents B from being attempted;
    # a failure of B propagates rather than returning a partial pair.  Each loader restores GPU0.
    a = load_secondary_resident_bank(placement.worker_a, banks, model_config, worker_a_device, resident_device=devices[0], **common)
    b = load_secondary_resident_bank(placement.worker_b, banks, model_config, worker_b_device, resident_device=devices[1], **common)
    if a.report.placement.remote_resident_bytes != 5_326_848_000 or b.report.placement.remote_resident_bytes != 5_326_848_000:
        raise RuntimeError("D3 worker resident byte contract disagreement")
    return D3ResidentExpertBanks(a, b)
