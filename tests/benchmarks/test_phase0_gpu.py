"""Physical-GPU provenance: resolve, record, and *prove*.

Criteria section 2.1 fixes Phase 0 to one physical RTX 3060. The failures this file guards
against are all variations of one thing: a record that names a card the run did not use.
"""

from __future__ import annotations

import pytest

from .fakes import FAKE_UUID

from inferswarm_phase0 import gpu as gpu_mod
from inferswarm_phase0 import provenance as prov

OTHER_UUID = "GPU-99999999-8888-7777-6666-555555555555"


# --- resolution ------------------------------------------------------------------------------

def test_no_gpu_selector_is_an_explicit_refusal_reason():
    selection = gpu_mod.resolve_gpu(None)
    assert selection.proven is False
    assert "criteria section 2.1" in selection.unavailable


def test_a_numeric_index_resolves_to_a_stable_uuid(monkeypatch):
    """An nvidia-smi index is accepted as input but never kept as the identity: indices move
    between boots and CUDA_DEVICE_ORDER renumbers them again inside a process."""
    monkeypatch.setattr(gpu_mod, "_resolve_uuids", lambda selector: (FAKE_UUID,))
    monkeypatch.setattr(gpu_mod, "_smi_index_for", lambda uuid: 3)
    selection = gpu_mod.resolve_gpu("0")
    assert selection.requested == "0"
    assert selection.resolved_uuid == FAKE_UUID
    assert selection.physical_index == 3
    record = selection.record()
    assert record["requested_is_uuid"] is False
    assert record["resolved_uuid"] == FAKE_UUID


def test_resolution_uses_freetokens_own_selector(monkeypatch):
    """No second selector policy: this must call freetoken.gpu_select, the same code
    ft serve and ft bench bw use."""
    seen = {}

    def spy(specs):
        seen["specs"] = list(specs)
        return (FAKE_UUID,)

    monkeypatch.setattr("freetoken.gpu_select.resolve_gpu_uuids", spy)
    monkeypatch.setattr(gpu_mod, "_smi_index_for", lambda uuid: 0)
    assert gpu_mod.resolve_gpu(FAKE_UUID).resolved_uuid == FAKE_UUID
    assert seen["specs"] == [FAKE_UUID]


def test_an_unresolvable_selector_records_the_reason(monkeypatch):
    monkeypatch.setattr(
        gpu_mod, "_resolve_uuids",
        lambda selector: (_ for _ in ()).throw(ValueError("not a unique prefix")),
    )
    selection = gpu_mod.resolve_gpu("GPU-nope")
    assert selection.proven is False
    assert "not a unique prefix" in selection.unavailable


def test_a_host_without_nvml_cannot_prove_the_selector(monkeypatch):
    monkeypatch.setattr(gpu_mod, "_resolve_uuids", lambda selector: None)
    selection = gpu_mod.resolve_gpu("0")
    assert selection.proven is False
    assert "NVML is unavailable" in selection.unavailable


# --- verification against the running engine -----------------------------------------------------

def _selection():
    return gpu_mod.GpuSelection(requested=FAKE_UUID, resolved_uuid=FAKE_UUID, physical_index=0)


def test_a_matching_engine_gpu_is_proof():
    block = gpu_mod.verify_engine_gpu(_selection(), [{"index": 0, "uuid": FAKE_UUID}])
    assert block["matches"] is True


def test_a_different_engine_gpu_is_a_proven_mismatch():
    block = gpu_mod.verify_engine_gpu(_selection(), [{"index": 0, "uuid": OTHER_UUID}])
    assert block["matches"] is False
    assert OTHER_UUID in block["mismatch"]


def test_an_engine_that_reports_no_gpu_is_unproven_not_assumed_good():
    block = gpu_mod.verify_engine_gpu(_selection(), [])
    assert block["matches"] is None
    assert "cannot be proven" in block["unavailable"]


def test_verification_without_a_resolved_uuid_is_unproven():
    selection = gpu_mod.GpuSelection(
        requested="0", resolved_uuid=None, physical_index=None, unavailable="no NVML"
    )
    block = gpu_mod.verify_engine_gpu(selection, [{"uuid": FAKE_UUID}])
    assert block["matches"] is None


# --- the nvidia-smi provenance query --------------------------------------------------------------

def test_the_smi_query_asks_for_the_gpu_index():
    """Without the index a numeric --gpu selector cannot be correlated with a row, so
    'which of these cards ran the benchmark' is unanswerable from the record."""
    assert prov._SMI_FIELDS[0] == "index"


def test_the_resolved_uuid_marks_the_selected_row(monkeypatch):
    rows = "\n".join([
        f"0, {OTHER_UUID}, NVIDIA GeForce RTX 3060, 12288 MiB, 580.00, 8.6, 3, 4, 16, 16",
        f"1, {FAKE_UUID}, NVIDIA GeForce RTX 3060, 12288 MiB, 580.00, 8.6, 3, 4, 16, 16",
    ])

    def fake_run(cmd, timeout=20.0):
        if "--query-gpu" in " ".join(cmd):
            return rows
        return "GPU0\tX"

    monkeypatch.setattr(prov, "_run", fake_run)
    doc = prov.gpu_provenance("1", FAKE_UUID)
    selected = [g for g in doc["gpus"] if g["selected"]]
    assert [g["uuid"] for g in selected] == [FAKE_UUID]
    assert doc["selected"] == {"requested": "1", "resolved_uuid": FAKE_UUID}


def test_a_bare_index_still_marks_a_row_when_nvml_could_not_resolve(monkeypatch):
    rows = f"0, {OTHER_UUID}, RTX 3060, 12288 MiB, 580.00, 8.6, 3, 4, 16, 16"
    monkeypatch.setattr(
        prov, "_run", lambda cmd, timeout=20.0: rows if "--query-gpu" in " ".join(cmd) else "X"
    )
    doc = prov.gpu_provenance("0", None)
    assert doc["gpus"][0]["selected"] is True


# --- binding a torch process ------------------------------------------------------------------------

class _FakeDevice:
    def __init__(self, index):
        self.index = index


def test_binding_records_the_device_it_actually_bound(monkeypatch):
    """The identity is read back from the BOUND device, not echoed from the request."""
    monkeypatch.setattr(gpu_mod, "_resolve_uuids", lambda selector: (FAKE_UUID,))
    monkeypatch.setattr(gpu_mod, "_smi_index_for", lambda uuid: 1)
    monkeypatch.setitem(__import__("sys").modules, "torch", object())
    monkeypatch.setattr("freetoken.gpu_select.assign_gpu", lambda spec: None)
    monkeypatch.setattr("freetoken.gpu_select.bind_assigned_gpu", lambda default=0: _FakeDevice(1))
    monkeypatch.setattr(
        "freetoken.gpu_select.gpu_identity",
        lambda index: {"index": index, "name": "RTX 3060", "uuid": FAKE_UUID,
                       "total_bytes": 12 << 30},
    )
    device, identity, verification = gpu_mod.bind_torch_device(FAKE_UUID)
    assert device.index == 1
    assert identity["uuid"] == FAKE_UUID
    assert verification["matches"] is True
    assert verification["bound_cuda_index"] == 1


def test_binding_refuses_when_the_bound_card_is_not_the_requested_one(monkeypatch):
    """A hardware measurement that benchmarks device 0 while labelling the result as another
    card is worse than no measurement, so this raises instead of degrading."""
    monkeypatch.setattr(gpu_mod, "_resolve_uuids", lambda selector: (FAKE_UUID,))
    monkeypatch.setattr(gpu_mod, "_smi_index_for", lambda uuid: 0)
    monkeypatch.setitem(__import__("sys").modules, "torch", object())
    monkeypatch.setattr("freetoken.gpu_select.assign_gpu", lambda spec: None)
    monkeypatch.setattr("freetoken.gpu_select.bind_assigned_gpu", lambda default=0: _FakeDevice(0))
    monkeypatch.setattr(
        "freetoken.gpu_select.gpu_identity",
        lambda index: {"index": index, "name": "RTX 3060", "uuid": OTHER_UUID},
    )
    with pytest.raises(gpu_mod.GpuBindError, match="Refusing to attribute"):
        gpu_mod.bind_torch_device(FAKE_UUID)


def test_a_binding_failure_is_raised_not_swallowed(monkeypatch):
    monkeypatch.setattr(gpu_mod, "_resolve_uuids", lambda selector: (FAKE_UUID,))
    monkeypatch.setattr(gpu_mod, "_smi_index_for", lambda uuid: 0)
    monkeypatch.setitem(__import__("sys").modules, "torch", object())
    monkeypatch.setattr("freetoken.gpu_select.assign_gpu", lambda spec: None)

    def boom(default=0):
        raise RuntimeError("GPU is not visible to CUDA in this process")

    monkeypatch.setattr("freetoken.gpu_select.bind_assigned_gpu", boom)
    with pytest.raises(gpu_mod.GpuBindError, match="not visible to CUDA"):
        gpu_mod.bind_torch_device(FAKE_UUID)
