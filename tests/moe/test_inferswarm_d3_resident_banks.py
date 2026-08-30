from __future__ import annotations

from types import SimpleNamespace

import pytest

import freetoken.moe.inferswarm_d3_resident_banks as d3_banks
from freetoken.moe.inferswarm_d3_placement import load_d3_placement


ARTIFACT = "/home/zutfen/inferswarm/docs/investigations/data/phase1r-d3-three-device-placement.json"


def _device(ordinal, label):
    return SimpleNamespace(secondary=SimpleNamespace(visible_ordinal=ordinal, uuid=label))


def _bank(placement, ordinal):
    return SimpleNamespace(report=SimpleNamespace(placement=placement, secondary_visible_ordinal=ordinal,
        total_live_resident_bytes=placement.remote_resident_bytes,
        verification=(SimpleNamespace(source_sha256="same", resident_sha256="same"),)))


def test_d3_loads_independent_disjoint_banks_with_exact_native_accounting(monkeypatch):
    placement = load_d3_placement(ARTIFACT)
    calls = []
    def fake_load(worker_placement, _banks, _config, device, **kwargs):
        calls.append((worker_placement, device.secondary.visible_ordinal, kwargs["primary_visible_ordinal"]))
        return _bank(worker_placement, device.secondary.visible_ordinal)
    monkeypatch.setattr(d3_banks, "load_secondary_resident_bank", fake_load)
    pair = d3_banks.load_d3_resident_banks(placement, object(), object(), _device(1, "A"), _device(2, "B"), primary_visible_ordinal=0)
    assert [(ordinal, primary) for _placement, ordinal, primary in calls] == [(1, 0), (2, 0)]
    assert pair.worker_a.report.placement.remote_slots == pair.worker_b.report.placement.remote_slots == 3000
    assert pair.total_native_expert_bytes == 10_653_696_000
    assert not set(pair.worker_a.report.placement.flat_ids_in_rank_order) & set(pair.worker_b.report.placement.flat_ids_in_rank_order)


def test_worker_a_failure_prevents_worker_b_load(monkeypatch):
    placement = load_d3_placement(ARTIFACT)
    calls = []
    def fail_a(worker_placement, *_args, **_kwargs):
        calls.append(worker_placement.canonical_placement)
        raise RuntimeError("A failed")
    monkeypatch.setattr(d3_banks, "load_secondary_resident_bank", fail_a)
    with pytest.raises(RuntimeError, match="A failed"):
        d3_banks.load_d3_resident_banks(placement, object(), object(), _device(1, "A"), _device(2, "B"), primary_visible_ordinal=0)
    assert calls == ["worker_a"]


def test_worker_b_failure_does_not_return_a_partial_pair(monkeypatch):
    placement = load_d3_placement(ARTIFACT)
    def fail_b(worker_placement, *_args, **_kwargs):
        if worker_placement.canonical_placement == "worker_b":
            raise RuntimeError("B failed")
        return _bank(worker_placement, 1)
    monkeypatch.setattr(d3_banks, "load_secondary_resident_bank", fail_b)
    with pytest.raises(RuntimeError, match="B failed"):
        d3_banks.load_d3_resident_banks(placement, object(), object(), _device(1, "A"), _device(2, "B"), primary_visible_ordinal=0)


def test_d3_rejects_same_worker_cuda_device_before_allocation(monkeypatch):
    monkeypatch.setattr(d3_banks, "load_secondary_resident_bank", lambda *_args, **_kwargs: pytest.fail("must not allocate"))
    with pytest.raises(ValueError, match="distinct CUDA devices"):
        d3_banks.load_d3_resident_banks(load_d3_placement(ARTIFACT), object(), object(), _device(1, "A"), _device(1, "B"), primary_visible_ordinal=0)


@pytest.mark.parametrize("active, expected", [(("a",), ["worker_a"]), (("b",), ["worker_b"])])
def test_single_worker_shape_never_allocates_inactive_bank(monkeypatch, active, expected):
    calls = []
    def fake_load(worker_placement, _banks, _config, device, **_kwargs):
        calls.append(worker_placement.canonical_placement)
        return _bank(worker_placement, device.secondary.visible_ordinal)
    monkeypatch.setattr(d3_banks, "load_secondary_resident_bank", fake_load)
    pair = d3_banks.load_d3_resident_banks(load_d3_placement(ARTIFACT), object(), object(), _device(1, "A"), _device(2, "B"), active_workers=active, primary_visible_ordinal=0)
    assert calls == expected
    assert pair.total_native_expert_bytes == 5_326_848_000
    assert (pair.worker_a is not None) == ("a" in active)
    assert (pair.worker_b is not None) == ("b" in active)
