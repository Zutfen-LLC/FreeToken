"""Regression coverage for repeated Phase-0 diagnostics on one physical GPU.

P0-C runs device-memory bandwidth and the expert microbenchmark sequentially in the same
Python process. The first diagnostic binds the published physical UUID and therefore fills
in its CUDA-visible ordinal. Republishing the same UUID for the second diagnostic must be
idempotent; changing the physical GPU must still be refused.
"""

from __future__ import annotations

import sys
import types
import uuid

import pytest

from freetoken import gpu_select

FAKE_UUID = "GPU-33333333-3333-3333-3333-333333333333"
OTHER_UUID = "GPU-99999999-8888-7777-6666-555555555555"


@pytest.fixture(autouse=True)
def _reset_assignment(monkeypatch):
    monkeypatch.setattr(gpu_select, "_assigned_physical", None)
    monkeypatch.setattr(gpu_select, "_assigned_visible", None)


def _install_fake_torch(monkeypatch):
    props = types.SimpleNamespace(
        uuid=uuid.UUID(FAKE_UUID[len("GPU-"):]),
        name="NVIDIA GeForce RTX 3060",
        total_memory=12 << 30,
    )
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        device_count=lambda: 1,
        get_device_properties=lambda index: props,
        set_device=lambda device: None,
    )
    torch.device = lambda kind, index: types.SimpleNamespace(type=kind, index=index)
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_same_uuid_can_be_republished_after_bind_without_losing_visible_ordinal(monkeypatch):
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(gpu_select, "resolve_gpu_uuids", lambda specs: tuple(specs))

    gpu_select.assign_gpu(FAKE_UUID)
    first = gpu_select.bind_assigned_gpu()
    assert first.index == 0
    assert gpu_select._assigned_physical == FAKE_UUID
    assert gpu_select._assigned_visible == 0

    # P0-C does this when the second GPU diagnostic starts. Before the fix this raised:
    # set_assigned_gpu called twice: (FAKE_UUID, 0) then FAKE_UUID.
    gpu_select.assign_gpu(FAKE_UUID)
    second = gpu_select.bind_assigned_gpu()

    assert second.index == 0
    assert gpu_select._assigned_physical == FAKE_UUID
    assert gpu_select._assigned_visible == 0


def test_different_uuid_after_bind_is_still_refused(monkeypatch):
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(gpu_select, "resolve_gpu_uuids", lambda specs: tuple(specs))

    gpu_select.assign_gpu(FAKE_UUID)
    gpu_select.bind_assigned_gpu()

    with pytest.raises(RuntimeError, match="set_assigned_gpu called twice"):
        gpu_select.assign_gpu(OTHER_UUID)

    assert gpu_select._assigned_physical == FAKE_UUID
    assert gpu_select._assigned_visible == 0


def test_direct_namespace_change_after_uuid_bind_is_refused(monkeypatch):
    _install_fake_torch(monkeypatch)
    gpu_select.set_assigned_gpu(FAKE_UUID)
    gpu_select.bind_assigned_gpu()

    with pytest.raises(RuntimeError, match="set_assigned_gpu called twice"):
        gpu_select.set_assigned_gpu("0")


def test_repeated_numeric_assignment_remains_idempotent():
    gpu_select.set_assigned_gpu("0")
    gpu_select.set_assigned_gpu("0")
    assert gpu_select._assigned_physical is None
    assert gpu_select._assigned_visible == 0
