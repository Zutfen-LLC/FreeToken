from __future__ import annotations

import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from freetoken.moe.inferswarm_secondary import (
    DIRECT_PEER_CAPABLE,
    HOST_STAGED_REQUIRED,
    probe_secondary_device,
)

PRIMARY_UUID = "GPU-11111111-1111-1111-1111-111111111111"
SECONDARY_UUID = "GPU-22222222-2222-2222-2222-222222222222"


def _raw_uuid(value: str) -> uuid.UUID:
    return uuid.UUID(value.removeprefix("GPU-"))


class _FakeCuda:
    def __init__(self, uuids=(PRIMARY_UUID, SECONDARY_UUID), current=0, peer=None):
        self.current = current
        self.peer = dict(peer or {})
        self.set_device_calls = []
        self.props = [
            SimpleNamespace(
                uuid=_raw_uuid(gpu_uuid),
                name=f"Fake RTX 3060 ordinal {i}",
                total_memory=(12 << 30) + i,
                major=8,
                minor=6,
            )
            for i, gpu_uuid in enumerate(uuids)
        ]

    def device_count(self):
        return len(self.props)

    def current_device(self):
        return self.current

    def set_device(self, index):
        self.current = int(index)
        self.set_device_calls.append(int(index))

    @contextmanager
    def device(self, index):
        previous = self.current
        self.current = int(index)
        try:
            yield
        finally:
            self.current = previous

    def get_device_properties(self, index):
        return self.props[int(index)]

    def mem_get_info(self):
        return (8 << 30) + self.current, self.props[self.current].total_memory

    def can_device_access_peer(self, source, destination):
        return self.peer.get((int(source), int(destination)), False)


def _torch(cuda):
    return SimpleNamespace(cuda=cuda)


def test_uuid_resolution_uses_cuda_visible_order_not_an_assumed_cuda_one():
    # CUDA order is deliberately reversed: the requested secondary is visible ordinal 0.
    cuda = _FakeCuda((SECONDARY_UUID, PRIMARY_UUID), current=1)
    info = probe_secondary_device(
        SECONDARY_UUID,
        resolved_uuid=SECONDARY_UUID,
        primary_visible_ordinal=1,
        primary_resolved_uuid=PRIMARY_UUID,
        torch_module=_torch(cuda),
    )
    assert info.primary.uuid == PRIMARY_UUID
    assert info.primary.visible_ordinal == 1
    assert info.secondary.uuid == SECONDARY_UUID
    assert info.secondary.visible_ordinal == 0
    assert cuda.current_device() == 1
    assert info.primary_current_after_probe is True


def test_false_peer_access_requires_host_staging_in_both_directional_record():
    info = probe_secondary_device(
        SECONDARY_UUID,
        resolved_uuid=SECONDARY_UUID,
        primary_visible_ordinal=0,
        torch_module=_torch(_FakeCuda()),
    )
    assert info.can_access_peer_primary_to_secondary is False
    assert info.can_access_peer_secondary_to_primary is False
    assert info.transport_classification == HOST_STAGED_REQUIRED


def test_bidirectional_peer_access_is_described_as_direct_capability():
    cuda = _FakeCuda(peer={(0, 1): True, (1, 0): True})
    info = probe_secondary_device(
        SECONDARY_UUID,
        resolved_uuid=SECONDARY_UUID,
        primary_visible_ordinal=0,
        torch_module=_torch(cuda),
    )
    assert info.can_access_peer_primary_to_secondary is True
    assert info.can_access_peer_secondary_to_primary is True
    assert info.transport_classification == DIRECT_PEER_CAPABLE


def test_asymmetric_peer_access_still_requires_host_staging_for_round_trip():
    cuda = _FakeCuda(peer={(0, 1): True, (1, 0): False})
    info = probe_secondary_device(
        SECONDARY_UUID,
        resolved_uuid=SECONDARY_UUID,
        primary_visible_ordinal=0,
        torch_module=_torch(cuda),
    )
    assert info.transport_classification == HOST_STAGED_REQUIRED


def test_same_primary_physical_uuid_is_rejected_explicitly():
    cuda = _FakeCuda()
    with pytest.raises(ValueError, match="same physical GPU as the primary"):
        probe_secondary_device(
            PRIMARY_UUID,
            resolved_uuid=PRIMARY_UUID,
            primary_visible_ordinal=0,
            torch_module=_torch(cuda),
        )
    assert cuda.current_device() == 0


def test_nonexistent_secondary_uuid_is_rejected_and_primary_restored():
    cuda = _FakeCuda(current=0)
    with pytest.raises(ValueError, match="not visible to CUDA"):
        probe_secondary_device(
            "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            resolved_uuid=None,
            primary_visible_ordinal=0,
            torch_module=_torch(cuda),
        )
    assert cuda.current_device() == 0


def test_one_visible_gpu_is_rejected_before_any_secondary_inspection():
    cuda = _FakeCuda((PRIMARY_UUID,), current=0)
    with pytest.raises(ValueError, match="at least two CUDA devices.*only 1"):
        probe_secondary_device(
            "1",
            resolved_uuid=None,
            primary_visible_ordinal=0,
            torch_module=_torch(cuda),
        )


def test_numeric_secondary_must_be_a_valid_visible_ordinal():
    cuda = _FakeCuda()
    with pytest.raises(ValueError, match="secondary CUDA ordinal 2 is invalid"):
        probe_secondary_device(
            "2",
            resolved_uuid=None,
            primary_visible_ordinal=0,
            torch_module=_torch(cuda),
        )
    assert cuda.current_device() == 0


def test_runtime_shape_contains_mechanical_validation_and_memory_fields():
    info = probe_secondary_device(
        SECONDARY_UUID,
        resolved_uuid=SECONDARY_UUID,
        primary_visible_ordinal=0,
        torch_module=_torch(_FakeCuda()),
    ).as_dict()
    assert info["configured"] is True
    assert info["validation_passed"] is True
    assert info["requested_secondary_spec"] == SECONDARY_UUID
    assert info["primary"]["free_vram_bytes_at_probe"] == 8 << 30
    assert info["secondary"]["total_vram_bytes"] == (12 << 30) + 1
    assert info["secondary"]["compute_capability"]["label"] == "sm_86"
    assert info["peer_access"] == {
        "primary_to_secondary": False,
        "secondary_to_primary": False,
    }
