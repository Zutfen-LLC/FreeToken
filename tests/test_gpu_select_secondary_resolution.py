from __future__ import annotations

import pytest
from freetoken import gpu_select

GPU_A = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
GPU_B = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
GPU_C = "GPU-cccccccc-cccc-cccc-cccc-cccccccccccc"


def test_uuid_to_visible_ordinal_is_stable_under_cuda_ordering():
    class Cuda:
        @staticmethod
        def device_count():
            return 3

        @staticmethod
        def get_device_properties(index):
            import uuid

            ordered = [GPU_C, GPU_A, GPU_B]
            return type(
                "Props",
                (),
                {"uuid": uuid.UUID(ordered[index].removeprefix("GPU-"))},
            )()

    torch = type("Torch", (), {"cuda": Cuda})()
    assert gpu_select.visible_gpu_for_uuid(GPU_A, torch_module=torch) == 1
    assert gpu_select.visible_gpu_for_uuid(GPU_B[:12], torch_module=torch) == 2


def test_cuda_visible_devices_numeric_order_is_respected(monkeypatch):
    monkeypatch.setattr(gpu_select, "_nvml_uuids", lambda: [GPU_A, GPU_B, GPU_C])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")
    assert gpu_select.resolve_gpu_uuids(["0"]) == (GPU_C,)
    assert gpu_select.resolve_gpu_uuids(["1"]) == (GPU_A,)
    with pytest.raises(ValueError, match=r"only 2 GPU\(s\) are visible"):
        gpu_select.resolve_gpu_uuids(["2"])


def test_cuda_visible_devices_uuid_quota_is_respected(monkeypatch):
    monkeypatch.setattr(gpu_select, "_nvml_uuids", lambda: [GPU_A, GPU_B, GPU_C])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{GPU_C},{GPU_A}")
    assert gpu_select.resolve_gpu_uuids([GPU_C[:16]]) == (GPU_C,)
    with pytest.raises(ValueError, match="not one of the GPUs visible"):
        gpu_select.resolve_gpu_uuids([GPU_B])
