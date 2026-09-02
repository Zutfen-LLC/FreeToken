from __future__ import annotations

import pytest

from inferswarm_r5a.compose_economics import compose
from inferswarm_r5a.runtime import require_current_local_split_devices


def _gate():
    return {
        "result": "LOCAL_SPLIT_PREFLIGHT_PASSED",
        "vram_headroom": {
            "gpu-a.vram": {
                "uuid": "GPU-a",
                "pci_bdf": "00000000:02:00.0",
                "capacity_bytes": 12_884_901_888,
                "required_bytes": 10_861_202_432,
                "reservation_bytes": 536_870_912,
            },
            "gpu-b.vram": {
                "uuid": "GPU-b",
                "pci_bdf": "00000000:03:00.0",
                "capacity_bytes": 12_884_901_888,
                "required_bytes": 11_170_278_912,
                "reservation_bytes": 536_870_912,
            },
        },
    }


def test_local_realizer_rechecks_frozen_devices_before_materialization(monkeypatch):
    monkeypatch.setattr(
        "inferswarm_r5a.runtime.subprocess.check_output",
        lambda *args, **kwargs: (
            "GPU-a, 00000000:02:00.0, 12288\n"
            "GPU-b, 00000000:03:00.0, 12288\n"
        ),
    )
    require_current_local_split_devices(_gate())


def test_local_realizer_fails_closed_on_secondary_bdf_drift(monkeypatch):
    monkeypatch.setattr(
        "inferswarm_r5a.runtime.subprocess.check_output",
        lambda *args, **kwargs: (
            "GPU-a, 00000000:02:00.0, 12288\n"
            "GPU-b, 00000000:04:00.0, 12288\n"
        ),
    )
    with pytest.raises(RuntimeError, match="BDF drifted"):
        require_current_local_split_devices(_gate())


def test_economics_is_matched_and_does_not_call_residual_pure_network():
    common = {
        "producer_freetoken_sha": "producer",
        "environment_digest": "environment",
    }
    local = {
        **common,
        "arm_id": "local",
        "summary": {
            "ttft_ms": {"median": 10},
            "complete_request_wall_ms": {"median": 20},
            "decode_tok_s": {"median": 30},
        },
    }
    network = {
        **common,
        "arm_id": "network",
        "summary": {
            "ttft_ms": {"median": 15},
            "complete_request_wall_ms": {"median": 25},
            "decode_tok_s": {"median": 20},
        },
    }
    result = compose(local, network)
    assert result["metrics"]["ttft_ms"]["two_node_minus_same_node"] == 5
    assert "not pure network time" in result["residual_interpretation"]
