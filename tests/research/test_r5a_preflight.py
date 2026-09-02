import pytest

from inferswarm_r5a.preflight import validate_local_split_environment
from freetoken.research.r5a_serving import checkpoint_identity_from_gate


def _local_plan():
    import json
    from pathlib import Path

    return json.loads(Path("docs/inferswarm_r2/frozen-plan.json").read_text())


def _profile(capacity=12_884_901_888):
    return {
        "gpus": [
            {
                "uuid": "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099",
                "name": "NVIDIA GeForce RTX 3060",
                "pci_bus_id": "00000000:02:00.0",
                "memory_total_bytes": capacity,
            },
            {
                "uuid": "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55",
                "name": "NVIDIA GeForce RTX 3060",
                "pci_bus_id": "00000000:03:00.0",
                "memory_total_bytes": capacity,
            },
        ],
        "memory": {"mem_total_bytes": 134_984_912_896},
    }


def test_r5a_reads_checkpoint_identity_from_retained_r4_gate_schema():
    identity = {"revision": "frozen", "identical_across_nodes": True}
    gate = {
        "result": "ALL_PREFLIGHT_CHECKS_PASSED",
        "checks": {"checkpoint_identity": identity},
    }
    assert checkpoint_identity_from_gate(gate) == identity


def test_r5a_refuses_missing_or_unsuccessful_checkpoint_gate():
    with pytest.raises(ValueError, match="unsuccessful"):
        checkpoint_identity_from_gate({"result": "FAILED", "checks": {}})
    with pytest.raises(ValueError, match="lacks checkpoint"):
        checkpoint_identity_from_gate(
            {"result": "ALL_PREFLIGHT_CHECKS_PASSED", "checks": {}}
        )


def test_local_split_preflight_freezes_secondary_identity_and_headroom():
    gate = validate_local_split_environment(
        plan=_local_plan(),
        profile_a=_profile(),
        identity_a={"producer_sha": "producer", "tree_clean": True},
        producer_sha="producer",
    )
    assert gate["result"] == "LOCAL_SPLIT_PREFLIGHT_PASSED"
    assert gate["compute_units"][1]["stable_device_id"].startswith("GPU-d5c")
    assert gate["compute_units"][1]["pci_bdf"] == "00000000:03:00.0"
    assert gate["vram_headroom"]["gpu-b.vram"]["remaining_bytes"] > 0
    assert gate["representation_backend"]["compatible"] is True


def test_local_split_preflight_rejects_dirty_source_or_insufficient_secondary_vram():
    with pytest.raises(RuntimeError, match="exact clean"):
        validate_local_split_environment(
            plan=_local_plan(),
            profile_a=_profile(),
            identity_a={"producer_sha": "producer", "tree_clean": False},
            producer_sha="producer",
        )
    with pytest.raises(RuntimeError, match="headroom"):
        validate_local_split_environment(
            plan=_local_plan(),
            profile_a=_profile(capacity=11_200_000_000),
            identity_a={"producer_sha": "producer", "tree_clean": True},
            producer_sha="producer",
        )
