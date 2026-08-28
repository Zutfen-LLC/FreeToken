from __future__ import annotations

from inferswarm_phase1 import device_probe


def test_missing_nvidia_smi_is_recorded_not_raised(monkeypatch):
    monkeypatch.setattr(device_probe.shutil, "which", lambda _name: None)
    topology = device_probe._external_topology()
    assert topology["nvidia_smi_topo_m"]["value"] is None
    assert "nvidia-smi" in topology["nvidia_smi_topo_m"]["unavailable"]
    assert topology["gpus"]["value"] is None


def test_nvidia_smi_gpu_rows_are_structured(monkeypatch):
    row = "GPU-a, NVIDIA GeForce RTX 3060, 00000000:03:00.0, 3, 3, 16, 16, 580.00"

    def fake_run(args):
        if any(value.startswith("--query-gpu=") for value in args):
            return {"value": row, "unavailable": None}
        return {"value": "topology", "unavailable": None}

    monkeypatch.setattr(device_probe, "_run_nvidia_smi", fake_run)
    topology = device_probe._external_topology()
    gpu = topology["gpus"]["value"][0]
    assert gpu["uuid"] == "GPU-a"
    assert gpu["pci.bus_id"] == "00000000:03:00.0"
    assert gpu["pcie.link.gen.current"] == "3"
    assert gpu["pcie.link.width.max"] == "16"
