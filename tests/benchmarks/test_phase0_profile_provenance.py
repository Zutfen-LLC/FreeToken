"""Regression coverage for canonical P0-C provenance."""

from __future__ import annotations

from benchmarks.inferswarm_phase0 import cli
from benchmarks.inferswarm_phase0 import hardware_profile
from benchmarks.inferswarm_phase0.gpu import GpuSelection

INFERSWARM_COMMIT = "1" * 40


def test_profile_refuses_missing_inferswarm_commit_before_measurement(monkeypatch):
    called = False

    def _should_not_capture(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("hardware measurement started before provenance validation")

    monkeypatch.setattr(hardware_profile, "capture_profile", _should_not_capture)

    assert cli.main(["profile", "--gpu", "0"]) == 2
    assert called is False


def test_profile_refuses_symbolic_or_short_inferswarm_commit_before_measurement(monkeypatch):
    called = False

    def _should_not_capture(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("hardware measurement started before provenance validation")

    monkeypatch.setattr(hardware_profile, "capture_profile", _should_not_capture)

    assert cli.main(["profile", "--gpu", "0", "--inferswarm-commit", "main"]) == 2
    assert cli.main(["profile", "--gpu", "0", "--inferswarm-commit", "deadbeef"]) == 2
    assert called is False


def test_capture_profile_records_exact_inferswarm_commit(monkeypatch):
    selection = GpuSelection(
        requested=None,
        resolved_uuid=None,
        physical_index=None,
        unavailable="mocked selection",
    )
    seen = {}

    monkeypatch.setattr(hardware_profile, "resolve_gpu", lambda gpu: selection)

    def _software_provenance(commit, harness_version):
        seen["commit"] = commit
        seen["harness_version"] = harness_version
        return {"inferswarm_commit": {"value": commit}}

    monkeypatch.setattr(hardware_profile.prov, "software_provenance", _software_provenance)
    monkeypatch.setattr(hardware_profile.prov, "host_provenance", lambda: {})
    monkeypatch.setattr(hardware_profile.prov, "gpu_provenance", lambda *args: {})

    doc = hardware_profile.capture_profile(
        inferswarm_commit=INFERSWARM_COMMIT,
        run_bench_bw=False,
        device_bandwidth=False,
        expert_microbench=False,
    )

    assert seen["commit"] == INFERSWARM_COMMIT
    assert doc["software"]["inferswarm_commit"]["value"] == INFERSWARM_COMMIT
