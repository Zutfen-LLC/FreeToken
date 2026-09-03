"""R6 single-GPU control post-failure diagnostic: gate-logic regressions.

CPU-safe, torch-free (single_gpu_control.py only imports torch inside
run_control()/_distributed_diffs(), so its preflight/validity-gate helpers
are directly testable without CUDA or even torch installed).  Covers the
findings from the post-failure-extension gate review:

- host-memory and frozen-hardware-identity preflight helpers fail closed
  (in particular: no committed identity file -> never authorized);
- InvalidNumericalEvidenceError exists and is a RuntimeError, so a token
  mismatch or NaN/Inf capture can be distinguished from a generic failure
  without ever being silently folded into Outcome A/B;
- _best_effort_partial_report never raises, even against an object whose
  report() itself raises (diagnostic capture must not mask the real error).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_MODULE_PATH = REPO / "benchmarks/inferswarm_r6/single_gpu_control.py"
_SPEC = importlib.util.spec_from_file_location(
    "r6_single_gpu_control_for_test", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
sgc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sgc)


# --------------------------------------------------------------------------
# host-memory preflight
# --------------------------------------------------------------------------


def test_meminfo_total_bytes_parses_kib_line():
    text = "MemTotal:       16384000 kB\nMemFree:         100000 kB\n"
    assert sgc._meminfo_total_bytes(text) == 16384000 * 1024


def test_meminfo_total_bytes_none_when_missing():
    assert sgc._meminfo_total_bytes("MemFree: 100 kB\n") is None


# --------------------------------------------------------------------------
# frozen hardware identity preflight (fail-closed)
# --------------------------------------------------------------------------


def test_gpu_identity_fields_parses_csv_header_row():
    machine_gpu = {
        "stdout": "RTX 3090, GPU-abcdef, 0000:41:00.0, 24576, 100, 200, "
        "550.54.15, P0, 4, 16"
    }
    fields = sgc._gpu_identity_fields(machine_gpu)
    assert fields[:4] == ["RTX 3090", "GPU-abcdef", "0000:41:00.0", "24576"]


def test_gpu_identity_fields_none_on_empty_stdout():
    assert sgc._gpu_identity_fields({"stdout": ""}) is None


def test_frozen_identity_fails_closed_when_file_absent(tmp_path):
    """No committed preflight amendment -> never authorized, regardless of
    what the live GPU query reports."""
    machine = {
        "gpu": {"stdout": "RTX 3090, GPU-real, 0000:41:00.0, 24576, 0, 0, 1, P0, 4, 16"}
    }
    missing = tmp_path / "PREFLIGHT-INFERSWARM04.json"
    assert not missing.exists()
    assert sgc._frozen_hardware_identity_matches(machine, missing) is False


def test_frozen_identity_matches_when_all_fields_equal(tmp_path):
    frozen_path = tmp_path / "PREFLIGHT-INFERSWARM04.json"
    frozen_path.write_text(
        json.dumps(
            {
                "gpu_name": "RTX 3090",
                "gpu_uuid": "GPU-real",
                "pci_bus_id": "0000:41:00.0",
                "memory_total_mib": "24576",
            }
        )
    )
    machine = {
        "gpu": {"stdout": "RTX 3090, GPU-real, 0000:41:00.0, 24576, 0, 0, 1, P0, 4, 16"}
    }
    assert sgc._frozen_hardware_identity_matches(machine, frozen_path) is True


def test_frozen_identity_rejects_mismatched_uuid(tmp_path):
    """A committed identity file must still be checked field-by-field, not
    merely required to exist."""
    frozen_path = tmp_path / "PREFLIGHT-INFERSWARM04.json"
    frozen_path.write_text(
        json.dumps(
            {
                "gpu_name": "RTX 3090",
                "gpu_uuid": "GPU-frozen",
                "pci_bus_id": "0000:41:00.0",
                "memory_total_mib": "24576",
            }
        )
    )
    machine = {
        "gpu": {
            "stdout": "RTX 3090, GPU-different, 0000:41:00.0, 24576, 0, 0, 1, P0, 4, 16"
        }
    }
    assert sgc._frozen_hardware_identity_matches(machine, frozen_path) is False


def test_committed_preflight_identity_file_exists_and_schema_valid():
    """inferswarm04 is provisioned and qualified: the frozen hardware
    identity preflight must exist in this checkout, carry its schema, and
    be valid JSON with the four runner-consumed fields present."""
    path = sgc.FROZEN_HARDWARE_IDENTITY_PATH
    assert path.exists(), (
        "PREFLIGHT-INFERSWARM04.json must be committed once inferswarm04 "
        "is provisioned (see SINGLE_GPU_CONTROL_METHODOLOGY.md)"
    )
    frozen = json.loads(path.read_text())
    assert frozen["schema"] == "inferswarm.r6.preflight-inferswarm04/1"
    for field in ("gpu_name", "gpu_uuid", "pci_bus_id", "memory_total_mib"):
        assert isinstance(frozen[field], str) and frozen[field], field
    assert frozen["hostname"] == "inferswarm04"
    assert frozen["gemma_realization_occurred_on_this_node"] is False


def test_committed_preflight_agrees_with_frozen_expected_record():
    """The committed preflight must carry exactly the qualified physical
    identity of inferswarm04 (RTX 3090, UUID, BDF, VRAM)."""
    frozen = json.loads(sgc.FROZEN_HARDWARE_IDENTITY_PATH.read_text())
    assert frozen["gpu_name"] == "NVIDIA GeForce RTX 3090"
    assert frozen["gpu_uuid"] == "GPU-ecda1aaa-0c66-857b-8218-3d511dc75c03"
    assert frozen["pci_bus_id"] == "00000000:01:00.0"
    assert frozen["memory_total_mib"] == "24576"


def test_frozen_identity_mismatched_live_uuid_fails_closed(tmp_path):
    frozen_path = tmp_path / "PREFLIGHT-INFERSWARM04.json"
    frozen_path.write_text(sgc.FROZEN_HARDWARE_IDENTITY_PATH.read_text())
    machine = {
        "gpu": {
            "stdout": (
                "NVIDIA GeForce RTX 3090, GPU-11111111-2222-3333-4444-"
                "555555555555, 00000000:01:00.0, 24576, 0, 0, 1, P0, 4, 16"
            )
        }
    }
    assert sgc._frozen_hardware_identity_matches(machine, frozen_path) is False


def test_frozen_identity_mismatched_live_bdf_fails_closed(tmp_path):
    frozen_path = tmp_path / "PREFLIGHT-INFERSWARM04.json"
    frozen_path.write_text(sgc.FROZEN_HARDWARE_IDENTITY_PATH.read_text())
    machine = {
        "gpu": {
            "stdout": (
                "NVIDIA GeForce RTX 3090, "
                "GPU-ecda1aaa-0c66-857b-8218-3d511dc75c03, "
                "00000000:02:00.0, 24576, 0, 0, 1, P0, 4, 16"
            )
        }
    }
    assert sgc._frozen_hardware_identity_matches(machine, frozen_path) is False


def test_frozen_identity_mismatched_live_vram_fails_closed(tmp_path):
    frozen_path = tmp_path / "PREFLIGHT-INFERSWARM04.json"
    frozen_path.write_text(sgc.FROZEN_HARDWARE_IDENTITY_PATH.read_text())
    machine = {
        "gpu": {
            "stdout": (
                "NVIDIA GeForce RTX 3090, "
                "GPU-ecda1aaa-0c66-857b-8218-3d511dc75c03, "
                "0000:01:00.0, 24576, 0, 0, 1, P0, 4, 16"
            ).replace(", 24576,", ", 8192,")
        }
    }
    assert sgc._frozen_hardware_identity_matches(machine, frozen_path) is False


def test_frozen_identity_live_machine_matching_committed_record_passes(tmp_path):
    """Positive arm: a live census that reproduces the committed record's
    exact nvidia-smi identity strings must be authorized."""
    frozen_path = tmp_path / "PREFLIGHT-INFERSWARM04.json"
    frozen_path.write_text(sgc.FROZEN_HARDWARE_IDENTITY_PATH.read_text())
    machine = {
        "gpu": {
            "stdout": (
                "NVIDIA GeForce RTX 3090, "
                "GPU-ecda1aaa-0c66-857b-8218-3d511dc75c03, "
                "00000000:01:00.0, 24576, 24123, 1, 610.57.04, P0, 1, 16"
            )
        }
    }
    assert sgc._frozen_hardware_identity_matches(machine, frozen_path) is True


# --------------------------------------------------------------------------
# fail-closed validity / failure reporting
# --------------------------------------------------------------------------


def test_invalid_numerical_evidence_error_is_a_runtime_error():
    assert issubclass(sgc.InvalidNumericalEvidenceError, RuntimeError)


def test_best_effort_partial_report_none_for_none_runtime():
    assert sgc._best_effort_partial_report(None) is None


def test_best_effort_partial_report_swallows_report_failure():
    class ExplodingRuntime:
        def report(self, _checkpoint):
            raise RuntimeError("cuda context torn down")

    assert sgc._best_effort_partial_report(ExplodingRuntime()) is None


def test_best_effort_partial_report_returns_real_report():
    class FakeRuntime:
        def report(self, checkpoint):
            return {"checkpoint": checkpoint, "ok": True}

    result = sgc._best_effort_partial_report(FakeRuntime())
    assert result == {"checkpoint": "P_failure_partial_evidence", "ok": True}


# --------------------------------------------------------------------------
# structural regressions for the fixed findings
# --------------------------------------------------------------------------


def test_run_control_never_machine_declares_outcome_a_from_drift_alone():
    """Regression: the removed constant classification
    OUTCOME_A_FREE_TOKEN_NUMERICAL_DIFFERENCE must not reappear; Outcome A
    is always reported as a candidate pending explicit maintainer
    adjudication of the second (unfrozen) condition."""
    source = _MODULE_PATH.read_text()
    assert "OUTCOME_A_FREE_TOKEN_NUMERICAL_DIFFERENCE" not in source
    assert "OUTCOME_A_CANDIDATE_REQUIRES_MAINTAINER_ADJUDICATION" in source


def test_run_control_gates_interpretation_on_exact_match_and_nan_inf():
    source = _MODULE_PATH.read_text()
    assert "InvalidNumericalEvidenceError" in source
    # the gate must run before aggregate_ref/interpretation are computed
    gate_index = source.index("raise InvalidNumericalEvidenceError")
    interpretation_index = source.index('interpretation = "OUTCOME_A_CANDIDATE')
    assert gate_index < interpretation_index
