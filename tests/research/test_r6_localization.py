"""#71 localization tests (CPU-pure: no torch, no GPU).

Structural/contract tests for the localization instrumentation:

- capture seam no-op safety: without an armed sink the stage executes the
  exact pre-#71 statement sequence (source-contract assertions, runnable
  on the torch-free coordinator);
- boundary sender/receiver record identity contract;
- analyzer comparison math on synthetic tensors-free fixtures where
  possible, plus the comparator row schema;
- methodology freeze presence and the historical-evidence-immutability
  fence (localization code must not touch docs/inferswarm_r6/ paths).

GPU/torch-dependent behavior (real captures, real stage execution) is
qualified on the compute nodes by the physical campaign itself.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LOC = REPO / "benchmarks" / "inferswarm_r6_localization"
R6_STAGE_RUNTIME = REPO / "benchmarks" / "inferswarm_r6" / "stage_runtime.py"
R6_STAGE_CHAIN = REPO / "benchmarks" / "inferswarm_r6" / "stage_chain.py"
R6_WIRE_CLIENT = REPO / "benchmarks" / "inferswarm_r6" / "wire_client.py"
R6_LAST_STAGE = REPO / "benchmarks" / "inferswarm_r6" / "last_stage_service.py"


def _source(path: Path) -> str:
    return path.read_text()


def test_no_torch_import_in_localization_test_support():
    # The analyzer must defer torch to call time so the summary-only parts
    # stay importable anywhere: assert NO module-level (body-of-module)
    # torch import; imports inside functions are the intended pattern.
    src = _source(LOC / "analyze.py")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name != "torch" for alias in node.names), (
                "analyzer must defer torch import to call time"
            )
        if isinstance(node, ast.ImportFrom):
            assert node.module != "torch", (
                "analyzer must defer torch import to call time"
            )


def test_capture_seam_is_noop_without_sink():
    src = _source(R6_STAGE_RUNTIME)
    # _emit guards on sink presence before any tensor work
    m = re.search(r"def _emit\(.*?\n(.*?)\n    def embed", src, re.S)
    assert m, "_emit must directly precede embed()"
    body = m.group(1)
    assert "if sink is None or self._capture_step is None:" in body
    assert "return" in body


def test_forward_layers_hook_contract():
    src = _source(R6_STAGE_RUNTIME)
    # hook only constructed when a sink AND after-layers are armed
    assert "if self._capture_sink is not None and self._capture_after_layers:" in src
    # execute_dense_layer_sequence keeps its signature extension optional
    m = re.search(
        r"def execute_dense_layer_sequence\(\s*layers, hidden: torch\.Tensor"
        r"(?:, \*, after_layer_hook=None)?\s*\)", src)
    assert m


def test_capture_default_is_none():
    src = _source(R6_STAGE_RUNTIME)
    assert "_capture_sink: Any = None" in src
    assert "_capture_step: int | None = None" in src
    assert "_capture_after_layers: frozenset = frozenset()" in src


def test_stage_chain_capture_ops_are_explicit():
    src = _source(R6_STAGE_CHAIN)
    for op in ("ARM_CAPTURE", "SET_CAPTURE_STEPS", "SAVE_CAPTURE"):
        assert f'"{op}"' in src
    # capture_step attribution brackets the prefill call and is reset after
    assert 'message.get("capture_step") is not None' in src


def test_wire_client_byte_log_opt_in():
    src = _source(R6_WIRE_CLIENT)
    assert "_boundary_byte_log: list[dict] | None = None" in src
    assert "if _boundary_byte_log is not None:" in src
    # the byte log records the exact bytes it sends (unchanged payload path)
    assert '"payload_bytes": bytes(payload),' in src


def test_last_stage_localization_opt_in():
    src = _source(R6_LAST_STAGE)
    assert "--localization-capture" in src
    assert "localization_rx_log is not None" in src
    # receiver-side capture fires BEFORE prefill/decode execution
    idx_rx = src.index("localization_rx_log.append")
    idx_exec = src.index('if header["operation"] == "prefill":')
    assert idx_rx < idx_exec


def test_wire_header_tolerates_capture_step():
    wire = _source(REPO / "python" / "freetoken" / "research" / "r4_wire.py")
    # required-fields check is a subset; capture_step is an extra field
    assert '"capture_step"' not in wire  # r4_wire itself stays untouched
    client = _source(R6_WIRE_CLIENT)
    assert 'header["capture_step"] = int(capture_step)' in client


def test_analyzer_row_schema():
    src = _source(LOC / "analyze.py")
    for field in ("exact_equal", "max_absdiff", "mean_absdiff", "rms_diff",
                  "max_coordinate_rows_then_dim", "s_at_max", "d_at_max"):
        assert f'"{field}"' in src
    for point in ("embedding_output", "after_layer_15", "after_layer_31",
                  "after_layer_47", "final_norm", "bf16_logits",
                  "final_row_fp32"):
        assert f'"{point}"' in src


def test_boundary_identity_records():
    src = _source(LOC / "analyze.py")
    for field in ("sender_sha256", "receiver_sha256", "bytes_identical",
                  "exact_equality"):
        assert f'"{field}"' in src


def test_methodology_frozen_before_physical():
    doc = (REPO / "docs" / "inferswarm_r6_localization" / "METHODOLOGY.md").read_text()
    assert "FROZEN BEFORE FIRST CANONICAL PHYSICAL CAPTURE" in doc
    assert "51e772ab88643df61888c8860c8e67e307190565" in doc
    # fail-closed interpretation retained
    assert "No tolerance is invented after seeing data" in doc


def test_historical_r6_evidence_fence():
    # localization code must not WRITE into the historical R6 evidence dir:
    # allowed mentions are read-only references to the frozen inputs
    # (reference-generation.json / canonical-prompt.json) and provenance
    # citations of the accepted chain plan.
    allow = re.compile(
        r'docs/inferswarm_r6/(reference-generation\.json|canonical-prompt\.json'
        r'|chain-plan\.json)'
    )
    for module in (LOC / "capture.py", LOC / "analyze.py", LOC / "run_single_arm.py",
                   LOC / "run_dist_arm.py", LOC / "build_plan.py",
                   LOC / "audit_config.py"):
        src = allow.sub("", _source(module))
        assert "docs/inferswarm_r6/" not in src, (
            f"{module.name} references historical R6 evidence beyond the "
            "frozen read-only inputs"
        )
    # reads of the reference inputs are allowed (they are frozen inputs, not
    # mutable evidence)
    single = _source(LOC / "run_single_arm.py")
    assert 'default="docs/inferswarm_r6/reference-generation.json"' in single


def test_s_arm_runner_gate_contract():
    src = _source(LOC / "run_single_arm.py")
    assert "S_ARM_BLOCKED_DIRTY_SOURCE" in src
    assert "exact_generated_token_match" in src
    assert "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d" in src


def test_d_arm_runner_gate_contract():
    src = _source(LOC / "run_dist_arm.py")
    assert "D_ARM_TOKEN_MISMATCH" in src
    assert '"op": "RESET"' in src
    # single-chunk replay at position 0 (accepted capture semantics)
    assert '"position": 0' in src


def test_build_plan_provenance():
    src = _source(LOC / "build_plan.py")
    assert "r6_localization_71" in src
    assert "supersedes_for_localization_only" in src
    assert "44d6c94" in src


def test_config_audit_fields():
    src = _source(LOC / "audit_config.py")
    for field in ("k_eq_v", "rope_base", "sliding_window", "num_kv_heads",
                  "head_dim"):
        assert f'"{field}"' in src
    assert "cross_arm_s_vs_d_mismatches" in src


def test_capture_record_schema_fields():
    src = _source(LOC / "capture.py")
    for field in ("checkpoint", "step", "global_layer", "shape", "dtype",
                  "byte_count", "sha256", "nan_count", "inf_count",
                  "position_range", "source_device", "role", "gpu_uuid"):
        assert f'"{field}"' in src
