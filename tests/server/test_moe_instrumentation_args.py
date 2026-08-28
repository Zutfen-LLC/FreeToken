from __future__ import annotations

from unittest.mock import patch

import pytest
from freetoken.server.args import parse_args


class _Config:
    def to_dict(self) -> dict:
        return {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "torch_dtype": "bfloat16",
        }


def _parse(*extra: str):
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _Config()):
        return parse_args(["--model", "/models/anon", *extra])[0]


def test_moe_instrumentation_cli_is_opt_in():
    args = _parse()
    assert args.moe_collect_stats is False
    assert args.moe_trace_max_steps == 0
    assert args.moe_layer_timing_max_steps == 0
    assert args.moe_layer_timing_role == "unspecified"
    assert args.inferswarm_correctness_diagnostics is False


def test_moe_instrumentation_cli_enables_stats_and_bounded_trace():
    args = _parse(
        "--moe-collect-stats",
        "--moe-trace-max-steps",
        "128",
        "--cuda-graph-max-bs",
        "0",
    )
    assert args.moe_collect_stats is True
    assert args.moe_trace_max_steps == 128


def test_exact_trace_rejects_cuda_graph_replay():
    with pytest.raises(SystemExit):
        _parse("--moe-trace-max-steps", "8", "--cuda-graph-max-bs", "1")

    with pytest.raises(SystemExit):
        _parse("--moe-trace-max-steps", "8")


def test_complete_layer_timing_is_bounded_and_graph_compatible():
    args = _parse(
        "--moe-layer-timing-max-steps",
        "16",
        "--moe-layer-timing-role",
        "baseline",
        "--cuda-graph-max-bs",
        "1",
    )
    assert args.moe_layer_timing_max_steps == 16
    assert args.moe_layer_timing_role == "baseline"
    assert args.cuda_graph_max_bs == 1


def test_c3_correctness_diagnostics_are_explicit_and_graph_compatible():
    args = _parse(
        "--inferswarm-correctness-diagnostics",
        "--cuda-graph-max-bs",
        "1",
    )
    assert args.inferswarm_correctness_diagnostics is True
    assert args.cuda_graph_max_bs == 1
