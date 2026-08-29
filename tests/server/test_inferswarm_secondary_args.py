from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
from freetoken.server.args import parse_args
from freetoken.server.launch import _resolve_server_gpu_args

PRIMARY_UUID = "GPU-11111111-1111-1111-1111-111111111111"
SECONDARY_UUID = "GPU-22222222-2222-2222-2222-222222222222"


class _Config:
    def to_dict(self) -> dict:
        return {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "torch_dtype": "bfloat16",
        }


def _parse(*extra: str):
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _Config()):
        return parse_args(["--model", "/models/anon", *extra])[0]


def test_parser_accepts_no_secondary_without_changing_defaults():
    args = _parse()
    assert args.inferswarm_secondary_gpu is None
    assert args.inferswarm_secondary_gpu_assigned is None
    assert args.inferswarm_placement is None
    assert args.inferswarm_remote_decode is False
    assert args.inferswarm_remote_mode == "overlap"
    assert args.inferswarm_mechanism_max_steps == 256


def test_parser_accepts_exactly_one_secondary_spec():
    args = _parse("--inferswarm-secondary-gpu", SECONDARY_UUID)
    assert args.inferswarm_secondary_gpu == SECONDARY_UUID


def test_placement_without_secondary_is_rejected_explicitly():
    with pytest.raises(SystemExit):
        _parse("--inferswarm-placement", "/tmp/placement.json")


def test_split_gpu0_diagnostic_permits_placement_without_secondary():
    args = _parse(
        "--inferswarm-placement",
        "/tmp/placement.json",
        "--inferswarm-correctness-diagnostics",
        "--inferswarm-c3-root-cause-mode",
        "DIAGNOSTIC_SPLIT_GPU0",
        "--cuda-graph-max-bs",
        "0",
        "--max-running-requests",
        "1",
    )
    assert args.inferswarm_secondary_gpu is None
    assert args.inferswarm_remote_decode is False
    assert args.inferswarm_c3_root_cause_mode == "DIAGNOSTIC_SPLIT_GPU0"


def test_split_gpu0_diagnostic_forbids_secondary_and_remote_dispatch():
    common = (
        "--inferswarm-placement",
        "/tmp/placement.json",
        "--inferswarm-correctness-diagnostics",
        "--inferswarm-c3-root-cause-mode",
        "DIAGNOSTIC_SPLIT_GPU0",
        "--cuda-graph-max-bs",
        "0",
        "--max-running-requests",
        "1",
    )
    with pytest.raises(SystemExit):
        _parse(*common, "--inferswarm-secondary-gpu", SECONDARY_UUID)
    with pytest.raises(SystemExit):
        _parse(*common, "--inferswarm-remote-decode")


def test_secondary_without_placement_preserves_probe_only_configuration():
    args = _parse("--inferswarm-secondary-gpu", SECONDARY_UUID)
    assert args.inferswarm_secondary_gpu == SECONDARY_UUID
    assert args.inferswarm_placement is None


def test_secondary_and_placement_enable_p2_configuration():
    args = _parse(
        "--inferswarm-secondary-gpu",
        SECONDARY_UUID,
        "--inferswarm-placement",
        "/tmp/placement.json",
    )
    assert args.inferswarm_secondary_gpu == SECONDARY_UUID
    assert args.inferswarm_placement == "/tmp/placement.json"
    assert args.inferswarm_remote_decode is False


def test_remote_decode_without_secondary_is_rejected_explicitly():
    with pytest.raises(SystemExit):
        _parse("--inferswarm-remote-decode")


def test_remote_decode_without_placement_is_rejected_explicitly():
    with pytest.raises(SystemExit):
        _parse(
            "--inferswarm-secondary-gpu",
            SECONDARY_UUID,
            "--inferswarm-remote-decode",
        )


def test_secondary_placement_and_eager_remote_decode_enable_p3():
    args = _parse(
        "--inferswarm-secondary-gpu",
        SECONDARY_UUID,
        "--inferswarm-placement",
        "/tmp/placement.json",
        "--inferswarm-remote-decode",
        "--cuda-graph-max-bs",
        "0",
    )
    assert args.inferswarm_remote_decode is True
    assert args.inferswarm_remote_mode == "overlap"


def test_remote_serialized_diagnostic_mode_is_explicit_and_bounded():
    args = _parse(
        "--inferswarm-secondary-gpu",
        SECONDARY_UUID,
        "--inferswarm-placement",
        "/tmp/placement.json",
        "--inferswarm-remote-decode",
        "--inferswarm-remote-mode",
        "serialized",
        "--inferswarm-mechanism-max-steps",
        "12",
        "--cuda-graph-max-bs",
        "0",
    )
    assert args.inferswarm_remote_mode == "serialized"
    assert args.inferswarm_mechanism_max_steps == 12


def test_remote_decode_refuses_cuda_graph_capture():
    with pytest.raises(SystemExit):
        _parse(
            "--inferswarm-secondary-gpu",
            SECONDARY_UUID,
            "--inferswarm-placement",
            "/tmp/placement.json",
            "--inferswarm-remote-decode",
            "--cuda-graph-max-bs",
            "1",
        )


@pytest.mark.parametrize("value", ["0,1", f"{PRIMARY_UUID},{SECONDARY_UUID}"])
def test_parser_rejects_multiple_secondary_entries(value):
    with pytest.raises(SystemExit):
        _parse("--inferswarm-secondary-gpu", value)


def test_secondary_is_not_repurposed_as_a_tp_rank():
    with pytest.raises(SystemExit):
        _parse(
            "--tensor-parallel-size",
            "2",
            "--gpu",
            "0,1",
            "--inferswarm-secondary-gpu",
            SECONDARY_UUID,
        )


def test_parent_records_resolved_secondary_uuid(monkeypatch):
    args = _parse("--gpu", PRIMARY_UUID, "--inferswarm-secondary-gpu", "1")

    def resolve(specs):
        return (PRIMARY_UUID,) if list(specs) == [PRIMARY_UUID] else (SECONDARY_UUID,)

    monkeypatch.setattr("freetoken.gpu_select.resolve_gpu_uuids", resolve)
    resolved = _resolve_server_gpu_args(args)
    assert resolved.gpu_assigned == (PRIMARY_UUID,)
    assert resolved.inferswarm_secondary_gpu == "1"
    assert resolved.inferswarm_secondary_gpu_assigned == SECONDARY_UUID


def test_parent_rejects_same_gpu_even_when_inputs_use_different_forms(monkeypatch):
    args = _parse("--gpu", "0", "--inferswarm-secondary-gpu", PRIMARY_UUID[:20])
    monkeypatch.setattr(
        "freetoken.gpu_select.resolve_gpu_uuids", lambda _specs: (PRIMARY_UUID,)
    )
    with pytest.raises(ValueError, match="same physical GPU"):
        _resolve_server_gpu_args(args)


def test_parent_preserves_a_clear_secondary_resolution_failure(monkeypatch):
    args = _parse("--inferswarm-secondary-gpu", "GPU-deadbeef")

    def fail(_specs):
        raise ValueError("--gpu GPU-deadbeef: not found or not a unique prefix")

    monkeypatch.setattr("freetoken.gpu_select.resolve_gpu_uuids", fail)
    with pytest.raises(ValueError, match="--inferswarm-secondary-gpu.*not found"):
        _resolve_server_gpu_args(args)


def test_resolver_does_not_change_absent_secondary_state(monkeypatch):
    args = _parse("--gpu", PRIMARY_UUID)
    monkeypatch.setattr(
        "freetoken.gpu_select.resolve_gpu_uuids", lambda _specs: (PRIMARY_UUID,)
    )
    resolved = _resolve_server_gpu_args(args)
    assert resolved.inferswarm_secondary_gpu is None
    assert resolved.inferswarm_secondary_gpu_assigned is None
    assert resolved == replace(args, gpu_assigned=(PRIMARY_UUID,))
