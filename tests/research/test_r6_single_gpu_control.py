from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from freetoken.models.loader import BoundedSafetensorsReader, MergeRule
from freetoken.research.r6_dense_census import DenseBlockSpec
from safetensors.torch import save_file

_STAGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "inferswarm_r6"
    / "stage_runtime.py"
)
_SPEC = importlib.util.spec_from_file_location("r6_stage_runtime_for_test", _STAGE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_STAGE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_STAGE)
GemmaDenseStage = _STAGE.GemmaDenseStage
execute_dense_layer_sequence = _STAGE.execute_dense_layer_sequence
materialize_dense_state = _STAGE.materialize_dense_state
semantic_boundaries_for_role = _STAGE.semantic_boundaries_for_role


def _reader(tmp_path, tensors):
    root = tmp_path / "model"
    root.mkdir()
    save_file(tensors, root / "model.safetensors")
    return BoundedSafetensorsReader(root, tensors)


def _merge_rule(key):
    rules = {
        ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
        ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
        ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
        ".gate_proj": MergeRule(".gate_up_proj", "gate", ("gate", "up")),
        ".up_proj": MergeRule(".gate_up_proj", "up", ("gate", "up")),
    }
    for suffix, rule in rules.items():
        if key.endswith(suffix + ".weight"):
            return key.replace(suffix, rule.fused_suffix), rule
    return None


def test_two_and_three_way_fusion_copy_in_final_device_order_without_cat(
    tmp_path, monkeypatch
):
    prefix = "model.layers.0"
    tensors = {
        f"{prefix}.self_attn.q_proj.weight": torch.full((2, 3), 1.0),
        f"{prefix}.self_attn.k_proj.weight": torch.full((1, 3), 2.0),
        f"{prefix}.self_attn.v_proj.weight": torch.full((1, 3), 3.0),
        f"{prefix}.mlp.gate_proj.weight": torch.full((2, 3), 4.0),
        f"{prefix}.mlp.up_proj.weight": torch.full((2, 3), 5.0),
        f"{prefix}.norm.weight": torch.tensor([6.0, 7.0, 8.0]),
    }
    reader = _reader(tmp_path, tensors)
    expected = {
        f"{prefix}.self_attn.qkv_proj.weight": torch.empty(4, 3),
        f"{prefix}.mlp.gate_up_proj.weight": torch.empty(4, 3),
        f"{prefix}.norm.weight": torch.empty(3),
    }

    monkeypatch.setattr(
        torch,
        "cat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("CPU torch.cat is forbidden")
        ),
    )
    loaded, fusion_peak = materialize_dense_state(
        expected_state=expected,
        reader=reader,
        device="cpu",
        rename_key=lambda key: key,
        merge_rule_for=_merge_rule,
    )

    torch.testing.assert_close(
        loaded[f"{prefix}.self_attn.qkv_proj.weight"][:, 0],
        torch.tensor([1.0, 1.0, 2.0, 3.0]),
    )
    torch.testing.assert_close(
        loaded[f"{prefix}.mlp.gate_up_proj.weight"][:, 0],
        torch.tensor([4.0, 4.0, 5.0, 5.0]),
    )
    torch.testing.assert_close(
        loaded[f"{prefix}.norm.weight"], torch.tensor([6.0, 7.0, 8.0])
    )
    assert fusion_peak == 2 * 3 * 4
    assert reader.host_staging_current_bytes == 0
    assert reader.host_staging_peak_live_tensor_bytes == 2 * 3 * 4


def test_k_eq_v_copies_one_k_source_to_two_destination_slices(tmp_path):
    prefix = "model.layers.0.self_attn"
    q_key = f"{prefix}.q_proj.weight"
    k_key = f"{prefix}.k_proj.weight"
    tensors = {
        q_key: torch.full((2, 2), 1.0),
        k_key: torch.full((1, 2), 9.0),
    }
    reader = _reader(tmp_path, tensors)

    loaded, _ = materialize_dense_state(
        expected_state={f"{prefix}.qkv_proj.weight": torch.empty(4, 2)},
        reader=reader,
        device="cpu",
        rename_key=lambda key: key,
        merge_rule_for=_merge_rule,
        k_eq_v_k_keys=frozenset({k_key}),
    )

    fused = loaded[f"{prefix}.qkv_proj.weight"]
    torch.testing.assert_close(fused[:, 0], torch.tensor([1.0, 1.0, 9.0, 9.0]))
    assert reader.fetched_keys.count(k_key) == 1


def test_merge_preflight_fails_closed_on_missing_component(tmp_path):
    prefix = "model.layers.0.self_attn"
    tensors = {f"{prefix}.q_proj.weight": torch.ones(2, 2)}
    reader = _reader(tmp_path, tensors)

    with pytest.raises(RuntimeError, match="incomplete merge group"):
        materialize_dense_state(
            expected_state={f"{prefix}.qkv_proj.weight": torch.empty(4, 2)},
            reader=reader,
            device="cpu",
            rename_key=lambda key: key,
            merge_rule_for=_merge_rule,
        )
    assert reader.fetched_keys == []


def test_full_range_single_role_contract_and_tied_embedding():
    spec = DenseBlockSpec(0, 48, True, True)
    spec.validate(48)
    assert semantic_boundaries_for_role("single") == []

    tied_weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    stage = object.__new__(GemmaDenseStage)
    stage.block = SimpleNamespace(embed_tokens=SimpleNamespace(weight=tied_weight))
    stage.full_config = SimpleNamespace(final_logit_softcapping=None)
    final = torch.tensor([[1.0, 2.0, 3.0]])
    logits = stage.lm_head_logits(final)
    torch.testing.assert_close(logits, final @ tied_weight.t())
    assert stage.block.embed_tokens.weight.data_ptr() == tied_weight.data_ptr()


def test_synthetic_full_range_execution_matches_equivalent_stages():
    class Affine:
        def __init__(self, scale, shift):
            self.scale = scale
            self.shift = shift

        def forward(self, value):
            return value * self.scale + self.shift

    layers = [Affine(1.5, 0.25), Affine(0.5, -1.0), Affine(2.0, 3.0)]
    hidden = torch.tensor([[1.0, -2.0]])

    single = execute_dense_layer_sequence(layers, hidden)
    boundary = execute_dense_layer_sequence(layers[:2], hidden)
    staged = execute_dense_layer_sequence(layers[2:], boundary)
    torch.testing.assert_close(single, staged)
