from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from freetoken.research.n0_model_block import (
    ModelBlockSpec,
    SelectiveTensorReader,
    checkpoint_census,
    freeze_two_block_plan,
    validate_complement,
)


def _write_indexed_checkpoint(tmp_path, tensors):
    shard = "model-00001-of-00001.safetensors"
    safetensors.torch.save_file(tensors, str(tmp_path / shard))
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total},
        "weight_map": {key: shard for key in tensors},
    }))
    return total


def test_census_split_and_exact_complement_are_deterministic(tmp_path):
    tensors = {
        "model.language_model.embed_tokens.weight": torch.zeros(5, 2),
        "model.language_model.layers.0.input_layernorm.weight": torch.zeros(2),
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight": torch.zeros(2, 1, dtype=torch.uint8),
        "model.language_model.layers.0.mlp.experts.0.gate_proj.input_scale": torch.zeros(1),
        "model.language_model.layers.1.input_layernorm.weight": torch.zeros(2),
        "model.language_model.layers.1.mlp.experts.0.gate_proj.weight": torch.zeros(2, 1, dtype=torch.uint8),
        "model.language_model.norm.weight": torch.zeros(2),
        "lm_head.weight": torch.zeros(5, 1, dtype=torch.uint8),
        "model.visual.patch_embed.weight": torch.zeros(7),
    }
    total = _write_indexed_checkpoint(tmp_path, tensors)
    census = checkpoint_census(tmp_path)
    plan1 = freeze_two_block_plan(census, ["linear_attention", "full_attention"])
    plan2 = freeze_two_block_plan(census, ["linear_attention", "full_attention"])

    assert census["total_checkpoint_bytes"] == total
    assert plan1 == plan2
    assert plan1["split_boundary"] == 1
    assert plan1["coverage_proof"]["required_key_union_is_all"]
    assert plan1["coverage_proof"]["ordinary_layer_key_intersection_is_empty"]
    assert "model.language_model.embed_tokens.weight" in plan1["block_a"]["allowed_tensor_keys"]
    assert "model.language_model.norm.weight" in plan1["block_b"]["allowed_tensor_keys"]
    assert "lm_head.weight" in plan1["block_b"]["allowed_tensor_keys"]
    assert "model.visual.patch_embed.weight" not in (
        plan1["block_a"]["allowed_tensor_keys"] + plan1["block_b"]["allowed_tensor_keys"]
    )


@pytest.mark.parametrize("spec", [
    ModelBlockSpec(-1, 2, True, False),
    ModelBlockSpec(2, 2, True, False),
    ModelBlockSpec(2, 1, True, False),
    ModelBlockSpec(0, 4, True, False),
])
def test_invalid_block_ranges_fail_closed(spec):
    with pytest.raises(ValueError):
        spec.validate(3)


def test_two_block_plan_rejects_gap_overlap_and_wrong_boundary_ownership():
    with pytest.raises(ValueError):
        validate_complement(
            ModelBlockSpec(0, 1, True, False),
            ModelBlockSpec(2, 3, False, True),
            3,
        )
    with pytest.raises(ValueError):
        validate_complement(
            ModelBlockSpec(0, 2, True, False),
            ModelBlockSpec(1, 3, False, True),
            3,
        )
    with pytest.raises(ValueError):
        validate_complement(
            ModelBlockSpec(0, 1, False, False),
            ModelBlockSpec(1, 3, True, True),
            3,
        )


def test_selective_reader_never_uses_whole_shard_helper(tmp_path, monkeypatch):
    tensors = {"wanted": torch.arange(4), "forbidden": torch.arange(100)}
    _write_indexed_checkpoint(tmp_path, tensors)
    monkeypatch.setattr(
        safetensors.torch, "load_file",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("whole-shard helper used")),
    )
    reader = SelectiveTensorReader(tmp_path, {"wanted"})
    loaded = dict(reader.tensors())
    assert list(loaded) == ["wanted"]
    assert reader.fetched_keys == ["wanted"]
    assert reader.fetched_bytes == tensors["wanted"].numel() * tensors["wanted"].element_size()


def test_block_expert_banks_have_exact_global_ids_and_skip_full_constructor(
    tmp_path, monkeypatch
):
    from freetoken.models import nvfp4_banks

    tensors = {}
    for layer in range(3):
        for expert in range(2):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                base = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"
                out_dim = 16
                in_dim = 16
                tensors[base + ".weight"] = torch.zeros(out_dim, in_dim // 2, dtype=torch.uint8)
                tensors[base + ".weight_scale"] = torch.zeros(out_dim, in_dim // 16, dtype=torch.float8_e4m3fn)
                tensors[base + ".weight_scale_2"] = torch.ones((), dtype=torch.float32)
    _write_indexed_checkpoint(tmp_path, tensors)
    spec = nvfp4_banks.Nvfp4ExpertSourceSpec(
        key_pattern=re.compile(
            r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
            r"(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\."
            r"(?P<kind>weight|weight_scale|weight_scale_2)$"
        ),
        proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
        layer_to_bank=lambda layer, config: layer,
        desc="test experts",
    )
    config = SimpleNamespace(
        num_layers=3, num_experts=2, hidden_size=16, moe_intermediate_size=16,
        first_k_dense_replace=0,
    )
    monkeypatch.setattr(
        nvfp4_banks, "load_nvfp4_expert_source_banks",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("full constructor used")),
    )
    fetched = []
    banks = nvfp4_banks.load_nvfp4_expert_source_banks_for_layers(
        str(tmp_path), config, spec, (1, 2), drop_page_cache=lambda path: None,
        primary=False, layer_sink=lambda layer, layer_banks: None,
        on_fetch=lambda key, tensor: fetched.append(key),
    )
    assert all(len(per_layer) == 2 for per_layer in banks.values())
    assert len(fetched) == 2 * 2 * 3 * 3
    assert all("layers.0." not in key for key in fetched)

