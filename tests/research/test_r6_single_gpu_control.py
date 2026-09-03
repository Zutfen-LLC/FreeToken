from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from freetoken.layers.base import BaseOP
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


class _TinyOwned(BaseOP):
    """Minimal BaseOP-backed stand-in for a Gemma sub-module: shares the
    exact load_state_dict(state, prefix, _internal) pop-based contract real
    modules (VocabParallelEmbedding/Gemma4DecoderLayer/GemmaRMSNorm) use,
    without needing a full model config."""

    def __init__(self, **tensors):
        for name, tensor in tensors.items():
            setattr(self, name, tensor)


class _FakeCudaTensor:
    """Duck-typed CUDA-resident tensor: only .device/.numel()/.element_size()
    are exercised by the residency helpers, so no real GPU is needed to
    prove the "fully resident" branch."""

    def __init__(self, n: int, element_size: int = 4):
        self.device = SimpleNamespace(type="cuda")
        self._n = n
        self._element_size = element_size

    def numel(self):
        return self._n

    def element_size(self):
        return self._element_size


def test_stage_modules_load_state_dict_fully_consumes_caller_dict():
    """Regression for the fatal real-runtime bug: _StageModules.load_state_dict
    used to defensively copy its input (`state = dict(state)`), but
    BaseOP.load_state_dict pops keys from whatever dict it is given -- so
    the copy drained instead of the caller's dict, and _selective_load's
    `if loaded: raise` fired unconditionally on every real (non-empty)
    load. This exercises the actual module-container consumption path
    (_StageModules, module-level and directly importable), not
    materialize_dense_state() in isolation."""
    _StageModules = _STAGE._StageModules
    spec = SimpleNamespace(start_layer=5, end_layer=7)
    modules = _StageModules(config=None, spec=spec, full_config=None)
    modules.embed_tokens = _TinyOwned(weight=torch.zeros(3, 3))
    modules.layers = [_TinyOwned(weight=torch.zeros(2, 2)) for _ in range(2)]
    modules.norm = _TinyOwned(weight=torch.zeros(2))

    state = {
        "model.embed_tokens.weight": torch.ones(3, 3),
        "model.layers.5.weight": torch.full((2, 2), 5.0),
        "model.layers.6.weight": torch.full((2, 2), 6.0),
        "model.norm.weight": torch.tensor([9.0, 9.0]),
    }
    modules.load_state_dict(state)

    # the caller's own dict object must be fully drained on success -- this
    # is exactly what _selective_load's post-load `if loaded: raise` checks.
    assert state == {}
    torch.testing.assert_close(modules.embed_tokens.weight, torch.ones(3, 3))
    torch.testing.assert_close(modules.layers[0].weight, torch.full((2, 2), 5.0))
    torch.testing.assert_close(modules.layers[1].weight, torch.full((2, 2), 6.0))
    torch.testing.assert_close(modules.norm.weight, torch.tensor([9.0, 9.0]))


def test_stage_modules_load_state_dict_raises_on_leftover_keys():
    _StageModules = _STAGE._StageModules
    spec = SimpleNamespace(start_layer=0, end_layer=1)
    modules = _StageModules(config=None, spec=spec, full_config=None)
    modules.layers = [_TinyOwned(weight=torch.zeros(2, 2))]

    state = {
        "model.layers.0.weight": torch.ones(2, 2),
        "model.layers.99.weight": torch.zeros(2, 2),
    }
    with pytest.raises(RuntimeError, match="unexpected selective block keys"):
        modules.load_state_dict(state)


def test_host_resident_bytes_flags_non_cuda_tensors():
    host_bytes, host_keys = _STAGE._host_resident_bytes(
        {
            "a": torch.zeros(4, dtype=torch.float32),
            "b": torch.zeros(2, dtype=torch.float32),
        }
    )
    assert host_bytes == (4 + 2) * 4
    assert set(host_keys) == {"a", "b"}


def test_host_resident_bytes_zero_when_fully_cuda_resident():
    host_bytes, host_keys = _STAGE._host_resident_bytes(
        {"a": _FakeCudaTensor(8), "b": _FakeCudaTensor(4)}
    )
    assert host_bytes == 0
    assert host_keys == []


def test_cpu_owned_decoder_layers_counts_layers_with_any_host_tensor():
    resident_layer = _TinyOwned(weight=_FakeCudaTensor(4))
    host_layer = _TinyOwned(weight=torch.zeros(4))
    count = _STAGE._cpu_owned_decoder_layers([10, 11], [resident_layer, host_layer])
    assert count == 1


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
