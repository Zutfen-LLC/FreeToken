"""N0 checkpoint census and selective contiguous model-block planning.

This module is deliberately outside the normal model-loading path.  Selection is
computed from the safetensors index/header metadata before ``get_tensor`` is called.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterator

import safetensors
import torch

MODEL_REPOSITORY = "nvidia/Qwen3.6-35B-A3B-NVFP4"
MODEL_REVISION = "491c2f1ea524c639598bf8fa787a93fed5a6fbce"
PLAN_SCHEMA = "inferswarm.n0.model-block-plan/1"
CENSUS_SCHEMA = "inferswarm.n0.checkpoint-census/1"

_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class ModelBlockSpec:
    start_layer: int
    end_layer: int
    owns_embeddings: bool
    owns_final_norm_head: bool
    model_repository: str = MODEL_REPOSITORY
    revision: str = MODEL_REVISION

    def validate(self, num_layers: int) -> None:
        if self.model_repository != MODEL_REPOSITORY or self.revision != MODEL_REVISION:
            raise ValueError("N0 block model/revision identity mismatch")
        if self.start_layer < 0 or self.end_layer < 0:
            raise ValueError("block layer range cannot be negative")
        if self.start_layer >= self.end_layer:
            raise ValueError("block layer range must be non-empty and increasing")
        if self.end_layer > num_layers:
            raise ValueError(f"block end {self.end_layer} exceeds {num_layers} layers")


@dataclass(frozen=True)
class TensorRecord:
    key: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    byte_count: int
    owner_category: str
    layer_id: int | None
    routed_expert: bool
    required_for_text_runtime: bool


def _classify(key: str) -> tuple[str, int | None, bool, bool]:
    match = _LAYER_RE.match(key)
    if match:
        layer = int(match.group(1))
        expert = _EXPERT_RE.search(key) is not None
        # ModelOpt activation calibration values are not consumed by FreeToken's W4A16
        # routed-expert implementation.  They remain in the complete checkpoint census.
        required = not (expert and key.endswith(".input_scale"))
        return f"layer {layer}", layer, expert, required
    if key == "model.language_model.embed_tokens.weight":
        return "embedding/input", None, False, True
    if key == "model.language_model.norm.weight":
        return "final norm", None, False, True
    if key.startswith("lm_head."):
        return "LM head", None, False, True
    if key.startswith(("model.visual.", "visual.", "mtp.")):
        return "optional/non-text", None, False, False
    if key.startswith("model.language_model."):
        return "global/shared", None, False, True
    return "optional/non-text", None, False, False


def checkpoint_census(model_path: str | os.PathLike[str]) -> dict:
    root = Path(model_path)
    index_path = root / "model.safetensors.index.json"
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard in index["weight_map"].items():
        by_shard[shard].append(key)

    records: list[TensorRecord] = []
    for shard in sorted(by_shard):
        # Parse the documented safetensors header directly. Calling get_slice once per
        # key is quadratic in safetensors 0.8 for this 124k-key checkpoint.
        with (root / shard).open("rb") as stream:
            header_len = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_len))
        header.pop("__metadata__", None)
        actual = set(header)
        expected = set(by_shard[shard])
        if actual != expected:
            raise ValueError(f"index/header key mismatch for {shard}")
        for key in sorted(expected):
                dtype = header[key]["dtype"]
                shape = tuple(header[key]["shape"])
                try:
                    width = _DTYPE_BYTES[dtype]
                except KeyError as exc:
                    raise ValueError(f"unsupported safetensors dtype {dtype!r}") from exc
                elements = 1
                for dim in shape:
                    elements *= dim
                category, layer, expert, required = _classify(key)
                records.append(TensorRecord(
                    key, shard, dtype, shape, elements * width, category, layer, expert, required
                ))

    total = sum(r.byte_count for r in records)
    if total != int(index["metadata"]["total_size"]):
        raise ValueError(f"header bytes {total} != index total_size")
    per_layer = []
    num_layers = max(r.layer_id for r in records if r.layer_id is not None) + 1
    for layer in range(num_layers):
        layer_records = [r for r in records if r.layer_id == layer]
        expert = sum(r.byte_count for r in layer_records if r.routed_expert)
        required_expert = sum(
            r.byte_count for r in layer_records if r.routed_expert and r.required_for_text_runtime
        )
        per_layer.append({
            "layer": layer,
            "checkpoint_bytes": sum(r.byte_count for r in layer_records),
            "routed_expert_checkpoint_bytes": expert,
            "routed_expert_required_bytes": required_expert,
            "non_expert_checkpoint_bytes": sum(
                r.byte_count for r in layer_records if not r.routed_expert
            ),
            "required_text_bytes": sum(
                r.byte_count for r in layer_records if r.required_for_text_runtime
            ),
        })
    category_bytes = defaultdict(int)
    for record in records:
        category_bytes[record.owner_category] += record.byte_count
    return {
        "schema": CENSUS_SCHEMA,
        "measurement_status": "MEASURED / checkpoint-derived",
        "model_repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "checkpoint_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "tensor_count": len(records),
        "total_checkpoint_bytes": total,
        "required_text_model_bytes": sum(
            r.byte_count for r in records if r.required_for_text_runtime
        ),
        "bytes_by_owner_category": dict(sorted(category_bytes.items())),
        "per_layer": per_layer,
        "tensors": [asdict(r) for r in records],
    }


def _owned(record: dict, block: ModelBlockSpec) -> bool:
    if not record["required_for_text_runtime"]:
        return False
    layer = record["layer_id"]
    if layer is not None:
        return block.start_layer <= layer < block.end_layer
    category = record["owner_category"]
    if category == "embedding/input":
        return block.owns_embeddings
    if category in {"final norm", "LM head"}:
        return block.owns_final_norm_head
    # Genuine global/shared text tensors are explicitly duplicated.
    return category == "global/shared"


def freeze_two_block_plan(census: dict, layer_types: list[str]) -> dict:
    num_layers = len(layer_types)
    if num_layers != len(census["per_layer"]):
        raise ValueError("config/checkpoint layer-count mismatch")
    records = census["tensors"]
    candidates = []
    for boundary in range(1, num_layers):
        a = ModelBlockSpec(0, boundary, True, False)
        b = ModelBlockSpec(boundary, num_layers, False, True)
        a_bytes = sum(r["byte_count"] for r in records if _owned(r, a))
        b_bytes = sum(r["byte_count"] for r in records if _owned(r, b))
        candidates.append((max(a_bytes, b_bytes), boundary, a_bytes, b_bytes))
    _, split, a_bytes, b_bytes = min(candidates)
    a = ModelBlockSpec(0, split, True, False)
    b = ModelBlockSpec(split, num_layers, False, True)
    a_keys = [r["key"] for r in records if _owned(r, a)]
    b_keys = [r["key"] for r in records if _owned(r, b)]
    duplicated = sorted(set(a_keys) & set(b_keys))
    required = {r["key"] for r in records if r["required_for_text_runtime"]}
    coverage = set(a_keys) | set(b_keys)
    if coverage != required:
        raise ValueError("two-block plan does not cover required text state")
    ordinary_overlap = [k for k in duplicated if _LAYER_RE.match(k)]
    if ordinary_overlap:
        raise ValueError("ordinary layer tensor duplicated between blocks")

    def block_dict(spec: ModelBlockSpec, keys: list[str], total: int) -> dict:
        key_set = set(keys)
        selected = [r for r in records if r["key"] in key_set]
        return {
            "spec": asdict(spec),
            "allowed_tensor_keys": keys,
            "owned_checkpoint_bytes": total,
            "bytes_by_state_category": {
                "embedding_input": sum(r["byte_count"] for r in selected if r["owner_category"] == "embedding/input"),
                "routed_experts": sum(r["byte_count"] for r in selected if r["routed_expert"]),
                "layer_non_expert": sum(r["byte_count"] for r in selected if r["layer_id"] is not None and not r["routed_expert"]),
                "final_norm": sum(r["byte_count"] for r in selected if r["owner_category"] == "final norm"),
                "lm_head": sum(r["byte_count"] for r in selected if r["owner_category"] == "LM head"),
                "global_shared": sum(r["byte_count"] for r in selected if r["owner_category"] == "global/shared"),
            },
        }

    return {
        "schema": PLAN_SCHEMA,
        "status": "FROZEN_BEFORE_N0_PHYSICAL_EXECUTION",
        "model_repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "checkpoint_index_sha256": census["checkpoint_index_sha256"],
        "number_of_layers": num_layers,
        "layer_types": layer_types,
        "split_rule": "minimize max(block_A_owned_bytes, block_B_owned_bytes); tie-break lowest boundary",
        "split_boundary": split,
        "block_a": block_dict(a, a_keys, a_bytes),
        "block_b": block_dict(b, b_keys, b_bytes),
        "duplicated_tensor_keys": duplicated,
        "duplicated_bytes": sum(r["byte_count"] for r in records if r["key"] in set(duplicated)),
        "total_checkpoint_bytes": census["total_checkpoint_bytes"],
        "required_text_model_bytes": census["required_text_model_bytes"],
        "coverage_proof": {
            "layer_union_is_all": True,
            "layer_intersection_is_empty": True,
            "required_key_union_is_all": coverage == required,
            "ordinary_layer_key_intersection_is_empty": not ordinary_overlap,
            "unowned_required_keys": sorted(required - coverage),
        },
    }


class SelectiveTensorReader:
    """Read only planned keys, one tensor at a time, from indexed shards."""

    def __init__(self, model_path: str | os.PathLike[str], allowed_keys: set[str]):
        self.root = Path(model_path)
        self.allowed_keys = frozenset(allowed_keys)
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        unknown = self.allowed_keys - set(index["weight_map"])
        if unknown:
            raise ValueError(f"allowed keys absent from checkpoint: {sorted(unknown)[:3]}")
        self._weight_map = index["weight_map"]
        self.fetched_keys: list[str] = []
        self.fetched_bytes = 0

    def tensors(self, device: torch.device | str = "cpu") -> Iterator[tuple[str, torch.Tensor]]:
        by_shard: dict[str, list[str]] = defaultdict(list)
        for key in sorted(self.allowed_keys):
            by_shard[self._weight_map[key]].append(key)
        for shard in sorted(by_shard):
            with safetensors.safe_open(self.root / shard, framework="pt", device=str(device)) as handle:
                for key in by_shard[shard]:
                    tensor = handle.get_tensor(key)
                    self.fetched_keys.append(key)
                    self.fetched_bytes += tensor.numel() * tensor.element_size()
                    yield key, tensor


@dataclass
class SelectiveBlockLoadResult:
    block: "SelectiveQwen35Block"
    expert_banks: dict[str, list[torch.Tensor]]
    global_layer_ids: tuple[int, ...]
    allowed_keys: frozenset[str]
    fetched_keys: list[str]
    fetched_bytes: int


class SelectiveQwen35Block:
    """Only the real modules owned by one N0 block; layer IDs remain global."""

    def __init__(self, config, spec: ModelBlockSpec):
        from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead
        from freetoken.layers import GemmaRMSNorm, ParallelLMHead, VocabParallelEmbedding
        from freetoken.models.qwen3_5_moe.model import Qwen3_5DecoderLayer

        spec.validate(config.num_layers)
        self.spec = spec
        self.config = config
        self.global_layer_ids = tuple(range(spec.start_layer, spec.end_layer))
        self.embed_tokens = (
            VocabParallelEmbedding(config.vocab_size, config.hidden_size)
            if spec.owns_embeddings else None
        )
        self.layers = [Qwen3_5DecoderLayer(config, layer) for layer in self.global_layer_ids]
        self.norm = (
            GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if spec.owns_final_norm_head else None
        )
        if spec.owns_final_norm_head:
            self.lm_head = (
                Nvfp4LMHead(config.vocab_size, config.hidden_size)
                if config.lm_head_quant == "nvfp4"
                else ParallelLMHead(config.vocab_size, config.hidden_size)
            )
        else:
            self.lm_head = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        if self.embed_tokens is not None:
            self.embed_tokens.state_dict(prefix="model.embed_tokens", result=result)
        for layer_id, layer in zip(self.global_layer_ids, self.layers, strict=True):
            layer.state_dict(prefix=f"model.layers.{layer_id}", result=result)
        if self.norm is not None:
            self.norm.state_dict(prefix="model.norm", result=result)
        if self.lm_head is not None:
            self.lm_head.state_dict(prefix="lm_head", result=result)
        return result

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        state = dict(state)
        if self.embed_tokens is not None:
            self.embed_tokens.load_state_dict(state, prefix="model.embed_tokens", _internal=True)
        for layer_id, layer in zip(self.global_layer_ids, self.layers, strict=True):
            layer.load_state_dict(state, prefix=f"model.layers.{layer_id}", _internal=True)
        if self.norm is not None:
            self.norm.load_state_dict(state, prefix="model.norm", _internal=True)
        if self.lm_head is not None:
            self.lm_head.load_state_dict(state, prefix="lm_head", _internal=True)
        if state:
            raise RuntimeError(f"unexpected selective block keys: {list(state)[:8]}")

    def forward_layers(
        self, hidden: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        for layer in self.layers:
            hidden, residual = layer.forward(hidden, residual)
        return hidden, residual

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is None:
            raise ValueError("this block does not own embeddings")
        return self.embed_tokens.forward(input_ids)

    def finalize(self, hidden: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
        if self.norm is None:
            raise ValueError("this block does not own final normalization")
        hidden, _ = self.norm.forward_add_residual(hidden, residual)
        return hidden


def load_selective_qwen35_block(
    model_path: str,
    block_spec: ModelBlockSpec,
    allowed_keys: frozenset[str],
    *,
    device: torch.device,
) -> SelectiveBlockLoadResult:
    """Materialize a pinned-Qwen3.6 block and its block-local native expert banks."""
    from freetoken.models.qwen3_5_moe.config import parse_config
    from freetoken.models.qwen3_5_moe.weight import (
        _NVFP4_SOURCE_SPEC,
        iter_block_weights,
    )
    from freetoken.models.nvfp4_banks import load_nvfp4_expert_source_banks_for_layers
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.utils import cached_load_hf_config, torch_dtype
    from freetoken.layers.rotary import set_rope_device

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    config = replace(parse_config(cached_load_hf_config(model_path)), moe_backend="offload")
    block_spec.validate(config.num_layers)
    set_rope_device(device)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        block = SelectiveQwen35Block(config, block_spec)

    fetched_keys: list[str] = []
    fetched_bytes = 0

    def note_fetch(key: str, tensor: torch.Tensor) -> None:
        nonlocal fetched_bytes
        if key not in allowed_keys:
            raise RuntimeError(f"selective loader fetched unplanned key {key}")
        fetched_keys.append(key)
        fetched_bytes += tensor.numel() * tensor.element_size()

    model_state = block.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    for key, weight in iter_block_weights(
        model_path, device, allowed_raw_keys=allowed_keys, on_fetch=note_fetch
    ):
        expected = model_state.get(key)
        loaded[key] = weight.to(device=device, dtype=expected.dtype if expected is not None else None)
    block.load_state_dict(loaded)

    # This calls the block-specific constructor directly, never the legacy full-bank API.
    from freetoken.models.qwen3_5_moe.weight import drop_page_cache

    banks = load_nvfp4_expert_source_banks_for_layers(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        block.global_layer_ids,
        drop_page_cache=drop_page_cache,
        primary=True,
        on_fetch=note_fetch,
    )
    return SelectiveBlockLoadResult(
        block, banks, block.global_layer_ids, allowed_keys, fetched_keys, fetched_bytes
    )


def validate_complement(a: ModelBlockSpec, b: ModelBlockSpec, num_layers: int) -> None:
    a.validate(num_layers)
    b.validate(num_layers)
    if a.start_layer != 0 or a.end_layer != b.start_layer or b.end_layer != num_layers:
        raise ValueError("two-block plan must be contiguous, complementary, and complete")
    if not a.owns_embeddings or b.owns_embeddings:
        raise ValueError("only block A may own embeddings")
    if a.owns_final_norm_head or not b.owns_final_norm_head:
        raise ValueError("only block B may own final norm/head")


def write_json_with_sha(path: str | os.PathLike[str], payload: dict) -> None:
    target = Path(path)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    target.with_suffix(target.suffix + ".sha256").write_text(
        f"{hashlib.sha256(data).hexdigest()}  {target.name}\n"
    )


__all__ = [
    "ModelBlockSpec", "SelectiveTensorReader", "checkpoint_census",
    "SelectiveBlockLoadResult", "SelectiveQwen35Block", "freeze_two_block_plan",
    "load_selective_qwen35_block", "validate_complement", "write_json_with_sha",
]
