"""R6 model-agnostic dense checkpoint census and selective block planning.

This module generalizes the accepted N0/Qwen checkpoint census to any
safetensors text checkpoint whose tensors follow the ``<prefix>.layers.N.``
convention, without expert, router, or model-family knowledge:

* the census derives layer identity, byte totals, and per-layer sizes purely
  from the safetensors header (single shard or sharded index);
* ownership classification comes from a small generic rule set over key
  names (embedding / final norm / per-layer / non-text), with the optional
  ``tied lm_head`` (no checkpoint tensor) handled by explicit declaration;
* block specs describe contiguous layer ranges plus embedding/head
  ownership, validated against the census;
* the selective loader reads ONLY planned keys, one tensor at a time, via
  a caller-supplied module factory — it never constructs or fetches model
  state outside the frozen allowed-key set.

Model-specific knowledge (key prefixes, module construction) lives in the
strategy adapter (benchmarks/), never here.  CPU-safe imports only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

CENSUS_SCHEMA = "inferswarm.r6.checkpoint-census/1"
PLAN_SCHEMA = "inferswarm.r6.model-block-plan/1"

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
class DenseBlockSpec:
    """One contiguous decoder-layer range plus global-state ownership."""

    start_layer: int
    end_layer: int
    owns_embeddings: bool
    owns_final_norm_head: bool

    def validate(self, num_layers: int) -> None:
        if self.start_layer < 0 or self.end_layer < 0:
            raise ValueError("block layer range cannot be negative")
        if self.start_layer >= self.end_layer:
            raise ValueError("block layer range must be non-empty and increasing")
        if self.end_layer > num_layers:
            raise ValueError(f"block end {self.end_layer} exceeds {num_layers} layers")


@dataclass(frozen=True)
class DenseTensorRecord:
    key: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    byte_count: int
    owner_category: str
    layer_id: int | None


def _layer_id_of(key: str, layer_re: re.Pattern[str]) -> int | None:
    match = layer_re.search(key)
    return int(match.group(1)) if match else None


def checkpoint_census(
    model_path: str | os.PathLike[str],
    *,
    text_prefix: str,
    non_text_prefixes: tuple[str, ...] = (),
    index_name: str = "model.safetensors.index.json",
) -> dict[str, Any]:
    """Census a safetensors checkpoint header set without reading tensor data.

    ``text_prefix`` is the key prefix of the text tower (everything under it
    is required text state); keys under ``non_text_prefixes`` are counted as
    explicitly non-text (vision/audio towers) and excluded from text state.
    """
    root = Path(model_path)
    index_path = root / index_name
    if index_path.exists():
        index_bytes = index_path.read_bytes()
        index = json.loads(index_bytes)
        by_shard: dict[str, list[str]] = defaultdict(list)
        for key, shard in index["weight_map"].items():
            by_shard[shard].append(key)
        shards = sorted(by_shard)
        total = int(index["metadata"]["total_size"])
        index_sha = hashlib.sha256(index_bytes).hexdigest()
    else:
        by_shard = {}
        shards = []
        total = 0
        index_sha = None
        for candidate in sorted(root.glob("*.safetensors")):
            if candidate.name.endswith(".index.safetensors"):
                continue
            shards.append(candidate.name)
        if not shards:
            raise ValueError(f"no safetensors shards under {root}")

    layer_re = re.compile(re.escape(text_prefix) + r"\.layers\.(\d+)\.")

    records: list[DenseTensorRecord] = []
    header_total = 0
    for shard in shards:
        with (root / shard).open("rb") as stream:
            header_len = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_len))
        header.pop("__metadata__", None)
        keys = sorted(header) if not by_shard else sorted(by_shard.get(shard, ()))
        if by_shard and set(header) != set(keys):
            raise ValueError(f"index/header key mismatch for {shard}")
        for key in keys:
            info = header[key]
            dtype = info["dtype"]
            shape = tuple(info["shape"])
            try:
                width = _DTYPE_BYTES[dtype]
            except KeyError as exc:
                raise ValueError(f"unsupported safetensors dtype {dtype!r}") from exc
            elements = 1
            for dim in shape:
                elements *= dim
            byte_count = elements * width
            header_total += byte_count
            layer_id = _layer_id_of(key, layer_re)
            if not key.startswith(text_prefix + "."):
                category = "non-text"
            elif key == f"{text_prefix}.embed_tokens.weight":
                category = "embedding/input"
            elif layer_id is not None:
                category = "decoder-layer"
            elif key == f"{text_prefix}.norm.weight":
                category = "final norm"
            else:
                category = "text-other"
            records.append(
                DenseTensorRecord(
                    key=key,
                    shard=shard,
                    dtype=dtype,
                    shape=shape,
                    byte_count=byte_count,
                    owner_category=category,
                    layer_id=layer_id,
                )
            )
    if index_path.exists() and header_total != total:
        raise ValueError(f"header bytes {header_total} != index total_size {total}")

    layer_ids = [r.layer_id for r in records if r.layer_id is not None]
    num_layers = max(layer_ids) + 1 if layer_ids else 0
    present = sorted({i for i in layer_ids})
    if present and present != list(range(num_layers)):
        raise ValueError("checkpoint layer ids are not contiguous from 0")

    per_layer = []
    for layer in range(num_layers):
        layer_records = [r for r in records if r.layer_id == layer]
        per_layer.append(
            {
                "layer": layer,
                "checkpoint_bytes": sum(r.byte_count for r in layer_records),
                "tensor_count": len(layer_records),
            }
        )
    category_bytes = defaultdict(int)
    for record in records:
        category_bytes[record.owner_category] += record.byte_count

    text_required = [
        r for r in records if r.owner_category in _TEXT_CATEGORIES
    ]
    return {
        "schema": CENSUS_SCHEMA,
        "measurement_status": "MEASURED / checkpoint-header-derived",
        "model_path": str(Path(model_path)),
        "text_prefix": text_prefix,
        "checkpoint_index_sha256": index_sha,
        "tensor_count": len(records),
        "total_checkpoint_bytes": header_total,
        "required_text_model_bytes": sum(r.byte_count for r in text_required),
        "bytes_by_owner_category": dict(sorted(category_bytes.items())),
        "per_layer": per_layer,
        "tensors": [asdict(r) for r in records],
    }


_TEXT_CATEGORIES = frozenset(
    {"embedding/input", "decoder-layer", "final norm", "text-other"}
)


def _owned(record: dict, block: DenseBlockSpec) -> bool:
    if record["owner_category"] not in _TEXT_CATEGORIES:
        return False
    layer = record["layer_id"]
    if layer is not None:
        return block.start_layer <= layer < block.end_layer
    category = record["owner_category"]
    if category == "embedding/input":
        return block.owns_embeddings
    return block.owns_final_norm_head  # final norm (+text-other globals)


def freeze_dense_block_plan(
    census: dict[str, Any],
    specs: list[DenseBlockSpec],
    *,
    declared_shared_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze an N-block contiguous plan over the census.

    ``declared_shared_state`` declares explicitly shared logical state that
    legally materializes on more than one participant (e.g. a tied
    embedding/lm_head), with its byte size and duplication policy.  Every
    other required text key must be owned by exactly one block.
    """
    num_layers = len(census["per_layer"])
    records = census["tensors"]
    required = {r["key"] for r in records if r["owner_category"] in _TEXT_CATEGORIES}

    previous_end = 0
    for spec in specs:
        spec.validate(num_layers)
        if spec.start_layer != previous_end:
            raise ValueError("block specs must be contiguous and ordered")
        previous_end = spec.end_layer
    if specs and specs[0].start_layer != 0:
        raise ValueError("first block must start at layer 0")
    if specs and specs[-1].end_layer != num_layers:
        raise ValueError("last block must end at the final layer")
    if not specs:
        raise ValueError("plan requires at least one block")
    if not specs[0].owns_embeddings:
        raise ValueError("only the first block may own embeddings")
    if specs[-1].owns_final_norm_head is False:
        raise ValueError("the last block must own the final norm/head")
    for spec in specs[1:]:
        if spec.owns_embeddings:
            raise ValueError("only the first block may own embeddings")
    for spec in specs[:-1]:
        if spec.owns_final_norm_head:
            raise ValueError("only the last block may own the final norm/head")

    shared = dict(declared_shared_state or {})
    shared_keys = set(shared.get("tensor_keys", []))

    owner_counts: dict[str, int] = defaultdict(int)
    for record in records:
        if record["key"] not in required:
            continue
        for spec in specs:
            if _owned(record, spec):
                owner_counts[record["key"]] += 1

    unowned = sorted(k for k in required if owner_counts[k] == 0)
    duplicated = sorted(k for k in required if owner_counts[k] > 1)
    undeclared_duplication = sorted(k for k in duplicated if k not in shared_keys)
    if unowned:
        raise ValueError(f"plan leaves required text keys unowned: {unowned[:5]}")
    if undeclared_duplication:
        raise ValueError(
            f"plan duplicates keys outside declared shared state: "
            f"{undeclared_duplication[:5]}"
        )

    def block_dict(spec: DenseBlockSpec) -> dict[str, Any]:
        keys = [r["key"] for r in records if _owned(r, spec)]
        selected = [r for r in records if r["key"] in set(keys)]
        return {
            "spec": asdict(spec),
            "allowed_tensor_keys": keys,
            "owned_checkpoint_bytes": sum(r["byte_count"] for r in selected),
        }

    return {
        "schema": PLAN_SCHEMA,
        "status": "FROZEN_BEFORE_R6_PHYSICAL_EXECUTION",
        "model_path": census["model_path"],
        "checkpoint_index_sha256": census["checkpoint_index_sha256"],
        "number_of_layers": num_layers,
        "blocks": [block_dict(spec) for spec in specs],
        "declared_shared_state": shared,
        "total_checkpoint_bytes": census["total_checkpoint_bytes"],
        "required_text_model_bytes": census["required_text_model_bytes"],
        "coverage_proof": {
            "required_key_union_is_all": True,
            "unowned_required_keys": unowned,
            "duplicated_keys_declared_shared": True,
        },
    }


class DenseSelectiveTensorReader:
    """Read only planned keys, one tensor at a time, from indexed shards."""

    def __init__(self, model_path: str | os.PathLike[str], allowed_keys):
        self.root = Path(model_path)
        self.allowed_keys = frozenset(allowed_keys)
        index_path = self.root / "model.safetensors.index.json"
        single = self.root / "model.safetensors"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            self._weight_map = index["weight_map"]
        elif single.exists():
            self._weight_map = None
            # Fail closed at construction even without an index: read the
            # single shard's header (metadata only) and reject any planned
            # key the checkpoint does not carry.
            with single.open("rb") as stream:
                header_len = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(header_len))
            header.pop("__metadata__", None)
            unknown = self.allowed_keys - set(header)
            if unknown:
                raise ValueError(
                    f"allowed keys absent from checkpoint: {sorted(unknown)[:3]}"
                )
        else:
            raise ValueError(f"no safetensors index or single shard under {self.root}")
        if self._weight_map is not None:
            unknown = self.allowed_keys - set(self._weight_map)
            if unknown:
                raise ValueError(
                    f"allowed keys absent from checkpoint: {sorted(unknown)[:3]}"
                )
        self.fetched_keys: list[str] = []
        self.fetched_bytes = 0

    def tensors(self, device="cpu") -> Iterator[tuple[str, Any]]:
        import safetensors

        if self._weight_map is None:
            by_shard = {None: sorted(self.allowed_keys)}
            shards = [None]
        else:
            by_shard = defaultdict(list)
            for key in sorted(self.allowed_keys):
                by_shard[self._weight_map[key]].append(key)
            shards = sorted(by_shard)
        for shard in shards:
            path = str(self.root / shard) if shard else str(self.root / "model.safetensors")
            with safetensors.safe_open(path, framework="pt", device=str(device)) as handle:
                for key in by_shard[shard]:
                    tensor = handle.get_tensor(key)
                    self.fetched_keys.append(key)
                    self.fetched_bytes += tensor.numel() * tensor.element_size()
                    yield key, tensor


def load_selective_dense_block(
    model_path: str,
    block: dict[str, Any],
    module_factory: Callable[[DenseBlockSpec], Any],
    *,
    device,
    key_adapter: Callable[[str], str] | None = None,
    reader: DenseSelectiveTensorReader | None = None,
) -> dict[str, Any]:
    """Materialize one dense block via a strategy-supplied module factory.

    The factory constructs only the modules this block owns (on ``meta`` or
    the target device, per strategy choice); this loader then streams ONLY
    the planned checkpoint keys into it and fails closed on any unplanned
    fetch, unexpected key, or missing key.  Tied/shared state explicitly
    listed in the plan's declared shared state may be loaded on every
    participant that declares it.
    """
    import torch  # local import: loader is compute-node-only

    spec = DenseBlockSpec(**block["spec"])
    allowed = frozenset(block["allowed_tensor_keys"])
    shared_keys = frozenset(
        (block.get("declared_shared_state") or {}).get("tensor_keys", [])
    )
    module = module_factory(spec)

    state = module.state_dict()
    unexpected = sorted(set(state) - allowed - shared_keys)
    if unexpected:
        raise RuntimeError(
            f"module factory built state outside the plan: {unexpected[:5]}"
        )
    target = reader or DenseSelectiveTensorReader(model_path, allowed | shared_keys)

    loaded: dict[str, Any] = {}
    planned = allowed | shared_keys
    for key, tensor in target.tensors(device="cpu"):
        module_key = key_adapter(key) if key_adapter else key
        if key not in planned:
            raise RuntimeError(f"selective loader fetched unplanned key {key}")
        expected = state.get(module_key)
        if expected is None:
            raise RuntimeError(f"checkpoint key {key} has no module destination")
        loaded[module_key] = tensor.to(device=device, dtype=expected.dtype)
    module.load_state_dict(loaded)
    leftovers = sorted((set(state) & planned) - set(loaded))
    if leftovers:
        raise RuntimeError(f"planned keys never loaded: {leftovers[:5]}")
    return {
        "module": module,
        "spec": spec,
        "fetched_keys": list(target.fetched_keys),
        "fetched_bytes": target.fetched_bytes,
        "fetched_unexpected": [],
    }


def write_json_with_sha(path: str, payload: dict) -> None:
    target = Path(path)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    target.with_suffix(target.suffix + ".sha256").write_text(
        f"{hashlib.sha256(data).hexdigest()}  {target.name}\n"
    )


__all__ = [
    "CENSUS_SCHEMA",
    "PLAN_SCHEMA",
    "DenseBlockSpec",
    "DenseTensorRecord",
    "DenseSelectiveTensorReader",
    "checkpoint_census",
    "freeze_dense_block_plan",
    "load_selective_dense_block",
    "write_json_with_sha",
]
