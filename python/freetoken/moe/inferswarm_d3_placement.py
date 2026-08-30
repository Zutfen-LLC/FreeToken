"""Strict parser for the frozen InferSwarm D3 placement artifact.

This deliberately does not expand the canonical Phase-1 placement allow-list.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .inferswarm_resident_bank import FrozenPlacement, LayerPlacement, PlacementIdentity

D3_SCHEMA = "inferswarm.phase1r.d3-placement/1"
D3_STATUS = "FROZEN_BEFORE_D3_PERFORMANCE"
D3_ARTIFACT_SHA256 = "6677fe1c506376a55aa8dcabb8d5761dc0373ced9d9b053209991059556d5887"
MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
REVISION = "491c2f1ea524c639598bf8fa787a93fed5a6fbce"


@dataclass(frozen=True)
class D3Placement:
    artifact_sha256: str
    worker_a: FrozenPlacement
    worker_b: FrozenPlacement
    local_remainder: tuple[int, ...]


def _identity_set(doc, key, digest, *, count, flat_field, remote_slots):
    raw = doc["partition"][key]
    ids = raw[flat_field]
    records = raw["identities"]
    if len(ids) != count or len(set(ids)) != count or len(records) != count:
        raise ValueError(f"D3 {key} must contain {count:,} unique identities")
    identities = []
    mapping = {}
    by_layer = [[] for _ in range(40)]
    for slot, (flat, record) in enumerate(zip(ids, records, strict=True)):
        layer, expert = record.get("layer"), record.get("expert_id")
        if not isinstance(flat, int) or record.get("flat_id") != flat or flat != layer * 256 + expert:
            raise ValueError(f"D3 {key} flat-ID arithmetic failed at slot {slot}")
        if not 0 <= layer < 40 or not 0 <= expert < 256 or (layer, expert) in mapping:
            raise ValueError(f"D3 {key} identity is invalid at slot {slot}")
        mapping[layer, expert] = slot
        identities.append(PlacementIdentity(flat, layer, expert, slot if remote_slots else -1))
        by_layer[layer].append(identities[-1])
    observed = raw["per_layer"]
    if len(observed) != 40:
        raise ValueError(f"D3 {key} must represent 40 layers")
    for layer, entry in enumerate(observed):
        if entry.get("layer") != layer or entry.get("expert_ids") != [i.expert_id for i in by_layer[layer]]:
            raise ValueError(f"D3 {key} per-layer identities disagree")
    per_layer = tuple(LayerPlacement(layer, tuple(i.expert_id for i in rows), tuple(i.remote_slot for i in rows)) for layer, rows in enumerate(by_layer))
    return tuple(ids), tuple(identities), per_layer, MappingProxyType(mapping)


def _worker(doc, key, digest):
    ids, identities, per_layer, mapping = _identity_set(doc, key, digest, count=3000, flat_field="flat_ids_in_rank_order", remote_slots=True)
    return FrozenPlacement(digest, D3_SCHEMA, "phase1r-d3-three-device", D3_STATUS, key, MODEL, REVISION, 40, 256, 10240, 1775616, 3000, 5326848000, 5326848000, ids, identities, per_layer, mapping)


def parse_d3_placement_bytes(raw: bytes) -> D3Placement:
    digest = hashlib.sha256(raw).hexdigest()
    if digest != D3_ARTIFACT_SHA256:
        raise ValueError(f"D3 placement SHA-256 disagreement: expected {D3_ARTIFACT_SHA256}, got {digest}")
    try:
        doc = json.loads(raw)
        geometry = doc["geometry"]
        if doc["schema"] != D3_SCHEMA or doc["status"] != D3_STATUS:
            raise ValueError("D3 placement schema/status disagreement")
        if doc["model"] != {"repository": MODEL, "revision": REVISION}:
            raise ValueError("D3 placement model provenance disagreement")
        if (geometry["num_moe_layers"], geometry["num_experts_per_layer"], geometry["logical_expert_identities"], geometry["gpu0_cache_slots"], geometry["worker_a_slots"], geometry["worker_b_slots"], geometry["bytes_per_native_nvfp4_identity"], geometry["worker_resident_bytes_each"], geometry["combined_worker_resident_bytes"]) != (40, 256, 10240, 3774, 3000, 3000, 1775616, 5326848000, 10653696000):
            raise ValueError("D3 placement geometry/byte arithmetic disagreement")
        semantics = doc["semantics"]
        if (semantics["gpu0_local_identity_count"], semantics["gpu0_cache_capacity_slots"], semantics["gpu0_cache_kind"], semantics["gpu0_cache_policy"], semantics["gpu0_cache_is_logical_ownership"]) != (4240, 3774, "dynamic", "FreeToken runtime cache", False):
            raise ValueError("D3 GPU0 local-ownership/cache semantics disagreement")
        union = doc["ranking"]["ranked_union_flat_ids"]
        a, b = doc["partition"]["worker_a"]["flat_ids_in_rank_order"], doc["partition"]["worker_b"]["flat_ids_in_rank_order"]
        if len(union) != 6000 or len(set(union)) != 6000 or set(union) != set(a) | set(b) or set(a) & set(b):
            raise ValueError("D3 placement A/B union/disjointness disagreement")
        local, _identities, _layers, _mapping = _identity_set(doc, "local_remainder", digest, count=4240, flat_field="flat_ids", remote_slots=False)
        if set(a) & set(local) or set(b) & set(local) or set(a) | set(b) | set(local) != set(range(10240)):
            raise ValueError("D3 logical ownership partition disagreement")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid D3 placement: {exc}") from exc
    return D3Placement(digest, _worker(doc, "worker_a", digest), _worker(doc, "worker_b", digest), local)


def load_d3_placement(path: str | Path) -> D3Placement:
    return parse_d3_placement_bytes(Path(path).read_bytes())
