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
D3_ARTIFACT_SHA256 = "6c9cb2c5dbd46cdd93049769427b3c5bec47cdbb0564050c8fedb7a90eac15d7"
MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
REVISION = "491c2f1ea524c639598bf8fa787a93fed5a6fbce"


@dataclass(frozen=True)
class D3Placement:
    artifact_sha256: str
    worker_a: FrozenPlacement
    worker_b: FrozenPlacement


def _worker(doc, key, digest):
    geometry = doc["geometry"]
    raw = doc["partition"][key]
    ids = raw["flat_ids_in_rank_order"]
    records = raw["identities"]
    if len(ids) != 3000 or len(set(ids)) != 3000 or len(records) != 3000:
        raise ValueError(f"D3 {key} must contain 3,000 unique identities")
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
        identities.append(PlacementIdentity(flat, layer, expert, slot))
        by_layer[layer].append(identities[-1])
    per_layer = tuple(LayerPlacement(layer, tuple(i.expert_id for i in rows), tuple(i.remote_slot for i in rows)) for layer, rows in enumerate(by_layer))
    return FrozenPlacement(digest, D3_SCHEMA, "phase1r-d3-three-device", D3_STATUS, key, MODEL, REVISION, 40, 256, 10240, 1775616, 3000, 5326848000, 5326848000, tuple(ids), tuple(identities), per_layer, MappingProxyType(mapping))


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
        union = doc["ranking"]["ranked_union_flat_ids"]
        a, b = doc["partition"]["worker_a"]["flat_ids_in_rank_order"], doc["partition"]["worker_b"]["flat_ids_in_rank_order"]
        if len(union) != 6000 or len(set(union)) != 6000 or set(union) != set(a) | set(b) or set(a) & set(b):
            raise ValueError("D3 placement A/B union/disjointness disagreement")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid D3 placement: {exc}") from exc
    return D3Placement(digest, _worker(doc, "worker_a", digest), _worker(doc, "worker_b", digest))


def load_d3_placement(path: str | Path) -> D3Placement:
    return parse_d3_placement_bytes(Path(path).read_bytes())
