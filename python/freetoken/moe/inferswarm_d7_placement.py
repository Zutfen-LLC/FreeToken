"""Strict parser for the frozen D7 fan-in-sparse placement artifact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from .inferswarm_d3_placement import D3Placement, MODEL, REVISION
from .inferswarm_resident_bank import FrozenPlacement, LayerPlacement, PlacementIdentity

D7_SCHEMA = "inferswarm.phase1r.d7-placement/1"
D7_STATUS = "FROZEN_BEFORE_D7_PERFORMANCE"
D7_ARTIFACT_SHA256 = "c360cad506fa4dbc2f768b24d8ad5dfd1d10956aafc88a5fa6e2736dfe0581d1"
D3_ARTIFACT_SHA256 = "6677fe1c506376a55aa8dcabb8d5761dc0373ced9d9b053209991059556d5887"


def _worker(doc, key: str, digest: str) -> FrozenPlacement:
    raw = doc["partition"][key]
    flat_ids = raw["flat_ids_in_rank_order"]
    records = raw["identities"]
    if len(flat_ids) != 3000 or len(set(flat_ids)) != 3000 or len(records) != 3000:
        raise ValueError(f"D7 {key} must contain 3,000 unique identities")
    identities = []
    mapping = {}
    layers = [[] for _ in range(40)]
    for slot, (flat, record) in enumerate(zip(flat_ids, records, strict=True)):
        layer, expert = record.get("layer"), record.get("expert_id")
        if (not isinstance(flat, int) or record.get("flat_id") != flat
                or flat != layer * 256 + expert or not 0 <= layer < 40
                or not 0 <= expert < 256 or (layer, expert) in mapping):
            raise ValueError(f"D7 {key} identity arithmetic failed at slot {slot}")
        identity = PlacementIdentity(flat, layer, expert, slot)
        identities.append(identity)
        mapping[layer, expert] = slot
        layers[layer].append(identity)
    per_layer = tuple(LayerPlacement(layer, tuple(row.expert_id for row in values),
                                     tuple(row.remote_slot for row in values))
                      for layer, values in enumerate(layers))
    expected = [{"layer": row.layer_id, "expert_ids": list(row.expert_ids)} for row in per_layer]
    if raw["per_layer"] != expected:
        raise ValueError(f"D7 {key} per-layer identities disagree")
    return FrozenPlacement(digest, D7_SCHEMA, "phase1r-d7-fanin-sparse", D7_STATUS, key,
                           MODEL, REVISION, 40, 256, 10240, 1775616, 3000,
                           5326848000, 5326848000, tuple(flat_ids), tuple(identities),
                           per_layer, MappingProxyType(mapping))


def parse_d7_placement_bytes(raw: bytes) -> D3Placement:
    digest = hashlib.sha256(raw).hexdigest()
    if digest != D7_ARTIFACT_SHA256:
        raise ValueError(f"D7 placement SHA-256 disagreement: expected {D7_ARTIFACT_SHA256}, got {digest}")
    try:
        doc = json.loads(raw)
        geometry = doc["geometry"]
        if doc["schema"] != D7_SCHEMA or doc["status"] != D7_STATUS:
            raise ValueError("D7 placement schema/status disagreement")
        if doc["model"] != {"repository": MODEL, "revision": REVISION}:
            raise ValueError("D7 placement model provenance disagreement")
        if doc["source"]["d3_placement_sha256"] != D3_ARTIFACT_SHA256:
            raise ValueError("D7 source D3 placement disagreement")
        if (geometry["num_moe_layers"], geometry["num_experts_per_layer"],
                geometry["logical_expert_identities"], geometry["gpu0_cache_slots"],
                geometry["gpu0_logical_local_identities"], geometry["worker_a_slots"],
                geometry["worker_b_slots"], geometry["bytes_per_native_nvfp4_identity"],
                geometry["worker_resident_bytes_each"], geometry["combined_worker_resident_bytes"]) != (
                    40, 256, 10240, 3774, 4240, 3000, 3000, 1775616, 5326848000, 10653696000):
            raise ValueError("D7 placement geometry disagreement")
        ownership = doc["derivation"]["per_layer_ownership"]
        if doc["derivation"]["split_layer_count"] != 0 or len(ownership) != 40:
            raise ValueError("D7 whole-layer ownership contract disagreement")
        if any(row != {"layer": layer, "owner": row.get("owner"),
                       "remote_identity_count": row.get("remote_identity_count"), "split": False}
               or row.get("owner") not in ("A", "B") for layer, row in enumerate(ownership)):
            raise ValueError("D7 per-layer ownership contract disagreement")
        worker_a, worker_b = _worker(doc, "worker_a", digest), _worker(doc, "worker_b", digest)
        a, b = set(worker_a.flat_ids_in_rank_order), set(worker_b.flat_ids_in_rank_order)
        union = doc["ranking"]["ranked_union_flat_ids"]
        local = tuple(doc["partition"]["local_remainder"]["flat_ids"])
        if len(union) != 6000 or len(set(union)) != 6000 or a | b != set(union) or a & b:
            raise ValueError("D7 exact D3 union/disjointness disagreement")
        if len(local) != 4240 or a & set(local) or b & set(local) or a | b | set(local) != set(range(10240)):
            raise ValueError("D7 logical ownership partition disagreement")
        for layer, row in enumerate(ownership):
            layer_a = any(flat // 256 == layer for flat in a)
            layer_b = any(flat // 256 == layer for flat in b)
            if layer_a == layer_b or ("A" if layer_a else "B") != row["owner"]:
                raise ValueError("D7 artifact is not whole-layer owned")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid D7 placement: {exc}") from exc
    return D3Placement(digest, worker_a, worker_b, local)


def load_d7_placement(path: str | Path) -> D3Placement:
    return parse_d7_placement_bytes(Path(path).read_bytes())
