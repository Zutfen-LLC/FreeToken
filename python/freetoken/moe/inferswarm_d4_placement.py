"""Strict parser for the frozen D4 capability-weighted placement."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from types import MappingProxyType
from .inferswarm_d3_placement import D3Placement,MODEL,REVISION
from .inferswarm_resident_bank import FrozenPlacement,LayerPlacement,PlacementIdentity
D4_SCHEMA="inferswarm.phase1r.d4-placement/1";D4_STATUS="FROZEN_BEFORE_D4_PERFORMANCE"
D4_ARTIFACT_SHA256="283595b7559bb3aa46a08c7d00cfef1e0a77eb62967d6392c618a63f35d34cdf"
def _worker(doc,key,digest):
 raw=doc["partition"][key];ids=raw["flat_ids_in_rank_order"]
 if len(ids)!=3000 or len(set(ids))!=3000 or len(raw["identities"])!=3000:raise ValueError(f"D4 {key} must contain 3,000 unique identities")
 identities=[];mapping={};layers=[[] for _ in range(40)]
 for slot,(flat,record) in enumerate(zip(ids,raw["identities"],strict=True)):
  layer,expert=record["layer"],record["expert_id"]
  if record["flat_id"]!=flat or flat!=layer*256+expert or (layer,expert) in mapping:raise ValueError(f"D4 {key} identity arithmetic failed at slot {slot}")
  identity=PlacementIdentity(flat,layer,expert,slot);identities.append(identity);mapping[layer,expert]=slot;layers[layer].append(identity)
 per_layer=tuple(LayerPlacement(layer,tuple(i.expert_id for i in values),tuple(i.remote_slot for i in values)) for layer,values in enumerate(layers))
 if raw["per_layer"]!=[{"layer":x.layer_id,"expert_ids":list(x.expert_ids)} for x in per_layer]:raise ValueError(f"D4 {key} per-layer identities disagree")
 return FrozenPlacement(digest,D4_SCHEMA,"phase1r-d4-capability-weighted",D4_STATUS,key,MODEL,REVISION,40,256,10240,1775616,3000,5326848000,5326848000,tuple(ids),tuple(identities),per_layer,MappingProxyType(mapping))
def parse_d4_placement_bytes(raw:bytes)->D3Placement:
 digest=hashlib.sha256(raw).hexdigest()
 if digest!=D4_ARTIFACT_SHA256:raise ValueError(f"D4 placement SHA-256 disagreement: expected {D4_ARTIFACT_SHA256}, got {digest}")
 doc=json.loads(raw);geometry=doc["geometry"]
 if doc["schema"]!=D4_SCHEMA or doc["status"]!=D4_STATUS or doc["model"]!={"repository":MODEL,"revision":REVISION}:raise ValueError("D4 placement schema/status/model disagreement")
 if (geometry["gpu0_cache_slots"],geometry["worker_a_slots"],geometry["worker_b_slots"],geometry["combined_worker_resident_bytes"])!=(3774,3000,3000,10653696000):raise ValueError("D4 placement geometry disagreement")
 a,b=_worker(doc,"worker_a",digest),_worker(doc,"worker_b",digest);local=tuple(doc["partition"]["local_remainder"]["flat_ids"])
 if len(local)!=4240 or set(a.flat_ids_in_rank_order)&set(b.flat_ids_in_rank_order) or set(a.flat_ids_in_rank_order)|set(b.flat_ids_in_rank_order)|set(local)!=set(range(10240)):raise ValueError("D4 logical ownership partition disagreement")
 return D3Placement(digest,a,b,local)
def load_d4_placement(path:str|Path)->D3Placement:return parse_d4_placement_bytes(Path(path).read_bytes())
