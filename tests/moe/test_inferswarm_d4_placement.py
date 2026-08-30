from __future__ import annotations
import hashlib
from pathlib import Path
import pytest
from freetoken.moe.inferswarm_d4_placement import D4_ARTIFACT_SHA256,parse_d4_placement_bytes
ARTIFACT=Path('/home/zutfen/inferswarm/docs/investigations/data/phase1r-d4-capability-weighted-placement.json')
def test_frozen_d4_artifact_is_exact_and_exhaustive():
 raw=ARTIFACT.read_bytes();assert hashlib.sha256(raw).hexdigest()==D4_ARTIFACT_SHA256
 p=parse_d4_placement_bytes(raw);a=set(p.worker_a.flat_ids_in_rank_order);b=set(p.worker_b.flat_ids_in_rank_order);local=set(p.local_remainder)
 assert (len(a),len(b),len(local))==(3000,3000,4240);assert not (a&b or a&local or b&local);assert a|b|local==set(range(10240))
def test_d4_parser_refuses_any_byte_change():
 with pytest.raises(ValueError,match='SHA-256'):parse_d4_placement_bytes(ARTIFACT.read_bytes()+b' ')
