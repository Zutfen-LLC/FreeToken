from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from freetoken.moe.inferswarm_d3_placement import D3_ARTIFACT_SHA256, load_d3_placement, parse_d3_placement_bytes

ARTIFACT = Path('/home/zutfen/inferswarm/docs/investigations/data/phase1r-d3-three-device-placement.json')


def test_frozen_d3_artifact_has_enforced_sha_and_disjoint_workers():
    raw = ARTIFACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == D3_ARTIFACT_SHA256
    placement = parse_d3_placement_bytes(raw)
    a, b = set(placement.worker_a.flat_ids_in_rank_order), set(placement.worker_b.flat_ids_in_rank_order)
    assert len(a) == len(b) == 3000
    assert not a & b


def test_d3_parser_refuses_any_sha_change():
    raw = ARTIFACT.read_bytes().replace(b'FROZEN_BEFORE_D3_PERFORMANCE', b'FROZEN_AFTER__D3_PERFORMANCE')
    with pytest.raises(ValueError, match='SHA-256'):
        parse_d3_placement_bytes(raw)
