from __future__ import annotations

import hashlib
from unittest.mock import patch
from pathlib import Path

import pytest

import freetoken.moe.inferswarm_d3_placement as d3_mod
from freetoken.moe.inferswarm_d3_placement import D3_ARTIFACT_SHA256, parse_d3_placement_bytes

ARTIFACT = Path('/home/zutfen/inferswarm/docs/investigations/data/phase1r-d3-three-device-placement.json')


def test_frozen_d3_artifact_has_enforced_sha_and_disjoint_workers():
    raw = ARTIFACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == D3_ARTIFACT_SHA256
    placement = parse_d3_placement_bytes(raw)
    a, b = set(placement.worker_a.flat_ids_in_rank_order), set(placement.worker_b.flat_ids_in_rank_order)
    assert len(a) == len(b) == 3000
    assert not a & b
    assert len(placement.local_remainder) == 4240
    assert a | b | set(placement.local_remainder) == set(range(10240))
    assert not (a & set(placement.local_remainder) or b & set(placement.local_remainder))
    assert placement.worker_a.remote_slot(*divmod(placement.worker_a.flat_ids_in_rank_order[0], 256)) == 0
    assert placement.worker_b.remote_slot(*divmod(placement.worker_b.flat_ids_in_rank_order[0], 256)) == 0


def test_d3_parser_refuses_any_sha_change():
    raw = ARTIFACT.read_bytes().replace(b'FROZEN_BEFORE_D3_PERFORMANCE', b'FROZEN_AFTER__D3_PERFORMANCE')
    with pytest.raises(ValueError, match='SHA-256'):
        parse_d3_placement_bytes(raw)


def test_d3_parser_refuses_the_superseded_placement_sha():
    class OldDigest:
        def hexdigest(self):
            return "6c9cb2c5dbd46cdd93049769427b3c5bec47cdbb0564050c8fedb7a90eac15d7"

    with patch.object(d3_mod.hashlib, "sha256", return_value=OldDigest()), pytest.raises(ValueError, match="SHA-256"):
        parse_d3_placement_bytes(ARTIFACT.read_bytes())
