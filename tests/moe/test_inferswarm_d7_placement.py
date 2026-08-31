from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from freetoken.moe.inferswarm_d7_placement import D7_ARTIFACT_SHA256, parse_d7_placement_bytes

ARTIFACT = Path("/home/zutfen/inferswarm/docs/investigations/data/phase1r-d7-fanin-sparse-placement.json")


def test_frozen_d7_parser_is_sha_pinned_exact_and_whole_layer_owned():
    raw = ARTIFACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == D7_ARTIFACT_SHA256
    placement = parse_d7_placement_bytes(raw)
    a = set(placement.worker_a.flat_ids_in_rank_order)
    b = set(placement.worker_b.flat_ids_in_rank_order)
    local = set(placement.local_remainder)
    assert (len(a), len(b), len(local)) == (3000, 3000, 4240)
    assert not (a & b or a & local or b & local)
    assert a | b | local == set(range(10240))
    for layer in range(40):
        assert bool({flat for flat in a if flat // 256 == layer}) != bool({flat for flat in b if flat // 256 == layer})


def test_d7_parser_refuses_any_byte_change():
    with pytest.raises(ValueError, match="SHA-256"):
        parse_d7_placement_bytes(ARTIFACT.read_bytes() + b" ")
