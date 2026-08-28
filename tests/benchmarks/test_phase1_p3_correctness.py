from types import SimpleNamespace

import pytest
import torch
from inferswarm_phase1.p3_correctness import _deviation, _route_ids


def _placement():
    return SimpleNamespace(
        num_experts=8,
        per_layer=(SimpleNamespace(expert_ids=(1, 3, 5, 7)),),
    )


def test_p3_fixture_builds_deterministic_local_remote_and_mixed_routes():
    placement = _placement()
    assert _route_ids(placement, 0, "remote_only", 4) == [1, 3, 5, 7]
    assert _route_ids(placement, 0, "local_only", 4) == [0, 2, 4, 6]
    assert _route_ids(placement, 0, "mixed", 4) == [1, 3, 0, 2]


def test_p3_fixture_deviation_reports_absolute_and_relative_values():
    absolute, relative = _deviation(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.5]))
    assert absolute == 0.5
    assert relative == pytest.approx(0.2)
