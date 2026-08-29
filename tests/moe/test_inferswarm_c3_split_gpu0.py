from __future__ import annotations

from types import SimpleNamespace

import torch
from freetoken.engine.correctness_diagnostics import CorrectnessDiagnostics
from freetoken.moe.inferswarm_remote_decode import SplitGpu0DiagnosticExecutor


class _Cache:
    def __init__(self):
        self.ensure_inputs = []
        self.copy_calls = 0
        self.routing_inputs = []
        self.bank_schema = ("weight",)
        self.bank_sources = {
            "weight": [torch.arange(8, dtype=torch.uint8).reshape(4, 2)]
        }

    def record_decode_routing(self, layer_id, ids):
        self.routing_inputs.append((layer_id, ids.clone()))

    def ensure_experts(self, layer_id, ids, *, record_routing=True):
        self.ensure_inputs.append((layer_id, ids.clone(), record_routing))
        ids.add_(10)

    def copy_missing(self):
        self.copy_calls += 1

    def bank_views(self):
        return ()

    def alphas_for_slots(self, layer_id):
        del layer_id


class _Layer:
    layer_id = 0

    def __init__(self):
        self.calls = []

    def _expert_gemm(self, cache, hidden, weights, ids, **kwargs):
        del cache, kwargs
        self.calls.append((weights.clone(), ids.clone()))
        return hidden * weights.sum(dim=1, keepdim=True).to(hidden.dtype)


def test_split_gpu0_uses_complete_route_set_two_production_calls_and_no_gpu1():
    diagnostics = CorrectnessDiagnostics(
        root_cause_mode="DIAGNOSTIC_SPLIT_GPU0",
        root_cause_expected_layers=1,
    )
    diagnostics.begin_decode_step(0)
    placement = SimpleNamespace(
        num_layers=1,
        num_experts=4,
        artifact_sha256="a" * 64,
    )
    route_lookup = torch.tensor([[-1, 7, -1, 8]], dtype=torch.int32)
    executor = SplitGpu0DiagnosticExecutor(
        placement=placement,
        primary_device=torch.device("cpu"),
        diagnostics=diagnostics,
        route_lookup=route_lookup,
    )
    cache, layer = _Cache(), _Layer()
    hidden = torch.tensor([[2.0, 4.0]])
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    raw_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    diagnostics.capture_moe_input(0, hidden, raw_ids, weights)
    output = executor.decode(layer, cache, hidden, weights, raw_ids)
    diagnostics.capture_moe_output(0, output)

    torch.testing.assert_close(output, hidden)
    assert len(layer.calls) == 2
    assert cache.copy_calls == 1
    assert cache.ensure_inputs[0][1].tolist() == [[0, 1, 2, 3]]
    assert cache.ensure_inputs[0][2] is False
    torch.testing.assert_close(layer.calls[0][0], torch.tensor([[0.1, 0.0, 0.3, 0.0]]))
    torch.testing.assert_close(layer.calls[1][0], torch.tensor([[0.0, 0.2, 0.0, 0.4]]))
    assert layer.calls[0][1].tolist() == [[10, 11, 12, 13]]
    assert layer.calls[1][1].tolist() == [[10, 11, 12, 13]]

    root = diagnostics.snapshot()["moe_root_cause"]
    weight_proof = root["layers"][0]["selected_expert_weights"]
    assert weight_proof["selected_raw_expert_ids"] == [0, 1, 2, 3]
    assert weight_proof["bank_order"] == ["weight"]
    assert len(weight_proof["banks"][0]["rows"]) == 4
    ownership = root["layers"][0]["ownership"]
    assert ownership == {
        "local_selection_count": 2,
        "remote_selection_count": 2,
        "total_routed_selections": 4,
        "masks_complete": True,
        "masks_disjoint": True,
        "every_route_exactly_once": True,
    }
    report = executor.snapshot()
    assert report["diagnostic_label"] == "DIAGNOSTIC_SPLIT_GPU0"
    assert report["uses_gpu1"] is False
    assert report["gpu1_dispatches"] == 0
    assert report["production_gemm_calls"] == 2
    assert report["combine_operations"] == 1
    assert report["f_gate_evidence_eligible"] is False
    assert report["f_gate_counters_incremented"] is False
