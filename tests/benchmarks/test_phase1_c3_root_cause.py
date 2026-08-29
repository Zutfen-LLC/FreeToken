from __future__ import annotations

import base64
import hashlib

import pytest
import torch
from inferswarm_phase1.c3_layer_replay import (
    fixed_input_hashes,
    localize_transport,
    replay_comparisons,
)
from inferswarm_phase1.c3_root_cause import (
    assert_performance_firewall,
    classify_stage1,
    externalize_trace_tensors,
    tensor_from_evidence,
    tensor_metrics,
)


def _row():
    return {
        "hidden_input": {
            "exact_raw_byte_equality": True,
            "within_c1_tolerance": True,
        },
        "routing": {"topk_ids_exact": True, "routing_weights_exact": True},
        "moe_output": {
            "exact_raw_byte_equality": True,
            "within_c1_tolerance": True,
        },
    }


def _pair():
    rows = [{"layer_id": layer_id, **_row()} for layer_id in range(40)]
    return {
        "first_divergence": {
            "hidden_input_bits": None,
            "hidden_input_exceeds_c1": None,
            "router_ids": None,
            "routing_weights": None,
            "moe_output_bits": None,
            "moe_output_exceeds_c1": None,
        },
        "layers": rows,
    }


def _pairs():
    return {
        name: _pair()
        for name in ("R_vs_O", "R_vs_S", "R_vs_G", "O_vs_S", "O_vs_G", "S_vs_G")
    }


def _differ(report, boundary, layer=0):
    report["first_divergence"][boundary] = layer
    if boundary == "hidden_input_bits":
        report["layers"][layer]["hidden_input"]["exact_raw_byte_equality"] = False
    elif boundary == "router_ids":
        report["layers"][layer]["routing"]["topk_ids_exact"] = False
    elif boundary == "routing_weights":
        report["layers"][layer]["routing"]["routing_weights_exact"] = False
    elif boundary == "moe_output_bits":
        report["layers"][layer]["moe_output"]["exact_raw_byte_equality"] = False


def test_stage1_classifies_overlap_specific_divergence():
    pairs = _pairs()
    for name in ("R_vs_O", "R_vs_S", "O_vs_S"):
        _differ(pairs[name], "moe_output_bits")
    assert (
        classify_stage1(pairs, first_mixed_layer=0)["classification"]
        == "OVERLAP_SPECIFIC"
    )


def test_stage1_classifies_remote_execution_or_transport():
    pairs = _pairs()
    for name in ("R_vs_O", "R_vs_S", "O_vs_G", "S_vs_G"):
        _differ(pairs[name], "moe_output_bits")
    assert (
        classify_stage1(pairs, first_mixed_layer=0)["classification"]
        == "REMOTE_EXECUTION_OR_TRANSPORT"
    )


def test_stage1_classifies_split_reduction_topology():
    pairs = _pairs()
    for name in ("R_vs_O", "R_vs_S", "R_vs_G"):
        _differ(pairs[name], "moe_output_bits")
    assert (
        classify_stage1(pairs, first_mixed_layer=0)["classification"]
        == "SPLIT_REDUCTION_TOPOLOGY"
    )


@pytest.mark.parametrize(
    "boundary", ["hidden_input_bits", "router_ids", "routing_weights"]
)
def test_stage1_classifies_upstream_hidden_or_router_divergence(boundary):
    pairs = _pairs()
    _differ(pairs["R_vs_O"], boundary)
    assert (
        classify_stage1(pairs, first_mixed_layer=0)["classification"]
        == "UPSTREAM_STATE_OR_ROUTING"
    )


def test_stage1_keeps_nonmatching_relationship_unresolved():
    pairs = _pairs()
    for name in ("R_vs_O", "R_vs_S", "R_vs_G", "O_vs_G"):
        _differ(pairs[name], "moe_output_bits")
    assert classify_stage1(pairs, first_mixed_layer=0)["classification"] == "UNRESOLVED"


def test_tensor_and_replay_metrics_use_only_frozen_c1_tolerance():
    u = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    gl = torch.tensor([[0.5, 1.0]], dtype=torch.bfloat16)
    gr = torch.tensor([[0.5, 1.0]], dtype=torch.bfloat16)
    gs = gl + gr
    rr = gr.clone()
    rr[0, 0] += torch.tensor(0.25, dtype=torch.bfloat16)
    variants = {
        "U": u,
        "GL": gl,
        "GR": gr,
        "GS": gs,
        "RL": gl.clone(),
        "RR": rr,
        "RC": gl + rr,
    }
    report = replay_comparisons(variants)
    assert report["U_vs_GS"]["exact_raw_byte_equality"] is True
    assert report["GL_vs_RL"]["exact_raw_byte_equality"] is True
    assert report["GR_vs_RR"]["exact_raw_byte_equality"] is False
    assert report["GR_vs_RR"]["within_c1_tolerance"] is False
    assert report["GR_vs_RR"]["rtol"] == 2e-3
    assert report["GR_vs_RR"]["atol"] == 2e-3
    assert tensor_metrics(u, u)["max_absolute_deviation"] == 0.0


def test_target_replay_base_input_identity_is_lossless_and_mutation_sensitive():
    hidden = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    raw_ids = torch.tensor([[3, 7]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    before = fixed_input_hashes(hidden, raw_ids, weights)

    # Every named arm derives from these exact base values; replay_comparisons also refuses
    # a missing U/GL/GR/GS/RL/RR/RC capture.
    variants = {
        name: hidden.clone() for name in ("U", "GL", "GR", "GS", "RL", "RR", "RC")
    }
    assert set(replay_comparisons(variants)) == {
        "U_vs_GS",
        "GL_vs_RL",
        "GR_vs_RR",
        "GS_vs_RC",
        "U_vs_RC",
    }
    assert fixed_input_hashes(hidden, raw_ids, weights) == before
    raw_ids[0, 0] += 1
    assert fixed_input_hashes(hidden, raw_ids, weights) != before


def test_tensor_metrics_distinguishes_numerical_equality_from_exact_bytes():
    positive_zero = torch.tensor([0.0], dtype=torch.float32)
    negative_zero = torch.tensor([-0.0], dtype=torch.float32)
    report = tensor_metrics(positive_zero, negative_zero)
    assert report["exact_raw_byte_equality"] is False
    assert report["max_absolute_deviation"] == 0.0
    assert report["within_c1_tolerance"] is True


def test_lossless_tensor_sidecar_round_trip(tmp_path):
    original = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    raw = original.view(torch.uint8).reshape(-1).numpy().tobytes()
    trace = {
        "layers": [
            {
                "layer_id": 0,
                "tensors": {
                    "hidden_input": {
                        "dtype": "bfloat16",
                        "shape": [1, 2],
                        "raw_byte_count": len(raw),
                        "raw_byte_sha256": hashlib.sha256(raw).hexdigest(),
                        "raw_bytes_base64": base64.b64encode(raw).decode("ascii"),
                    }
                },
            }
        ]
    }
    document_path = tmp_path / "trace.json"
    clean, manifests = externalize_trace_tensors(
        trace, output_path=document_path, class_id="W3"
    )
    evidence = clean["layers"][0]["tensors"]["hidden_input"]
    assert "raw_bytes_base64" not in evidence
    assert manifests[0]["sha256"] == hashlib.sha256(raw).hexdigest()
    reconstructed = tensor_from_evidence(evidence, document_path=document_path)
    assert tensor_metrics(original, reconstructed)["exact_raw_byte_equality"] is True


def test_performance_firewall_rejects_forbidden_fields_recursively():
    assert_performance_firewall(
        {"performance_fields_collected": False, "nested": [{"correctness": True}]}
    )
    with pytest.raises(RuntimeError, match="forbidden performance field"):
        assert_performance_firewall({"nested": {"request_latency_ms": 1.0}})


def _transport_tensors():
    zero_f = torch.zeros(1, 2)
    zero_i = torch.zeros(1, 2, dtype=torch.int32)
    return {
        "transport_gpu0_source_hidden_activation": zero_f.clone(),
        "transport_pinned_host_staged_activation": zero_f.clone(),
        "transport_gpu1_activation_after_h2d": zero_f.clone(),
        "transport_gpu0_source_routing_weights": zero_f.clone(),
        "transport_pinned_host_routing_weights": zero_f.clone(),
        "transport_gpu1_routing_weights": zero_f.clone(),
        "transport_expected_remote_slot_ids": zero_i.clone(),
        "transport_pinned_host_remote_slot_ids": zero_i.clone(),
        "transport_gpu1_remote_slot_ids": zero_i.clone(),
        "transport_gpu1_remote_partial_before_d2h": zero_f.clone(),
        "transport_pinned_host_returned_partial": zero_f.clone(),
        "transport_gpu0_returned_remote_partial": zero_f.clone(),
    }


@pytest.mark.parametrize(
    ("mutated", "expected"),
    [
        ("transport_pinned_host_staged_activation", "gpu0_to_host_activation"),
        ("transport_gpu1_activation_after_h2d", "host_to_gpu1_activation"),
        ("transport_gpu1_remote_partial_before_d2h", "gpu1_remote_expert_execution"),
        ("transport_pinned_host_returned_partial", "gpu1_to_host_return"),
        ("transport_gpu0_returned_remote_partial", "host_to_gpu0_return"),
    ],
)
def test_transport_localization_attributes_injected_first_mutation(mutated, expected):
    tensors = _transport_tensors()
    tensors[mutated][0, 0] = 1
    # Preserve later stages after the injected mutation so the first changed boundary is clear.
    if mutated == "transport_gpu1_remote_partial_before_d2h":
        tensors["transport_pinned_host_returned_partial"][0, 0] = 1
        tensors["transport_gpu0_returned_remote_partial"][0, 0] = 1
    elif mutated == "transport_pinned_host_returned_partial":
        tensors["transport_gpu0_returned_remote_partial"][0, 0] = 1
    result = localize_transport(tensors, gpu0_remote_subset=torch.zeros(1, 2))
    assert result["first_corrupt_boundary"] == expected


def test_transport_localization_has_no_silent_fallback_for_missing_stage():
    tensors = _transport_tensors()
    del tensors["transport_gpu1_remote_slot_ids"]
    with pytest.raises(KeyError):
        localize_transport(tensors, gpu0_remote_subset=torch.zeros(1, 2))
