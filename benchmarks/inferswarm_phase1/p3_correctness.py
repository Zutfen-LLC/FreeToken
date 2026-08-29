"""Physical InferSwarm P3 C1/C2/C4 engineering fixture.

This command loads the canonical model once, compares complete single-GPU routed-layer
execution with the serialized P3 partition using identical native NVFP4 rows, then runs
an exact multi-layer ownership smoke.  It intentionally records no duration, throughput,
TTFT, or speedup value.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from types import MethodType
from typing import Any

import torch
from freetoken.engine.engine import Engine
from freetoken.gpu_select import set_assigned_gpu
from freetoken.moe.offload_cache import iter_offload_moe_layers
from freetoken.server.args import parse_args
from freetoken.server.launch import _resolve_server_gpu_args


def _route_ids(placement, layer_id: int, mode: str, top_k: int) -> list[int]:
    remote = list(placement.per_layer[layer_id].expert_ids)
    remote_set = set(remote)
    local = [
        expert for expert in range(placement.num_experts) if expert not in remote_set
    ]
    if mode == "mixed":
        remote_count = top_k // 2
        ids = remote[:remote_count] + local[: top_k - remote_count]
    elif mode == "remote_only":
        ids = remote[:top_k]
    elif mode == "local_only":
        ids = local[:top_k]
    else:
        raise ValueError(f"unknown fixture mode {mode!r}")
    if len(ids) != top_k:
        raise RuntimeError(
            f"layer {layer_id} cannot provide {top_k} {mode} fixture routes"
        )
    return ids


def _deviation(candidate: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    candidate_f = candidate.float()
    reference_f = reference.float()
    delta = (candidate_f - reference_f).abs()
    max_abs = float(delta.max().item())
    max_rel = float((delta / reference_f.abs().clamp_min(1e-8)).max().item())
    return max_abs, max_rel


def _one_case(engine: Engine, layers, mode: str, layer_id: int = 0) -> dict[str, Any]:
    cache = engine.moe_offload_cache
    executor = engine.inferswarm_remote_decode
    placement = engine.inferswarm_placement
    assert cache is not None and executor is not None and placement is not None
    layer = layers[layer_id]
    top_k = int(engine.config.model_config.num_experts_per_tok)
    raw = torch.tensor(
        [_route_ids(placement, layer_id, mode, top_k)],
        dtype=torch.int32,
        device=engine.device,
    )
    original_raw = raw.clone()
    weights = torch.arange(1, top_k + 1, dtype=torch.float32, device=engine.device)
    weights = (weights / weights.sum()).unsqueeze(0).contiguous()
    generator = torch.Generator(device=engine.device).manual_seed(1701 + layer_id)
    hidden = torch.randn(
        (1, engine.config.model_config.hidden_size),
        dtype=engine.dtype,
        device=engine.device,
        generator=generator,
    )

    # Complete routed-layer reference through the ordinary GPU0 cache + exact same
    # production kernel dispatch.  This intentionally services all raw identities.
    cache.reset()
    attached = layer.inferswarm_remote_decode
    layer.inferswarm_remote_decode = None
    try:
        reference = layer._decode_routed(hidden, weights, raw.clone()).clone()
    finally:
        layer.inferswarm_remote_decode = attached
    torch.cuda.synchronize(engine.device)

    # Candidate starts cold so its GPU0 service/copy input is mechanically inspectable.
    cache.reset()
    executor.reset()
    ensure_inputs: list[list[int]] = []
    copy_calls = 0
    original_ensure = cache.ensure_experts
    original_copy = cache.copy_missing

    def record_ensure(self, requested_layer, ids, *, record_routing=True):
        ensure_inputs.append(ids.detach().cpu().reshape(-1).tolist())
        return original_ensure(requested_layer, ids, record_routing=record_routing)

    def record_copy(self):
        nonlocal copy_calls
        copy_calls += 1
        return original_copy()

    cache.ensure_experts = MethodType(record_ensure, cache)
    cache.copy_missing = MethodType(record_copy, cache)
    candidate_ids = raw.clone()
    try:
        candidate = executor.decode(
            layer, cache, hidden, weights, candidate_ids
        ).clone()
    finally:
        cache.ensure_experts = original_ensure
        cache.copy_missing = original_copy
    torch.cuda.synchronize(engine.device)

    torch.testing.assert_close(
        candidate.float(), reference.float(), rtol=2e-3, atol=2e-3
    )
    max_abs, max_rel = _deviation(candidate, reference)
    remote_lookup = executor.route_lookup[layer_id][original_raw.long()]
    remote_mask = remote_lookup >= 0
    remote_raw = original_raw[remote_mask].detach().cpu().tolist()
    local_raw = original_raw[~remote_mask].detach().cpu().tolist()
    serviced = [expert for call in ensure_inputs for expert in call]
    num_copied = int(cache.num_indices.item()) if ensure_inputs else 0
    copy_plan = (
        cache.src_indices[:num_copied].detach().cpu().tolist() if ensure_inputs else []
    )
    snap = executor.snapshot()
    aggregate = snap["aggregate"]

    raw_ids_unchanged = candidate_ids.equal(original_raw)
    if remote_raw:
        assert raw_ids_unchanged, "P3 partition changed the raw router IDs"
    assert set(serviced).isdisjoint(remote_raw), (
        "remote identity entered GPU0 residency"
    )
    assert set(copy_plan).isdisjoint(remote_raw), (
        "remote identity entered GPU0 copy plan"
    )
    assert aggregate["executed_on_gpu0"] + aggregate["executed_on_gpu1"] == top_k
    assert aggregate["fallback_elsewhere"] == 0
    expected_dispatches = 1 if remote_raw else 0
    assert aggregate["remote_dispatches"] == expected_dispatches
    if mode == "remote_only":
        assert ensure_inputs == [] and copy_calls == 0
    return {
        "mode": mode,
        "layer_id": layer_id,
        "raw_router_ids": original_raw.detach().cpu().tolist(),
        "raw_router_ids_unchanged": raw_ids_unchanged,
        "raw_router_ids_note": (
            "P3 participating route tensor preserved"
            if remote_raw
            else "ordinary local-only FreeToken path rewrites its caller tensor to cache slots"
        ),
        "remote_raw_ids": remote_raw,
        "local_raw_ids": local_raw,
        "remote_slot_ids": remote_lookup[remote_mask].detach().cpu().tolist(),
        "gpu0_residency_inputs": ensure_inputs,
        "gpu0_copy_plan_expert_ids": copy_plan,
        "gpu0_copy_calls": copy_calls,
        "max_absolute_deviation": max_abs,
        "max_relative_deviation": max_rel,
        "assert_close": {"rtol": 2e-3, "atol": 2e-3, "passed": True},
        "nan_count": int(torch.isnan(candidate.float()).sum().item()),
        "inf_count": int(torch.isinf(candidate.float()).sum().item()),
        "ownership": aggregate,
    }


def _multi_layer_smoke(engine: Engine, layers) -> dict[str, Any]:
    cache = engine.moe_offload_cache
    executor = engine.inferswarm_remote_decode
    placement = engine.inferswarm_placement
    assert cache is not None and executor is not None and placement is not None
    top_k = int(engine.config.model_config.num_experts_per_tok)
    hidden_size = int(engine.config.model_config.hidden_size)
    cache.reset()
    executor.reset()
    nan_count = 0
    inf_count = 0
    for layer_id, layer in enumerate(layers):
        raw = torch.tensor(
            [_route_ids(placement, layer_id, "mixed", top_k)],
            dtype=torch.int32,
            device=engine.device,
        )
        weights = torch.full(
            (1, top_k), 1.0 / top_k, dtype=torch.float32, device=engine.device
        )
        generator = torch.Generator(device=engine.device).manual_seed(9000 + layer_id)
        hidden = torch.randn(
            (1, hidden_size),
            dtype=engine.dtype,
            device=engine.device,
            generator=generator,
        )
        output = executor.decode(layer, cache, hidden, weights, raw)
        nan_count += int(torch.isnan(output.float()).sum().item())
        inf_count += int(torch.isinf(output.float()).sum().item())
    torch.cuda.synchronize(engine.device)
    aggregate = executor.snapshot()["aggregate"]
    expected = top_k * len(layers)
    assert aggregate["total_router_selections"] == expected
    assert aggregate["executed_on_gpu0"] + aggregate["executed_on_gpu1"] == expected
    assert aggregate["remote_dispatches"] == len(layers)
    assert aggregate["fallback_elsewhere"] == 0
    assert nan_count == 0 and inf_count == 0
    return {
        "top_k": top_k,
        "num_layers": len(layers),
        "decode_steps": 1,
        "expected_router_selections": expected,
        "ownership": aggregate,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def run_fixture(
    model: str, primary: str, secondary: str, placement: str
) -> dict[str, Any]:
    server_args, _ = parse_args(
        [
            "--model-path",
            model,
            "--gpu",
            primary,
            "--inferswarm-secondary-gpu",
            secondary,
            "--inferswarm-placement",
            placement,
            "--inferswarm-remote-decode",
            "--inferswarm-remote-mode",
            "serialized",
            "--moe-backend",
            "offload",
            "--nvfp4-backend",
            "triton",
            "--moe-cache-size",
            "3774",
            "--kv-reserve-tokens",
            "17075",
            "--memory-ratio",
            "0.85",
            "--cuda-graph-max-bs",
            "0",
            "--max-running-requests",
            "1",
            "--sampling-defaults",
            "none",
        ]
    )
    server_args = _resolve_server_gpu_args(server_args)
    assigned = server_args.gpu_assigned or server_args.gpu
    set_assigned_gpu(assigned[0])
    engine = Engine(server_args)
    try:
        layers = list(iter_offload_moe_layers(engine.model))
        assert len(layers) == 40
        cases = [
            _one_case(engine, layers, "mixed"),
            _one_case(engine, layers, "local_only"),
            _one_case(engine, layers, "remote_only"),
        ]
        multi_layer = _multi_layer_smoke(engine, layers)
        resident = engine.inferswarm_resident_bank
        executor = engine.inferswarm_remote_decode
        assert resident is not None and executor is not None
        secondary_ordinal = resident.report.secondary_visible_ordinal
        resident_report = resident.report.as_dict()
        return {
            "schema": "inferswarm.phase1.p3-correctness-fixture/1",
            "evidence_label": "MEASURED ENGINEERING FIXTURE (not canonical C3)",
            "performance_measurement_taken": False,
            "model": {
                "repository": resident.placement.model_repository,
                "revision": resident.placement.model_revision,
            },
            "placement_sha256": resident.placement.artifact_sha256,
            "runtime": {
                "primary_uuid": engine.inferswarm_secondary_device.primary.uuid,
                "secondary_uuid": engine.inferswarm_secondary_device.secondary.uuid,
                "gpu0_cache_slots": engine.moe_offload_cache.cache_size,
                "quant_format": resident.report.layout.quant_format,
                "nvfp4_backend": resident.report.layout.nvfp4_backend,
                "bank_layout": resident.report.layout.bank_layout,
                "execution_mode": executor.mode,
                "transport": "host_staged",
                "cuda_graph_max_bs": engine.config.cuda_graph_max_bs,
            },
            "resident_bank": {
                "resident_slots": resident_report["resident_slots"],
                "expert_bank_tensor_bytes": resident_report["accounting"][
                    "expert_bank_tensor_bytes"
                ],
                "total_live_resident_bank_bytes": resident_report["accounting"][
                    "total_live_resident_bank_bytes"
                ],
                "source_byte_verification": {
                    "status": resident_report["source_byte_verification"]["status"],
                    "verified_rows": resident_report["source_byte_verification"][
                        "verified_rows"
                    ],
                    "verified_bytes": resident_report["source_byte_verification"][
                        "verified_bytes"
                    ],
                },
                "cuda_memory": resident_report["cuda_memory"],
                "startup_expert_weight_bytes_host_to_gpu1": resident_report[
                    "startup_expert_weight_bytes_host_to_gpu1"
                ],
                "steady_state_expert_weight_bytes_host_to_gpu1": resident_report[
                    "steady_state_expert_weight_bytes_host_to_gpu1"
                ],
            },
            "cases": cases,
            "multi_layer_smoke": multi_layer,
            "memory_after_fixture": {
                "gpu1_allocated_bytes": int(
                    torch.cuda.memory_allocated(secondary_ordinal)
                ),
                "gpu1_reserved_bytes": int(
                    torch.cuda.memory_reserved(secondary_ordinal)
                ),
                "transport_buffers": executor.transport.report(),
                "nvidia_smi_compute_apps": subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=gpu_uuid,pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                .stdout.strip()
                .splitlines(),
            },
            "primary_current_after_fixture": (
                torch.cuda.current_device() == engine.device.index
            ),
        }
    finally:
        engine.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--primary-gpu", required=True)
    parser.add_argument("--secondary-gpu", required=True)
    parser.add_argument("--placement", required=True)
    args = parser.parse_args(argv)
    document = run_fixture(
        args.model, args.primary_gpu, args.secondary_gpu, args.placement
    )
    json.dump(document, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
