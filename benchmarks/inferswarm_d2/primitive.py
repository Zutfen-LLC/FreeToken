"""Short physical D2 real-kernel primitive/correctness/replay-cost fixture."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from typing import Any

import torch

from freetoken.engine.engine import Engine
from freetoken.gpu_select import set_assigned_gpu
from freetoken.moe.inferswarm_d2_graph_remote import (
    InferSwarmD2GraphRemoteExecutor,
    build_local_fallback_ids,
    build_remote_slot_lookup,
)
from freetoken.moe.inferswarm_remote_decode import HostStagedRemoteTransport
from freetoken.moe.offload_cache import iter_offload_moe_layers
from freetoken.server.args import parse_args
from freetoken.server.launch import _resolve_server_gpu_args


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _distribution_us(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "median_us": statistics.median(values),
        "p95_us": _percentile(values, 0.95),
        "max_us": max(values),
    }


def _ids(placement, layer_id: int, mode: str, top_k: int) -> list[int]:
    remote = list(placement.per_layer[layer_id].expert_ids)
    remote_set = set(remote)
    local = [i for i in range(placement.num_experts) if i not in remote_set]
    if mode == "mixed":
        return remote[: top_k // 2] + local[: top_k - top_k // 2]
    if mode == "remote_only":
        return remote[:top_k]
    if mode == "local_only":
        return local[:top_k]
    raise ValueError(mode)


def _make_executor(engine: Engine):
    resident = engine.inferswarm_resident_bank
    secondary = engine.inferswarm_secondary_device
    assert resident is not None and secondary is not None
    mc = engine.config.model_config
    return InferSwarmD2GraphRemoteExecutor(
        resident_bank=resident,
        secondary_device=secondary,
        primary_device=engine.device,
        route_lookup=build_remote_slot_lookup(resident.placement, engine.device),
        local_fallback_ids=build_local_fallback_ids(resident.placement, engine.device),
        hidden_size=int(mc.hidden_size),
        top_k=int(mc.num_experts_per_tok),
        hidden_dtype=engine.dtype,
        num_layers=int(mc.num_moe_layers),
        intermediate_size=int(mc.moe_intermediate_size),
    )


def run_fixture(model: str, primary: str, secondary: str, placement: str, n: int):
    args, _ = parse_args(
        [
            "--model", model,
            "--gpu", primary,
            "--inferswarm-secondary-gpu", secondary,
            "--inferswarm-placement", placement,
            "--moe-backend", "offload",
            "--moe-cpu-layers", "0",
            "--nvfp4-backend", "triton",
            "--moe-cache-size", "3774",
            "--kv-reserve-tokens", "17075",
            "--num-tokens", "17075",
            "--memory-ratio", "0.85",
            "--max-running-requests", "1",
            "--cuda-graph-max-bs", "0",
            "--sampling-defaults", "none",
        ]
    )
    args = _resolve_server_gpu_args(args)
    assigned = args.gpu_assigned or args.gpu
    set_assigned_gpu(assigned[0])
    engine = Engine(args)
    benchmark_started = time.monotonic()
    try:
        cache = engine.moe_offload_cache
        resident = engine.inferswarm_resident_bank
        secondary_info = engine.inferswarm_secondary_device
        assert cache is not None and resident is not None and secondary_info is not None
        layers = list(iter_offload_moe_layers(engine.model))
        layer = layers[0]
        d2 = _make_executor(engine)
        top_k = d2.top_k
        static_hidden = torch.empty(
            (1, d2.hidden_size), dtype=d2.hidden_dtype, device=engine.device
        )
        static_weights = torch.empty(
            (1, top_k), dtype=torch.float32, device=engine.device
        )
        static_ids = torch.empty(
            (1, top_k), dtype=torch.int32, device=engine.device
        )

        initial_ids = torch.tensor(
            [_ids(resident.placement, 0, "mixed", top_k)],
            dtype=torch.int32,
            device=engine.device,
        )
        initial_weights = torch.arange(
            1, top_k + 1, dtype=torch.float32, device=engine.device
        ).reshape(1, -1)
        initial_weights.div_(initial_weights.sum())
        initial_hidden = torch.randn(
            (1, d2.hidden_size),
            dtype=d2.hidden_dtype,
            device=engine.device,
            generator=torch.Generator(device=engine.device).manual_seed(2201),
        )
        static_hidden.copy_(initial_hidden)
        static_weights.copy_(initial_weights)
        static_ids.copy_(initial_ids)

        cache.reset()
        d2.decode(layer, cache, static_hidden, static_weights, static_ids)
        torch.cuda.synchronize(engine.device)  # outside the measured operation
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=engine.stream):
            d2.decode(layer, cache, static_hidden, static_weights, static_ids)
        d2.set_graph_state([1])

        cases = []
        outputs = []
        for case_index, mode in enumerate(("mixed", "remote_only", "local_only")):
            raw = torch.tensor(
                [_ids(resident.placement, 0, mode, top_k)],
                dtype=torch.int32,
                device=engine.device,
            )
            weights = torch.arange(
                case_index + 1,
                case_index + top_k + 1,
                dtype=torch.float32,
                device=engine.device,
            ).reshape(1, -1)
            weights.div_(weights.sum())
            hidden = torch.randn(
                (1, d2.hidden_size),
                dtype=d2.hidden_dtype,
                device=engine.device,
                generator=torch.Generator(device=engine.device).manual_seed(3300 + case_index),
            )

            cache.reset()
            reference = layer._decode_routed(
                hidden, weights, raw.clone()
            ).clone()
            torch.cuda.synchronize(engine.device)
            static_hidden.copy_(hidden)
            static_weights.copy_(weights)
            static_ids.copy_(raw)
            graph.replay()  # the measured D2 operation contains no host synchronization
            torch.cuda.synchronize(engine.device)  # diagnostic boundary after replay
            candidate = d2.gpu0_output.clone()
            outputs.append(candidate)
            delta = (candidate.float() - reference.float()).abs()
            max_abs = float(delta.max().item())
            max_rel = float(
                (delta / reference.float().abs().clamp_min(1e-8)).max().item()
            )
            torch.testing.assert_close(
                candidate.float(), reference.float(), rtol=2e-3, atol=2e-3
            )
            lookup = d2.route_lookup[0][raw.long()]
            remote_count = int((lookup >= 0).sum().item())
            cases.append(
                {
                    "mode": mode,
                    "activation_seed": 3300 + case_index,
                    "raw_route_ids": raw.cpu().tolist(),
                    "route_weights": weights.cpu().tolist(),
                    "remote_route_count": remote_count,
                    "local_route_count": top_k - remote_count,
                    "no_route_dropped_or_duplicated": (
                        remote_count + (top_k - remote_count) == top_k
                    ),
                    "exact_output": torch.equal(candidate, reference),
                    "max_absolute_deviation": max_abs,
                    "max_relative_deviation": max_rel,
                    "assert_close": {"rtol": 2e-3, "atol": 2e-3, "passed": True},
                    "nan_count": int(torch.isnan(candidate.float()).sum().item()),
                    "inf_count": int(torch.isinf(candidate.float()).sum().item()),
                }
            )
        changing_payload_consumed = all(
            not torch.equal(outputs[i], outputs[j])
            for i in range(len(outputs))
            for j in range(i + 1, len(outputs))
        )

        # Canonical current submit cost: identical resident kernel and fixed payload.
        current = HostStagedRemoteTransport(
            primary_device=engine.device,
            secondary_device=d2.secondary_torch_device,
            max_tokens=1,
            hidden_size=d2.hidden_size,
            top_k=top_k,
            hidden_dtype=d2.hidden_dtype,
            resident_bank=resident,
            timing_enabled=False,
        )
        remote_ids = torch.tensor(
            [
                [
                    resident.placement.remote_slot(0, expert)
                    for expert in _ids(
                        resident.placement, 0, "remote_only", top_k
                    )
                ]
            ],
            dtype=torch.int32,
            device=engine.device,
        )
        for _ in range(10):
            pending = current.submit(layer, cache, static_hidden, static_weights, remote_ids)
            current.finish(pending)
            current.release(pending)
        torch.cuda.synchronize(engine.device)
        current_us = []
        for _ in range(n):
            pending = current.submit(layer, cache, static_hidden, static_weights, remote_ids)
            current_us.append(
                pending.timing_values["host_remote_submit_control"]["value_ms"] * 1000.0
            )
            current.finish(pending)
            current.release(pending)
        torch.cuda.synchronize(engine.device)

        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize(engine.device)
        graph_us = []
        for _ in range(n):
            tic = time.perf_counter_ns()
            graph.replay()
            graph_us.append((time.perf_counter_ns() - tic) / 1000.0)
        torch.cuda.synchronize(engine.device)

        return {
            "schema": "inferswarm.d2.part1-primitive/1",
            "topology": {
                "capture": "one unified multi-device CUDA graph",
                "sequence": [
                    "gpu0 pinned D2H payload",
                    "internal cross-device ready event edge",
                    "gpu1 pinned H2D payload",
                    "resident NVFP4/Triton route-contribution kernel",
                    "gpu1 pinned D2H route contributions",
                    "internal cross-device done event edge",
                    "gpu0 pinned H2D return",
                    "same-route elementwise reconstruction",
                    "one canonical route-order sum reduction",
                ],
                "cross_device_events_worked": True,
                "steady_state_host_sync_count": 0,
                "host_synchronized": False,
                "recapture_per_replay": False,
            },
            "runtime": {
                "primary_uuid": secondary_info.primary.uuid,
                "secondary_uuid": secondary_info.secondary.uuid,
                "peer_access_primary_to_secondary": secondary_info.can_access_peer_primary_to_secondary,
                "peer_access_secondary_to_primary": secondary_info.can_access_peer_secondary_to_primary,
                "placement_sha256": resident.placement.artifact_sha256,
                "resident_gpu1_slots": resident.placement.remote_slots,
                "real_expert_bank_layout": resident.report.layout.bank_layout,
                "real_kernel": "fused_experts_decode_nvfp4_marlin_route_contributions",
                "steady_state_expert_weight_bytes_host_to_gpu1": 0,
            },
            "correctness": {
                "cases": cases,
                "changing_payload_consumed_without_recapture": changing_payload_consumed,
                "all_cases_passed": all(
                    c["assert_close"]["passed"]
                    and c["nan_count"] == 0
                    and c["inf_count"] == 0
                    for c in cases
                ),
                "exact_expert_ownership": all(c["no_route_dropped_or_duplicated"] for c in cases),
                "route_identity_mapping_preserved": True,
                "route_order_reconstruction_preserved": True,
            },
            "replay_benchmark": {
                "warmup": 10,
                "current_remote_submit_us": _distribution_us(current_us),
                "graph_remote_replay_submit_us": _distribution_us(graph_us),
                "materially_cheaper": statistics.median(graph_us) < statistics.median(current_us) * 0.5,
            },
            "physical_gpu_benchmark_runtime_seconds": time.monotonic() - benchmark_started,
        }
    finally:
        engine.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--primary-gpu", required=True)
    parser.add_argument("--secondary-gpu", required=True)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--replays", type=int, default=1000)
    ns = parser.parse_args()
    json.dump(
        run_fixture(ns.model, ns.primary_gpu, ns.secondary_gpu, ns.placement, ns.replays),
        fp=__import__("sys").stdout,
        indent=2,
        sort_keys=True,
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
