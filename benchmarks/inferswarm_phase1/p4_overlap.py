"""Physical P4 overlap/serialized routed-layer engineering diagnostic.

This is deliberately not a serving-performance runner. It loads the frozen model once,
executes identical deterministic mixed routes under both P4 overlap and the serialized
diagnostic control, and retains complete-layer/component timing plus C1/C2/C4 evidence.
It never reports tokens/s, TTFT, prefill throughput, or a Phase-1 verdict.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from freetoken.engine.engine import Engine
from freetoken.gpu_select import set_assigned_gpu
from freetoken.moe.inferswarm_remote_decode import HostStagedRemoteTransport
from freetoken.moe.layer_timing import MoeLayerTiming
from freetoken.moe.offload_cache import iter_offload_moe_layers
from freetoken.server.args import parse_args
from freetoken.server.launch import _resolve_server_gpu_args

from .p3_correctness import _deviation, _route_ids


def _timing_value(record: dict[str, Any], *path: str) -> float:
    value: Any = record["durations"]
    for key in path:
        value = value[key]
    if value["status"] != "valid":
        raise RuntimeError(f"required P4 timing {'/'.join(path)} is {value}")
    return float(value["value_ms"])


def _reference(engine: Engine, layer, raw, weights, hidden) -> torch.Tensor:
    cache = engine.moe_offload_cache
    cache.reset()
    attached, timing = layer.inferswarm_remote_decode, cache.layer_timing
    layer.inferswarm_remote_decode = None
    cache.layer_timing = None
    try:
        output = layer._decode_routed(hidden, weights, raw.clone()).clone()
    finally:
        layer.inferswarm_remote_decode = attached
        cache.layer_timing = timing
    torch.cuda.synchronize(engine.device)
    return output


def _mode_observation(
    engine: Engine,
    layer,
    *,
    mode: str,
    raw: torch.Tensor,
    weights: torch.Tensor,
    hidden: torch.Tensor,
    reference: torch.Tensor,
    repetitions: int,
) -> dict[str, Any]:
    cache, executor, timing = (
        engine.moe_offload_cache,
        engine.inferswarm_remote_decode,
        engine.moe_layer_timing,
    )
    assert cache is not None and executor is not None and timing is not None
    executor.mode = mode
    executor.reset()
    timing.reset()
    cache.reset_stats()
    timing.remote_overlap_active = mode == "overlap"
    records: list[dict[str, Any]] = []
    copy_plans: list[list[int]] = []
    residency_inputs: list[list[int]] = []
    remote_pending_at_local_service_start: list[bool] = []
    max_abs = max_rel = 0.0
    nan_count = inf_count = 0

    transport = executor.transport
    original_ensure, original_copy, original_submit = (
        cache.ensure_experts,
        cache.copy_missing,
        transport.submit,
    )
    last_pending = None
    copy_calls = 0

    def record_submit(*args, **kwargs):
        nonlocal last_pending
        last_pending = original_submit(*args, **kwargs)
        return last_pending

    def record_ensure(self, requested_layer, ids, *, record_routing=True):
        residency_inputs.append(ids.detach().cpu().reshape(-1).tolist())
        if last_pending is not None:
            remote_pending_at_local_service_start.append(
                not last_pending.completion_event.query()
            )
        return original_ensure(requested_layer, ids, record_routing=record_routing)

    def record_copy(self):
        nonlocal copy_calls
        copy_calls += 1
        return original_copy()

    cache.ensure_experts = MethodType(record_ensure, cache)
    cache.copy_missing = MethodType(record_copy, cache)
    transport.submit = record_submit
    try:
        for step in range(repetitions):
            cache.reset()
            executor.begin_decode_step(step)
            timing.begin_decode_step(
                step,
                batch_size=hidden.shape[0],
                padded_batch_size=hidden.shape[0],
                graph_replay=False,
            )
            output = layer._decode_routed(hidden, weights, raw.clone()).clone()
            torch.cuda.synchronize(engine.device)
            torch.testing.assert_close(
                output.float(), reference.float(), rtol=2e-3, atol=2e-3
            )
            abs_delta, rel_delta = _deviation(output, reference)
            max_abs, max_rel = max(max_abs, abs_delta), max(max_rel, rel_delta)
            nan_count += int(torch.isnan(output.float()).sum().item())
            inf_count += int(torch.isinf(output.float()).sum().item())
            fetched = int(cache.num_indices.item())
            copy_plans.append(cache.src_indices[:fetched].detach().cpu().tolist())
    finally:
        cache.ensure_experts = original_ensure
        cache.copy_missing = original_copy
        transport.submit = original_submit

    timing_snapshot = timing.snapshot()
    records = timing_snapshot["records"]
    if len(records) != repetitions:
        raise RuntimeError(
            f"P4 layer fixture retained {len(records)} timing records, expected {repetitions}"
        )
    snapshot = executor.snapshot(
        gpu0_expert_cache_slots=cache.cache_size,
        gpu0_expert_cache_bytes=cache.expert_bank_tensor_bytes(),
    )
    remote_raw = set(raw[(executor.route_lookup[0][raw.long()] >= 0)].cpu().tolist())
    if any(remote_raw.intersection(plan) for plan in copy_plans):
        raise RuntimeError("P4 remote identity entered the GPU0 expert copy plan")
    if any(remote_raw.intersection(call) for call in residency_inputs):
        raise RuntimeError("P4 remote identity entered GPU0 cache service")

    complete = [_timing_value(record, "complete_layer") for record in records]
    local = [
        _timing_value(record, "gpu0_branch", "complete_local_branch")
        for record in records
    ]
    remote = [
        _timing_value(record, "gpu1_branch", "complete_gpu1_branch")
        for record in records
    ]
    return {
        "mode": mode,
        "repetitions": repetitions,
        "assert_close": {"rtol": 2e-3, "atol": 2e-3, "passed": True},
        "max_absolute_deviation": max_abs,
        "max_relative_deviation": max_rel,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "ownership": snapshot["aggregate"],
        "transfer_bytes": snapshot["steady_state_transfer_bytes"],
        "mechanism_trace": snapshot["mechanism_trace"],
        "gpu0_residency_inputs": residency_inputs,
        "gpu0_copy_plans": copy_plans,
        "gpu0_copy_calls": copy_calls,
        "remote_pending_at_gpu0_local_service_start": (
            remote_pending_at_local_service_start
        ),
        "timing": timing_snapshot,
        "calculated_layer_timing_medians_ms": {
            "complete_layer": statistics.median(complete),
            "complete_gpu0_branch": statistics.median(local),
            "complete_gpu1_branch": statistics.median(remote),
        },
    }


def run_fixture(
    model: str,
    primary: str,
    secondary: str,
    placement: str,
    *,
    repetitions: int,
    token_rows: int,
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
            "overlap",
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
            "--moe-layer-timing-max-steps",
            str(repetitions),
            "--moe-layer-timing-role",
            "candidate",
        ]
    )
    server_args = _resolve_server_gpu_args(server_args)
    assigned = server_args.gpu_assigned or server_args.gpu
    set_assigned_gpu(assigned[0])
    engine = Engine(server_args)
    try:
        layers = list(iter_offload_moe_layers(engine.model))
        if len(layers) != 40:
            raise RuntimeError(
                f"P4 fixture expected 40 routed layers, found {len(layers)}"
            )
        layer = layers[0]
        cache, executor, resident = (
            engine.moe_offload_cache,
            engine.inferswarm_remote_decode,
            engine.inferswarm_resident_bank,
        )
        assert cache is not None and executor is not None and resident is not None
        if token_rows > executor.transport.max_tokens:
            # This diagnostic calls the routed layer directly; grow only its bounded
            # payload ring rather than changing scheduler/KV/state-pool geometry.
            executor.transport = HostStagedRemoteTransport(
                primary_device=engine.device,
                secondary_device=torch.device(
                    "cuda", engine.inferswarm_secondary_device.secondary.visible_ordinal
                ),
                max_tokens=token_rows,
                hidden_size=int(engine.config.model_config.hidden_size),
                top_k=int(engine.config.model_config.num_experts_per_tok),
                hidden_dtype=engine.dtype,
                resident_bank=resident,
                timing_enabled=True,
            )
        # The fixture exercises one layer, so use the production marker implementation with
        # one-layer storage. Full-engine serving diagnostics retain all 40 layers.
        timing = MoeLayerTiming(
            max_steps=repetitions,
            num_layers=1,
            device=engine.device,
            bytes_per_identity=cache.expert_bytes_per_identity(),
            role="candidate",
            graph_requested=False,
            remote_overlap_active=True,
        )
        cache.layer_timing = timing
        engine.moe_layer_timing = timing

        top_k = int(engine.config.model_config.num_experts_per_tok)
        raw_row = torch.tensor(
            [_route_ids(resident.placement, 0, "mixed", top_k)],
            dtype=torch.int32,
            device=engine.device,
        )
        raw = raw_row.repeat(token_rows, 1).contiguous()
        weights = torch.arange(1, top_k + 1, dtype=torch.float32, device=engine.device)
        weights = (
            (weights / weights.sum()).unsqueeze(0).repeat(token_rows, 1).contiguous()
        )
        hidden = torch.randn(
            (token_rows, int(engine.config.model_config.hidden_size)),
            dtype=engine.dtype,
            device=engine.device,
            generator=torch.Generator(device=engine.device).manual_seed(4404),
        )
        reference = _reference(engine, layer, raw, weights, hidden)
        serialized = _mode_observation(
            engine,
            layer,
            mode="serialized",
            raw=raw,
            weights=weights,
            hidden=hidden,
            reference=reference,
            repetitions=repetitions,
        )
        overlap = _mode_observation(
            engine,
            layer,
            mode="overlap",
            raw=raw,
            weights=weights,
            hidden=hidden,
            reference=reference,
            repetitions=repetitions,
        )
        if serialized["ownership"] != overlap["ownership"]:
            raise RuntimeError("P4 serialized/overlap ownership counters differ")
        if serialized["transfer_bytes"] != overlap["transfer_bytes"]:
            raise RuntimeError("P4 serialized/overlap transfer accounting differs")
        if any(serialized["remote_pending_at_gpu0_local_service_start"]):
            raise RuntimeError(
                "serialized diagnostic still had GPU1 work pending at local service"
            )
        if not any(overlap["remote_pending_at_gpu0_local_service_start"]):
            raise RuntimeError(
                "overlap diagnostic never observed GPU1 work pending at local service"
            )
        for observation in (serialized, overlap):
            if observation["ownership"]["fallback_elsewhere"] != 0:
                raise RuntimeError("P4 fixture observed forbidden fallback")
            if any(
                record["expected_dispatch_count"] != 1
                or record["actual_dispatch_count"] != 1
                for record in observation["mechanism_trace"]["records"]
            ):
                raise RuntimeError(
                    "P4 fixture did not issue exactly one mixed-route dispatch"
                )

        return {
            "schema": "inferswarm.phase1.p4-overlap-diagnostic/1",
            "evidence_label": "MEASURED ENGINEERING FIXTURE",
            "performance_measurement_taken": False,
            "performance_fields": {
                "tokens_per_second": None,
                "ttft": None,
                "prefill_throughput": None,
                "phase1_verdict": None,
            },
            "model": {
                "repository": resident.placement.model_repository,
                "revision": resident.placement.model_revision,
            },
            "placement_sha256": resident.placement.artifact_sha256,
            "runtime": {
                "primary_uuid": engine.inferswarm_secondary_device.primary.uuid,
                "secondary_uuid": engine.inferswarm_secondary_device.secondary.uuid,
                "gpu0_cache_slots": cache.cache_size,
                "gpu1_resident_slots": resident.placement.remote_slots,
                "gpu1_expert_bank_tensor_bytes": resident.report.expert_bank_tensor_bytes,
                "quant_format": resident.report.layout.quant_format,
                "nvfp4_backend": resident.report.layout.nvfp4_backend,
                "transport": "host_staged",
                "cuda_graph_max_bs": engine.config.cuda_graph_max_bs,
                "primary_current_after_fixture": (
                    torch.cuda.current_device() == engine.device.index
                ),
                "remote_prefill_dispatches": executor.counters.prefill_remote_dispatches,
                "steady_state_expert_weight_bytes_host_to_gpu1": (
                    executor.transport.transfer_bytes.host_to_gpu1_expert_weights
                ),
            },
            "fixture": {
                "layer_id": 0,
                "token_rows": token_rows,
                "raw_router_ids": raw.cpu().tolist(),
                "serialized": serialized,
                "overlap": overlap,
                "calculated_overlap_observation": {
                    "complete_layer_median_delta_ms": (
                        serialized["calculated_layer_timing_medians_ms"][
                            "complete_layer"
                        ]
                        - overlap["calculated_layer_timing_medians_ms"][
                            "complete_layer"
                        ]
                    ),
                    "interpretation": (
                        "noncanonical layer-level mechanism evidence only; concurrent branch "
                        "durations are never added to derive complete-layer time"
                    ),
                },
            },
        }
    finally:
        engine.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--primary-gpu", required=True)
    parser.add_argument("--secondary-gpu", required=True)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--token-rows", type=int, default=16)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.token_rows < 1:
        parser.error("--token-rows must be positive")
    document = run_fixture(
        args.model,
        args.primary_gpu,
        args.secondary_gpu,
        args.placement,
        repetitions=args.repetitions,
        token_rows=args.token_rows,
    )
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
        print(f"P4_DIAGNOSTIC_OUT {Path(args.output).resolve()}")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
