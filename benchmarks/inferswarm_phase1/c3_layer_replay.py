"""Exact fixed-input replay for the first divergent Phase-1 C3 MoE layer.

The engineering diagnostic evaluates U/GL/GR/GS and RL/RR/RC from one losslessly captured
input/routing triple. It runs the distributed arm serialized first. Transport-stage capture
and selected resident-row revalidation are activated only when GR and RR actually differ.
No request-performance field is collected or reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from freetoken.engine.engine import Engine
from freetoken.gpu_select import set_assigned_gpu
from freetoken.moe.offload_cache import iter_offload_moe_layers
from freetoken.server.args import parse_args
from freetoken.server.launch import _resolve_server_gpu_args

from .c3_root_cause import (
    C1_ATOL,
    C1_RTOL,
    ROOT_CAUSE_EVIDENCE_SCHEMA,
    _sha256_path,
    assert_performance_firewall,
    tensor_from_evidence,
    tensor_metrics,
)

LAYER_REPLAY_SCHEMA = "inferswarm.phase1.c3-root-cause-layer-replay/1"


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().contiguous().cpu()
    return cpu.view(torch.uint8).reshape(-1).numpy().tobytes()


def fixed_input_hashes(
    hidden: torch.Tensor, raw_ids: torch.Tensor, routing_weights: torch.Tensor
) -> dict[str, str]:
    """Lossless identity of the three immutable base tensors used by every replay arm."""

    return {
        "hidden_input": hashlib.sha256(_raw_bytes(hidden)).hexdigest(),
        "raw_topk_ids": hashlib.sha256(_raw_bytes(raw_ids)).hexdigest(),
        "routing_weights": hashlib.sha256(_raw_bytes(routing_weights)).hexdigest(),
    }


def _tensor_manifest(
    tensor: torch.Tensor,
    *,
    output_path: Path,
    class_id: str,
    layer_id: int,
    name: str,
) -> dict[str, Any]:
    cpu = tensor.detach().contiguous().cpu()
    raw = _raw_bytes(cpu)
    digest = hashlib.sha256(raw).hexdigest()
    root = output_path.parent / f"{output_path.stem}.tensors" / class_id
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"layer-{layer_id:02d}-{name}.bin"
    path.write_bytes(raw)
    return {
        "dtype": str(cpu.dtype).removeprefix("torch."),
        "shape": list(cpu.shape),
        "raw_byte_count": len(raw),
        "raw_byte_sha256": digest,
        "sidecar": path.relative_to(output_path.parent).as_posix(),
    }


def _layer_record(
    document: dict[str, Any], *, class_id: str, layer_id: int
) -> dict[str, Any]:
    class_row = next(
        (row for row in document["classes"] if row["class_id"] == class_id), None
    )
    if class_row is None:
        raise RuntimeError(f"reference trace has no {class_id} record")
    layer = next(
        (
            item
            for item in class_row["trace"]["layers"]
            if int(item["layer_id"]) == layer_id
        ),
        None,
    )
    if layer is None:
        raise RuntimeError(f"reference trace has no layer {layer_id}")
    return layer


def _selected_resident_row_validation(
    engine: Engine,
    *,
    layer_id: int,
    raw_ids: torch.Tensor,
    remote_mask: torch.Tensor,
) -> dict[str, Any]:
    cache, resident, executor = (
        engine.moe_offload_cache,
        engine.inferswarm_resident_bank,
        engine.inferswarm_remote_decode,
    )
    assert cache is not None and resident is not None and executor is not None
    rows = []
    seen: set[int] = set()
    for expert in raw_ids[remote_mask].detach().cpu().tolist():
        expert = int(expert)
        if expert in seen:
            continue
        seen.add(expert)
        remote_slot = int(executor.route_lookup[layer_id, expert].item())
        banks = []
        for name, remote_view in zip(
            cache.bank_schema, resident.bank_views(), strict=True
        ):
            source = cache.bank_sources[name][layer_id][expert]
            remote = remote_view[remote_slot]
            source_hash = hashlib.sha256(_raw_bytes(source)).hexdigest()
            remote_hash = hashlib.sha256(_raw_bytes(remote)).hexdigest()
            banks.append(
                {
                    "bank": name,
                    "source_sha256": source_hash,
                    "gpu1_resident_sha256": remote_hash,
                    "exact": source_hash == remote_hash,
                }
            )
        rows.append(
            {
                "expert_id": expert,
                "remote_slot": remote_slot,
                "banks": banks,
                "all_banks_exact": all(bank["exact"] for bank in banks),
            }
        )
    return {
        "performed": True,
        "scope": "selected remote identities in targeted layer only",
        "rows": rows,
        "all_selected_rows_exact": all(row["all_banks_exact"] for row in rows),
    }


def localize_transport(
    tensors: dict[str, torch.Tensor], *, gpu0_remote_subset: torch.Tensor
) -> dict[str, Any]:
    """Attribute the first exact mutation across the conditional §11 staging chain."""

    comparisons = (
        (
            "gpu0_to_host_activation",
            "transport_gpu0_source_hidden_activation",
            "transport_pinned_host_staged_activation",
            "REMOTE_TRANSPORT",
        ),
        (
            "gpu0_to_host_routing_weights",
            "transport_gpu0_source_routing_weights",
            "transport_pinned_host_routing_weights",
            "REMOTE_TRANSPORT",
        ),
        (
            "gpu0_to_host_remote_slot_ids",
            "transport_expected_remote_slot_ids",
            "transport_pinned_host_remote_slot_ids",
            "REMOTE_TRANSPORT",
        ),
        (
            "host_to_gpu1_activation",
            "transport_pinned_host_staged_activation",
            "transport_gpu1_activation_after_h2d",
            "REMOTE_TRANSPORT",
        ),
        (
            "host_to_gpu1_routing_weights",
            "transport_pinned_host_routing_weights",
            "transport_gpu1_routing_weights",
            "REMOTE_TRANSPORT",
        ),
        (
            "host_to_gpu1_remote_slot_ids",
            "transport_pinned_host_remote_slot_ids",
            "transport_gpu1_remote_slot_ids",
            "REMOTE_TRANSPORT",
        ),
    )
    reports = []
    for boundary, left, right, classification in comparisons:
        metrics = tensor_metrics(tensors[left], tensors[right])
        reports.append({"boundary": boundary, **metrics})
        if not metrics["exact_raw_byte_equality"]:
            return {
                "classification": classification,
                "first_corrupt_boundary": boundary,
                "comparisons": reports,
            }

    gpu1 = tensors["transport_gpu1_remote_partial_before_d2h"]
    execution = tensor_metrics(gpu0_remote_subset, gpu1)
    reports.append({"boundary": "gpu0_vs_gpu1_remote_execution", **execution})
    if not execution["exact_raw_byte_equality"]:
        return {
            "classification": "REMOTE_GPU_EXECUTION",
            "first_corrupt_boundary": "gpu1_remote_expert_execution",
            "comparisons": reports,
        }
    for boundary, left, right in (
        (
            "gpu1_to_host_return",
            "transport_gpu1_remote_partial_before_d2h",
            "transport_pinned_host_returned_partial",
        ),
        (
            "host_to_gpu0_return",
            "transport_pinned_host_returned_partial",
            "transport_gpu0_returned_remote_partial",
        ),
    ):
        metrics = tensor_metrics(tensors[left], tensors[right])
        reports.append({"boundary": boundary, **metrics})
        if not metrics["exact_raw_byte_equality"]:
            return {
                "classification": "REMOTE_TRANSPORT",
                "first_corrupt_boundary": boundary,
                "comparisons": reports,
            }
    return {
        "classification": "UNRESOLVED",
        "first_corrupt_boundary": None,
        "comparisons": reports,
    }


def _largest_deviations(
    left: torch.Tensor, right: torch.Tensor, *, count: int = 8
) -> list[dict[str, Any]]:
    difference = (left.float() - right.float()).abs()
    flat = difference.reshape(-1)
    if not flat.numel():
        return []
    count = min(count, flat.numel())
    values, indices = torch.topk(flat, count)
    rows = []
    for value, flat_index in zip(values.tolist(), indices.tolist(), strict=True):
        coordinates = []
        remainder = int(flat_index)
        for size in reversed(left.shape):
            coordinates.append(remainder % int(size))
            remainder //= int(size)
        coordinates.reverse()
        index = tuple(coordinates)
        rows.append(
            {
                "coordinates": coordinates,
                "left": float(left[index].float().item()),
                "right": float(right[index].float().item()),
                "absolute_deviation": float(value),
            }
        )
    return rows


def _bf16_ulp_observation(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.dtype != torch.bfloat16 or right.dtype != torch.bfloat16:
        return {
            "output_dtype": str(left.dtype).removeprefix("torch."),
            "bf16_ulp_at_largest_coordinate": None,
            "deviation_in_local_bf16_ulps": None,
        }
    difference = (left.float() - right.float()).abs()
    index = int(difference.reshape(-1).argmax().item())
    anchor = right.reshape(-1)[index]
    toward = torch.tensor(float("inf"), dtype=torch.bfloat16)
    next_value = torch.nextafter(anchor.cpu(), toward)
    ulp = abs(float(next_value.float().item()) - float(anchor.float().item()))
    delta = float(difference.reshape(-1)[index].item())
    return {
        "output_dtype": "bfloat16",
        "bf16_ulp_at_largest_coordinate": ulp,
        "deviation_in_local_bf16_ulps": delta / ulp if ulp else None,
        "interpretation": (
            "factual local-ULP characterization only; no new correctness threshold is applied"
        ),
    }


def _diagnostic_tensor_map(
    snapshot: dict[str, Any], layer_id: int
) -> dict[str, torch.Tensor]:
    root = snapshot["moe_root_cause"]
    layer = next(item for item in root["layers"] if item["layer_id"] == layer_id)
    # Snapshot tensors are inline here; use a dummy document path because no sidecars exist.
    return {
        name: tensor_from_evidence(record, document_path=Path("."))
        for name, record in layer["tensors"].items()
    }


def replay_comparisons(variants: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    """Compute the five predeclared fixed-input replay comparisons."""

    required = {"U", "GL", "GR", "GS", "RL", "RR", "RC"}
    if set(variants) != required:
        raise ValueError(
            f"replay variants must be exactly {sorted(required)}, got {sorted(variants)}"
        )
    return {
        "U_vs_GS": tensor_metrics(variants["U"], variants["GS"]),
        "GL_vs_RL": tensor_metrics(variants["GL"], variants["RL"]),
        "GR_vs_RR": tensor_metrics(variants["GR"], variants["RR"]),
        "GS_vs_RC": tensor_metrics(variants["GS"], variants["RC"]),
        "U_vs_RC": tensor_metrics(variants["U"], variants["RC"]),
    }


def run_replay(
    *,
    model: str,
    primary: str,
    secondary: str,
    placement: str,
    reference_trace_path: Path,
    class_id: str,
    layer_id: int,
    output_path: Path,
) -> dict[str, Any]:
    reference_document = json.loads(reference_trace_path.read_text(encoding="utf-8"))
    if (
        reference_document.get("schema") != ROOT_CAUSE_EVIDENCE_SCHEMA
        or reference_document.get("shape") != "R"
    ):
        raise RuntimeError("target replay requires a root-cause R trace")
    captured = _layer_record(reference_document, class_id=class_id, layer_id=layer_id)
    hidden_cpu = tensor_from_evidence(
        captured["tensors"]["hidden_input"], document_path=reference_trace_path
    )
    ids_cpu = tensor_from_evidence(
        captured["tensors"]["raw_topk_ids"], document_path=reference_trace_path
    )
    weights_cpu = tensor_from_evidence(
        captured["tensors"]["routing_weights"], document_path=reference_trace_path
    )

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
            "--moe-cpu-layers",
            "0",
            "--nvfp4-backend",
            "triton",
            "--moe-cache-size",
            "3774",
            "--kv-reserve-tokens",
            "17075",
            "--num-tokens",
            "17075",
            "--memory-ratio",
            "0.85",
            "--cuda-graph-max-bs",
            "0",
            "--max-running-requests",
            "1",
            "--sampling-defaults",
            "none",
            "--inferswarm-correctness-diagnostics",
            "--inferswarm-c3-root-cause-mode",
            "trace",
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
                f"target replay expected 40 MoE layers, found {len(layers)}"
            )
        layer = layers[layer_id]
        cache, executor, recorder = (
            engine.moe_offload_cache,
            engine.inferswarm_remote_decode,
            engine.inferswarm_correctness_diagnostics,
        )
        assert cache is not None and executor is not None and recorder is not None
        hidden = hidden_cpu.to(engine.device)
        raw_ids = ids_cpu.to(engine.device)
        weights = weights_cpu.to(engine.device)
        input_hashes = fixed_input_hashes(hidden, raw_ids, weights)

        recorder.reset()
        recorder.begin_decode_step(0)
        recorder.capture_selected_expert_weights(layer_id, raw_ids, cache)
        replay_weight_proof = recorder.snapshot()["moe_root_cause"]["layers"][0][
            "selected_expert_weights"
        ]
        reference_weight_proof = captured.get("selected_expert_weights")
        if (
            reference_weight_proof is None
            or replay_weight_proof != reference_weight_proof
        ):
            raise RuntimeError(
                "target replay model rows differ from the exact R-trace selected weights"
            )

        remote_slot_ids = executor.route_lookup[layer_id][raw_ids.long()]
        remote_mask = remote_slot_ids >= 0
        local_mask = ~remote_mask
        if not bool(torch.all(local_mask | remote_mask).item()) or bool(
            torch.any(local_mask & remote_mask).item()
        ):
            raise RuntimeError("target replay ownership masks are not complementary")

        # U/GL/GR/GS: one cache service makes the complete exact route set resident on GPU0.
        cache.reset()
        gpu0_slots = raw_ids.clone()
        cache.ensure_experts(layer_id, gpu0_slots)
        cache.copy_missing()
        views, alphas = cache.bank_views(), cache.alphas_for_slots(layer_id)
        zero = weights.new_zeros(())
        local_weights = torch.where(local_mask, weights, zero).contiguous()
        remote_weights = torch.where(remote_mask, weights, zero).contiguous()
        u = layer._expert_gemm(
            cache,
            hidden,
            weights,
            gpu0_slots,
            views=views,
            n=None,
            alphas=alphas,
            is_prefill=False,
        ).clone()
        gl = layer._expert_gemm(
            cache,
            hidden,
            local_weights,
            gpu0_slots,
            views=views,
            n=None,
            alphas=alphas,
            is_prefill=False,
        ).clone()
        gr = layer._expert_gemm(
            cache,
            hidden,
            remote_weights,
            gpu0_slots,
            views=views,
            n=None,
            alphas=alphas,
            is_prefill=False,
        ).clone()
        gs = (gl + gr).clone()

        # RL/RR/RC: actual serialized candidate implementation from the same tensor objects.
        cache.reset()
        executor.reset()
        recorder.reset()
        recorder.begin_decode_step(0)
        executor.begin_decode_step(0)
        recorder.capture_moe_input(layer_id, hidden, raw_ids, weights)
        rc = executor.decode(layer, cache, hidden, weights, raw_ids.clone()).clone()
        torch.cuda.synchronize(engine.device)
        actual_tensors = _diagnostic_tensor_map(recorder.snapshot(), layer_id)
        rl = actual_tensors["local_partial"]
        rr = actual_tensors["remote_partial"]
        recorded_rc = actual_tensors["combined_partial"]
        if not torch.equal(rc.cpu(), recorded_rc):
            raise RuntimeError(
                "recorded RC does not equal the actual distributed combine"
            )

        variants = {
            "U": u.cpu(),
            "GL": gl.cpu(),
            "GR": gr.cpu(),
            "GS": gs.cpu(),
            "RL": rl,
            "RR": rr,
            "RC": rc.cpu(),
        }
        comparisons = replay_comparisons(variants)
        if fixed_input_hashes(hidden, raw_ids, weights) != input_hashes:
            raise RuntimeError(
                "a target replay arm mutated its fixed base input/routing"
            )

        transport = {"performed": False, "reason": "GR and RR are exactly equal"}
        resident_validation = {
            "performed": False,
            "reason": "GR and RR are exactly equal; §11 is intentionally not run",
        }
        if not comparisons["GR_vs_RR"]["exact_raw_byte_equality"]:
            cache.reset()
            executor.reset()
            recorder.reset()
            recorder.begin_decode_step(0)
            recorder.enable_transport_capture(layer_id)
            executor.begin_decode_step(0)
            recorder.capture_moe_input(layer_id, hidden, raw_ids, weights)
            replay_rc = executor.decode(
                layer, cache, hidden, weights, raw_ids.clone()
            ).clone()
            torch.cuda.synchronize(engine.device)
            transport_tensors = _diagnostic_tensor_map(recorder.snapshot(), layer_id)
            transport = {
                "performed": True,
                "localization": localize_transport(
                    transport_tensors, gpu0_remote_subset=gr.cpu()
                ),
                "tensors": {
                    name: _tensor_manifest(
                        tensor,
                        output_path=output_path,
                        class_id=class_id,
                        layer_id=layer_id,
                        name=name,
                    )
                    for name, tensor in transport_tensors.items()
                    if name.startswith("transport_")
                },
                "repeat_RC_matches_first_RC": bool(torch.equal(replay_rc, rc)),
            }
            resident_validation = _selected_resident_row_validation(
                engine,
                layer_id=layer_id,
                raw_ids=raw_ids,
                remote_mask=remote_mask,
            )

        if (
            not comparisons["U_vs_GS"]["exact_raw_byte_equality"]
            and comparisons["GL_vs_RL"]["exact_raw_byte_equality"]
            and comparisons["GR_vs_RR"]["exact_raw_byte_equality"]
            and comparisons["GS_vs_RC"]["exact_raw_byte_equality"]
        ):
            classification = "SPLIT_REDUCTION_TOPOLOGY"
        elif not comparisons["GR_vs_RR"]["exact_raw_byte_equality"]:
            classification = transport["localization"]["classification"]
        else:
            classification = "UNRESOLVED"

        tensor_manifests = {
            name: _tensor_manifest(
                tensor,
                output_path=output_path,
                class_id=class_id,
                layer_id=layer_id,
                name=name,
            )
            for name, tensor in variants.items()
        }
        return {
            "schema": LAYER_REPLAY_SCHEMA,
            "evidence_label": "MEASURED FIXED-INPUT C3 LAYER REPLAY",
            "correctness_only": True,
            "performance_fields_collected": False,
            "classification": classification,
            "class_id": class_id,
            "layer_id": layer_id,
            "fixed_input": {
                "reference_trace_sha256": _sha256_path(reference_trace_path),
                "hashes": input_hashes,
                "same_exact_base_input_and_routing_reused_by_every_variant": True,
                "selected_expert_weights_match_reference_trace": True,
                "selected_expert_weight_proof": replay_weight_proof,
            },
            "ownership": {
                "raw_route_count": int(raw_ids.numel()),
                "local_count": int(local_mask.sum().item()),
                "remote_count": int(remote_mask.sum().item()),
                "complete": True,
                "disjoint": True,
            },
            "execution": {
                "U": "one production GPU0 expert GEMM over all routes",
                "GL": "production GPU0 expert GEMM with remote-owned weights zeroed",
                "GR": "production GPU0 expert GEMM with local-owned weights zeroed",
                "GS": "GL + GR once on GPU0",
                "RL": "actual candidate GPU0 local partial",
                "RR": "actual GPU1 resident remote partial returned through host staging",
                "RC": "actual distributed RL + RR combine",
                "distributed_mode": "serialized",
            },
            "tensors": tensor_manifests,
            "comparisons": comparisons,
            "split_characterization": {
                "U_vs_GS_largest_deviations": _largest_deviations(u.cpu(), gs.cpu()),
                "bf16_addition_order_observation": _bf16_ulp_observation(
                    u.cpu(), gs.cpu()
                ),
                "GL": {
                    "nan_count": int(torch.isnan(gl.float()).sum().item()),
                    "inf_count": int(torch.isinf(gl.float()).sum().item()),
                },
                "GR": {
                    "nan_count": int(torch.isnan(gr.float()).sum().item()),
                    "inf_count": int(torch.isinf(gr.float()).sum().item()),
                },
            },
            "transport_localization": transport,
            "selected_remote_resident_row_validation": resident_validation,
            "runtime": {
                "primary_uuid": engine.inferswarm_secondary_device.primary.uuid,
                "secondary_uuid": engine.inferswarm_secondary_device.secondary.uuid,
                "placement_sha256": engine.inferswarm_placement.artifact_sha256,
                "gpu0_cache_slots": cache.cache_size,
                "quant_format": cache.quant_format,
                "nvfp4_backend": "triton",
                "transport": "host_staged",
                "model_repository": engine.inferswarm_placement.model_repository,
                "model_revision": engine.inferswarm_placement.model_revision,
            },
            "c1_tolerance_unchanged": {"rtol": C1_RTOL, "atol": C1_ATOL},
        }
    finally:
        engine.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--primary-gpu", required=True)
    parser.add_argument("--secondary-gpu", required=True)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--reference-trace", required=True)
    parser.add_argument(
        "--class", dest="class_id", required=True, choices=("W1", "W2", "W3", "W4")
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if not 0 <= args.layer < 40:
        parser.error("--layer must be in [0, 40)")
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = run_replay(
        model=args.model,
        primary=args.primary_gpu,
        secondary=args.secondary_gpu,
        placement=args.placement,
        reference_trace_path=Path(args.reference_trace).resolve(),
        class_id=args.class_id,
        layer_id=args.layer,
        output_path=output_path,
    )
    assert_performance_firewall(document)
    output_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"C3_LAYER_REPLAY_OUT {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
