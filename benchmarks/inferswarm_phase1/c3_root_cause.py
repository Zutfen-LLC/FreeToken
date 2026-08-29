"""Lossless four-way Phase-1 C3 numerical trace capture and comparison.

This is a correctness-only tool. It drives the frozen request bodies without recording wall
time, throughput, TTFT, or any Phase-1 performance/verdict field. Tensor bytes are removed
from the human-readable JSON and written to SHA-addressed raw sidecars.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from inferswarm_phase0.manifest import REQUIRED_CLASSES, load_manifest

from .c3_correctness import (
    FROZEN_WARMUPS,
    V2_ARTIFACT_SHA256,
    _generation_state,
)
from .p4_workload_smoke import _greedy_generation, _instrumentation

ROOT_CAUSE_EVIDENCE_SCHEMA = "inferswarm.phase1.c3-root-cause-trace/1"
ROOT_CAUSE_COMPARISON_SCHEMA = "inferswarm.phase1.c3-root-cause-comparison/1"
C1_RTOL = 2e-3
C1_ATOL = 2e-3
FROZEN_MODEL_REPOSITORY = "nvidia/Qwen3.6-35B-A3B-NVFP4"
FROZEN_MODEL_REVISION = "491c2f1ea524c639598bf8fa787a93fed5a6fbce"
FROZEN_PLACEMENT_POLICY = "phase1-qwen36-placement-v2"
FROZEN_PRIMARY_UUID = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
FROZEN_SECONDARY_UUID = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
FROZEN_GPU0_CACHE_SLOTS = 3_774
SHAPES = ("R", "O", "S", "G")
FORBIDDEN_PERFORMANCE_KEY_FRAGMENTS = (
    "latency",
    "throughput",
    "ttft",
    "tokens_per_second",
    "tokens_per_sec",
    "speedup",
    "candidate_baseline_ratio",
    "prefill_speed",
    "kernel_performance",
)
PAIR_NAMES = (
    ("R", "O"),
    ("R", "S"),
    ("R", "G"),
    ("O", "S"),
    ("O", "G"),
    ("S", "G"),
)

_DTYPES = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_performance_firewall(document: Any, *, path: str = "$") -> None:
    """Reject accidental request/kernel performance fields anywhere in an artifact."""

    if isinstance(document, dict):
        for key, value in document.items():
            lowered = str(key).lower()
            if any(
                fragment in lowered for fragment in FORBIDDEN_PERFORMANCE_KEY_FRAGMENTS
            ):
                raise RuntimeError(
                    f"correctness evidence contains forbidden performance field {path}.{key}"
                )
            assert_performance_firewall(value, path=f"{path}.{key}")
    elif isinstance(document, list):
        for index, value in enumerate(document):
            assert_performance_firewall(value, path=f"{path}[{index}]")


def tensor_from_evidence(
    record: dict[str, Any], *, document_path: Path
) -> torch.Tensor:
    """Reconstruct an exact CPU tensor from inline or external raw-byte evidence."""

    dtype_name = record["dtype"]
    if dtype_name not in _DTYPES:
        raise ValueError(f"unsupported tensor evidence dtype {dtype_name!r}")
    if "raw_bytes_base64" in record:
        raw = base64.b64decode(record["raw_bytes_base64"], validate=True)
    else:
        sidecar = record.get("sidecar")
        if not isinstance(sidecar, str):
            raise ValueError("tensor evidence has neither inline bytes nor a sidecar")
        path = (document_path.parent / sidecar).resolve()
        raw = path.read_bytes()
        if _sha256_path(path) != record["raw_byte_sha256"]:
            raise ValueError(f"tensor sidecar hash mismatch: {path}")
    if len(raw) != int(record["raw_byte_count"]):
        raise ValueError("tensor raw-byte length disagrees with its manifest")
    if _sha256_bytes(raw) != record["raw_byte_sha256"]:
        raise ValueError("tensor raw-byte SHA-256 disagrees with its manifest")
    tensor = torch.frombuffer(bytearray(raw), dtype=_DTYPES[dtype_name]).clone()
    return tensor.reshape(record["shape"])


def externalize_trace_tensors(
    trace: dict[str, Any], *, output_path: Path, class_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Write bounded tensor bytes to deterministic raw sidecars and return a clean manifest."""

    clean = copy.deepcopy(trace)
    sidecar_root = output_path.parent / f"{output_path.stem}.tensors" / class_id
    sidecars: list[dict[str, Any]] = []
    for layer in clean["layers"]:
        layer_id = int(layer["layer_id"])
        for name, tensor in layer["tensors"].items():
            encoded = tensor.pop("raw_bytes_base64", None)
            if not isinstance(encoded, str):
                raise TypeError(
                    f"root-cause tensor {class_id}/layer{layer_id}/{name} has no raw bytes"
                )
            raw = base64.b64decode(encoded, validate=True)
            digest = _sha256_bytes(raw)
            if (
                digest != tensor["raw_byte_sha256"]
                or len(raw) != tensor["raw_byte_count"]
            ):
                raise RuntimeError(
                    "root-cause tensor bytes disagree with recorder metadata"
                )
            sidecar_root.mkdir(parents=True, exist_ok=True)
            path = sidecar_root / f"layer-{layer_id:02d}-{name}.bin"
            path.write_bytes(raw)
            relative = path.relative_to(output_path.parent).as_posix()
            tensor["sidecar"] = relative
            sidecars.append(
                {
                    "class_id": class_id,
                    "layer_id": layer_id,
                    "tensor": name,
                    "path": relative,
                    "sha256": digest,
                    "bytes": len(raw),
                }
            )
    return clean, sidecars


def tensor_metrics(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    rtol: float = C1_RTOL,
    atol: float = C1_ATOL,
) -> dict[str, Any]:
    shape_equal = tuple(left.shape) == tuple(right.shape)
    dtype_equal = left.dtype == right.dtype
    exact = (
        shape_equal
        and dtype_equal
        and torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )
    result: dict[str, Any] = {
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "exact_raw_byte_equality": exact,
        "rtol": rtol,
        "atol": atol,
    }
    if not shape_equal:
        result.update(
            max_absolute_deviation=None,
            max_relative_deviation=None,
            within_c1_tolerance=False,
        )
        return result
    left_f, right_f = left.float(), right.float()
    difference = (left_f - right_f).abs()
    if difference.numel() == 0:
        max_abs = max_rel = 0.0
    else:
        max_abs = float(difference.max().item())
        nonzero = right_f.abs() > 0
        relative = torch.where(
            nonzero,
            difference / right_f.abs(),
            torch.where(difference == 0, 0.0, float("inf")),
        )
        max_rel = float(relative.max().item())
    result.update(
        max_absolute_deviation=max_abs,
        max_relative_deviation=max_rel,
        within_c1_tolerance=bool(
            torch.isclose(left_f, right_f, rtol=rtol, atol=atol).all().item()
        ),
    )
    return result


def _layer_map(class_record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    layers = class_record["trace"]["layers"]
    return {int(layer["layer_id"]): layer for layer in layers}


def _tensor(layer: dict[str, Any], name: str, *, document_path: Path) -> torch.Tensor:
    return tensor_from_evidence(layer["tensors"][name], document_path=document_path)


def compare_class_pair(
    left_record: dict[str, Any],
    right_record: dict[str, Any],
    *,
    left_path: Path,
    right_path: Path,
    left_shape: str,
    right_shape: str,
) -> dict[str, Any]:
    left_layers, right_layers = _layer_map(left_record), _layer_map(right_record)
    if set(left_layers) != set(range(40)) or set(right_layers) != set(range(40)):
        raise RuntimeError("four-way comparison requires exactly MoE layers 0..39")
    rows = []
    for layer_id in range(40):
        left, right = left_layers[layer_id], right_layers[layer_id]
        hidden = tensor_metrics(
            _tensor(left, "hidden_input", document_path=left_path),
            _tensor(right, "hidden_input", document_path=right_path),
        )
        left_ids = _tensor(left, "raw_topk_ids", document_path=left_path)
        right_ids = _tensor(right, "raw_topk_ids", document_path=right_path)
        ids_exact = bool(torch.equal(left_ids, right_ids))
        left_weights = _tensor(left, "routing_weights", document_path=left_path)
        right_weights = _tensor(right, "routing_weights", document_path=right_path)
        weights = tensor_metrics(left_weights, right_weights)
        output = tensor_metrics(
            _tensor(left, "moe_output", document_path=left_path),
            _tensor(right, "moe_output", document_path=right_path),
        )
        partial_digests = {}
        for side, layer in ((left_shape, left), (right_shape, right)):
            partial_digests[side] = {
                name: tensor["raw_byte_sha256"]
                for name, tensor in layer["tensors"].items()
                if name in ("local_partial", "remote_partial", "combined_partial")
            }
        rows.append(
            {
                "layer_id": layer_id,
                "hidden_input": hidden,
                "routing": {
                    "topk_ids_exact": ids_exact,
                    "routing_weights_exact": weights["exact_raw_byte_equality"],
                    "routing_weight_max_absolute_deviation": weights[
                        "max_absolute_deviation"
                    ],
                    "routing_weight_max_relative_deviation": weights[
                        "max_relative_deviation"
                    ],
                },
                "moe_output": output,
                "candidate_partial_digests": partial_digests,
            }
        )

    def first(predicate) -> int | None:
        return next((row["layer_id"] for row in rows if predicate(row)), None)

    return {
        "left": left_shape,
        "right": right_shape,
        "first_divergence": {
            "hidden_input_bits": first(
                lambda row: not row["hidden_input"]["exact_raw_byte_equality"]
            ),
            "hidden_input_exceeds_c1": first(
                lambda row: not row["hidden_input"]["within_c1_tolerance"]
            ),
            "router_ids": first(lambda row: not row["routing"]["topk_ids_exact"]),
            "routing_weights": first(
                lambda row: not row["routing"]["routing_weights_exact"]
            ),
            "moe_output_bits": first(
                lambda row: not row["moe_output"]["exact_raw_byte_equality"]
            ),
            "moe_output_exceeds_c1": first(
                lambda row: not row["moe_output"]["within_c1_tolerance"]
            ),
        },
        "layers": rows,
    }


def route_geometry(class_record: dict[str, Any]) -> dict[str, int]:
    layers = class_record["trace"]["layers"]
    ownership = [layer["ownership"] for layer in layers]
    if any(item is None for item in ownership):
        return {
            "total_routed_selections": 0,
            "remote_owned_selections": 0,
            "mixed_layers": 0,
            "local_only_layers": 0,
            "remote_only_layers": 0,
        }
    return {
        "total_routed_selections": sum(
            int(item["total_routed_selections"]) for item in ownership
        ),
        "remote_owned_selections": sum(
            int(item["remote_selection_count"]) for item in ownership
        ),
        "mixed_layers": sum(
            int(
                item["local_selection_count"] > 0 and item["remote_selection_count"] > 0
            )
            for item in ownership
        ),
        "local_only_layers": sum(
            int(
                item["local_selection_count"] > 0
                and item["remote_selection_count"] == 0
            )
            for item in ownership
        ),
        "remote_only_layers": sum(
            int(
                item["local_selection_count"] == 0
                and item["remote_selection_count"] > 0
            )
            for item in ownership
        ),
    }


def classify_stage1(
    pair_reports: dict[str, dict[str, Any]], *, first_mixed_layer: int | None
) -> dict[str, Any]:
    """Apply the predeclared first-pass decision tree to mechanical pair reports."""

    ro, rs, rg = (
        pair_reports["R_vs_O"],
        pair_reports["R_vs_S"],
        pair_reports["R_vs_G"],
    )
    os_report, og, sg = (
        pair_reports["O_vs_S"],
        pair_reports["O_vs_G"],
        pair_reports["S_vs_G"],
    )
    candidates = [
        value
        for value in (
            ro["first_divergence"]["moe_output_bits"],
            rs["first_divergence"]["moe_output_bits"],
        )
        if value is not None
    ]
    relevant = min(candidates) if candidates else None
    upstream_candidates = [
        value
        for report in (ro, rs)
        for key in ("hidden_input_bits", "router_ids", "routing_weights")
        if (value := report["first_divergence"][key]) is not None
    ]
    first_upstream = min(upstream_candidates) if upstream_candidates else None
    if (
        first_upstream is not None
        and first_mixed_layer is not None
        and first_upstream <= first_mixed_layer
    ):
        return {
            "classification": "UPSTREAM_STATE_OR_ROUTING",
            "first_relevant_layer": first_upstream,
            "reason": "reference/candidate input or routing diverged no later than the first mixed layer",
        }
    if relevant is None:
        return {
            "classification": "UNRESOLVED",
            "first_relevant_layer": None,
            "reason": "the stage-1 trace did not reproduce a distributed MoE-output divergence",
        }

    def exact_through(report: dict[str, Any], layer_id: int) -> bool:
        for row in report["layers"][: layer_id + 1]:
            if not (
                row["hidden_input"]["exact_raw_byte_equality"]
                and row["routing"]["topk_ids_exact"]
                and row["routing"]["routing_weights_exact"]
                and row["moe_output"]["exact_raw_byte_equality"]
            ):
                return False
        return True

    raw_os_equal = all(
        row["routing"]["topk_ids_exact"] for row in os_report["layers"][: relevant + 1]
    )
    if not exact_through(os_report, relevant) and raw_os_equal:
        return {
            "classification": "OVERLAP_SPECIFIC",
            "first_relevant_layer": relevant,
            "reason": "O and S differ at or before the first reference divergence with identical raw routing",
        }

    row_rg = rg["layers"][relevant]
    row_ro = ro["layers"][relevant]
    row_rs = rs["layers"][relevant]
    if (
        exact_through(os_report, relevant)
        and row_rg["moe_output"]["exact_raw_byte_equality"]
        and not row_ro["moe_output"]["exact_raw_byte_equality"]
        and not row_rs["moe_output"]["exact_raw_byte_equality"]
    ):
        return {
            "classification": "REMOTE_EXECUTION_OR_TRANSPORT",
            "first_relevant_layer": relevant,
            "reason": "G matches R while the identical O/S result does not",
        }
    if (
        exact_through(os_report, relevant)
        and not row_rg["moe_output"]["exact_raw_byte_equality"]
        and og["layers"][relevant]["moe_output"]["exact_raw_byte_equality"]
        and sg["layers"][relevant]["moe_output"]["exact_raw_byte_equality"]
    ):
        return {
            "classification": "SPLIT_REDUCTION_TOPOLOGY",
            "first_relevant_layer": relevant,
            "reason": "O, S, and G agree while all differ from R",
        }
    return {
        "classification": "UNRESOLVED",
        "first_relevant_layer": relevant,
        "reason": "the exact four-way relationship does not match a predeclared stage-1 case",
    }


def _validate_trace_snapshot(snapshot: dict[str, Any], shape: str) -> dict[str, Any]:
    correctness = snapshot.get("inferswarm_correctness_diagnostics") or {}
    trace = correctness.get("moe_root_cause") or {}
    if (
        trace.get("enabled") is not True
        or trace.get("exactly_expected_layers") is not True
    ):
        raise RuntimeError(f"C3 root-cause trace is incomplete: {trace}")
    if trace.get("truncated") or trace.get("performance_fields_collected") is not False:
        raise RuntimeError(
            "C3 root-cause trace overflowed or contains performance fields"
        )
    remote = snapshot.get("inferswarm_remote_decode") or {}
    split = snapshot.get("inferswarm_split_gpu0_diagnostic") or {}
    runtime = snapshot.get("inferswarm_c3_root_cause_runtime") or {}
    if (
        runtime.get("primary_gpu_uuid") != FROZEN_PRIMARY_UUID
        or runtime.get("gpu0_cache_slots") != FROZEN_GPU0_CACHE_SLOTS
        or runtime.get("quant_format") != "nvfp4"
        or runtime.get("production_decode_kernel")
        != "fused_experts_decode_nvfp4_marlin"
    ):
        raise RuntimeError(f"shape {shape} violates the frozen GPU0 runtime: {runtime}")
    expected_mode = "DIAGNOSTIC_SPLIT_GPU0" if shape == "G" else "trace"
    if trace.get("mode") != expected_mode:
        raise RuntimeError(
            f"shape {shape} root-cause mode mismatch: {trace.get('mode')!r}"
        )
    if shape == "R" and (remote.get("enabled") or split.get("enabled")):
        raise RuntimeError("R must be ordinary unsplit GPU0 execution")
    if shape == "R" and (
        runtime.get("secondary_gpu_uuid") is not None
        or runtime.get("placement_sha256") is not None
    ):
        raise RuntimeError("R unexpectedly configured a secondary GPU or placement")
    if shape in ("O", "S"):
        expected_remote = "overlap" if shape == "O" else "serialized"
        if (
            remote.get("enabled") is not True
            or remote.get("execution_mode") != expected_remote
        ):
            raise RuntimeError(f"shape {shape} remote mode mismatch")
        if split.get("enabled"):
            raise RuntimeError(f"shape {shape} unexpectedly enabled split-GPU0")
        if (
            runtime.get("secondary_gpu_uuid") != FROZEN_SECONDARY_UUID
            or runtime.get("placement_sha256") != V2_ARTIFACT_SHA256
            or runtime.get("placement_policy") != FROZEN_PLACEMENT_POLICY
            or remote.get("placement_sha256") != V2_ARTIFACT_SHA256
        ):
            raise RuntimeError(f"shape {shape} violates the frozen distributed runtime")
    if shape == "G":
        if remote.get("enabled"):
            raise RuntimeError("G must not enable distributed remote decode")
        if (
            split.get("enabled") is not True
            or split.get("diagnostic_label") != "DIAGNOSTIC_SPLIT_GPU0"
            or split.get("uses_gpu1") is not False
            or split.get("gpu1_dispatches") != 0
            or split.get("f_gate_evidence_eligible") is not False
        ):
            raise RuntimeError(f"G split-GPU0 firewall failed: {split}")
        if (
            runtime.get("secondary_gpu_uuid") is not None
            or runtime.get("placement_sha256") != V2_ARTIFACT_SHA256
            or runtime.get("placement_policy") != FROZEN_PLACEMENT_POLICY
            or split.get("placement_sha256") != V2_ARTIFACT_SHA256
        ):
            raise RuntimeError("G violates the frozen split-GPU0 runtime")
    return trace


def capture_shape(
    *,
    origin: str,
    manifest_path: str,
    model_id: str,
    shape: str,
    classes: list[str],
    timeout: float,
    output_path: Path,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, canonical=True)
    provenance = manifest.record().get("provenance") or {}
    if (
        provenance.get("model_repository") != FROZEN_MODEL_REPOSITORY
        or provenance.get("model_revision") != FROZEN_MODEL_REVISION
        or model_id != FROZEN_MODEL_REPOSITORY
    ):
        raise RuntimeError(
            "root-cause capture requires the exact frozen model/revision"
        )
    workloads = manifest.by_class()
    results, sidecars = [], []
    for class_id in classes:
        workload = workloads[class_id]
        body = workload.greedy_reference_body(model_id)
        warmups = []
        for _ in range(FROZEN_WARMUPS):
            observation = _greedy_generation(origin, body, timeout=timeout)
            observation.pop("_output_text")
            if observation["completion_tokens"] != workload.output_tokens:
                raise RuntimeError(
                    "root-cause warmup did not preserve frozen output length"
                )
            warmups.append(observation)
        _instrumentation(origin, "reset", timeout)
        observation = _greedy_generation(origin, body, timeout=timeout)
        observation.pop("_output_text")
        if observation["completion_tokens"] != workload.output_tokens:
            raise RuntimeError(
                "root-cause request did not preserve frozen output length"
            )
        snapshot = _instrumentation(origin, "snapshot", timeout)
        # Validate that this remained the exact-token/greedy diagnostic path, but retain no
        # duplicate rounded logit payload in the root-cause artifact.
        generation = _generation_state(snapshot, workload.output_tokens)
        trace = _validate_trace_snapshot(snapshot, shape)
        trace, class_sidecars = externalize_trace_tensors(
            trace, output_path=output_path, class_id=class_id
        )
        sidecars.extend(class_sidecars)
        runtime = {
            "root_cause": snapshot.get("inferswarm_c3_root_cause_runtime"),
            "remote_decode": {
                key: (snapshot.get("inferswarm_remote_decode") or {}).get(key)
                for key in (
                    "enabled",
                    "execution_mode",
                    "overlap_active",
                    "serialized_diagnostic_only",
                    "transport",
                    "placement_sha256",
                    "resolved_quant_format",
                    "resolved_nvfp4_backend",
                    "resolved_bank_layout",
                )
            },
            "split_gpu0": snapshot.get("inferswarm_split_gpu0_diagnostic"),
            "resident_artifact": (
                (snapshot.get("inferswarm_resident_bank") or {}).get("artifact")
            ),
        }
        results.append(
            {
                "class_id": class_id,
                "warmups": warmups,
                "observation": observation,
                "generated_token_count": generation["generated_token_count"],
                "generated_token_ids_sha256": hashlib.sha256(
                    json.dumps(
                        generation["generated_token_ids"], separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "runtime": runtime,
                "trace": trace,
                "route_geometry": route_geometry({"trace": trace}),
            }
        )
    return {
        "schema": ROOT_CAUSE_EVIDENCE_SCHEMA,
        "evidence_label": "MEASURED C3 ROOT-CAUSE CORRECTNESS TRACE",
        "shape": shape,
        "shape_description": {
            "R": "ordinary unsplit correctness reference on GPU0",
            "O": "actual distributed host-staged overlap",
            "S": "actual distributed host-staged serialized",
            "G": "DIAGNOSTIC_SPLIT_GPU0",
        }[shape],
        "manifest": manifest.record(),
        "requested_model_id": model_id,
        "protocol": {
            "warmups": FROZEN_WARMUPS,
            "greedy": True,
            "ignore_eos": True,
            "fixed_output_length": True,
            "decode_step_captured": 0,
            "expected_moe_layers": 40,
        },
        "correctness_only": True,
        "performance_fields_collected": False,
        "classes": results,
        "tensor_sidecars": sidecars,
    }


def compare_shapes(paths: dict[str, Path]) -> dict[str, Any]:
    documents = {
        shape: json.loads(path.read_text(encoding="utf-8"))
        for shape, path in paths.items()
    }
    for shape, document in documents.items():
        if document.get("schema") != ROOT_CAUSE_EVIDENCE_SCHEMA:
            raise RuntimeError(f"shape {shape} has the wrong trace schema")
        if document.get("shape") != shape:
            raise RuntimeError(f"shape label mismatch in {paths[shape]}")
        if document.get("performance_fields_collected") is not False:
            raise RuntimeError(f"shape {shape} is not a performance-free artifact")
    by_shape = {
        shape: {row["class_id"]: row for row in document["classes"]}
        for shape, document in documents.items()
    }
    classes = set.intersection(*(set(rows) for rows in by_shape.values()))
    if not classes:
        raise RuntimeError("four-way traces have no common workload class")
    class_reports = []
    for class_id in sorted(classes):
        pairs = {}
        for left, right in PAIR_NAMES:
            name = f"{left}_vs_{right}"
            pairs[name] = compare_class_pair(
                by_shape[left][class_id],
                by_shape[right][class_id],
                left_path=paths[left],
                right_path=paths[right],
                left_shape=left,
                right_shape=right,
            )
        geometry = {
            shape: route_geometry(by_shape[shape][class_id]) for shape in SHAPES
        }
        mixed = [
            int(layer["layer_id"])
            for layer in by_shape["O"][class_id]["trace"]["layers"]
            if layer["ownership"]
            and layer["ownership"]["local_selection_count"] > 0
            and layer["ownership"]["remote_selection_count"] > 0
        ]
        r_runtime = by_shape["R"][class_id]["runtime"]["root_cause"] or {}
        g_runtime = by_shape["G"][class_id]["runtime"]["root_cause"] or {}
        rg_rows = pairs["R_vs_G"]["layers"]
        r_layers = _layer_map(by_shape["R"][class_id])
        g_layers = _layer_map(by_shape["G"][class_id])
        split_isolation = []
        for layer_id, rg_row in enumerate(rg_rows):
            ownership = g_layers[layer_id]["ownership"] or {}
            split_isolation.append(
                {
                    "layer_id": layer_id,
                    "same_raw_expert_ids": rg_row["routing"]["topk_ids_exact"],
                    "same_routing_weights": rg_row["routing"]["routing_weights_exact"],
                    "same_expert_weight_source": (
                        documents["R"].get("requested_model_id")
                        == documents["G"].get("requested_model_id")
                    ),
                    "same_expert_weights": (
                        r_layers[layer_id].get("selected_expert_weights") is not None
                        and r_layers[layer_id].get("selected_expert_weights")
                        == g_layers[layer_id].get("selected_expert_weights")
                    ),
                    "same_production_kernel": (
                        r_runtime.get("production_decode_kernel")
                        == g_runtime.get("production_decode_kernel")
                        == "fused_experts_decode_nvfp4_marlin"
                    ),
                    "both_execute_on_gpu0": (
                        r_runtime.get("gpu1_execution_enabled") is False
                        and g_runtime.get("gpu1_execution_enabled") is False
                    ),
                    "masks_complete": ownership.get("masks_complete"),
                    "masks_disjoint": ownership.get("masks_disjoint"),
                    "every_route_exactly_once": ownership.get(
                        "every_route_exactly_once"
                    ),
                }
            )
        decision = classify_stage1(
            pairs, first_mixed_layer=min(mixed) if mixed else None
        )
        relevant_layer = decision["first_relevant_layer"]
        proof_keys = (
            "same_raw_expert_ids",
            "same_routing_weights",
            "same_expert_weight_source",
            "same_expert_weights",
            "same_production_kernel",
            "both_execute_on_gpu0",
            "masks_complete",
            "masks_disjoint",
            "every_route_exactly_once",
        )
        class_reports.append(
            {
                "class_id": class_id,
                "route_geometry": geometry,
                "split_gpu0_isolation": {
                    "intended_only_difference": (
                        "one full routed reduction versus two complementary production "
                        "GPU0 reductions plus one add"
                    ),
                    "layers": split_isolation,
                    "all_layers_proven": all(
                        all(row[key] is True for key in proof_keys)
                        for row in split_isolation
                    ),
                    "all_layers_structural_split_proven": all(
                        all(
                            row[key] is True
                            for key in (
                                "same_expert_weight_source",
                                "same_production_kernel",
                                "both_execute_on_gpu0",
                                "masks_complete",
                                "masks_disjoint",
                                "every_route_exactly_once",
                            )
                        )
                        for row in split_isolation
                    ),
                    "first_relevant_layer": relevant_layer,
                    "first_relevant_layer_proven": (
                        relevant_layer is not None
                        and all(
                            split_isolation[relevant_layer][key] is True
                            for key in proof_keys
                        )
                    ),
                    "downstream_state_note": (
                        "After the first MoE-output bit divergence, later R/G hidden state, "
                        "routing weights, or IDs may differ as downstream propagation; the "
                        "targeted replay supplies the required fixed-input per-layer proof."
                    ),
                },
                "pairs": pairs,
                "stage1_decision": decision,
            }
        )
    reports_by_class = {row["class_id"]: row for row in class_reports}
    route_geometry_analysis = {}
    for class_id, report in reports_by_class.items():
        candidate_geometry = report["route_geometry"]["O"]
        ro_layers = report["pairs"]["R_vs_O"]["layers"]
        mixed = candidate_geometry["mixed_layers"]
        comparisons_to_failed_classes = {}
        for failed_class in ("W3", "W4"):
            failed = reports_by_class.get(failed_class)
            failed_geometry = failed["route_geometry"]["O"] if failed else None
            comparisons_to_failed_classes[failed_class] = (
                {
                    "mixed_layer_delta": mixed - failed_geometry["mixed_layers"],
                    "remote_selection_delta": (
                        candidate_geometry["remote_owned_selections"]
                        - failed_geometry["remote_owned_selections"]
                    ),
                }
                if failed_geometry is not None
                else None
            )
        route_geometry_analysis[class_id] = {
            "no_remote_selections_at_step0": (
                candidate_geometry["remote_owned_selections"] == 0
            ),
            "has_mixed_split_execution_at_step0": mixed > 0,
            "all_R_vs_O_moe_outputs_bit_exact": all(
                row["moe_output"]["exact_raw_byte_equality"] for row in ro_layers
            ),
            "genuine_split_execution_with_zero_observed_moe_deviation": (
                mixed > 0
                and all(
                    row["moe_output"]["exact_raw_byte_equality"] for row in ro_layers
                )
            ),
            "comparison_to_failed_classes": comparisons_to_failed_classes,
        }
    return {
        "schema": ROOT_CAUSE_COMPARISON_SCHEMA,
        "evidence_label": "CALCULATED FROM LOSSLESS C3 ROOT-CAUSE TENSORS",
        "correctness_only": True,
        "performance_fields_collected": False,
        "c1_tolerance": {"rtol": C1_RTOL, "atol": C1_ATOL},
        "inputs": {
            shape: {"path": str(path), "sha256": _sha256_path(path)}
            for shape, path in paths.items()
        },
        "route_geometry_analysis": route_geometry_analysis,
        "classes": class_reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--origin", required=True)
    capture.add_argument("--manifest", required=True)
    capture.add_argument("--model-id", required=True)
    capture.add_argument("--shape", choices=SHAPES, required=True)
    capture.add_argument(
        "--class", dest="classes", action="append", choices=REQUIRED_CLASSES
    )
    capture.add_argument("--timeout", type=float, default=3600.0)
    capture.add_argument("--output", required=True)
    compare = subparsers.add_parser("compare")
    for shape, flag in (
        ("R", "reference"),
        ("O", "overlap"),
        ("S", "serialized"),
        ("G", "split-gpu0"),
    ):
        compare.add_argument(f"--{flag}", dest=shape, required=True)
    compare.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "capture":
        document = capture_shape(
            origin=args.origin.rstrip("/"),
            manifest_path=args.manifest,
            model_id=args.model_id,
            shape=args.shape,
            classes=args.classes or list(REQUIRED_CLASSES),
            timeout=args.timeout,
            output_path=output_path,
        )
    else:
        document = compare_shapes(
            {shape: Path(getattr(args, shape)).resolve() for shape in SHAPES}
        )
    assert_performance_firewall(document)
    output_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"C3_ROOT_CAUSE_OUT {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
