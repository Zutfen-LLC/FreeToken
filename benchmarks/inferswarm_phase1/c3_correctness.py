"""Exact, performance-free InferSwarm C3 capture and comparison.

This tool drives the frozen greedy/fixed-length W1-W4 protocol but reads token IDs and
step-0 logits only from the opt-in engine correctness recorder. It never re-tokenizes text,
records request timing, or emits a performance verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from inferswarm_phase0.manifest import REQUIRED_CLASSES, load_manifest

from .p4_workload_smoke import _greedy_generation, _instrumentation

C3_SCHEMA = "inferswarm.phase1.c3-correctness/1"
C3_RTOL = 2e-3
C3_ATOL = 2e-3
C3_TOKEN_WINDOW = 64
FROZEN_WARMUPS = 2
V2_ARTIFACT_SHA256 = "2f62bb84df40d4cc5649e940a39cb53d2975eadecbc320fb97d2b037d4e005f4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generation_state(snapshot: dict[str, Any], expected_tokens: int) -> dict[str, Any]:
    diagnostics = snapshot.get("inferswarm_correctness_diagnostics") or {}
    if diagnostics.get("enabled") is not True:
        raise RuntimeError("server did not enable InferSwarm correctness diagnostics")
    if diagnostics.get("truncated") or diagnostics.get("overflow_requests") != 0:
        raise RuntimeError(f"C3 correctness diagnostics overflowed: {diagnostics}")
    records = diagnostics.get("records") or []
    if len(records) != 1:
        raise RuntimeError(
            f"C3 reset-delimited window requires exactly one request record, got {len(records)}"
        )
    record = records[0]
    token_ids = record.get("generated_token_ids")
    step0 = record.get("step0") or {}
    if not isinstance(token_ids, list) or len(token_ids) != expected_tokens:
        raise RuntimeError(
            f"C3 generated-token count {len(token_ids) if isinstance(token_ids, list) else None} "
            f"!= frozen {expected_tokens}"
        )
    if step0.get("available") is not True or not isinstance(
        step0.get("full_logits"), list
    ):
        raise RuntimeError("C3 step-0 full logits are unavailable")
    if len(step0["full_logits"]) != step0.get("vocab_size"):
        raise RuntimeError(
            "C3 full-logit vector length disagrees with recorded vocabulary"
        )
    if int(token_ids[0]) != int(step0.get("argmax")):
        raise RuntimeError(
            "frozen greedy protocol token 0 disagrees with the recorded step-0 argmax"
        )
    return record


def _first_divergence(candidate: list[int], reference: list[int]) -> int | None:
    divergence = next(
        (
            index
            for index, pair in enumerate(zip(candidate, reference, strict=False))
            if pair[0] != pair[1]
        ),
        None,
    )
    if divergence is None and len(candidate) != len(reference):
        return min(len(candidate), len(reference))
    return divergence


def score_c3(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    candidate_tokens = [int(value) for value in candidate["generated_token_ids"]]
    reference_tokens = [int(value) for value in reference["generated_token_ids"]]
    candidate_step0, reference_step0 = candidate["step0"], reference["step0"]
    candidate_logits = torch.tensor(candidate_step0["full_logits"], dtype=torch.float32)
    reference_logits = torch.tensor(reference_step0["full_logits"], dtype=torch.float32)
    shapes_equal = candidate_logits.shape == reference_logits.shape
    if shapes_equal:
        difference = (candidate_logits - reference_logits).abs()
        close = torch.isclose(
            candidate_logits, reference_logits, rtol=C3_RTOL, atol=C3_ATOL
        )
        logits_close = bool(close.all().item())
        max_absolute = float(difference.max().item())
        nonzero = reference_logits.abs() > 0
        relative = torch.where(
            nonzero,
            difference / reference_logits.abs(),
            torch.where(difference == 0, 0.0, float("inf")),
        )
        max_relative = float(relative.max().item())
        nan_count = int(torch.isnan(candidate_logits).sum().item())
        inf_count = int(torch.isinf(candidate_logits).sum().item())
    else:
        logits_close = False
        max_absolute = max_relative = None
        nan_count = inf_count = None

    first64_equal = (
        len(candidate_tokens) >= C3_TOKEN_WINDOW
        and len(reference_tokens) >= C3_TOKEN_WINDOW
        and candidate_tokens[:C3_TOKEN_WINDOW] == reference_tokens[:C3_TOKEN_WINDOW]
    )
    argmax_equal = candidate_step0["argmax"] == reference_step0["argmax"]
    top5_equal = candidate_step0["top5_order"] == reference_step0["top5_order"]
    passed = (
        first64_equal
        and argmax_equal
        and top5_equal
        and logits_close
        and nan_count == 0
        and inf_count == 0
    )
    return {
        "evaluated": True,
        "passed": passed,
        "token_gate": {
            "window": C3_TOKEN_WINDOW,
            "first64_equal": first64_equal,
            "candidate_token_count": len(candidate_tokens),
            "reference_token_count": len(reference_tokens),
            "first_divergence_token_index": _first_divergence(
                candidate_tokens, reference_tokens
            ),
            "beyond_token_64_divergence_is_diagnostic_only": True,
        },
        "step0": {
            "argmax_equal": argmax_equal,
            "candidate_argmax": candidate_step0["argmax"],
            "reference_argmax": reference_step0["argmax"],
            "top5_order_equal": top5_equal,
            "candidate_top5_order": candidate_step0["top5_order"],
            "reference_top5_order": reference_step0["top5_order"],
            "full_logits_shape_equal": shapes_equal,
            "full_logits_within_tolerance": logits_close,
            "rtol": C3_RTOL,
            "atol": C3_ATOL,
            "max_absolute_deviation": max_absolute,
            "max_relative_deviation": max_relative,
            "candidate_nan_count": nan_count,
            "candidate_inf_count": inf_count,
        },
    }


def _generate_and_capture(
    *, origin: str, body: dict[str, Any], output_tokens: int, timeout: float
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    observation = _greedy_generation(origin, body, timeout=timeout)
    observation.pop("_output_text")
    if observation["completion_tokens"] != output_tokens:
        raise RuntimeError(
            f"completion length {observation['completion_tokens']} != frozen {output_tokens}"
        )
    snapshot = _instrumentation(origin, "snapshot", timeout)
    runtime = {
        "resident_bank": snapshot.get("inferswarm_resident_bank"),
        "remote_decode": snapshot.get("inferswarm_remote_decode"),
    }
    return observation, _generation_state(snapshot, output_tokens), runtime


def _validate_role_runtime(runtime: dict[str, Any], role: str) -> None:
    resident = runtime.get("resident_bank") or {}
    remote = runtime.get("remote_decode") or {}
    if role == "reference":
        if remote.get("enabled"):
            raise RuntimeError(
                "C3 correctness reference unexpectedly enabled remote decode"
            )
        return
    artifact = resident.get("artifact") or {}
    if artifact.get("sha256") != V2_ARTIFACT_SHA256:
        raise RuntimeError(
            "C3 candidate did not load the exact frozen v2 artifact: "
            f"{artifact.get('sha256')!r}"
        )
    if artifact.get("policy") != "phase1-qwen36-placement-v2":
        raise RuntimeError("C3 candidate runtime did not report the v2 policy ID")
    if remote.get("enabled") is not True or remote.get("overlap_active") is not True:
        raise RuntimeError("C3 candidate did not use the frozen P4 overlap mechanism")
    if remote.get("placement_sha256") != V2_ARTIFACT_SHA256:
        raise RuntimeError("C3 remote executor placement SHA disagrees with v2")


def _warm_and_reset(
    *, origin: str, body: dict[str, Any], output_tokens: int, timeout: float
) -> list[dict[str, Any]]:
    warmups = []
    for _ in range(FROZEN_WARMUPS):
        observation = _greedy_generation(origin, body, timeout=timeout)
        observation.pop("_output_text")
        if observation["completion_tokens"] != output_tokens:
            raise RuntimeError("C3 warmup did not use the frozen fixed output length")
        warmups.append(observation)
    _instrumentation(origin, "reset", timeout)
    return warmups


def run_reference(
    *, origin: str, manifest_path: str, model_id: str, timeout: float
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, canonical=True)
    results = []
    for class_id in REQUIRED_CLASSES:
        workload = manifest.by_class()[class_id]
        body = workload.greedy_reference_body(model_id)
        warmups = _warm_and_reset(
            origin=origin,
            body=body,
            output_tokens=workload.output_tokens,
            timeout=timeout,
        )
        first_observation, first, first_runtime = _generate_and_capture(
            origin=origin,
            body=body,
            output_tokens=workload.output_tokens,
            timeout=timeout,
        )
        _instrumentation(origin, "reset", timeout)
        second_observation, second, second_runtime = _generate_and_capture(
            origin=origin,
            body=body,
            output_tokens=workload.output_tokens,
            timeout=timeout,
        )
        _validate_role_runtime(first_runtime, "reference")
        _validate_role_runtime(second_runtime, "reference")
        token_equal = first["generated_token_ids"] == second["generated_token_ids"]
        results.append(
            {
                "class_id": class_id,
                "warmups": warmups,
                "reference_observations": [first_observation, second_observation],
                "self_consistency": {
                    "passed": token_equal,
                    "exact_generated_token_sequences_equal": token_equal,
                    "first_divergence_token_index": _first_divergence(
                        first["generated_token_ids"], second["generated_token_ids"]
                    ),
                },
                "generation_state": first,
                "second_generation_state": second,
                "runtime": first_runtime,
            }
        )
    return {
        "schema": C3_SCHEMA,
        "evidence_label": "MEASURED CORRECTNESS REFERENCE (no performance fields)",
        "role": "reference",
        "manifest": manifest.record(),
        "protocol": {
            "warmups": FROZEN_WARMUPS,
            "greedy": True,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "ignore_eos": True,
            "fixed_length": True,
            "token_source": "scheduler generation state; never re-tokenized text",
            "step0_source": "actual model logits before sampling",
        },
        "all_reference_self_consistency_passed": all(
            row["self_consistency"]["passed"] for row in results
        ),
        "performance_fields_collected": False,
        "classes": results,
    }


def run_candidate(
    *,
    origin: str,
    manifest_path: str,
    model_id: str,
    timeout: float,
    reference_path: Path,
) -> dict[str, Any]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("schema") != C3_SCHEMA or reference.get("role") != "reference":
        raise RuntimeError("C3 reference evidence has the wrong schema or role")
    if reference.get("all_reference_self_consistency_passed") is not True:
        raise RuntimeError("C3 reference self-consistency precondition did not pass")
    reference_by_class = {row["class_id"]: row for row in reference["classes"]}
    if set(reference_by_class) != set(REQUIRED_CLASSES):
        raise RuntimeError("C3 reference evidence does not contain exactly W1-W4")

    manifest = load_manifest(manifest_path, canonical=True)
    results = []
    for class_id in REQUIRED_CLASSES:
        workload = manifest.by_class()[class_id]
        body = workload.greedy_reference_body(model_id)
        warmups = _warm_and_reset(
            origin=origin,
            body=body,
            output_tokens=workload.output_tokens,
            timeout=timeout,
        )
        observation, candidate, runtime = _generate_and_capture(
            origin=origin,
            body=body,
            output_tokens=workload.output_tokens,
            timeout=timeout,
        )
        _validate_role_runtime(runtime, "candidate")
        c3 = score_c3(candidate, reference_by_class[class_id]["generation_state"])
        results.append(
            {
                "class_id": class_id,
                "warmups": warmups,
                "candidate_observation": observation,
                "candidate_generation_state": candidate,
                "runtime": runtime,
                "C3": c3,
            }
        )
    return {
        "schema": C3_SCHEMA,
        "evidence_label": "MEASURED DISTRIBUTED C3 CORRECTNESS (no performance fields)",
        "role": "candidate",
        "manifest": manifest.record(),
        "reference_evidence_sha256": _sha256(reference_path),
        "reference_self_consistency_precondition": "passed",
        "protocol": reference["protocol"],
        "C3_passed": all(row["C3"]["passed"] for row in results),
        "performance_fields_collected": False,
        "performance_fields": {
            "tokens_per_second": None,
            "ttft": None,
            "prefill_speed_ratio": None,
            "aggregate_speedup": None,
            "phase1_verdict": None,
        },
        "classes": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--role", choices=("reference", "candidate"), required=True)
    parser.add_argument("--reference-evidence")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.role == "candidate" and not args.reference_evidence:
        parser.error("--role candidate requires --reference-evidence")
    if args.role == "reference" and args.reference_evidence:
        parser.error("--reference-evidence is only valid for --role candidate")

    common = {
        "origin": args.origin.rstrip("/"),
        "manifest_path": args.manifest,
        "model_id": args.model_id,
        "timeout": args.timeout,
    }
    if args.role == "reference":
        document = run_reference(**common)
    else:
        document = run_candidate(
            **common, reference_path=Path(args.reference_evidence).resolve()
        )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))
    return (
        0
        if (
            document.get("all_reference_self_consistency_passed") is True
            or document.get("C3_passed") is True
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
