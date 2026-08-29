"""Correctness/mechanism-only W1-W4 diagnostic against an already-running server.

The client deliberately records no request timestamps or throughput. For each selected
frozen class it establishes the two-warmup prefix/cache state used by the correctness
reference, resets the idle engine instrumentation window, performs one greedy fixed-length
generation, and snapshots the common timing plus InferSwarm mechanism gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from inferswarm_phase0.manifest import REQUIRED_CLASSES, load_manifest

REFERENCE_HASHES = {
    "W1": "59b9b9dc2cb001576a156e39fa5141d454253e8550babb795567b546e3fa0f84",
    "W2": "1e601a5673bab480a371d8d558912598f28f33cd59efda18ebda61f3cbd467bd",
    "W3": "0102f179f1479573dd11d8bf429e5ddc1869b6c5b0903962aff252ad16519f8e",
    "W4": "02804bb980bd21cf7c3b189512d8ec4b504cfcb809e7abac0032603048f80414",
}


def _post_json(url: str, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from {url}: {exc.read()[:500]!r}") from exc


def _greedy_generation(
    origin: str, body: dict[str, Any], *, timeout: float
) -> dict[str, Any]:
    """Consume FreeToken SSE without collecting any wall-clock measurements."""
    request = urllib.request.Request(
        f"{origin}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    pieces: list[str] = []
    usage: dict[str, Any] | None = None
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"generation HTTP {exc.code}: {exc.read()[:500]!r}") from exc
    with response:
        for raw in response:
            line = raw.strip()
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:") :].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                text = delta.get("reasoning_content") or delta.get("content")
                if text:
                    pieces.append(text)
    if usage is None:
        raise RuntimeError("generation ended without a FreeToken usage chunk")
    text = "".join(pieces)
    return {
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": int(usage["completion_tokens"]),
        "_output_text": text,
    }


def _reference_outputs(path: str) -> dict[str, dict[str, Any]]:
    """Load one retained warmed reference output per class and verify its published hash."""
    outputs: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            class_id = record.get("class_id")
            if (
                class_id in REFERENCE_HASHES
                and class_id not in outputs
                and record.get("phase") == "measured"
                and isinstance(record.get("output_text"), str)
            ):
                observed = hashlib.sha256(
                    record["output_text"].encode("utf-8")
                ).hexdigest()
                if observed != REFERENCE_HASHES[class_id]:
                    raise RuntimeError(
                        f"retained {class_id} reference text hash {observed} does not match "
                        f"published {REFERENCE_HASHES[class_id]}"
                    )
                outputs[class_id] = record
    missing = sorted(set(REFERENCE_HASHES) - set(outputs))
    if missing:
        raise RuntimeError(f"reference JSONL has no warmed output text for {missing}")
    return outputs


def _score_c3(tokenizer, candidate_text: str, reference_text: str) -> dict[str, Any]:
    """Return a text round-trip diagnostic without misrepresenting it as exact C3.

    The OpenAI-compatible serving stream exposes decoded text, not generated token IDs
    or the step-0 logits.  Tokenization is not injective, so re-encoding that text cannot
    prove the frozen exact-token criterion even when the decoded strings agree.
    """
    candidate = tokenizer.encode(candidate_text, add_special_tokens=False)
    reference = tokenizer.encode(reference_text, add_special_tokens=False)
    first_divergence = next(
        (
            index
            for index, pair in enumerate(zip(candidate, reference))
            if pair[0] != pair[1]
        ),
        None,
    )
    if first_divergence is None and len(candidate) != len(reference):
        first_divergence = min(len(candidate), len(reference))
    first64_reencoded_equal = len(candidate) >= 64 and candidate[:64] == reference[:64]
    return {
        "passed": None,
        "evaluated": False,
        "status": "exact_generated_token_ids_and_step0_logits_not_available",
        "gate": "first 64 greedy generated token IDs exactly equal",
        "first64_equal": None,
        "text_reencoding_diagnostic": {
            "first64_reencoded_equal": first64_reencoded_equal,
            "first_divergence_reencoded_token_index": first_divergence,
            "candidate_reencoded_token_count": len(candidate),
            "reference_reencoded_token_count": len(reference),
        },
        "beyond_token_64_divergence_is_diagnostic_only": True,
    }


def _instrumentation(origin: str, operation: str, timeout: float) -> dict[str, Any]:
    response = _post_json(
        f"{origin}/v1/moe/instrumentation",
        {"operation": operation, "timeout": timeout},
        timeout=timeout + 5,
    )
    if response.get("status") != "ok" or "payload" not in response:
        raise RuntimeError(f"MoE instrumentation {operation} failed: {response}")
    return response["payload"]


def evaluate_candidate_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    remote = snapshot["inferswarm_remote_decode"]
    gates = remote["gates"]
    observed = {name: gates[name] for name in ("F1", "F2", "F3", "F5", "F6")}
    failures = [
        name for name, gate in observed.items() if gate.get("passed") is not True
    ]
    aggregate = remote["aggregate"]
    if aggregate["prefill_remote_dispatches"] != 0:
        failures.append("remote_prefill_zero")
    expert_bytes = remote["steady_state_transfer_bytes"]["host_to_gpu1"][
        "expert_weights"
    ]
    if expert_bytes != 0:
        failures.append("steady_state_remote_expert_weights_zero")
    return {
        "passed": not failures,
        "failures": failures,
        "gates": observed,
        "remote_prefill_dispatches": aggregate["prefill_remote_dispatches"],
        "steady_state_expert_weight_bytes_host_to_gpu1": expert_bytes,
    }


def _validate_runtime(snapshot: dict[str, Any], role: str) -> None:
    timing = snapshot["moe_layer_timing"]
    if not timing["enabled"] or timing["role"] != role:
        raise RuntimeError(f"server timing role/configuration mismatch: {timing}")
    if not timing["validity"]["complete_layer_timing_valid"]:
        raise RuntimeError("complete MoE layer timing is invalid")
    remote = snapshot["inferswarm_remote_decode"]
    if role == "candidate":
        if not remote["enabled"] or not remote["overlap_active"]:
            raise RuntimeError("candidate server is not using P4 overlap")
        if timing["graph"]["active"]:
            raise RuntimeError(
                "P4 candidate timing unexpectedly used CUDA graph replay"
            )
    elif role == "baseline":
        if remote["enabled"]:
            raise RuntimeError("B1 timing server unexpectedly enabled remote decode")
        if not timing["graph"]["active"]:
            raise RuntimeError("B1 timing did not use CUDA graph replay")


def run_smoke(
    *,
    origin: str,
    manifest_path: str,
    model_id: str,
    role: str,
    classes: list[str],
    warmups: int,
    timeout: float,
    reference_jsonl: str,
    tokenizer_model: str,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    manifest = load_manifest(manifest_path, canonical=True)
    workloads = manifest.by_class()
    references = _reference_outputs(reference_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, local_files_only=True)
    results: list[dict[str, Any]] = []
    for class_id in classes:
        workload = workloads[class_id]
        body = workload.greedy_reference_body(model_id)
        warmup_records = []
        for _ in range(warmups):
            record = _greedy_generation(origin, body, timeout=timeout)
            record.pop("_output_text")
            warmup_records.append(record)
        _instrumentation(origin, "reset", timeout)
        observation = _greedy_generation(origin, body, timeout=timeout)
        candidate_text = observation.pop("_output_text")
        snapshot = _instrumentation(origin, "snapshot", timeout)
        _validate_runtime(snapshot, role)
        expected_hash = REFERENCE_HASHES[class_id]
        c3 = _score_c3(tokenizer, candidate_text, references[class_id]["output_text"])
        if observation["completion_tokens"] != workload.output_tokens:
            raise RuntimeError(
                f"{class_id} completion length {observation['completion_tokens']} != "
                f"{workload.output_tokens}"
            )
        result = {
            "class_id": class_id,
            "warmups": warmup_records,
            "measurement": observation,
            "reference_output_sha256": expected_hash,
            "whole_output_hash_matches_reference": (
                observation["output_sha256"] == expected_hash
            ),
            "C3": {
                **c3,
                "protocol": "two warmups then greedy fixed length",
                "reference_self_consistency_precondition": "published passed",
                "step0_argmax_equal": None,
                "step0_top5_and_full_logits": "not_collected_by_serving_smoke",
            },
            "timing_enabled": snapshot["moe_layer_timing"]["enabled"],
            "instrumentation": snapshot,
        }
        if role == "candidate":
            result["mechanism_summary"] = evaluate_candidate_snapshot(snapshot)
        results.append(result)
    failures = []
    for result in results:
        if result["C3"]["passed"] is not True:
            failures.append(f"{result['class_id']}:C3")
        mechanism = result.get("mechanism_summary")
        if mechanism is not None:
            failures.extend(
                f"{result['class_id']}:{name}" for name in mechanism["failures"]
            )
    return {
        "schema": "inferswarm.phase1.p4-workload-smoke/1",
        "evidence_label": (
            "MEASURED NONCANONICAL MECHANISM SMOKE"
            if role == "candidate"
            else "MEASURED ISSUE-5 TIMING DIAGNOSTIC"
        ),
        "role": role,
        "manifest": manifest.record(),
        "warmups_per_class": warmups,
        "sampling": {
            "greedy": True,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "ignore_eos": True,
        },
        "performance_fields_collected": False,
        "performance_fields": {
            "tokens_per_second": None,
            "ttft": None,
            "prefill_throughput": None,
            "phase1_verdict": None,
        },
        "all_requested_gates_passed": not failures,
        "gate_failures": failures,
        "classes": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--reference-jsonl", required=True)
    parser.add_argument("--role", choices=("candidate", "baseline"), required=True)
    parser.add_argument(
        "--class", dest="classes", action="append", choices=REQUIRED_CLASSES
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")
    classes = args.classes or list(REQUIRED_CLASSES)
    document = run_smoke(
        origin=args.origin.rstrip("/"),
        manifest_path=args.manifest,
        model_id=args.model_id,
        role=args.role,
        classes=classes,
        warmups=args.warmups,
        timeout=args.timeout,
        reference_jsonl=args.reference_jsonl,
        tokenizer_model=args.tokenizer_model,
    )
    output = Path(args.output)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"P4_WORKLOAD_SMOKE_OUT {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
