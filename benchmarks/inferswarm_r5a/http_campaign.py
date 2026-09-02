"""Drive matched R5A workloads only through FreeToken's ordinary HTTP API."""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from freetoken.research.n0_model_block import write_json_with_sha


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - operator URL
        return json.load(response)


def stream_chat(origin: str, body: dict[str, Any], *, timeout: int = 300) -> dict[str, Any]:
    payload = json.dumps(body).encode()
    request = urllib.request.Request(  # noqa: S310 - operator URL
        origin.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter_ns()
    event_times = []
    content = []
    reasoning = []
    usage = None
    finish_reason = None
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            row = json.loads(data)
            if row.get("usage"):
                usage = row["usage"]
            choices = row.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            piece = delta.get("content") or ""
            thought = delta.get("reasoning_content") or ""
            if piece or thought:
                event_times.append(time.perf_counter_ns())
                content.append(piece)
                reasoning.append(thought)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    ended = time.perf_counter_ns()
    first = event_times[0] if event_times else ended
    intervals = [right - left for left, right in zip(event_times, event_times[1:])]
    completion_tokens = int((usage or {}).get("completion_tokens", 0))
    decode_wall = max(0, ended - first)
    return {
        "request_started_ns": started,
        "first_output_ns": first,
        "request_ended_ns": ended,
        "ttft_ns": first - started,
        "complete_request_wall_ns": ended - started,
        "decode_observation_wall_ns": decode_wall,
        "decode_tok_s": (
            max(0, completion_tokens - 1) / (decode_wall / 1e9)
            if decode_wall and completion_tokens > 1
            else None
        ),
        "output_event_count": len(event_times),
        "output_event_intervals_ns": intervals,
        "output_event_interval_p50_ns": statistics.median(intervals) if intervals else None,
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "usage": usage,
        "finish_reason": finish_reason,
    }


def _body(workload, model: str, max_tokens: int) -> dict[str, Any]:
    value = workload.greedy_reference_body(model)
    value["max_tokens"] = max_tokens
    value["ignore_eos"] = True
    value["stream"] = True
    value["stream_options"] = {"include_usage": True}
    return value


def run_campaign(
    *,
    origin: str,
    manifest_path: Path,
    model: str,
    classes: list[str],
    concurrency: int,
    warmups: int = 0,
    repetitions: int = 1,
) -> dict[str, Any]:
    from inferswarm_phase0.manifest import load_manifest

    if concurrency not in (1, 2):
        raise ValueError("R5A bounded concurrency is frozen to 1 or 2")
    manifest = load_manifest(manifest_path, canonical=True)
    health = _get_json(origin.rstrip("/") + "/health")
    if health.get("status") != "ok":
        raise RuntimeError(f"server is not ready: {health}")
    before = _get_json(origin.rstrip("/") + "/v1/instrumentation?limit=256")
    by_class = manifest.by_class()
    requests = []
    lock = threading.Lock()

    def one(class_id: str, ordinal: int, sample_kind: str, repetition: int) -> None:
        result = stream_chat(origin, _body(by_class[class_id], model, 32))
        result["class_id"] = class_id
        result["request_ordinal"] = ordinal
        result["sample_kind"] = sample_kind
        result["repetition"] = repetition
        with lock:
            requests.append(result)

    if concurrency == 1:
        schedule = []
        for repetition in range(warmups):
            schedule.extend((class_id, "warmup", repetition) for class_id in classes)
        for repetition in range(repetitions):
            schedule.extend((class_id, "measured", repetition) for class_id in classes)
        for ordinal, (class_id, sample_kind, repetition) in enumerate(schedule):
            one(class_id, ordinal, sample_kind, repetition)
    else:
        if warmups != 0 or repetitions != 1:
            raise ValueError("bounded concurrency arm is frozen to zero warmups and one repetition")
        if len(classes) != 2:
            raise ValueError("bounded concurrency arm requires exactly two requests")
        threads = [
            threading.Thread(target=one, args=(class_id, ordinal, "measured", 0))
            for ordinal, class_id in enumerate(classes)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    after = _get_json(origin.rstrip("/") + "/v1/instrumentation?limit=256")
    previous = int(before.get("prefill", {}).get("observed", 0))
    prefills = [
        row
        for row in after.get("prefill", {}).get("records", [])
        if int(row.get("seq", 0)) > previous
    ]
    return {
        "schema": "inferswarm.r5a.http-serving-arm/1",
        "origin": origin,
        "server_instance_id": health.get("instance_id"),
        "model": model,
        "workload_manifest_sha256": manifest.manifest_sha256,
        "classes": classes,
        "bounded_concurrency": concurrency,
        "warmups_per_class": warmups,
        "measured_repetitions_per_class": repetitions,
        "request_path": "/v1/chat/completions",
        "request_entry_contract": "ordinary FreeToken OpenAI serving adapter -> GenSpec -> TokenizeMsg",
        "requests": sorted(requests, key=lambda row: row["request_ordinal"]),
        "instrumentation_before": before,
        "instrumentation_after": after,
        "new_prefill_records": prefills,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--classes", default="W2,W4")
    parser.add_argument("--concurrency", type=int, choices=(1, 2), default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_campaign(
        origin=args.origin,
        manifest_path=args.manifest,
        model=args.model,
        classes=args.classes.split(","),
        concurrency=args.concurrency,
        warmups=args.warmups,
        repetitions=args.repetitions,
    )
    write_json_with_sha(args.out, result)
    print(json.dumps({"out": str(args.out), "requests": len(result["requests"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
