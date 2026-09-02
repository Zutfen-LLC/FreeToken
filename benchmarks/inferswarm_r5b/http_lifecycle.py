"""Drive a bounded R5B lifecycle through ordinary HTTP plus the local event seam."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import shlex
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from freetoken.research.n0_model_block import write_json_with_sha


def send_event(socket_path: Path, secret: str, payload: dict[str, Any]) -> int:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    envelope = {
        "payload": payload,
        "hmac": hmac.new(secret.encode(), canonical, hashlib.sha256).hexdigest(),
    }
    data = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        channel.sendto(data, str(socket_path))
    finally:
        channel.close()
    return time.perf_counter_ns()


def _wait_tcp(host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"replacement participant did not listen on {host}:{port}")


def run_lifecycle(
    *,
    origin: str,
    body: dict[str, Any],
    event_socket: Path,
    event_secret: str,
    before_loss_command: str | None,
    peer_host: str,
    peer_port: int,
) -> dict[str, Any]:
    payload = json.dumps(body).encode()
    request = urllib.request.Request(
        origin.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events = []
    output_times = []
    content = []
    reasoning = []
    usage = None
    started = time.perf_counter_ns()
    with urllib.request.urlopen(request, timeout=1800) as response:  # noqa: S310
        for raw in response:
            line = raw.decode().strip()
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
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or ""
            thought = delta.get("reasoning_content") or ""
            if not (piece or thought):
                continue
            output_times.append(time.perf_counter_ns())
            content.append(piece)
            reasoning.append(thought)
            count = len(output_times)
            if count == 1:
                event = {
                    "event_id": "r5b-scale-up",
                    "kind": "AVAILABLE",
                    "resource_id": "gpu.node-a.1",
                    "expected_remaining_requests": 64,
                    "resource_stability_confidence": 1.0,
                    "trusted_history_available": True,
                    "inject_late_result": True,
                }
                sent = send_event(event_socket, event_secret, event)
                events.append({"payload": event, "sent_at_ns": sent})
            elif count == 3:
                launch_started = None
                if before_loss_command:
                    launch_started = time.perf_counter_ns()
                    subprocess.run(shlex.split(before_loss_command), check=True)
                    _wait_tcp(peer_host, peer_port, 180)
                event = {
                    "event_id": "r5b-scale-down-loss",
                    "kind": "PARTICIPANT_LOST",
                    "resource_id": "gpu.node-a.1",
                    "expected_remaining_requests": 64,
                    "resource_stability_confidence": 1.0,
                    "trusted_history_available": True,
                }
                sent = send_event(event_socket, event_secret, event)
                events.append(
                    {
                        "payload": event,
                        "replacement_participant_launch_started_at_ns": launch_started,
                        "sent_at_ns": sent,
                    }
                )
            elif count == 5:
                event = {
                    "event_id": "r5b-scale-back-up",
                    "kind": "RETURNED",
                    "resource_id": "gpu.node-a.1",
                    "expected_remaining_requests": 64,
                    "resource_stability_confidence": 1.0,
                    "trusted_history_available": True,
                }
                sent = send_event(event_socket, event_secret, event)
                events.append({"payload": event, "sent_at_ns": sent})
    ended = time.perf_counter_ns()
    gaps = [right - left for left, right in zip(output_times, output_times[1:])]
    return {
        "schema": "inferswarm.r5b.http-lifecycle/1",
        "request_path": "/v1/chat/completions",
        "request_entry_contract": (
            "ordinary FreeToken OpenAI adapter -> GenSpec -> TokenizeMsg -> "
            "epoch-aware controller"
        ),
        "request_body": body,
        "request_started_ns": started,
        "request_ended_ns": ended,
        "request_wall_ns": ended - started,
        "output_event_times_ns": output_times,
        "inter_output_gaps_ns": gaps,
        "maximum_client_visible_inter_token_gap_ns": max(gaps) if gaps else None,
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "usage": usage,
        "resource_events_sent": events,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--event-socket", type=Path, required=True)
    parser.add_argument("--event-secret", required=True)
    parser.add_argument("--before-loss-command")
    parser.add_argument("--peer-host", default="10.0.0.219")
    parser.add_argument("--peer-port", type=int, default=18485)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_lifecycle(
        origin=args.origin,
        body=json.loads(args.body.read_text()),
        event_socket=args.event_socket,
        event_secret=args.event_secret,
        before_loss_command=args.before_loss_command,
        peer_host=args.peer_host,
        peer_port=args.peer_port,
    )
    write_json_with_sha(args.out, result)
    print(json.dumps({"out": str(args.out), "events": len(result["resource_events_sent"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
