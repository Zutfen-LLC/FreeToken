"""R4 transport-only microbenchmark: 8,192 and 524,288 byte service frames.

Measures the framed request/response round trip over the exact persistent
TCP path with no model involvement.  Uses the same wire module and payload
sizes as the real boundary.  Retained separately from model execution.
"""

from __future__ import annotations

import socket
import statistics
import time
from typing import Any

from freetoken.research.r4_wire import (
    WireError,
    encode_frame,
    recv_frame,
    send_exact,
)

WIRE_PROTOCOL_ID = "inferswarm.r4.boundary-wire/1"
PAYLOAD_SIZES = (8192, 524288)
REPETITIONS = 200
WARMUP = 20


def run_transport_client(
    *, peer_host: str, peer_port: int, repetitions: int = REPETITIONS
) -> dict[str, Any]:
    """Client side: send N framed payloads of each size, receive echo acks."""

    sock = socket.create_connection((peer_host, peer_port), timeout=30)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    results: dict[str, Any] = {"tcp_nodelay": True, "sizes": {}}
    total_bytes = 0
    try:
        for size in PAYLOAD_SIZES:
            payload = bytes(i % 251 for i in range(size))
            # warmup
            for _ in range(WARMUP):
                _roundtrip(sock, size, payload, size)
            samples = []
            for _ in range(repetitions):
                started = time.perf_counter_ns()
                _roundtrip(sock, size, payload, repetitions)
                samples.append(time.perf_counter_ns() - started)
                total_bytes += size
            effective_bw = [
                size * 8 / (ns / 1e9) / 1e6 for ns in samples
            ]
            results["sizes"][str(size)] = {
                "repetitions": repetitions,
                "rtt_ns_p50": statistics.median(samples),
                "rtt_ns_p95": sorted(samples)[int(len(samples) * 0.95)],
                "rtt_ns_mean": round(statistics.mean(samples)),
                "effective_mbps_p50": round(statistics.median(effective_bw), 2),
                "payload_bytes_total": repetitions * size,
            }
        results["total_payload_bytes"] = total_bytes
    finally:
        sock.close()
    return results


def _roundtrip(
    sock: socket.socket, size: int, payload: bytes, count: int
) -> None:
    header = {
        "kind": "request",
        "protocol": WIRE_PROTOCOL_ID,
        "experiment_id": "transport-microbenchmark",
        "session_id": 0,
        "op": "TRANSPORT_ECHO",
        "size": size,
        "payload_len": len(payload),
    }
    send_exact(sock, encode_frame(header, payload))
    reply, echo = recv_frame(
        sock, {"protocol": WIRE_PROTOCOL_ID, "session_id": 0}
    )
    if reply.get("op") != "ECHO" or len(echo) != size:
        raise WireError("transport echo mismatch")


def run_transport_server(*, listen_host: str, listen_port: int) -> None:
    """Server side: echo framed payloads back until disconnect."""

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((listen_host, listen_port))
    server.listen(1)
    try:
        conn, _addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            header, payload = recv_frame(conn)
            if header.get("op") == "TRANSPORT_ECHO":
                reply = {
                    "kind": "response",
                    "protocol": WIRE_PROTOCOL_ID,
                    "experiment_id": "transport-microbenchmark",
                    "session_id": 0,
                    "op": "ECHO",
                    "payload_len": len(payload),
                }
                send_exact(conn, encode_frame(reply, bytes(payload)))
            elif header.get("op") == "TRANSPORT_QUIT":
                break
            else:
                raise WireError(f"unsupported microbenchmark op {header.get('op')!r}")
    except WireError:
        pass
    finally:
        server.close()


__all__ = [
    "PAYLOAD_SIZES",
    "REPETITIONS",
    "run_transport_client",
    "run_transport_server",
]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--server", action="store_true")
    mode.add_argument("--client", action="store_true")
    parser.add_argument("--peer", default="10.0.0.219")
    parser.add_argument("--listen-host", default="10.0.0.219")
    parser.add_argument("--port", type=int, default=18490)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.server:
        run_transport_server(listen_host=args.listen_host, listen_port=args.port)
        return 0
    result = run_transport_client(
        peer_host=args.peer, peer_port=args.port, repetitions=args.repetitions
    )
    if args.out:
        from freetoken.research.n0_model_block import write_json_with_sha

        write_json_with_sha(args.out, result)
    print(json.dumps({"sizes": {k: v["rtt_ns_p50"] for k, v in result["sizes"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
