"""R4 network characterization: direct peer measurements before the campaign.

Ordinary tools only: ping, iperf3, ethtool, ip, /proc/net/netstat.  Raw
command outputs plus machine-readable summaries are retained.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

IPERF3 = "/usr/bin/iperf3"
ETHTOOL = "/usr/sbin/ethtool"


def _run(command: list[str], timeout: int = 120) -> str:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT, timeout=timeout
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return f"<failed: {exc}>"


def _tcp_retransmits(interface: str) -> dict[str, int]:
    counters = {}
    try:
        for table in ("netstat", "snmp"):
            text = _run(["cat", "/proc/net/" + table])
        text = _run(["cat", "/proc/net/netstat"])
        lines = text.splitlines()
        for i in range(0, len(lines) - 1, 2):
            header = lines[i].split()
            values = lines[i + 1].split()
            if header and header[0] == "TcpExt:":
                pairs = dict(zip(header[1:], values[1:], strict=False))
                for key in ("TCPLostRetransmit", "TCPRetransFail", "TCPTimeouts"):
                    if key in pairs:
                        counters[key] = int(pairs[key])
    except Exception:  # noqa: BLE001
        pass
    nic = _run([ETHTOOL, "-S", interface])
    for line in nic.splitlines():
        match = re.match(r"\s*(\w*(?:retrans|err|drop)\w*):\s*(\d+)", line, re.I)
        if match:
            counters["nic_" + match.group(1)] = int(match.group(2))
    return counters


def characterize(
    *,
    peer_ipv4: str,
    interface: str,
    out_dir: Path,
    iperf_seconds: int = 10,
    iperf_omit: int = 2,
    iperf_port: int = 5201,
    as_server: bool = False,
) -> dict[str, Any]:
    """One node's half of the characterization.  iperf3 runs client/server
    on both nodes are orchestrated by the campaign runner over SSH."""

    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema": "inferswarm.r4.network-characterization/1",
        "captured_at_unix": time.time(),
        "peer": peer_ipv4,
        "interface": interface,
    }
    ping_out = _run(["ping", "-c", "100", "-i", "0.05", peer_ipv4], timeout=60)
    (out_dir / "ping-100.txt").write_text(ping_out)
    rtts = [float(m.group(1)) for m in re.finditer(r"time=([\d.]+) ms", ping_out)]
    result["idle_rtt_ms"] = {
        "samples": len(rtts),
        "min": min(rtts) if rtts else None,
        "p50": sorted(rtts)[len(rtts) // 2] if rtts else None,
        "p95": sorted(rtts)[int(len(rtts) * 0.95)] if rtts else None,
        "max": max(rtts) if rtts else None,
        "loss_summary": next(
            (l.strip() for l in ping_out.splitlines() if "packet loss" in l), ""
        ),
    }
    result["counters_before"] = _tcp_retransmits(interface)
    result["link_settings"] = _run([ETHTOOL, interface])
    result["link_driver"] = _run([ETHTOOL, "-i", interface])
    result["route_to_peer"] = _run(["ip", "route", "get", peer_ipv4])
    result["sockets"] = _run(["ss", "-tinp", "dst", peer_ipv4])
    return result


def iperf_client(
    peer_ipv4: str,
    *,
    reverse: bool,
    bidirectional: bool,
    seconds: int,
    out_path: Path,
    port: int = 5201,
) -> dict[str, Any]:
    command = [
        IPERF3,
        "-c",
        peer_ipv4,
        "-t",
        str(seconds),
        "-O",
        "2",
        "-P",
        "1",
        "-p",
        str(port),
        "--json",
    ]
    if reverse:
        command.append("-R")
    if bidirectional:
        command.append("--bidir")
    output = _run(command, timeout=seconds * 4)
    out_path.write_text(output)
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return {"error": "iperf3 json parse failed", "raw_ref": out_path.name}
    summary: dict[str, Any] = {}
    endpoints = []
    if "client" in payload or "end" in payload:
        endpoints.append(("forward", payload.get("end")))
    if payload.get("server_output_json"):
        server = payload["server_output_json"]
        if "end" in server:
            endpoints.append(("server_sum", server["end"]))
    for label, end in endpoints:
        if isinstance(end, dict) and "sum_sent" in end:
            summary[label + "_mbps"] = round(
                end["sum_sent"]["bits_per_second"] / 1e6, 2
            )
        elif isinstance(end, dict) and "sum_received" in end:
            summary[label + "_mbps"] = round(
                end["sum_received"]["bits_per_second"] / 1e6, 2
            )
        elif isinstance(end, dict) and "sum" in end:
            summary[label + "_mbps"] = round(end["sum"]["bits_per_second"] / 1e6, 2)
    retrans = payload.get("end", {})
    if isinstance(retrans, dict) and "sum_sent" in retrans:
        summary["retransmits"] = retrans["sum_sent"].get("retransmits")
    summary["raw_ref"] = out_path.name
    return summary


def iperf_server_start(port: int = 5201):
    process = subprocess.Popen(
        [IPERF3, "-s", "-p", str(port), "--one-off"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)
    return process


__all__ = [
    "characterize",
    "iperf_client",
    "iperf_server_start",
]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peer", required=True)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = characterize(peer_ipv4=args.peer, interface=args.interface, out_dir=args.out_dir)
    from freetoken.research.n0_model_block import write_json_with_sha

    write_json_with_sha(args.out_dir / "characterization.json", result)
    print(json.dumps({"rtt_p50_ms": result["idle_rtt_ms"]["p50"], "out": str(args.out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
