"""Hardware/network freeze capture and drift rejection for R4.

Runs locally on each Node; captures the retained freeze profile required by
InferSwarm issue #57 and mechanically rejects drift against the frozen plan
inputs before canonical execution.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from freetoken.research.n0_model_block import write_json_with_sha

DEFAULT_ETHTOOL = "/usr/sbin/ethtool"
NODE_A_LAN_IPV4 = "10.0.0.141"
NODE_B_LAN_IPV4 = "10.0.0.219"
GPU_FIELDS = (
    "index,uuid,name,memory.total,pci.bus_id,pcie.link.gen.current,"
    "pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,driver_version"
)


def _run(command: list[str], ethtool_path: str = DEFAULT_ETHTOOL) -> str:
    replaced = [
        ethtool_path if token == "ethtool" else token for token in command
    ]
    try:
        return subprocess.check_output(replaced, text=True, stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return f"<failed: {exc}>"


def _logical_cpu_count() -> int:
    """Count online logical CPUs from sysfs (handles ranges like 0-7)."""

    total = 0
    for part in Path("/sys/devices/system/cpu/online").read_text().strip().split(","):
        if "-" in part:
            lo, hi = part.split("-")
            total += int(hi) - int(lo) + 1
        elif part.strip():
            total += 1
    return total


def _cpu_topology() -> dict[str, Any]:
    """Correct CPU topology: physical cores deduplicated across threads.

    Uses (physical_id, core_id) pairs from /proc/cpuinfo so hyperthreads do
    not inflate the physical count (the previous `grep -c '^core id'`
    implementation reported threads as cores and a broken logical count).
    Falls back to lscpu parsing when cpuinfo lacks topology fields.
    """

    model = None
    pairs: set[tuple[str, str]] = set()
    processor_count = 0
    current: dict[str, str] = {}
    try:
        lines = Path("/proc/cpuinfo").read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        if not line.strip():
            if current:
                processor_count += 1
                if "physical id" in current and "core id" in current:
                    pairs.add((current["physical id"], current["core id"]))
            current = {}
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "model name":
            model = value.strip()
        elif key in ("physical id", "core id"):
            current[key] = value.strip()
    if current:
        processor_count += 1
        if "physical id" in current and "core id" in current:
            pairs.add((current["physical id"], current["core id"]))
    physical = len(pairs) if pairs else None
    logical = _logical_cpu_count() or processor_count or None
    if physical is None:
        # lscpu fallback
        output = _run(["lscpu"])
        if not output.startswith("<failed"):
            sockets = cores = None
            for line in output.splitlines():
                if line.startswith("Socket(s):"):
                    sockets = int(line.split(":")[1])
                elif line.startswith("Core(s) per socket:"):
                    cores = int(line.split(":")[1])
                elif line.startswith("CPU(s):") and physical is None:
                    pass
            if sockets and cores:
                physical = sockets * cores
    return {
        "model": model,
        "physical_cores": physical,
        "logical_cpus": logical,
    }


DMI_MEMORY_DEVICE_SIZE = re.compile(r"^\s*Size:\s*(\d+)\s*([MG]B)\s*$", re.MULTILINE)


def _physical_installed_ram_bytes() -> int | None:
    """Physically installed RAM from DMI (dmidecode), not /proc/meminfo.

    Linux MemTotal excludes firmware/hardware reservations (a 16 GiB machine
    commonly reports ~15.48 GiB), so issue #57's 16 GiB installed-RAM
    requirement must be proven from DMI memory-device sizes.  Uses the
    absolute path pinned in the r4 sudoers entry (90-r4-dmidecode).
    """

    output = _run(["sudo", "-n", "/usr/sbin/dmidecode", "-t", "memory"])
    if output.startswith("<failed"):
        return None
    total = 0
    blocks = output.split("Memory Device")
    for block in blocks[1:]:
        found = False
        for size, unit in DMI_MEMORY_DEVICE_SIZE.findall(block):
            if unit == "GB":
                total += int(size) * 1024**3
            else:
                total += int(size) * 1024**2
            found = True
        # blocks without a Size line (empty sockets) contribute nothing
        _ = found
    return total or None


def _memory() -> dict[str, Any]:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        fields = line.split()
        if fields[0].rstrip(":") in (
            "MemTotal",
            "MemAvailable",
            "SwapTotal",
            "SwapFree",
            "MemFree",
        ):
            info[fields[0].rstrip(":")] = int(fields[1])
    vmstat = {}
    for line in Path("/proc/vmstat").read_text().splitlines():
        fields = line.split()
        if fields[0] in ("pswpin", "pswpout"):
            vmstat[fields[0]] = int(fields[1])
    swap_lines = _run(["cat", "/proc/swaps"])
    return {
        "mem_total_kib": info.get("MemTotal"),
        "mem_available_kib": info.get("MemAvailable"),
        "mem_free_kib": info.get("MemFree"),
        # issue #57 separates physical installed RAM (DMI) from Linux-visible
        # MemTotal (excludes firmware/hardware reservations)
        "physical_installed_ram_bytes": _physical_installed_ram_bytes(),
        "mem_total_bytes": (
            info.get("MemTotal") * 1024 if info.get("MemTotal") is not None else None
        ),
        "mem_available_bytes": (
            info.get("MemAvailable") * 1024
            if info.get("MemAvailable") is not None
            else None
        ),
        "swap_total_kib": info.get("SwapTotal"),
        "swap_free_kib": info.get("SwapFree"),
        "swap_activity": vmstat,
        "swaps": swap_lines.strip(),
        "measured_at_unix": time.time(),
    }


def _gpus() -> list[dict[str, str]]:
    output = _run(["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"])
    records = []
    for line in output.splitlines():
        if line.startswith("<failed"):
            raise RuntimeError(f"nvidia-smi failed: {line}")
        item = dict(
            zip(
                GPU_FIELDS.split(","),
                (value.strip() for value in line.split(",")),
                strict=True,
            )
        )
        item["memory_total_bytes"] = str(int(item["memory.total"]) * 1024 * 1024)
        # normalized BDF key: profiles historically recorded only the raw
        # nvidia-smi `pci.bus_id` while R4 code expects `pci_bus_id`
        item["pci_bus_id"] = item["pci.bus_id"]
        records.append(item)
    return records


def _interface(
    interface: str, peer_ipv4: str, ethtool_path: str = DEFAULT_ETHTOOL
) -> dict[str, Any]:
    return {
        "name": interface,
        "addr": _run(["ip", "-j", "addr", "show", interface]),
        "route_to_peer": _run(["ip", "route", "get", peer_ipv4]),
        "ethtool_settings": _run(["ethtool", interface]),
        "ethtool_driver": _run(["ethtool", "-i", interface]),
        "ethtool_statistics": _run(["ethtool", "-S", interface]),
    }


def _ethtool_field(settings: str, key: str) -> str | None:
    match = re.search(rf"{key}:\s*(.+)", settings)
    return match.group(1).strip() if match else None


def capture_node_profile(
    *,
    node_id: str,
    peer_ipv4: str,
    interface: str,
    model_path: str,
    ethtool_path: str = DEFAULT_ETHTOOL,
) -> dict[str, Any]:
    """Capture one Node's retained freeze profile."""

    profile = {
        "schema": "inferswarm.r4.node-hardware-profile/1",
        "captured_at_unix": time.time(),
        "node_id": node_id,
        "hostname": _run(["hostname"]).strip(),
        "os_release": _run(["cat", "/etc/os-release"]).strip(),
        "kernel": _run(["uname", "-r"]).strip(),
        "cpu": _cpu_topology(),
        "memory": _memory(),
        "gpus": _gpus(),
        "gpu_topology": _run(["nvidia-smi", "topo", "-m"]),
        "nvidia_driver": _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"]).strip(),
        "cuda_runtime": _cuda_runtime(),
        "interface": _interface(interface, peer_ipv4, ethtool_path),
        "model_checkpoint": {
            "path": model_path,
            "revision_dir": Path(model_path).name,
        },
        "freetoken_sha": _git_sha(),
    }
    settings = profile["interface"]["ethtool_settings"]
    profile["link"] = {
        "speed": _ethtool_field(settings, "Speed"),
        "duplex": _ethtool_field(settings, "Duplex"),
        "mtu": _mtu(interface),
    }
    return profile


def _grep_first(path: str, key: str) -> str | None:
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(key):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _mtu(interface: str) -> str | None:
    try:
        return _run(["cat", f"/sys/class/net/{interface}/mtu"]).strip()
    except Exception:  # noqa: BLE001
        return None


def _cuda_runtime() -> str:
    try:
        import torch

        return f"torch={torch.__version__} cuda={torch.version.cuda}"
    except Exception:  # noqa: BLE001
        return "unavailable"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def require_canonical_link(profile: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed proof that this Node's peer-facing link is canonical 1 GbE."""

    link = profile["link"]
    errors = []
    if link.get("speed") != "1000Mb/s":
        errors.append(f"negotiated speed is {link.get('speed')!r}, not 1000Mb/s")
    if link.get("duplex") != "Full":
        errors.append(f"duplex is {link.get('duplex')!r}, not Full")
    if str(link.get("mtu")) != "1500":
        errors.append(f"MTU is {link.get('mtu')!r}, not 1500")
    route = profile["interface"]["route_to_peer"]
    interface = profile["interface"]["name"]
    if f"dev {interface}" not in route:
        errors.append(f"route to peer does not use {interface}: {route.strip()!r}")
    if "tailscale" in route or "lo" in route.split():
        errors.append("route to peer traverses VPN/loopback")
    if errors:
        raise RuntimeError(
            "canonical 1 GbE path not proven:\n- " + "\n- ".join(errors)
        )
    return {
        "speed": link["speed"],
        "duplex": link["duplex"],
        "mtu": int(link["mtu"]),
        "interface": interface,
        "route": route.strip(),
    }


def require_gpu(profile: dict[str, Any], uuid: str) -> dict[str, Any]:
    for gpu in profile["gpus"]:
        if gpu["uuid"] == uuid:
            return gpu
    raise RuntimeError(f"frozen GPU {uuid} absent from node {profile['node_id']}")


def require_vram_headroom(
    gpu: dict[str, Any], required_bytes: int, safety_bytes: int
) -> dict[str, Any]:
    capacity = int(gpu["memory_total_bytes"])
    if capacity - required_bytes < safety_bytes:
        raise RuntimeError(
            f"VRAM headroom failed for {gpu['uuid']}: capacity {capacity}, "
            f"required {required_bytes}, safety {safety_bytes}"
        )
    return {
        "uuid": gpu["uuid"],
        "capacity_bytes": capacity,
        "required_bytes": required_bytes,
        "safety_bytes": safety_bytes,
        "remaining_bytes": capacity - required_bytes,
    }


def require_host_ram_for_block_b(profile: dict[str, Any]) -> dict[str, Any]:
    """Issue #57: >= 16 GiB physical RAM and >= 12 GiB MemAvailable immediately
    before realization on the Block B Node; no swap reliance."""

    memory = profile["memory"]
    total_gib = memory["mem_total_kib"] / 1024 / 1024
    available_gib = memory["mem_available_kib"] / 1024 / 1024
    errors = []
    if total_gib < 16:
        errors.append(f"physical RAM {total_gib:.2f} GiB < 16 GiB")
    if available_gib < 12:
        errors.append(f"MemAvailable {available_gib:.2f} GiB < 12 GiB")
    if errors:
        raise RuntimeError(
            "Block B host-RAM precondition failed:\n- " + "\n- ".join(errors)
        )
    return {
        "mem_total_gib": round(total_gib, 3),
        "mem_available_gib": round(available_gib, 3),
        "measured_at_unix": profile.get("captured_at_unix"),
    }


def verify_checkpoint_revision(model_path: str, revision: str) -> dict[str, Any]:
    if Path(model_path).name != revision:
        raise RuntimeError(
            f"model path {model_path} is not the pinned revision {revision}"
        )
    files = sorted(p.name for p in Path(model_path).iterdir() if p.is_file())
    weights = [name for name in files if name.endswith(".safetensors")]
    if not weights:
        raise RuntimeError("no safetensors weights found in pinned checkpoint dir")
    return {"path": model_path, "revision": revision, "files": files}


def verify_same_producer(sha_a: str, sha_b: str) -> None:
    if sha_a != sha_b or not sha_a or sha_a == "unknown":
        raise RuntimeError(
            f"producer SHA mismatch across nodes: {sha_a!r} vs {sha_b!r}"
        )


__all__ = [
    "capture_node_profile",
    "require_canonical_link",
    "require_gpu",
    "require_host_ram_for_block_b",
    "require_vram_headroom",
    "verify_checkpoint_revision",
    "verify_same_producer",
    "write_json_with_sha",
]
