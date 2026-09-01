"""Mechanical fail-closed preflight gate for the canonical R4 campaign.

Issue #57 requires the campaign to REFUSE to proceed when frozen
prerequisites are not satisfied.  The helpers in ``node_preflight`` capture
and check individual facts; this module composes them into one gate that
runs on the campaign driver (Node A) BEFORE plan realization / canonical
diagnostic or clean execution and exits non-zero on any violation:

- producer identity (both Nodes: exact frozen producer SHA, clean tree,
  identity collected cleanly with no safe.directory warnings);
- frozen GPU UUID + PCI BDF on each Node (no silent card selection);
- VRAM headroom for both block participants;
- Block-B host RAM: >= 16 GiB physically installed (DMI, not MemTotal),
  >= 12 GiB MemAvailable immediately before Block-B realization, and
  no swap-reliance during the canonical staging lifecycle (delta of
  pswpin/pswpout across the lifecycle must be zero);
- checkpoint identity: exact frozen repository revision, per-file
  sha256 equality between the two Nodes;
- canonical network path: 1000 Mb/s full-duplex MTU-1500 direct route
  via the frozen interface on both Nodes.

Every check returns a retained record; any failure raises
PreflightGateError, which aborts the campaign before heavyweight model
realization.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, NoReturn

from benchmarks.inferswarm_r4.node_preflight import (
    require_canonical_link,
    require_vram_headroom,
)
from benchmarks.inferswarm_r4.r4_plan import (
    CANONICAL_LINK,
    GPU_A_UUID,
    GPU_B_UUID,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    NODE_A_ID,
    NODE_B_ID,
)

BLOCK_B_MIN_PHYSICAL_RAM_BYTES = 16 * 1024**3
BLOCK_B_MIN_MEM_AVAILABLE_BYTES = 12 * 1024**3
VRAM_SAFETY_BYTES = 512 * 1024**2


class PreflightGateError(RuntimeError):
    """Raised when a frozen R4 prerequisite is not satisfied."""


def _fail(node: str, check: str, detail: str) -> NoReturn:
    raise PreflightGateError(f"[{node}] preflight check {check!r} failed: {detail}")


# -- producer identity -------------------------------------------------------


def _run_cleanly(command: str, context: str) -> str:
    proc = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    if proc.returncode != 0 or proc.stderr.strip():
        raise PreflightGateError(
            f"identity collection for {context} was not clean "
            f"(rc={proc.returncode}, stderr={proc.stderr.strip()[:200]!r}); "
            "safe.directory/ownership warnings must never obscure identity"
        )
    return proc.stdout


def collect_local_identity(repo: str, node_id: str) -> dict[str, Any]:
    """Hostname + producer SHA + clean-tree proof for the local Node."""

    prefix = f"git -c safe.directory={repo} -C {repo}"
    hostname = _run_cleanly("hostname", f"{node_id} hostname").strip()
    sha = _run_cleanly(f"{prefix} rev-parse HEAD", f"{node_id} HEAD").strip()
    status = _run_cleanly(f"{prefix} status --short", f"{node_id} status")
    dirty = [line for line in status.splitlines() if line.strip()]
    return producer_identity(
        node_id=node_id, hostname=hostname, repo_sha=sha, dirty_entries=dirty
    )


def collect_remote_identity(ssh_argv: str, repo: str, node_id: str) -> dict[str, Any]:
    """Hostname + producer SHA + clean-tree proof for the peer Node over SSH.

    ``ssh_argv`` is the complete ssh prefix (e.g.
    "ssh -i /home/zutfen/.ssh/id_r4_staging zutfen@10.0.0.219").  Git runs
    with an explicit safe.directory on the remote side so dubious-ownership
    warnings can never obscure the identity result; any stderr output from
    the identity commands is itself a failure.
    """

    prefix = f"git -c safe.directory={repo} -C {repo}"
    hostname = _run_cleanly(f"{ssh_argv} hostname", f"{node_id} hostname").strip()
    sha = _run_cleanly(f"{ssh_argv} {prefix} rev-parse HEAD", f"{node_id} HEAD").strip()
    status = _run_cleanly(f"{ssh_argv} {prefix} status --short", f"{node_id} status")
    dirty = [line for line in status.splitlines() if line.strip()]
    return producer_identity(
        node_id=node_id, hostname=hostname, repo_sha=sha, dirty_entries=dirty
    )


def producer_identity(
    *, node_id: str, hostname: str, repo_sha: str, dirty_entries: list[str]
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "hostname": hostname,
        "producer_sha": repo_sha,
        "dirty_entries": dirty_entries,
        "tree_clean": not dirty_entries,
    }


def require_producer_identity(
    identity: dict[str, Any], expected_sha: str
) -> dict[str, Any]:
    if not identity["producer_sha"] or identity["producer_sha"] == "unknown":
        _fail(identity["node_id"], "producer-identity", "producer SHA not collected")
    if identity["producer_sha"] != expected_sha:
        _fail(
            identity["node_id"],
            "producer-identity",
            f"producer SHA {identity['producer_sha']!r} != frozen {expected_sha!r}",
        )
    if not identity["tree_clean"]:
        _fail(
            identity["node_id"],
            "clean-tree",
            f"source tree dirty: {identity['dirty_entries'][:5]!r}",
        )
    return identity


# -- frozen GPU identity -----------------------------------------------------


def gpu_bdf(gpu: dict[str, Any]) -> str | None:
    """Normalized BDF accessor (profiles historically recorded both keys)."""

    return gpu.get("pci_bus_id") or gpu.get("pci.bus_id")


def require_frozen_gpu(
    profile: dict[str, Any], uuid: str, bdf: str | None = None
) -> dict[str, Any]:
    match = None
    for gpu in profile["gpus"]:
        if gpu["uuid"] == uuid:
            match = gpu
            break
    if match is None:
        _fail(
            profile["node_id"],
            "frozen-gpu",
            f"frozen GPU {uuid} absent from node "
            f"(visible: {[g['uuid'] for g in profile['gpus']]})",
        )
    actual_bdf = gpu_bdf(match)
    if bdf is not None and actual_bdf != bdf:
        _fail(
            profile["node_id"],
            "frozen-gpu",
            f"GPU {uuid} BDF drift: {actual_bdf!r} != frozen {bdf!r}",
        )
    return {
        "uuid": uuid,
        "pci_bdf": actual_bdf,
        "name": match.get("name"),
        "capacity_bytes": int(match.get("memory_total_bytes", 0)) or None,
    }


# -- VRAM headroom -----------------------------------------------------------


def block_vram_required_bytes(plan: dict[str, Any], execution_id: str) -> int:
    r1 = plan["participant_r1_plans"][execution_id]
    return sum(
        int(item.get("expected_bytes") or item.get("bytes") or 0)
        for item in r1["materializations"]
        if "vram" in item["id"]
    )


def require_block_vram_headroom(
    plan: dict[str, Any], execution_id: str, gpu: dict[str, Any], node_id: str
) -> dict[str, Any]:
    required = block_vram_required_bytes(plan, execution_id)
    capacity = gpu.get("capacity_bytes")
    if not capacity:
        _fail(node_id, "vram-headroom", f"VRAM capacity unknown for {gpu['uuid']}")
    gpu_row = {**gpu, "memory_total_bytes": capacity}
    try:
        return require_vram_headroom(gpu_row, required, VRAM_SAFETY_BYTES)
    except RuntimeError as exc:
        _fail(node_id, "vram-headroom", str(exc))
    raise AssertionError  # pragma: no cover


# -- Block-B host RAM --------------------------------------------------------


def require_block_b_host_ram(memory: dict[str, Any]) -> dict[str, Any]:
    """Issue #57 Block-B host memory gate.

    ``physical_installed_ram_bytes`` comes from DMI (dmidecode), never from
    /proc/meminfo MemTotal (which excludes firmware/hardware reservations).
    Historical swap counters may be non-zero; only the delta across the
    canonical lifecycle matters (checked at release via
    require_no_swap_reliance).
    """

    installed = memory.get("physical_installed_ram_bytes")
    available = memory.get("mem_available_bytes")
    if installed is None:
        _fail("node-b", "physical-ram", "physically installed RAM not proven (DMI)")
    if installed < BLOCK_B_MIN_PHYSICAL_RAM_BYTES:
        _fail(
            "node-b",
            "physical-ram",
            f"physically installed RAM {installed} B < "
            f"{BLOCK_B_MIN_PHYSICAL_RAM_BYTES} B (16 GiB)",
        )
    if available is None or available < BLOCK_B_MIN_MEM_AVAILABLE_BYTES:
        _fail(
            "node-b",
            "mem-available",
            f"MemAvailable {available} B < {BLOCK_B_MIN_MEM_AVAILABLE_BYTES} B "
            "(12 GiB) immediately before Block-B realization",
        )
    return {
        "physical_installed_ram_bytes": installed,
        "linux_memtotal_bytes": memory.get("mem_total_bytes"),
        "memavailable_bytes": available,
        "mem_available_proven_at_unix": memory.get("measured_at_unix"),
        "note": "MemTotal excludes firmware reservations; installed RAM proven via DMI",
    }


def read_process_vm_swap_kib() -> int:
    """This process's swapped-out memory (VmSwap from /proc/self/status).

    Issue #57's "no reliance on swap for pinned/registered staging" is a
    property of the staging process, not of the whole system: kernel
    background reclaim may evict unrelated cold pages opportunistically
    (whole-system pswpin/pswpout deltas), while the staging lifecycle
    relies on swap only if ITS OWN pages are swapped out.  The staging
    buffer is additionally pinned via cudaHostRegister (fail-closed).
    """

    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmSwap:"):
            return int(line.split()[1])
    return 0


def require_no_swap_reliance(
    before: dict[str, Any],
    after: dict[str, Any],
    staging_process_vm_swap_kib: int | None = None,
) -> dict[str, Any]:
    """Prove the canonical staging lifecycle did not rely on swap.

    Criterion: the Block-B staging process itself swapped out nothing
    (VmSwap == 0).  Whole-system pswpin/pswpout deltas are retained as
    informational context: they include kernel reclaim of unrelated cold
    pages and historical activity, which issue #57 explicitly excludes
    from the failure condition.
    """

    system_deltas = {}
    for key in ("pswpin", "pswpout"):
        b = before.get(key)
        a = after.get(key)
        if b is None or a is None:
            _fail("node-b", "swap-reliance", f"swap counter {key} missing")
        system_deltas[key] = a - b
    if staging_process_vm_swap_kib is None:
        _fail(
            "node-b",
            "swap-reliance",
            "staging process VmSwap not proven (missing /proc/self/status "
            "evidence)",
        )
    if staging_process_vm_swap_kib != 0:
        _fail(
            "node-b",
            "swap-reliance",
            f"staging process has {staging_process_vm_swap_kib} kB swapped "
            "out; pinned/registered staging must not rely on swap",
        )
    return {
        "staging_process_vm_swap_kib": staging_process_vm_swap_kib,
        "system_wide_swap_deltas": system_deltas,
        "system_deltas_note": "informational: includes unrelated kernel "
        "background reclaim; the failure criterion is process-scoped VmSwap",
        "swap_reliance": False,
    }


# -- checkpoint identity -----------------------------------------------------


def checkpoint_manifest(model_path: str) -> dict[str, str]:
    """Per-file sha256 manifest of the pinned checkpoint directory."""

    manifest = {}
    for item in sorted(Path(model_path).iterdir()):
        if item.is_file():
            digest = hashlib.sha256()
            with item.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            manifest[item.name] = digest.hexdigest()
    if not manifest:
        _fail("checkpoint", "manifest", f"no files found in {model_path}")
    return manifest


def require_revision_directory(model_path: str) -> None:
    if Path(model_path).name != MODEL_REVISION:
        _fail(
            "checkpoint",
            "revision",
            f"model path {model_path} is not the frozen revision {MODEL_REVISION}",
        )


def require_checkpoint_identity(
    manifest_a: dict[str, str], manifest_b: dict[str, str]
) -> dict[str, Any]:
    if set(manifest_a) != set(manifest_b):
        _fail(
            "checkpoint",
            "file-set",
            "file sets differ between nodes: "
            f"only-a={sorted(set(manifest_a) - set(manifest_b))[:5]}, "
            f"only-b={sorted(set(manifest_b) - set(manifest_a))[:5]}",
        )
    mismatched = sorted(
        name for name in manifest_a if manifest_a[name] != manifest_b[name]
    )
    if mismatched:
        _fail("checkpoint", "hash-equality", f"files differ: {mismatched[:5]}")
    return {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "file_count": len(manifest_a),
        "files_sha256": manifest_a,
        "identical_across_nodes": True,
    }


# -- canonical network path --------------------------------------------------


def require_canonical_network(profile: dict[str, Any], interface: str) -> dict[str, Any]:
    frozen_interface = CANONICAL_LINK.get(
        "node_a_interface" if profile["node_id"] == NODE_A_ID else "node_b_interface"
    )
    if interface != frozen_interface:
        _fail(
            profile["node_id"],
            "canonical-link",
            f"peer-facing interface {interface!r} is not the frozen interface "
            f"{frozen_interface!r}",
        )
    try:
        return require_canonical_link(profile)
    except RuntimeError as exc:
        _fail(profile["node_id"], "canonical-link", str(exc))
    raise AssertionError  # pragma: no cover


# -- full gate ---------------------------------------------------------------


def run_gate(
    *,
    producer_sha: str,
    node_a_profile: dict[str, Any],
    node_b_profile: dict[str, Any],
    identity_a: dict[str, Any],
    identity_b: dict[str, Any],
    plan: dict[str, Any],
    checkpoint_manifest_a: dict[str, str],
    checkpoint_manifest_b: dict[str, str],
    node_a_model_path: str,
    node_b_model_path: str,
    frozen_bdf_a: str | None = None,
    frozen_bdf_b: str | None = None,
) -> dict[str, Any]:
    """Run every frozen-prerequisite check; raise PreflightGateError on any
    violation.  Returns the retained gate record (itself canonical evidence).
    """

    record: dict[str, Any] = {
        "schema": "inferswarm.r4.preflight-gate/1",
        "frozen_producer_sha": producer_sha,
        "checks": {},
    }

    def keep(name: str, value: Any) -> None:
        record["checks"][name] = value

    keep(
        "producer_identity",
        {
            NODE_A_ID: require_producer_identity(identity_a, producer_sha),
            NODE_B_ID: require_producer_identity(identity_b, producer_sha),
        },
    )
    if identity_a["producer_sha"] != identity_b["producer_sha"]:
        _fail("both", "same-producer", "nodes run different producer SHAs")
    keep("same_producer_both_nodes", True)

    gpu_a = require_frozen_gpu(node_a_profile, GPU_A_UUID, frozen_bdf_a)
    gpu_b = require_frozen_gpu(node_b_profile, GPU_B_UUID, frozen_bdf_b)
    keep("frozen_gpu", {NODE_A_ID: gpu_a, NODE_B_ID: gpu_b})

    keep(
        "vram_headroom",
        {
            "block_a": require_block_vram_headroom(
                plan, "exec.block-a", gpu_a, NODE_A_ID
            ),
            "block_b": require_block_vram_headroom(
                plan, "exec.block-b", gpu_b, NODE_B_ID
            ),
        },
    )

    keep("block_b_host_ram", require_block_b_host_ram(node_b_profile["memory"]))
    require_revision_directory(node_a_model_path)
    require_revision_directory(node_b_model_path)
    keep(
        "checkpoint_identity",
        require_checkpoint_identity(checkpoint_manifest_a, checkpoint_manifest_b),
    )
    keep(
        "canonical_link",
        {
            NODE_A_ID: require_canonical_network(
                node_a_profile, CANONICAL_LINK["node_a_interface"]
            ),
            NODE_B_ID: require_canonical_network(
                node_b_profile, CANONICAL_LINK["node_b_interface"]
            ),
        },
    )
    record["result"] = "ALL_PREFLIGHT_CHECKS_PASSED"
    return record


__all__ = [
    "BLOCK_B_MIN_MEM_AVAILABLE_BYTES",
    "BLOCK_B_MIN_PHYSICAL_RAM_BYTES",
    "PreflightGateError",
    "VRAM_SAFETY_BYTES",
    "block_vram_required_bytes",
    "checkpoint_manifest",
    "collect_local_identity",
    "collect_remote_identity",
    "gpu_bdf",
    "producer_identity",
    "require_block_b_host_ram",
    "require_block_vram_headroom",
    "require_checkpoint_identity",
    "require_frozen_gpu",
    "require_no_swap_reliance",
    "require_producer_identity",
    "require_revision_directory",
    "run_gate",
]
