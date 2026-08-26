"""Provenance capture for a Phase-0 run.

The rule from InferSwarm's benchmark contract: *"If a field cannot be filled, record why. A
benchmark with silent holes in its provenance is not a benchmark."* Every getter here
therefore returns either a value or ``{"value": None, "unavailable": "<reason>"}``; nothing
is ever silently omitted, and nothing is ever guessed.

Two things are deliberately refused rather than approximated:

* a **model revision** that is not a full 40-hex commit SHA, for a canonical run. Branch
  names and ``main`` are not revisions; the criteria (section 1.1) require the exact
  upstream commit to be pinned *before* the first Phase-0 measurement.
* the **InferSwarm commit**, which this repository cannot know. It is supplied to the
  harness, and a canonical run without it fails to start.
"""

from __future__ import annotations

import functools
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_HEX40 = re.compile(r"^[0-9a-f]{40}$")

# Environment variables worth recording (they change what the runtime does). Anything whose
# name looks like a credential is dropped by name -- values are never inspected or copied.
_ENV_PREFIXES = ("FREETOKEN_", "CUDA_", "NVIDIA_", "PYTORCH_", "TORCH", "TRITON_", "OMP_", "MKL_")
_SENSITIVE_MARKERS = ("SECRET", "PASSWORD", "PASSWD", "API_KEY", "APIKEY", "AUTH", "CREDENTIAL")
_SENSITIVE_EXACT = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"}


def unavailable(reason: str) -> Dict[str, Any]:
    return {"value": None, "unavailable": reason}


def _run(cmd: List[str], timeout: float = 20.0) -> str | None:
    """Best-effort command capture. Returns None when the tool is missing or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def is_sensitive_env(name: str) -> bool:
    upper = name.upper()
    if upper in _SENSITIVE_EXACT:
        return True
    return any(marker in upper for marker in _SENSITIVE_MARKERS)


def relevant_env() -> Dict[str, str]:
    """Runtime-relevant environment variables, credentials excluded by name."""
    return {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith(_ENV_PREFIXES) and not is_sensitive_env(name)
    }


# --------------------------------------------------------------------------------------
# repository / software
# --------------------------------------------------------------------------------------

def git_commit(repo_dir: str | Path) -> Dict[str, Any]:
    repo_dir = str(repo_dir)
    head = _run(["git", "-C", repo_dir, "rev-parse", "HEAD"])
    if head is None:
        return unavailable(f"git rev-parse failed in {repo_dir} (not a git checkout?)")
    status = _run(["git", "-C", repo_dir, "status", "--porcelain"])
    return {
        "value": head,
        "dirty": bool(status),
        "dirty_paths": (status or "").splitlines()[:50],
    }


def freetoken_repo_root() -> Path:
    """This file lives at <repo>/benchmarks/inferswarm_phase0/provenance.py."""
    return Path(__file__).resolve().parents[2]


def software_provenance(inferswarm_commit: str | None, harness_version: str) -> Dict[str, Any]:
    freetoken = git_commit(freetoken_repo_root())
    return {
        "freetoken_commit": freetoken,
        "freetoken_repo_root": str(freetoken_repo_root()),
        "inferswarm_commit": (
            {"value": inferswarm_commit}
            if inferswarm_commit
            else unavailable(
                "not supplied; pass --inferswarm-commit (this repository cannot know it)"
            )
        ),
        "harness_version": harness_version,
        "command_line": list(sys.argv),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": _torch_versions(),
    }


@functools.lru_cache(maxsize=1)
def _torch_versions() -> Dict[str, Any]:
    """torch / CUDA runtime versions, read out of the interpreter that will run the server.

    Cached: it costs a subprocess that imports torch, and it cannot change within a run."""
    code = (
        "import json, torch; "
        "print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, "
        "'cudnn': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}))"
    )
    out = _run([sys.executable, "-c", code], timeout=180)
    if out is None:
        return unavailable("could not import torch in this interpreter")
    try:
        return json.loads(out.splitlines()[-1])
    except (ValueError, IndexError):
        return unavailable(f"unparseable torch version output: {out[:200]!r}")


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPin:
    repository: str
    revision: str
    local_path: str | None


def _snapshot_identity(local_path: str | None) -> Dict[str, Any]:
    """The Hugging Face snapshot directory's own identity, when the path is one.

    A ``.../snapshots/<sha>`` layout names the revision in the path, which is an
    independent check on the declared ``--model-revision``.
    """
    if not local_path:
        return unavailable("no local model path supplied")
    p = Path(local_path)
    if not p.exists():
        return unavailable(f"model path does not exist: {local_path}")
    parts = p.resolve().parts
    if "snapshots" in parts:
        idx = len(parts) - 1 - parts[::-1].index("snapshots")
        if idx + 1 < len(parts):
            return {"value": parts[idx + 1], "source": "huggingface snapshot directory name"}
    return unavailable(
        "path is not a huggingface snapshots/<revision> directory; revision cannot be "
        "cross-checked from the filesystem"
    )


def validate_revision(revision: str | None, *, canonical: bool) -> None:
    """Reject a symbolic/unpinned revision for a canonical run (criteria section 1.1)."""
    if not canonical:
        return
    if not revision:
        raise ValueError(
            "--model-revision is required for a canonical run: no Phase-0 measurement may "
            "begin until the exact upstream revision is recorded (criteria section 1.1)"
        )
    if not _HEX40.match(revision.strip().lower()):
        raise ValueError(
            f"--model-revision {revision!r} is not a 40-hex commit SHA. Branch names, tags "
            "and 'main' are not revisions -- resolve the exact upstream commit and pass it. "
            "Never invent one."
        )


def model_provenance(pin: ModelPin, *, expert_quant: Any = None) -> Dict[str, Any]:
    return {
        "repository": pin.repository,
        "revision": pin.revision if pin.revision else unavailable("not supplied"),
        "revision_is_pinned_sha": bool(pin.revision and _HEX40.match(pin.revision.lower())),
        # The local path is host-specific; it is recorded in the LOCAL run artifact for
        # reproduction on this machine and must not be copied into a committed fixture.
        "local_path": pin.local_path or unavailable("not supplied"),
        "snapshot_identity": _snapshot_identity(pin.local_path),
        # Resolved weight format, taken from the engine's own report when the server is up
        # (the flag text is not the format).
        "expert_quant_resolved": (
            expert_quant if expert_quant is not None else unavailable("server not started yet")
        ),
    }


# --------------------------------------------------------------------------------------
# host / GPU
# --------------------------------------------------------------------------------------

def host_provenance() -> Dict[str, Any]:
    uname = platform.uname()
    return {
        "os": f"{uname.system} {uname.release}",
        "kernel_version": uname.version,
        "machine": uname.machine,
        "hostname_recorded": False,  # deliberately not captured
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": _physical_cores(),
        "ram_total_bytes": _ram_total_bytes(),
        "environment": relevant_env(),
        "environment_note": (
            "names matching credential patterns (HF_TOKEN, *SECRET*, *API_KEY*, ...) are "
            "excluded by name; no value is inspected"
        ),
    }


def _cpu_model() -> Any:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return unavailable("no 'model name' line in /proc/cpuinfo")


def _physical_cores() -> Any:
    out = _run(["lscpu", "-p=Core,Socket"])
    if out is None:
        return unavailable("lscpu unavailable")
    cores = {line for line in out.splitlines() if line and not line.startswith("#")}
    return len(cores) or unavailable("lscpu returned no core rows")


def _ram_total_bytes() -> Any:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return unavailable("could not read MemTotal from /proc/meminfo")


_SMI_FIELDS = (
    "uuid", "name", "memory.total", "driver_version", "compute_cap",
    "pcie.link.gen.current", "pcie.link.gen.max",
    "pcie.link.width.current", "pcie.link.width.max",
)


def gpu_provenance(gpu_selector: str | None = None) -> Dict[str, Any]:
    """Every visible GPU's stable identity and PCIe link state, plus the topology matrix.

    ``gpu_selector`` is the ``ft serve --gpu`` value (a UUID or an nvidia-smi index); the
    matching entry is marked ``selected`` so a multi-GPU box cannot leave the reader
    guessing which card the run used.
    """
    query = _run(
        ["nvidia-smi", f"--query-gpu={','.join(_SMI_FIELDS)}", "--format=csv,noheader"]
    )
    selected = gpu_selector or unavailable("--gpu not supplied; the runtime chose")
    if query is None:
        return {
            "gpus": unavailable("nvidia-smi unavailable or failed"),
            "topology": unavailable("nvidia-smi unavailable or failed"),
            "topology_p2p": unavailable("nvidia-smi unavailable or failed"),
            "selected": selected,
            "nvcc": _run(["nvcc", "--version"]) or unavailable("nvcc not on PATH"),
        }
    gpus: List[Dict[str, Any]] = []
    for line in query.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != len(_SMI_FIELDS):
            continue
        entry = dict(zip(_SMI_FIELDS, cells))
        entry["selected"] = bool(
            gpu_selector and gpu_selector.strip() in (entry["uuid"], entry.get("index", ""))
        )
        gpus.append(entry)
    topo = _run(["nvidia-smi", "topo", "-m"])
    return {
        "gpus": gpus or unavailable("nvidia-smi returned no parseable GPU rows"),
        "topology": topo if topo is not None else unavailable("nvidia-smi topo -m failed"),
        "topology_p2p": _run(["nvidia-smi", "topo", "-p2p", "r"])
        or unavailable("nvidia-smi topo -p2p r failed"),
        "selected": selected,
        "nvcc": _run(["nvcc", "--version"]) or unavailable("nvcc not on PATH"),
    }


# --------------------------------------------------------------------------------------
# completeness gate
# --------------------------------------------------------------------------------------

# Dotted paths into the assembled provenance document that a canonical run must have.
REQUIRED_FOR_CANONICAL = (
    "software.freetoken_commit",
    "software.inferswarm_commit",
    "model.repository",
    "model.revision",
    "host.cpu_model",
    "host.ram_total_bytes",
    "gpu.gpus",
    "gpu.topology",
)


def _lookup(doc: Dict[str, Any], dotted: str) -> Any:
    node: Any = doc
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def missing_required(doc: Dict[str, Any]) -> List[str]:
    """Required provenance paths that are absent or explicitly unavailable."""
    missing = []
    for path in REQUIRED_FOR_CANONICAL:
        value = _lookup(doc, path)
        if value is None:
            missing.append(path)
        elif isinstance(value, dict) and "unavailable" in value and value.get("value") is None:
            missing.append(f"{path} ({value['unavailable']})")
    return missing


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
