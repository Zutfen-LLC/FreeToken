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
            out: Dict[str, Any] = {
                "value": parts[idx + 1],
                "source": "huggingface snapshot directory name",
            }
            # The cache repo directory is `models--<org>--<name>` one level above
            # `snapshots/`. Reading it costs nothing, downloads nothing, and is an
            # independent check that this checkout is the declared repository and not a
            # same-shaped path from another model.
            if idx >= 1 and parts[idx - 1].startswith("models--"):
                out["repository"] = parts[idx - 1][len("models--"):].replace("--", "/")
                out["repository_source"] = "huggingface cache directory name"
            else:
                out["repository"] = None
                out["repository_unavailable"] = (
                    "the snapshot's parent is not a models--<org>--<name> cache directory, "
                    "so the repository cannot be cross-checked from the filesystem"
                )
            return out
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


def validate_inferswarm_commit(commit: str | None, *, canonical: bool) -> None:
    """Reject an InferSwarm commit that is not a full 40-hex SHA, for a canonical run.

    A non-empty string is not a provenance record: "main", "phase0" or a short SHA all name
    something that moves or is ambiguous, and the benchmark contract requires the exact
    InferSwarm commit a result belongs to.
    """
    if not canonical:
        return
    if not commit:
        raise ValueError(
            "--inferswarm-commit is required for a canonical run: this repository cannot "
            "know which InferSwarm commit the campaign belongs to (BENCHMARKING.md, "
            "required provenance)"
        )
    if not _HEX40.match(commit.strip().lower()):
        raise ValueError(
            f"--inferswarm-commit {commit!r} is not a 40-hex commit SHA. Branch names, tags "
            "and abbreviated SHAs are not commits -- pass the full one. Never invent one."
        )


def check_snapshot_revision(pin: ModelPin) -> str | None:
    """Reconcile the declared revision/repository with the local HF snapshot layout.

    Returns the reason a canonical run must be refused, or None. When the path is not a
    Hugging Face snapshot the answer is None with an explicit "cannot cross-check" recorded
    in ``model.snapshot_identity`` -- an unverifiable path is not the same as a contradicted
    one, and guessing either way would be worse than saying so.
    """
    identity = _snapshot_identity(pin.local_path)
    snapshot = identity.get("value")
    if not snapshot:
        return None
    if pin.revision and snapshot.strip().lower() != pin.revision.strip().lower():
        return (
            f"the local checkpoint resolves to snapshot {snapshot}, which disagrees with "
            f"--model-revision {pin.revision}. One of the two is wrong; a Phase-0 record "
            "cannot be reproduced from a revision the checkpoint does not have."
        )
    repository = identity.get("repository")
    if repository and pin.repository and repository.lower() != pin.repository.strip().lower():
        return (
            f"the local checkpoint lives under the Hugging Face cache entry for "
            f"{repository!r}, which disagrees with --model-repository {pin.repository!r}"
        )
    return None


def check_clean_working_tree(commit_block: Dict[str, Any]) -> str | None:
    """Refuse a canonical measurement from a modified FreeToken checkout.

    A dirty tree cannot be reproduced from its commit SHA, and recording the modified
    filenames does not make it reproducible -- the *contents* are what changed and they are
    nowhere in the artifact. Returns the refusal reason, or None when the tree is clean.
    """
    if not isinstance(commit_block, dict) or not commit_block.get("dirty"):
        return None
    paths = commit_block.get("dirty_paths") or []
    listed = "; ".join(str(p) for p in paths[:20]) or "(git reported no paths)"
    more = f" (+{len(paths) - 20} more)" if len(paths) > 20 else ""
    return (
        "the FreeToken working tree is dirty, so this measurement cannot be reproduced from "
        f"commit {commit_block.get('value')}. Modified paths: {listed}{more}. Commit or stash "
        "them, or run with --dev-smoke (which produces a NON-CANONICAL record)."
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
            expert_quant if expert_quant is not None
            else unavailable(
                "only knowable once an engine has loaded the banks; backfilled after the "
                "campaign as model_expert_quant_resolved, and recorded per arm under "
                "resolved_configuration[<arm>].instrumentation.runtime_config.model"
            )
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


# ``index`` first: without it a numeric --gpu selector cannot be correlated with a row, so
# "which of these cards ran the benchmark" stays unanswerable from the record alone.
_SMI_FIELDS = (
    "index", "uuid", "name", "memory.total", "driver_version", "compute_cap",
    "pcie.link.gen.current", "pcie.link.gen.max",
    "pcie.link.width.current", "pcie.link.width.max",
)


def _selects(selector: str | None, entry: Dict[str, str], resolved_uuid: str | None) -> bool:
    """Whether this nvidia-smi row is the card the campaign declared.

    A resolved UUID wins outright. Failing that, a selector is compared against both the row's
    UUID (as a prefix, the form ``nvidia-smi -L`` accepts) and its index -- and the index is
    only in the row because ``_SMI_FIELDS`` now asks for it.
    """
    if resolved_uuid and entry.get("uuid"):
        return entry["uuid"].upper() == resolved_uuid.upper()
    if not selector:
        return False
    spec = selector.strip()
    uuid = entry.get("uuid", "")
    return bool(
        (uuid and spec.upper().startswith("GPU-") and uuid.upper().startswith(spec.upper()))
        or spec == entry.get("index", "")
    )


def gpu_provenance(
    gpu_selector: str | None = None, resolved_uuid: str | None = None
) -> Dict[str, Any]:
    """Every visible GPU's stable identity and PCIe link state, plus the topology matrix.

    ``gpu_selector`` is the ``ft serve --gpu`` value (a UUID or an nvidia-smi index) and
    ``resolved_uuid`` is what ``gpu.resolve_gpu`` turned it into. The matching entry is
    marked ``selected`` so a multi-GPU box cannot leave the reader guessing which card the
    run used -- but note that a *selected* row is a statement about the flag, not proof of
    execution. The proof is ``gpu.verify_engine_gpu``, which compares the resolved UUID with
    the identity the running engine reports for itself.
    """
    query = _run(
        ["nvidia-smi", f"--query-gpu={','.join(_SMI_FIELDS)}", "--format=csv,noheader"]
    )
    selected = (
        {"requested": gpu_selector, "resolved_uuid": resolved_uuid}
        if gpu_selector
        else unavailable("--gpu not supplied; the runtime chose")
    )
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
        entry["selected"] = _selects(gpu_selector, entry, resolved_uuid)
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
