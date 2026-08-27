"""The session-level ``ft bench bw`` prerequisite, and the profile artifact it produces.

Criteria section 2.1 requires B2 to run "after a fresh ``ft bench bw`` profile for this GPU
+ expert format". Two facts make that a **session** prerequisite rather than a B2-local side
effect:

* **B2 consumes it** for its per-decode-step fetch split
  (``Engine._resolve_hybrid_fetch`` -> ``bench_profile.load_hybrid_fetch_fraction``).
* **B3 consumes it too.** ``--moe-backend auto`` reads the same profile to decide whether to
  upgrade the offload default to hybrid (``engine._adjust_config`` ->
  ``bench_profile.load_backend_recommendation``). In a reversed session (``--reverse-order``,
  the section-10 second session) B3 runs *before* B2, so a B2-local refresh would let B3
  consume a stale profile -- or none at all -- and the artifact would not say so.

So the refresh happens once, before the sweep traversal starts, in either direction. What is
recorded is not "we ran the command" but the exact bytes the engine will read: the resolved
profile path, the file's contents, its sha256, and a check that the profile's own ``gpu.uuid``
is the card this campaign declared.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from . import provenance as prov
from .gpu import GpuSelection

# `ft bench bw` prints this line (with FREETOKEN_BENCH_PROGRESS=1) naming the file it wrote.
# Parsing it is exact; the default-path computation below is only the fallback.
_OUT_MARKER = "FTBENCH_OUT "


def _default_profile_path(gpu_uuid: str | None) -> str:
    """Indirection point for tests: FreeToken's own profile-path rule, not a copy of it."""
    from freetoken.moe.bench_profile import default_profile_path

    return default_profile_path(gpu_uuid)


def bench_bw_command(
    python_executable: str, gpu: str | None, dtype: str = "nvfp4"
) -> List[str]:
    """``ft bench bw`` for the selected GPU and expert format (criteria section 2.1)."""
    cmd = [python_executable, "-m", "freetoken.cli", "bench", "bw", "--dtype", dtype]
    if gpu:
        cmd += ["--gpu", gpu]
    return cmd


@dataclass
class BenchBwResult:
    """One ``ft bench bw`` invocation plus the profile artifact it left behind."""

    record: Dict[str, Any]

    @property
    def ok(self) -> bool:
        return bool(self.record.get("ok"))

    @property
    def profile_usable(self) -> bool:
        profile = self.record.get("profile") or {}
        calibration = profile.get("nvfp4_calibration") or {}
        return (
            bool(profile.get("sha256"))
            and profile.get("gpu_matches") is True
            and calibration.get("usable") is True
        )

    @property
    def failure_reason(self) -> str:
        if not self.ok:
            rc = self.record.get("returncode")
            err = self.record.get("error")
            tail = (self.record.get("stderr_tail") or "").strip().splitlines()[-3:]
            detail = err or f"returncode {rc}"
            return f"`ft bench bw` failed ({detail}){'; ' + ' | '.join(tail) if tail else ''}"
        profile = self.record.get("profile") or {}
        if profile.get("unavailable"):
            return str(profile["unavailable"])
        if profile.get("gpu_matches") is False:
            return str(profile.get("gpu_mismatch") or "the profile was benched on another GPU")
        if profile.get("gpu_matches") is not True:
            return str(
                profile.get("gpu_unverified")
                or "the profile could not be positively tied to the selected GPU"
            )
        calibration = profile.get("nvfp4_calibration") or {}
        if calibration.get("usable") is not True:
            return str(
                calibration.get("unavailable")
                or "the profile does not contain a machine-usable NVFP4 calibration"
            )
        return ""


def _parse_out_path(stdout: str) -> str | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_OUT_MARKER):
            path = line[len(_OUT_MARKER):].strip()
            if path:
                return path
    return None


def _load_backend_recommendation(path: str, gpu_name: str | None, gpu_uuid: str | None):
    """Call the same exact-path reader used by Engine._adjust_config."""
    from freetoken.moe.bench_profile import load_backend_recommendation

    return load_backend_recommendation(
        "nvfp4", gpu_name=gpu_name, gpu_uuid=gpu_uuid, path=path
    )


def _load_hybrid_fetch_fraction(path: str, gpu_name: str | None, gpu_uuid: str | None):
    """Call the same exact-path reader used by Engine._resolve_hybrid_fetch."""
    from freetoken.moe.bench_profile import load_hybrid_fetch_fraction

    return load_hybrid_fetch_fraction(
        "nvfp4", gpu_name=gpu_name, gpu_uuid=gpu_uuid, path=path
    )


def is_positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > 0
    )


def validate_nvfp4_calibration(
    path: str, contents: Any, selection: GpuSelection
) -> Dict[str, Any]:
    """Prove the captured profile is usable by the Phase-0 runtime paths.

    ``ft bench bw`` intentionally preserves a profile when an individual kernel benchmark
    fails.  A successful process and readable JSON therefore do not prove that B2 can derive
    its fetch split or that B3 can make an evidence-backed backend choice.  Structural checks
    prove the required CPU-MoE/PCIe measurements exist; the two reader calls below prove the
    exact captured file has the same meaning to the harness that it will have to the engine.
    """
    block: Dict[str, Any] = {"dtype": "nvfp4", "usable": False}
    problems: List[str] = []
    if not isinstance(contents, dict):
        problems.append("profile root is not an object")
        block["unavailable"] = "; ".join(problems)
        return block

    dtypes = contents.get("dtypes")
    if not isinstance(dtypes, dict) or "nvfp4" not in dtypes:
        problems.append('profile has no dtypes["nvfp4"] tuning result')
    elif dtypes.get("nvfp4") not in ("hybrid", "offload"):
        problems.append(
            f'dtypes["nvfp4"] is not a usable backend verdict: {dtypes.get("nvfp4")!r}'
        )

    dtype_kernels = contents.get("dtype_kernels")
    kernels = dtype_kernels.get("nvfp4") if isinstance(dtype_kernels, dict) else None
    if not isinstance(kernels, dict):
        problems.append('profile has no dtype_kernels["nvfp4"] measurements')
    else:
        for field in ("cpu_moe_gbs", "pcie_gather_gbs"):
            value = kernels.get(field)
            if not is_positive_finite(value):
                problems.append(
                    f'dtype_kernels["nvfp4"]["{field}"] must be a positive finite number; '
                    f"got {value!r}"
                )

    gpu = contents.get("gpu")
    gpu_name = gpu.get("name") if isinstance(gpu, dict) else None
    try:
        recommendation = _load_backend_recommendation(
            path, gpu_name, selection.resolved_uuid
        )
    except Exception as e:  # noqa: BLE001 -- reader failure is captured benchmark evidence
        recommendation = None
        block["backend_reader_error"] = repr(e)
    block["backend_recommendation"] = recommendation
    if recommendation not in ("hybrid", "offload"):
        problems.append(
            "load_backend_recommendation(\"nvfp4\") returned no usable recommendation"
        )

    try:
        fraction = _load_hybrid_fetch_fraction(path, gpu_name, selection.resolved_uuid)
    except Exception as e:  # noqa: BLE001 -- reader failure is captured benchmark evidence
        fraction = None
        block["hybrid_fraction_reader_error"] = repr(e)
    block["hybrid_fetch_fraction"] = fraction
    # B2 is explicitly hybrid even when B3's recommendation is offload, so every canonical
    # sweep needs a real split.  Zero is the engine's fixed-cap fallback, not a calibration.
    if not is_positive_finite(fraction) or float(fraction) > 1.0:
        problems.append(
            "load_hybrid_fetch_fraction(\"nvfp4\") did not derive a fraction in (0, 1]"
        )

    if problems:
        block["unavailable"] = "; ".join(problems)
        return block
    block["usable"] = True
    return block


def capture_profile(path: str | None, selection: GpuSelection) -> Dict[str, Any]:
    """Read back the exact profile file the engine will consult, and pin it.

    Records the resolved path, the file's own sha256 (over the raw bytes, so the pin is of
    the bytes and not of a re-serialization), the parsed contents, and whether the profile's
    ``gpu.uuid`` is the card this campaign declared. An unreadable or missing file is an
    explicit reason, never an empty block.
    """
    resolved_path = os.path.expanduser(path) if path else None
    block: Dict[str, Any] = {"path": resolved_path}
    if not path:
        block["unavailable"] = (
            "could not resolve the profile path `ft bench bw` wrote; the exact profile the "
            "engine will read cannot be pinned"
        )
        return block
    try:
        raw = Path(resolved_path).read_bytes()
    except OSError as e:
        block["unavailable"] = f"profile file could not be read: {e!r}"
        return block
    block["sha256"] = hashlib.sha256(raw).hexdigest()
    block["bytes"] = len(raw)
    try:
        contents = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        block["unavailable"] = f"profile file is not readable JSON: {e!r}"
        return block
    block["contents"] = contents
    raw_profile_gpu = contents.get("gpu") if isinstance(contents, dict) else None
    profile_gpu = raw_profile_gpu if isinstance(raw_profile_gpu, dict) else {}
    block["profile_gpu"] = profile_gpu
    want = selection.resolved_uuid
    got = profile_gpu.get("uuid")
    if want is None:
        block["gpu_matches"] = None
        block["gpu_unverified"] = (
            "no resolved GPU UUID for this campaign, so the profile cannot be tied to a card"
        )
    elif not got:
        block["gpu_matches"] = None
        block["gpu_unverified"] = "the profile does not name the GPU it was benched on"
    else:
        block["gpu_matches"] = str(got).upper() == want.upper()
        if not block["gpu_matches"]:
            block["gpu_mismatch"] = (
                f"profile was benched on {got}, but this campaign declared {want}"
            )
    block["nvfp4_calibration"] = validate_nvfp4_calibration(
        resolved_path, contents, selection
    )
    return block


def run_bench_bw(
    *,
    python_executable: str,
    selection: GpuSelection,
    dtype: str = "nvfp4",
    timeout: float = 3600.0,
    env: Dict[str, str] | None = None,
) -> BenchBwResult:
    """Refresh the bandwidth profile the hybrid split and the ``auto`` backend pick read.

    Runs on the *resolved* UUID rather than the raw selector where one is available, so the
    bench, the sweep and the microbenchmark provably touch the same physical card.
    """
    gpu_arg = selection.resolved_uuid or selection.requested
    cmd = bench_bw_command(python_executable, gpu_arg, dtype)
    started = prov.utc_now_iso()
    child_env = dict(os.environ)
    child_env.update(env or {})
    # Makes `ft bench bw` print FTBENCH_OUT <path>: the exact file it wrote, rather than a
    # path this module would otherwise have to re-derive.
    child_env["FREETOKEN_BENCH_PROGRESS"] = "1"
    record: Dict[str, Any] = {
        "command": cmd,
        "dtype": dtype,
        "gpu_selector_used": gpu_arg,
        "gpu_resolved_uuid": selection.resolved_uuid,
        "started_at": started,
    }
    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False, env=child_env
        )
    except (OSError, subprocess.SubprocessError) as e:
        record.update({"finished_at": prov.utc_now_iso(), "ok": False, "error": repr(e)})
        record["profile"] = capture_profile(None, selection)
        return BenchBwResult(record)
    record.update(
        {
            "finished_at": prov.utc_now_iso(),
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-8000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    )
    path = _parse_out_path(completed.stdout)
    record["profile_path_source"] = "FTBENCH_OUT" if path else "default_profile_path"
    if path is None:
        try:
            path = _default_profile_path(selection.resolved_uuid)
        except Exception as e:  # noqa: BLE001 -- a missing import is a recorded fact
            record["profile_path_error"] = repr(e)
            path = None
    record["profile"] = capture_profile(path, selection)
    return BenchBwResult(record)


def skipped_record(reason: str) -> Dict[str, Any]:
    """What goes in the artifact when the refresh was deliberately not run."""
    return {"ok": False, "skipped": True, "reason": reason, "profile": {"unavailable": reason}}


def consuming_arms(arms: Sequence[Any]) -> List[str]:
    """Arm ids that can consume the bandwidth profile, in declaration order.

    B2 resolves its fetch split from it; B3's ``--moe-backend auto`` reads it to decide
    whether to upgrade offload to hybrid. Both must therefore run *after* the refresh,
    whichever order the session traverses them in.
    """
    return [a.id for a in arms if getattr(a, "consumes_bench_bw", False)]
