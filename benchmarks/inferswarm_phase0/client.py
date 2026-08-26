"""Server lifecycle and one measured generation, over the real serving path.

Same measurement path as ``benchmarks/bench_decode_moe.py`` -- spawn ``ft serve``, wait for
``/health``, stream ``/v1/chat/completions`` and timestamp every SSE event -- so the numbers
include the scheduler, detokenizer and HTTP/SSE hop, i.e. what a client actually sees. This
module adds only what the Phase-0 protocol needs: full per-token timing retention, the
``/v1/instrumentation`` prefill read, and a VRAM observation per generation.

**Prefill throughput is not derived from TTFT here, and must not be.** TTFT is
first-token wall-clock from the client: it contains request transport, tokenization, chat
template rendering, queueing, the prefill forward, sampling, detokenization and the SSE
hop. Dividing prompt tokens by it would attribute all of that to prefill. The prefill
number this module reports comes from ``/v1/instrumentation``, which is CUDA-event elapsed
time around the prefill model forward(s) only (see ``python/freetoken/engine/engine.py``
``forward_batch`` and ``python/freetoken/scheduler/scheduler.py`` ``_accumulate_prefill``).
When the server was not started with ``FREETOKEN_INSTRUMENT_PREFILL=1`` the field is an
explicit null with a reason -- never a TTFT-derived substitute.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence


class ServerError(RuntimeError):
    """The server could not be started, or died. Carries a log tail for the artifact."""

    def __init__(self, message: str, log_tail: str = "") -> None:
        super().__init__(message)
        self.log_tail = log_tail


class GenerationError(RuntimeError):
    """One generation failed. Recorded as a failure; never silently skipped."""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_json(url: str, timeout: float = 10.0) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    origin: str
    log_path: str
    command: Sequence[str]
    env_overrides: Dict[str, str] = field(default_factory=dict)

    def log_tail(self, lines: int = 40) -> str:
        try:
            return "".join(Path(self.log_path).read_text(errors="replace").splitlines(keepends=True)[-lines:])
        except OSError:
            return ""


def _pump(src, log_f, echo: bool) -> None:
    for chunk in iter(lambda: src.read1(65536), b""):
        log_f.write(chunk)
        log_f.flush()
        if echo:
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()


def start_server(
    command: Sequence[str],
    origin: str,
    log_path: str,
    *,
    env_overrides: Dict[str, str] | None = None,
    ready_timeout: float = 1800.0,
    echo: bool = True,
) -> ServerHandle:
    """Spawn ``ft serve`` and block until it reports ``maintenance == "serving"``.

    "Fully ready before measurement" is the first line of the criteria's section-10
    protocol, so readiness is a hard gate: a timeout raises rather than proceeding to
    measure a still-loading server.
    """
    env = dict(os.environ)
    env.update(env_overrides or {})
    log_f = open(log_path, "wb")
    proc = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    threading.Thread(target=_pump, args=(proc.stdout, log_f, echo), daemon=True).start()
    handle = ServerHandle(proc, origin, log_path, list(command), dict(env_overrides or {}))
    deadline = time.monotonic() + ready_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise ServerError(
                f"server exited with code {proc.returncode} during startup", handle.log_tail()
            )
        try:
            health = get_json(f"{origin}/health", timeout=5)
        except (OSError, ValueError):
            time.sleep(1.0)
            continue
        if health.get("status") == "error":
            raise ServerError(f"server reported startup error: {health}", handle.log_tail())
        if health.get("maintenance") == "serving":
            return handle
        time.sleep(1.0)
    raise ServerError(f"server not ready after {ready_timeout:.0f}s", handle.log_tail())


def stop_server(handle: ServerHandle) -> None:
    """SIGTERM the whole process group, escalate, then let the driver reclaim VRAM.

    Best-effort by design: it runs in ``finally`` and must not mask the real error. killpg
    runs even when the frontend already exited -- a crashed frontend leaves live non-daemon
    workers in the group, and they hold the GPU.
    """
    for sig, wait_s in ((signal.SIGTERM, 90), (signal.SIGKILL, 30)):
        try:
            os.killpg(handle.proc.pid, sig)
        except ProcessLookupError:
            pass
        try:
            handle.proc.wait(timeout=wait_s)
            break
        except subprocess.TimeoutExpired:
            continue
    time.sleep(3)


def fetch_instrumentation(origin: str, limit: int = 8) -> Dict[str, Any]:
    """``/v1/instrumentation``: resolved engine configuration + measured prefill log.

    A server predating this endpoint returns 404; that is recorded as unavailable rather
    than crashing a campaign, because the endpoint is provenance, not measurement.
    """
    try:
        return get_json(f"{origin}/v1/instrumentation?limit={limit}", timeout=15)
    except urllib.error.HTTPError as e:
        return {"unavailable": f"HTTP {e.code} from /v1/instrumentation (server too old?)"}
    except (OSError, ValueError) as e:
        return {"unavailable": f"could not read /v1/instrumentation: {e!r}"}


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile over an already-sorted list (n is small; exactness beats
    interpolation subtleties, and the raw values are preserved anyway)."""
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def stream_generation(
    origin: str, body: Dict[str, Any], *, timeout: float = 3600.0
) -> Dict[str, Any]:
    """One streamed chat completion, with every token-event arrival timestamped."""
    req = urllib.request.Request(
        f"{origin}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    stamps: List[float] = []
    pieces: List[str] = []
    usage: Dict[str, Any] | None = None
    response_id: str | None = None
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise GenerationError(f"HTTP {e.code}: {e.read()[:500]!r}") from e
    except OSError as e:
        raise GenerationError(f"request failed: {e!r}") from e
    # Iterate the SSE stream as bytes; json.loads decodes UTF-8 itself. (A text-mode reader
    # keyed off the content-type would decode latin-1: the server sends ensure_ascii=False
    # JSON with no charset on text/event-stream.)
    with resp:
        for raw in resp:
            line = raw.strip()
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:"):].strip()
            if payload == b"[DONE]":
                break
            now = time.perf_counter()
            chunk = json.loads(payload)
            if response_id is None and chunk.get("id"):
                response_id = str(chunk["id"])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                text = delta.get("reasoning_content") or delta.get("content")
                if text:
                    stamps.append(now)
                    pieces.append(text)
    if usage is None:
        raise GenerationError("stream ended without a usage chunk; is this a FreeToken server?")
    return {"t0": t0, "t_end": time.perf_counter(), "stamps": stamps,
            "text": "".join(pieces), "usage": usage,
            "response_id": response_id, "request_uid": request_uid(response_id)}


def request_uid(response_id: str | None) -> int | None:
    """The scheduler uid behind an OpenAI response id, or None when it is not one.

    ``/v1/chat/completions`` ids are ``chatcmpl-<uid>`` where ``uid`` is the frontend's own
    request id (``python/freetoken/server/openai_api.py``), and the instrumentation prefill
    log stamps each record with the same ``uid`` (``StatsTracker.record``). That is enough
    request identity to attribute a prefill record to *this* generation instead of inferring
    it from "newest record above a sequence floor" -- and it needs no API change. Returns
    None on any other id shape, and the caller then falls back to the sequence rule.
    """
    if not response_id:
        return None
    _, _, tail = response_id.rpartition("-")
    try:
        return int(tail)
    except ValueError:
        return None


def measure_generation(
    origin: str,
    body: Dict[str, Any],
    *,
    prefill_seq_floor: int,
    store_text: bool,
    text_preview_chars: int = 400,
    timeout: float = 3600.0,
) -> Dict[str, Any]:
    """One generation plus every metric the Phase-0 record needs.

    ``prefill_seq_floor`` is the ``observed`` counter read from ``/v1/instrumentation``
    *before* this generation. The prefill record is accepted only if its sequence number is
    strictly above that floor, which is what stops a stale record from being reported as
    this generation's -- a silent way to fabricate a measurement.

    Decode window: at batch 1 the server emits one delta event per decode step and the
    final usage chunk reports exact token counts, so

        decode_tok_s = (completion_tokens - 1) / (t_last_event - t_first_event)

    stays correct even when the detokenizer coalesces a few tokens into one event
    (multibyte characters): the window is still anchored on the first and last token's
    arrival.
    """
    result = stream_generation(origin, body, timeout=timeout)
    stamps, usage = result["stamps"], result["usage"]
    if len(stamps) < 2:
        raise GenerationError(f"need >=2 token events to measure decode, got {len(stamps)}")

    completion = int(usage["completion_tokens"])
    prompt_tokens = int(usage["prompt_tokens"])
    steps = completion - 1
    decode_window_s = stamps[-1] - stamps[0]
    # Raw inter-event gaps, kept in arrival order and in full: p50/p95/max are derived from
    # them here for convenience, but the list is what a later bootstrap needs.
    gaps_ms = [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:])]
    ordered = sorted(gaps_ms)
    text = result["text"]

    record: Dict[str, Any] = {
        "ttft_ms": (stamps[0] - result["t0"]) * 1e3,
        "wall_total_ms": (result["t_end"] - result["t0"]) * 1e3,
        "decode_window_s": decode_window_s,
        "decode_steps": steps,
        "decode_tok_s": steps / decode_window_s if decode_window_s > 0 else None,
        "ms_per_token_mean": (decode_window_s / steps * 1e3) if steps > 0 else None,
        "inter_token_ms": gaps_ms,
        "inter_token_ms_p50": _percentile(ordered, 0.50),
        "inter_token_ms_p95": _percentile(ordered, 0.95),
        "inter_token_ms_max": ordered[-1] if ordered else None,
        "token_events": len(stamps),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "requested_max_tokens": body.get("max_tokens"),
        "completion_matches_request": completion == body.get("max_tokens"),
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "output_chars": len(text),
        "output_preview": text[:text_preview_chars],
        "output_text": text if store_text else None,
        "output_text_stored": bool(store_text),
        "response_id": result.get("response_id"),
        "request_uid": result.get("request_uid"),
        # Named so nobody reaches for the obvious wrong division; see the module docstring.
        "prefill_tps_from_ttft_deliberately_absent": (
            "TTFT includes transport, tokenization, template rendering, queueing, sampling "
            "and the SSE hop; prompt_tokens/TTFT is not prefill throughput"
        ),
    }
    record.update(
        _observe_after(
            origin, prefill_seq_floor, prompt_tokens, result.get("request_uid")
        )
    )
    return record


# Stable prefill status codes, mirrored in validity.py. A canonical run requires
# ``PREFILL_OK``; every other value is a campaign-invalidating condition, because a Phase-0
# record without prefill throughput is not the record the criteria asked for.
PREFILL_OK = "ok"
PREFILL_UNAVAILABLE = "instrumentation_unavailable"
PREFILL_DISABLED = "instrumentation_disabled"
PREFILL_MISSING = "no_fresh_record"
PREFILL_AMBIGUOUS = "ambiguous_records"
PREFILL_SHARED_BATCH = "shared_batch"
PREFILL_UNUSABLE = "unusable_timing"


def _prefill_status(code: str, reason: str, **extra: Any) -> Dict[str, Any]:
    return {"ok": code == PREFILL_OK, "code": code, "reason": reason or None, **extra}


def select_prefill_record(
    prefill_block: Dict[str, Any], prefill_seq_floor: int, request_uid: int | None
) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
    """Pick THE prefill record belonging to this generation, or say why there isn't one.

    Two attribution routes, preferred in this order:

    ``uid``
        The response id carries the frontend's request uid and each prefill record is
        stamped with the same uid, so the record is matched by request identity. Exact, and
        immune to a concurrent request landing in the log between the floor read and here.

    ``sequence``
        Fallback when the id shape is unrecognized: records newer than the floor read before
        this generation. Here ``len(fresh) != 1`` is **not** resolved by taking the newest --
        "probably this one" is how a measurement gets attributed to another request. It is
        reported as ambiguous, which invalidates a canonical block.

    Returns ``(record | None, status)``. The status carries a stable code either way.
    """
    records = [r for r in prefill_block.get("records", []) if isinstance(r, dict)]
    if request_uid is not None:
        matched = [r for r in records if r.get("uid") == request_uid]
        if len(matched) == 1:
            return dict(matched[0]), _prefill_status(
                PREFILL_OK, "", attribution="uid", request_uid=request_uid
            )
        if not matched:
            return None, _prefill_status(
                PREFILL_MISSING,
                f"no prefill record carries this request's uid {request_uid}; refusing to "
                "attribute another request's measurement to it",
                attribution="uid",
                request_uid=request_uid,
            )
        return None, _prefill_status(
            PREFILL_AMBIGUOUS,
            f"{len(matched)} prefill records carry uid {request_uid}; the interval cannot be "
            "attributed unambiguously",
            attribution="uid",
            request_uid=request_uid,
            candidates=len(matched),
        )

    fresh = [r for r in records if int(r.get("seq", 0)) > prefill_seq_floor]
    if not fresh:
        return None, _prefill_status(
            PREFILL_MISSING,
            f"instrumentation is enabled but no prefill record newer than seq "
            f"{prefill_seq_floor} appeared; refusing to attribute a stale record",
            attribution="sequence",
        )
    if len(fresh) > 1:
        return None, _prefill_status(
            PREFILL_AMBIGUOUS,
            f"{len(fresh)} prefill records appeared above seq {prefill_seq_floor} and the "
            "response id carried no usable request uid; selecting the newest would be a "
            "guess, so no record is attributed",
            attribution="sequence",
            candidates=len(fresh),
        )
    return dict(fresh[0]), _prefill_status(PREFILL_OK, "", attribution="sequence")


def _observe_after(
    origin: str, prefill_seq_floor: int, prompt_tokens: int, request_uid: int | None = None
) -> Dict[str, Any]:
    """VRAM and prefill observations taken right after a generation completes."""
    out: Dict[str, Any] = {}
    try:
        stats = get_json(f"{origin}/v1/stats", timeout=10)
        out["vram_bytes"] = stats.get("vram_bytes")
        out["kv"] = stats.get("kv")
    except (OSError, ValueError) as e:
        out["vram_bytes"] = None
        out["vram_unavailable"] = f"/v1/stats read failed: {e!r}"

    instr = fetch_instrumentation(origin)
    prefill_block = instr.get("prefill") if isinstance(instr, dict) else None
    if not isinstance(prefill_block, dict):
        reason = (
            instr.get("unavailable", "no prefill block in /v1/instrumentation")
            if isinstance(instr, dict) else "no instrumentation document"
        )
        out["prefill"] = None
        out["prefill_unavailable"] = reason
        out["prefill_status"] = _prefill_status(PREFILL_UNAVAILABLE, reason)
        return out
    if not prefill_block.get("enabled"):
        reason = (
            "server was not started with FREETOKEN_INSTRUMENT_PREFILL=1; prefill was not "
            "measured, and TTFT is not a substitute"
        )
        out["prefill"] = None
        out["prefill_unavailable"] = reason
        out["prefill_status"] = _prefill_status(PREFILL_DISABLED, reason)
        return out

    rec, status = select_prefill_record(prefill_block, prefill_seq_floor, request_uid)
    if rec is None:
        out["prefill"] = None
        # "stale" is named explicitly: a missing fresh record is exactly the case where a
        # previous generation's record would otherwise be reported as this one's.
        out["prefill_unavailable"] = status["reason"]
        out["prefill_status"] = status
        return out

    gpu_ms = float(rec.get("gpu_ms") or 0.0)
    new_tokens = int(rec.get("new_tokens") or 0)
    if rec.get("shared_batch"):
        reason = (
            "the prefill batch carried more than one request; the interval belongs to the "
            "batch and cannot be attributed to this request"
        )
        rec["prefill_tok_s"] = None
        rec["prefill_tok_s_unavailable"] = reason
        status = _prefill_status(PREFILL_SHARED_BATCH, reason, **{
            k: v for k, v in status.items() if k in ("attribution", "request_uid")
        })
    elif gpu_ms > 0 and new_tokens > 0:
        rec["prefill_tok_s"] = new_tokens / (gpu_ms / 1e3)
    else:
        reason = f"gpu_ms={gpu_ms} new_tokens={new_tokens}; nothing to divide"
        rec["prefill_tok_s"] = None
        rec["prefill_tok_s_unavailable"] = reason
        status = _prefill_status(PREFILL_UNUSABLE, reason, **{
            k: v for k, v in status.items() if k in ("attribution", "request_uid")
        })
    rec["prompt_tokens_reported_by_usage"] = prompt_tokens
    rec["measurement_boundary"] = prefill_block.get("measurement")
    out["prefill"] = rec
    out["prefill_status"] = status
    return out


def prefill_seq_floor(origin: str) -> int:
    """The instrumentation prefill counter before a generation, used to reject stale records."""
    instr = fetch_instrumentation(origin, limit=1)
    block = instr.get("prefill") if isinstance(instr, dict) else None
    if isinstance(block, dict):
        try:
            return int(block.get("observed") or 0)
        except (TypeError, ValueError):
            return 0
    return 0
