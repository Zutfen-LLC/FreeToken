"""Internal R4 persistent network transport seam (research-only).

This module is deliberately model-opaque: it frames and validates opaque
binary payloads plus a small validated metadata record.  It knows nothing
about any model, strategy, or block noun.  Strategy-specific
meaning stays in the experiment adapter (benchmarks/inferswarm_r4).

Wire format (all integers little-endian, struct format ``<4sHHQ``):

    magic      4 bytes   b"FTR4"
    version    u16       WIRE_VERSION (1)
    header_len u16       length of the JSON header bytes
    reserved   u64       0 (alignment / future flags)

    header     header_len bytes  UTF-8 JSON, canonical (sorted keys)
    payload    N bytes          exact bulk activation bytes (optional)

Fail-closed rules: unknown magic/version, header over budget, session or
plan/experiment identity mismatch, malformed JSON, unknown frame kind,
payload length disagreeing with metadata, checksum mismatch when a
checksum is declared, and any mid-frame disconnect all raise
:class:`WireError`.  Partial socket reads/writes are handled by looping.
"""

from __future__ import annotations

import hashlib
import json
import socket
import struct
from collections.abc import Mapping
from typing import Any

WIRE_MAGIC = b"FTR4"
WIRE_VERSION = 1
HEADER_STRUCT = struct.Struct("<4sHHQ")
HEADER_BUDGET = 4096
PAYLOAD_BUDGET = 1 << 20  # 1 MiB: above the largest R4 boundary payload
FRAME_KINDS = frozenset({"hello", "request", "response"})
IDENTITY_KEYS = ("protocol", "experiment_id", "session_id")
REQUIRED_REQUEST_FIELDS = (
    "kind",
    "operation",
    "position",
    "token_count",
    "dtype",
    "layout",
    "payload_len",
)


class WireError(RuntimeError):
    """Any protocol violation; the connection must be torn down."""


def canonical_header(header: Mapping[str, Any]) -> bytes:
    return json.dumps(header, sort_keys=True, separators=(",", ":")).encode()


def payload_checksum(payload: bytes | bytearray | memoryview) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def encode_frame(header: Mapping[str, Any], payload: bytes = b"") -> bytes:
    """Encode one validated frame; returns the exact wire bytes."""

    kind = header.get("kind")
    if kind not in FRAME_KINDS:
        raise WireError(f"unknown frame kind {kind!r}")
    body = canonical_header(header)
    if len(body) > HEADER_BUDGET:
        raise WireError("header exceeds budget")
    if len(payload) > PAYLOAD_BUDGET:
        raise WireError("payload exceeds budget")
    if int(header.get("payload_len", len(payload))) != len(payload):
        raise WireError("header payload_len disagrees with payload")
    return HEADER_STRUCT.pack(WIRE_MAGIC, WIRE_VERSION, len(body), 0) + body + payload


def read_exact(sock: socket.socket, count: int) -> bytearray:
    """Read exactly ``count`` bytes, looping over partial reads."""

    if count < 0:
        raise WireError("negative read count")
    buffer = bytearray(count)
    view = memoryview(buffer)
    filled = 0
    while filled < count:
        try:
            chunk = sock.recv_into(view[filled:], count - filled)
        except (ConnectionResetError, BrokenPipeError, OSError) as exc:
            raise WireError(f"disconnect during read: {exc}") from exc
        if chunk == 0:
            raise WireError(
                f"disconnect mid-frame after {filled}/{count} bytes"
            )
        filled += chunk
    return buffer


def send_exact(sock: socket.socket, data: bytes | bytearray | memoryview) -> None:
    """Send all bytes, looping over partial writes."""

    view = memoryview(data)
    sent = 0
    total = len(data)
    while sent < total:
        try:
            written = sock.send(view[sent:])
        except (ConnectionResetError, BrokenPipeError, OSError) as exc:
            raise WireError(f"disconnect during send: {exc}") from exc
        if written <= 0:
            raise WireError("socket send made no progress")
        sent += written


def _decode_header(body: bytes | bytearray, identity: Mapping[str, Any] | None) -> dict[str, Any]:
    try:
        header = json.loads(body)
    except json.JSONDecodeError as exc:
        raise WireError(f"malformed header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise WireError("header is not an object")
    if identity is not None:
        for key in IDENTIFY_KEYS_CHECK:
            expected = identity.get(key)
            observed = header.get(key)
            if expected is not None and observed != expected:
                raise WireError(f"{key} mismatch: {observed!r} != {expected!r}")
    return header


IDENTIFY_KEYS_CHECK = IDENTITY_KEYS


def decode_header(
    body: bytes, identity: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return _decode_header(body, identity)


def recv_frame(
    sock: socket.socket, identity: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], bytearray]:
    """Receive one frame; validates structure and session identity.

    ``identity`` binds protocol/experiment/session expectations.  Payload
    semantic validation (dtype/layout/token count/checksum) is deferred to
    :func:`validate_request` / :func:`validate_response` in the adapter,
    which know the frozen boundary contract.
    """

    prefix = read_exact(sock, HEADER_STRUCT.size)
    magic, version, header_len, _reserved = HEADER_STRUCT.unpack(prefix)
    if magic != WIRE_MAGIC:
        raise WireError(f"bad magic {magic!r}")
    if version != WIRE_VERSION:
        raise WireError(f"unsupported wire version {version}")
    if header_len == 0 or header_len > HEADER_BUDGET:
        raise WireError(f"header length out of budget: {header_len}")
    body = read_exact(sock, header_len)
    header = _decode_header(body, identity)
    payload_len = int(header.get("payload_len", 0))
    if payload_len < 0 or payload_len > PAYLOAD_BUDGET:
        raise WireError(f"payload length out of budget: {payload_len}")
    payload = read_exact(sock, payload_len) if payload_len else bytearray()
    return header, payload


def validate_request(
    header: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    checksum: str | None = None,
    payload: bytes | bytearray | memoryview | None = None,
) -> int:
    """Fail-closed validation of one boundary request against the frozen
    boundary contract (dtype/layout/planes/row width/token count/bytes).

    Returns the expected payload byte count.
    """

    for field in REQUIRED_REQUEST_FIELDS:
        if field not in header:
            raise WireError(f"request missing field {field!r}")
    dtype = header["dtype"]
    layout = header["layout"]
    if dtype != contract["dtype"] or layout != contract["layout"]:
        raise WireError(f"dtype/layout mismatch: {dtype}/{layout}")
    token_count = int(header["token_count"])
    if token_count <= 0 or token_count > contract["max_token_count"]:
        raise WireError(f"token count out of bounds: {token_count}")
    expected = (
        token_count
        * contract["planes"]
        * contract["row_width"]
        * contract["element_bytes"]
    )
    if int(header["payload_len"]) != expected:
        raise WireError(
            f"payload length {header['payload_len']} != contract {expected}"
        )
    if payload is not None and len(payload) != expected:
        raise WireError(f"actual payload bytes {len(payload)} != contract {expected}")
    if checksum is not None:
        if "payload_sha256" not in header:
            raise WireError("diagnostic mode requires a payload checksum")
        if header["payload_sha256"] != checksum:
            raise WireError("boundary payload checksum mismatch")
    return expected


__all__ = [
    "FRAME_KINDS",
    "HEADER_BUDGET",
    "HEADER_STRUCT",
    "IDENTITY_KEYS",
    "PAYLOAD_BUDGET",
    "WIRE_MAGIC",
    "WIRE_VERSION",
    "WireError",
    "canonical_header",
    "decode_header",
    "encode_frame",
    "payload_checksum",
    "read_exact",
    "recv_frame",
    "send_exact",
    "validate_request",
]
