"""Research-internal remote realization/execution wire for the InferSwarm
external-Coordinator proof (inferswarm #67).

This module is deliberately model-opaque and control-plane-only: it frames and
validates JSON records between a CPU-only Coordinator and Node-local execution
agents.  It knows nothing about any model, strategy, block, CUDA, or GPU noun.
Backend-specific realization stays behind the Node-agent boundary; the
Coordinator speaks Execution Plan / session / epoch / operation semantics only.

Transport: ordinary TCP, one request/response pair per frame pair, explicit
length framing.  Research-internal only: this is not a public Node-agent API
and not a production wire protocol.

Wire format (all integers little-endian, struct format ``<4sHI``):

    magic      4 bytes   b"ISXC"  (InferSwarm eXternal Coordinator)
    version    u16       WIRE_VERSION (1)
    body_len   u32       length of the UTF-8 JSON body bytes

    body       body_len bytes  canonical JSON (sorted keys)

Fail-closed rules: unknown magic/version, body over budget, malformed JSON,
unknown message kind, unknown operation, missing or inconsistent identity
fields, declared checksum mismatch, negative or oversized counts, and any
mid-frame disconnect all raise :class:`XCWireError`.  Partial socket
reads/writes are handled by looping.  There are no retries: a failed
correctness-bearing exchange tears the connection down.
"""

from __future__ import annotations

import hashlib
import json
import socket
import struct
from collections.abc import Mapping
from typing import Any

WIRE_MAGIC = b"ISXC"
WIRE_VERSION = 1
HEADER_STRUCT = struct.Struct("<4sHI")
BODY_BUDGET = 24 * 1024 * 1024  # 24 MiB: realized observations/reports, no tensors
FRAME_KINDS = frozenset({"hello", "request", "response"})

OPERATIONS = frozenset(
    {
        "REALIZE",  # frozen Execution Plan -> realized observation
        "GENERATE",  # authorized execution operation
        "REPORT",  # runtime report
        "CLOSE",  # retire + reclaim the epoch runtime
    }
)

# Every request/response carries this immutable identity.  The Node agent
# rejects a request whose identity disagrees with its currently authorized
# realization; the Coordinator rejects a result that does not match its
# current session/epoch/plan/operation position.
IDENTITY_FIELDS = (
    "protocol",  # wire protocol id + version
    "scope_id",  # Swarm/test-scope identity
    "session_id",  # Coordinator session identity (null for REALIZE/CLOSE)
    "epoch_id",  # Coordinator-authorized epoch/generation
    "plan_digest",  # frozen Execution Plan digest
    "operation",  # operation identity
    "position",  # expected execution/commit position where applicable
)

PROTOCOL_ID = "inferswarm.external-coordinator.realization-wire/1"


class XCWireError(RuntimeError):
    """Any protocol violation; the connection must be torn down."""


def canonical_body(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def body_checksum(value: Mapping[str, Any]) -> str:
    """Strategy-semantic payload digest over the canonical record body."""
    return "sha256:" + hashlib.sha256(canonical_body(value)).hexdigest()


def encode_frame(body: Mapping[str, Any]) -> bytes:
    """Encode one validated frame; returns the exact wire bytes."""
    kind = body.get("kind")
    if kind not in FRAME_KINDS:
        raise XCWireError(f"unknown frame kind {kind!r}")
    data = canonical_body(body)
    if len(data) > BODY_BUDGET:
        raise XCWireError("body exceeds budget")
    return HEADER_STRUCT.pack(WIRE_MAGIC, WIRE_VERSION, len(data)) + data


def read_exact(sock: socket.socket, count: int) -> bytearray:
    """Read exactly ``count`` bytes, looping over partial reads."""
    if count < 0:
        raise XCWireError("negative read count")
    buffer = bytearray(count)
    view = memoryview(buffer)
    filled = 0
    while filled < count:
        try:
            chunk = sock.recv_into(view[filled:], count - filled)
        except (ConnectionResetError, BrokenPipeError, OSError) as exc:
            raise XCWireError(f"disconnect during read: {exc}") from exc
        if chunk == 0:
            raise XCWireError(f"disconnect mid-frame after {filled}/{count} bytes")
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
            raise XCWireError(f"disconnect during send: {exc}") from exc
        if written <= 0:
            raise XCWireError("socket send made no progress")
        sent += written


def recv_frame(sock: socket.socket) -> dict[str, Any]:
    """Receive, decode, and structurally validate one frame. Fail-closed."""
    header = read_exact(sock, HEADER_STRUCT.size)
    magic, version, body_len = HEADER_STRUCT.unpack(bytes(header))
    if magic != WIRE_MAGIC:
        raise XCWireError(f"unknown wire magic {bytes(magic)!r}")
    if version != WIRE_VERSION:
        raise XCWireError(f"unsupported wire version {version}")
    if body_len > BODY_BUDGET:
        raise XCWireError("declared body length exceeds budget")
    raw = bytes(read_exact(sock, body_len))
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise XCWireError(f"malformed JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise XCWireError("frame body is not a JSON object")
    if body.get("kind") not in FRAME_KINDS:
        raise XCWireError(f"unknown frame kind {body.get('kind')!r}")
    return body


def _require_int(body: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = body.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise XCWireError(f"{key} must be an integer")
    if value < minimum:
        raise XCWireError(f"{key} must be >= {minimum}")
    return value


def _require_str(body: Mapping[str, Any], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value:
        raise XCWireError(f"{key} must be a non-empty string")
    return value


def validate_request(body: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed structural validation of a Coordinator->Node request.

    Returns the identity record the Node agent must bind the realization to.
    Semantic authorization (does this epoch/plan digest match what this agent
    realized?) is the agent's job; this only proves the record is internally
    consistent and complete.
    """
    if body.get("kind") != "request":
        raise XCWireError("expected a request frame")
    operation = _require_str(body, "operation")
    if operation not in OPERATIONS:
        raise XCWireError(f"unknown operation {operation!r}")
    identity: dict[str, Any] = {
        "protocol": _require_str(body, "protocol"),
        "scope_id": _require_str(body, "scope_id"),
        "epoch_id": _require_str(body, "epoch_id"),
        "plan_digest": _require_str(body, "plan_digest"),
        "operation": operation,
    }
    if identity["protocol"] != PROTOCOL_ID:
        raise XCWireError(f"unsupported protocol id {identity['protocol']!r}")
    for key in ("session_id", "position", "max_new_tokens"):
        if key in body and body[key] is not None:
            _require_int(body, key)
    if operation == "GENERATE":
        # A correctness-bearing execution request must identify its session
        # and expected execution inputs exactly.
        _require_int(body, "session_id", minimum=1)
        _require_int(body, "max_new_tokens", minimum=1)
        prompt = body.get("prompt_token_ids")
        if (
            not isinstance(prompt, list)
            or not prompt
            or any(
                isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in prompt
            )
        ):
            raise XCWireError("prompt_token_ids must be a non-empty list of ints")
        identity["session_id"] = int(body["session_id"])
        position = body.get("position")
        identity["position"] = None if position is None else int(position)
    return identity


def validate_response(body: Mapping[str, Any]) -> None:
    """Fail-closed structural validation of a Node->Coordinator response."""
    if body.get("kind") != "response":
        raise XCWireError("expected a response frame")
    if not isinstance(body.get("ok"), bool):
        raise XCWireError("response lacks a boolean ok")
    if body["ok"]:
        for key in ("protocol", "epoch_id", "plan_digest", "operation"):
            _require_str(body, key)
    else:
        _require_str(body, "error")
    if "result" in body and not body["ok"]:
        raise XCWireError("failed response must not carry a result")
    if body.get("ok") and body.get("operation") == "GENERATE":
        tokens = body["result"].get("generated_token_ids")
        if not isinstance(tokens, list) or not tokens:
            raise XCWireError("GENERATE result lacks generated_token_ids")
        if body.get("result_checksum") is not None:
            expected = body["result_checksum"]
            actual = body_checksum(body["result"])
            if expected != actual:
                raise XCWireError("GENERATE result checksum mismatch")


def identity_of(body: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the comparable identity fields from any frame."""
    return {
        "protocol": body.get("protocol"),
        "scope_id": body.get("scope_id"),
        "session_id": body.get("session_id"),
        "epoch_id": body.get("epoch_id"),
        "plan_digest": body.get("plan_digest"),
        "operation": body.get("operation"),
        "position": body.get("position"),
    }


__all__ = [
    "BODY_BUDGET",
    "FRAME_KINDS",
    "IDENTITY_FIELDS",
    "OPERATIONS",
    "PROTOCOL_ID",
    "WIRE_MAGIC",
    "WIRE_VERSION",
    "XCWireError",
    "body_checksum",
    "canonical_body",
    "encode_frame",
    "identity_of",
    "recv_frame",
    "send_exact",
    "validate_request",
    "validate_response",
]
