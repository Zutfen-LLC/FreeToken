"""N1-only process-safe local split-block boundary.

The codec deliberately contains no generic RPC abstraction.  It frames one Qwen3.6
experiment over a persistent byte stream so a later experiment can replace the AF_UNIX
socket without changing model-block semantics.
"""

from __future__ import annotations

import enum
import socket
import struct
from dataclasses import dataclass


MAGIC = b"ISN1"
PROTOCOL_VERSION = 1
HIDDEN_SIZE = 2048
BOUNDARY_PLANES = 2
BF16_DTYPE_CODE = 1
BF16_ELEMENT_BYTES = 2
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
FLAG_PREFILL_FINAL = 1
KNOWN_FLAGS = FLAG_PREFILL_FINAL
HEADER = struct.Struct("!4sHHQQqIIIIIQ")
TOKEN_RESULT = struct.Struct("!i")


class BoundaryProtocolError(RuntimeError):
    pass


class MessageType(enum.IntEnum):
    HELLO = 1
    HELLO_ACK = 2
    OPEN_SESSION = 3
    ACK = 4
    PREFILL = 5
    DECODE = 6
    TOKEN_RESULT = 7
    CLOSE_SESSION = 8
    RESET_SESSION = 9
    ERROR = 10


@dataclass(frozen=True)
class Frame:
    message_type: MessageType
    session_id: int = 0
    step_id: int = 0
    absolute_start_position: int = -1
    token_count: int = 0
    batch_size: int = 1
    hidden_size: int = HIDDEN_SIZE
    hidden_dtype_code: int = BF16_DTYPE_CODE
    flags: int = 0
    payload: bytes = b""

    @property
    def payload_bytes(self) -> int:
        return len(self.payload)


def _validate_frame(frame: Frame) -> None:
    if not 0 <= frame.session_id < 1 << 64 or not 0 <= frame.step_id < 1 << 64:
        raise BoundaryProtocolError("session_id and step_id must be unsigned 64-bit values")
    if frame.flags & ~KNOWN_FLAGS:
        raise BoundaryProtocolError(f"unknown frame flags 0x{frame.flags:x}")
    if frame.payload_bytes > MAX_PAYLOAD_BYTES:
        raise BoundaryProtocolError("frame payload exceeds the N1 limit")
    if frame.message_type in (MessageType.PREFILL, MessageType.DECODE):
        if frame.batch_size != 1:
            raise BoundaryProtocolError("N1 requires batch_size=1")
        if frame.hidden_size != HIDDEN_SIZE:
            raise BoundaryProtocolError(f"wrong hidden size {frame.hidden_size}")
        if frame.hidden_dtype_code != BF16_DTYPE_CODE:
            raise BoundaryProtocolError(f"wrong hidden dtype code {frame.hidden_dtype_code}")
        if frame.token_count < 1:
            raise BoundaryProtocolError("hidden frame token_count must be positive")
        expected = frame.token_count * BOUNDARY_PLANES * HIDDEN_SIZE * BF16_ELEMENT_BYTES
        if frame.payload_bytes != expected:
            raise BoundaryProtocolError(
                f"hidden payload length {frame.payload_bytes} != implied {expected}"
            )
        if frame.message_type is MessageType.DECODE and frame.token_count != 1:
            raise BoundaryProtocolError("DECODE must carry exactly one hidden vector")
        if frame.message_type is MessageType.DECODE and frame.flags:
            raise BoundaryProtocolError("DECODE does not accept flags")
    elif frame.message_type is MessageType.TOKEN_RESULT:
        if frame.payload_bytes != TOKEN_RESULT.size:
            raise BoundaryProtocolError("TOKEN_RESULT must contain one signed int32")
    elif frame.flags:
        raise BoundaryProtocolError("flags are only valid on PREFILL")


def encode_frame(frame: Frame) -> bytes:
    _validate_frame(frame)
    return HEADER.pack(
        MAGIC,
        PROTOCOL_VERSION,
        int(frame.message_type),
        frame.session_id,
        frame.step_id,
        frame.absolute_start_position,
        frame.token_count,
        frame.batch_size,
        frame.hidden_size,
        frame.hidden_dtype_code,
        frame.flags,
        frame.payload_bytes,
    ) + frame.payload


def read_exact(stream: socket.socket, count: int) -> bytes:
    if count < 0:
        raise BoundaryProtocolError("negative read length")
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.recv(remaining)
        if not chunk:
            raise BoundaryProtocolError(
                f"truncated frame: needed {remaining} more of {count} bytes"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def write_all(stream: socket.socket, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = stream.send(view)
        if written <= 0:
            raise BoundaryProtocolError("byte stream closed during frame write")
        view = view[written:]


def recv_frame(stream: socket.socket) -> Frame:
    raw = read_exact(stream, HEADER.size)
    (
        magic,
        version,
        raw_type,
        session_id,
        step_id,
        absolute_start_position,
        token_count,
        batch_size,
        hidden_size,
        hidden_dtype_code,
        flags,
        payload_bytes,
    ) = HEADER.unpack(raw)
    if magic != MAGIC:
        raise BoundaryProtocolError("boundary magic mismatch")
    if version != PROTOCOL_VERSION:
        raise BoundaryProtocolError(
            f"protocol version mismatch: peer={version}, local={PROTOCOL_VERSION}"
        )
    try:
        message_type = MessageType(raw_type)
    except ValueError as exc:
        raise BoundaryProtocolError(f"unknown message type {raw_type}") from exc
    if payload_bytes > MAX_PAYLOAD_BYTES:
        raise BoundaryProtocolError("declared payload exceeds the N1 limit")
    frame = Frame(
        message_type=message_type,
        session_id=session_id,
        step_id=step_id,
        absolute_start_position=absolute_start_position,
        token_count=token_count,
        batch_size=batch_size,
        hidden_size=hidden_size,
        hidden_dtype_code=hidden_dtype_code,
        flags=flags,
        payload=read_exact(stream, payload_bytes),
    )
    _validate_frame(frame)
    return frame


class UnixSocketTransport:
    """A persistent AF_UNIX byte stream; tensor semantics stay in :class:`Frame`."""

    def __init__(self, stream: socket.socket):
        if stream.family != socket.AF_UNIX or stream.type != socket.SOCK_STREAM:
            raise BoundaryProtocolError("N1 transport requires AF_UNIX SOCK_STREAM")
        self.stream = stream

    @classmethod
    def connect(cls, path: str) -> "UnixSocketTransport":
        stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stream.connect(path)
        return cls(stream)

    def send(self, frame: Frame) -> None:
        write_all(self.stream, encode_frame(frame))

    def receive(self) -> Frame:
        return recv_frame(self.stream)

    def close(self) -> None:
        self.stream.close()


def _pack_text(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) >= 1 << 16:
        raise BoundaryProtocolError("handshake string is too long")
    return struct.pack("!H", len(encoded)) + encoded


def _unpack_text(payload: bytes, offset: int) -> tuple[str, int]:
    if offset + 2 > len(payload):
        raise BoundaryProtocolError("truncated handshake string length")
    length = struct.unpack_from("!H", payload, offset)[0]
    offset += 2
    if offset + length > len(payload):
        raise BoundaryProtocolError("truncated handshake string")
    try:
        value = payload[offset:offset + length].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BoundaryProtocolError("invalid UTF-8 handshake field") from exc
    return value, offset + length


@dataclass(frozen=True)
class Handshake:
    model_repository: str
    model_revision: str
    n0_plan_sha256: str
    block_a_start: int
    block_a_end: int
    block_b_start: int
    block_b_end: int
    hidden_size: int
    hidden_dtype_code: int
    process_a_device_uuid: str
    process_b_device_uuid: str

    _NUMERIC = struct.Struct("!HHHHII")

    def encode(self) -> bytes:
        return b"".join((
            _pack_text(self.model_repository),
            _pack_text(self.model_revision),
            _pack_text(self.n0_plan_sha256),
            self._NUMERIC.pack(
                self.block_a_start, self.block_a_end, self.block_b_start,
                self.block_b_end, self.hidden_size, self.hidden_dtype_code,
            ),
            _pack_text(self.process_a_device_uuid),
            _pack_text(self.process_b_device_uuid),
        ))

    @classmethod
    def decode(cls, payload: bytes) -> "Handshake":
        offset = 0
        model_repository, offset = _unpack_text(payload, offset)
        model_revision, offset = _unpack_text(payload, offset)
        n0_plan_sha256, offset = _unpack_text(payload, offset)
        if offset + cls._NUMERIC.size > len(payload):
            raise BoundaryProtocolError("truncated handshake numeric fields")
        numeric = cls._NUMERIC.unpack_from(payload, offset)
        offset += cls._NUMERIC.size
        process_a_device_uuid, offset = _unpack_text(payload, offset)
        process_b_device_uuid, offset = _unpack_text(payload, offset)
        if offset != len(payload):
            raise BoundaryProtocolError("trailing handshake bytes")
        return cls(
            model_repository, model_revision, n0_plan_sha256,
            *numeric, process_a_device_uuid, process_b_device_uuid,
        )

    def require_exact(self, expected: "Handshake") -> None:
        if self != expected:
            raise BoundaryProtocolError(
                f"handshake mismatch: peer={self!r}, expected={expected!r}"
            )


class SessionPhase(enum.Enum):
    CONNECTED = "connected"
    OPEN = "open"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    CLOSED = "closed"


class SessionStateMachine:
    """Strict B-side request ordering; A uses the same checks before sending."""

    def __init__(self) -> None:
        self.phase = SessionPhase.CONNECTED
        self.session_id: int | None = None
        self.next_step_id = 0
        self.expected_position = 0

    def open(self, session_id: int) -> None:
        if not session_id:
            raise BoundaryProtocolError("session_id zero is reserved")
        if self.phase not in (SessionPhase.CONNECTED, SessionPhase.CLOSED):
            raise BoundaryProtocolError("session is already open")
        self.phase = SessionPhase.OPEN
        self.session_id = session_id
        self.next_step_id = 0
        self.expected_position = 0

    def accept_hidden(self, frame: Frame) -> None:
        if frame.message_type not in (MessageType.PREFILL, MessageType.DECODE):
            raise BoundaryProtocolError("expected PREFILL or DECODE")
        if self.phase not in (SessionPhase.OPEN, SessionPhase.PREFILLING, SessionPhase.DECODING):
            raise BoundaryProtocolError("hidden frame received without an open session")
        if frame.session_id != self.session_id:
            raise BoundaryProtocolError("wrong session ID")
        if frame.step_id != self.next_step_id:
            raise BoundaryProtocolError(
                f"stale or skipped step ID {frame.step_id}; expected {self.next_step_id}"
            )
        if frame.absolute_start_position != self.expected_position:
            raise BoundaryProtocolError(
                f"non-monotone position {frame.absolute_start_position}; expected {self.expected_position}"
            )
        if frame.message_type is MessageType.PREFILL:
            if self.phase is SessionPhase.DECODING:
                raise BoundaryProtocolError("PREFILL after DECODE is forbidden")
            self.phase = (
                SessionPhase.DECODING
                if frame.flags & FLAG_PREFILL_FINAL
                else SessionPhase.PREFILLING
            )
        else:
            if self.phase is not SessionPhase.DECODING:
                raise BoundaryProtocolError("DECODE before final PREFILL is forbidden")
        self.next_step_id += 1
        self.expected_position += frame.token_count

    def close(self, session_id: int) -> None:
        if self.session_id != session_id or self.phase in (SessionPhase.CONNECTED, SessionPhase.CLOSED):
            raise BoundaryProtocolError("cannot close unknown or inactive session")
        self.phase = SessionPhase.CLOSED
        self.session_id = None
        self.next_step_id = 0
        self.expected_position = 0

    def reset(self, session_id: int) -> None:
        if self.session_id != session_id:
            raise BoundaryProtocolError("cannot reset unknown session")
        self.phase = SessionPhase.OPEN
        self.next_step_id = 0
        self.expected_position = 0


def encode_token_result(token_id: int) -> bytes:
    return TOKEN_RESULT.pack(token_id)


def decode_token_result(payload: bytes) -> int:
    if len(payload) != TOKEN_RESULT.size:
        raise BoundaryProtocolError("token result has the wrong byte length")
    return TOKEN_RESULT.unpack(payload)[0]


__all__ = [
    "BF16_DTYPE_CODE", "BOUNDARY_PLANES", "BoundaryProtocolError", "FLAG_PREFILL_FINAL", "Frame",
    "Handshake", "HEADER", "HIDDEN_SIZE", "MessageType", "PROTOCOL_VERSION",
    "SessionPhase", "SessionStateMachine", "UnixSocketTransport", "decode_token_result",
    "encode_frame", "encode_token_result", "read_exact", "recv_frame", "write_all",
]
