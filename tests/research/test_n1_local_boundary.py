from __future__ import annotations

import socket
import struct

import pytest

from freetoken.research.n1_local_boundary import (
    BF16_DTYPE_CODE,
    BoundaryProtocolError,
    FLAG_PREFILL_FINAL,
    Frame,
    Handshake,
    HEADER,
    HIDDEN_SIZE,
    MessageType,
    SessionPhase,
    SessionStateMachine,
    decode_token_result,
    encode_frame,
    encode_token_result,
    recv_frame,
)


def _hidden(tokens: int = 1) -> bytes:
    return bytes((index % 251 for index in range(tokens * HIDDEN_SIZE * 2)))


def _frame(kind=MessageType.DECODE, *, session=7, step=0, position=0, tokens=1, flags=0):
    return Frame(kind, session, step, position, tokens, 1, HIDDEN_SIZE,
                 BF16_DTYPE_CODE, flags, _hidden(tokens))


def _socket_round_trip(frame: Frame) -> Frame:
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        left.sendall(encode_frame(frame))
        left.shutdown(socket.SHUT_WR)
        return recv_frame(right)
    finally:
        left.close()
        right.close()


def test_header_and_hidden_payload_round_trip_is_byte_exact():
    frame = _frame(MessageType.PREFILL, tokens=3, flags=FLAG_PREFILL_FINAL)
    assert _socket_round_trip(frame) == frame
    assert _socket_round_trip(frame).payload == frame.payload


def test_bf16_bytes_are_never_numeric_serialized():
    # Includes every byte value and BF16 bit patterns for NaN/Inf/subnormal payloads.
    payload = bytes(range(256)) * (HIDDEN_SIZE * 2 // 256)
    frame = Frame(MessageType.DECODE, 3, 9, 12, 1, 1, HIDDEN_SIZE,
                  BF16_DTYPE_CODE, 0, payload)
    assert _socket_round_trip(frame).payload == payload


@pytest.mark.parametrize("cut", [0, 1, HEADER.size - 1, HEADER.size + 10])
def test_malformed_or_truncated_frame_is_rejected(cut):
    raw = encode_frame(_frame())
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        left.sendall(raw[:cut])
        left.shutdown(socket.SHUT_WR)
        with pytest.raises(BoundaryProtocolError, match="truncated"):
            recv_frame(right)
    finally:
        left.close()
        right.close()


def test_payload_length_validation_rejects_mismatch():
    with pytest.raises(BoundaryProtocolError, match="payload length"):
        encode_frame(Frame(MessageType.DECODE, 1, 0, 0, 1, payload=b"short"))


def test_wrong_dtype_and_hidden_size_are_rejected():
    with pytest.raises(BoundaryProtocolError, match="dtype"):
        encode_frame(Frame(MessageType.DECODE, 1, 0, 0, 1, 1, HIDDEN_SIZE, 99, 0, _hidden()))
    with pytest.raises(BoundaryProtocolError, match="hidden size"):
        encode_frame(Frame(MessageType.DECODE, 1, 0, 0, 1, 1, 2048, BF16_DTYPE_CODE, 0, _hidden()))


def test_protocol_version_mismatch_is_rejected_before_payload_read():
    raw = bytearray(encode_frame(_frame()))
    struct.pack_into("!H", raw, 4, 2)
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        left.sendall(raw)
        with pytest.raises(BoundaryProtocolError, match="version mismatch"):
            recv_frame(right)
    finally:
        left.close()
        right.close()


def test_handshake_round_trip_and_mismatch_fail_closed():
    expected = Handshake("repo", "r" * 40, "a" * 64, 0, 19, 19, 40, 4096, 1,
                         "GPU-a", "GPU-b")
    assert Handshake.decode(expected.encode()) == expected
    wrong = Handshake("repo", "x" * 40, "a" * 64, 0, 19, 19, 40, 4096, 1,
                      "GPU-a", "GPU-b")
    with pytest.raises(BoundaryProtocolError, match="handshake mismatch"):
        wrong.require_exact(expected)


def test_valid_open_prefill_decode_close_and_token_result():
    state = SessionStateMachine()
    state.open(7)
    state.accept_hidden(_frame(MessageType.PREFILL, tokens=3, flags=FLAG_PREFILL_FINAL))
    state.accept_hidden(_frame(MessageType.DECODE, step=1, position=3))
    assert state.phase is SessionPhase.DECODING
    assert decode_token_result(encode_token_result(71093)) == 71093
    state.close(7)
    assert state.phase is SessionPhase.CLOSED


def test_decode_before_prefill_wrong_session_stale_step_and_position_are_rejected():
    state = SessionStateMachine()
    state.open(7)
    with pytest.raises(BoundaryProtocolError, match="before final PREFILL"):
        state.accept_hidden(_frame())
    with pytest.raises(BoundaryProtocolError, match="wrong session"):
        state.accept_hidden(_frame(MessageType.PREFILL, session=8, flags=FLAG_PREFILL_FINAL))
    state.accept_hidden(_frame(MessageType.PREFILL, flags=FLAG_PREFILL_FINAL))
    with pytest.raises(BoundaryProtocolError, match="step ID"):
        state.accept_hidden(_frame(step=0, position=1))
    with pytest.raises(BoundaryProtocolError, match="position"):
        state.accept_hidden(_frame(step=1, position=2))


def test_reset_and_second_session_cannot_see_first_session_state():
    state = SessionStateMachine()
    state.open(11)
    state.accept_hidden(_frame(MessageType.PREFILL, session=11, tokens=2,
                               flags=FLAG_PREFILL_FINAL))
    state.reset(11)
    assert state.expected_position == 0 and state.next_step_id == 0
    state.close(11)
    state.open(12)
    assert state.expected_position == 0 and state.next_step_id == 0
    with pytest.raises(BoundaryProtocolError, match="wrong session"):
        state.accept_hidden(_frame(MessageType.PREFILL, session=11,
                                   flags=FLAG_PREFILL_FINAL))
