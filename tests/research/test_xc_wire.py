"""Tests for the external-Coordinator research wire (inferswarm #67)."""

from __future__ import annotations

import json
import socket
import struct
import threading

import pytest

from freetoken.research.xc_wire import (
    BODY_BUDGET,
    HEADER_STRUCT,
    PROTOCOL_ID,
    WIRE_MAGIC,
    WIRE_VERSION,
    XCWireError,
    body_checksum,
    encode_frame,
    recv_frame,
    send_exact,
    validate_request,
    validate_response,
)


def _request(operation: str = "REPORT", **extra):
    body = {
        "kind": "request",
        "protocol": PROTOCOL_ID,
        "scope_id": "scope/test",
        "epoch_id": "remote-realization:abc123",
        "plan_digest": "sha256:" + "a" * 64,
        "operation": operation,
        **extra,
    }
    return body


class TestFraming:
    def test_round_trip(self):
        import socket as s

        pair = s.socketpair()
        frame = encode_frame(_request())
        send_exact(pair[0], frame)
        got = recv_frame(pair[1])
        assert got["operation"] == "REPORT"
        pair[0].close()
        pair[1].close()

    def test_unknown_kind_rejected_at_encode(self):
        with pytest.raises(XCWireError):
            encode_frame({"kind": "banana"})

    def test_header_layout_is_exact(self):
        frame = encode_frame(_request())
        magic, version, body_len = HEADER_STRUCT.unpack(frame[: HEADER_STRUCT.size])
        assert magic == WIRE_MAGIC
        assert version == WIRE_VERSION
        body = frame[HEADER_STRUCT.size :]
        assert body_len == len(body)
        assert json.loads(body)["kind"] == "request"

    def test_bad_magic_fails_closed(self):
        pair = socket.socketpair()
        body = b"{}"
        pair[0].sendall(HEADER_STRUCT.pack(b"XXXX", 1, len(body)) + body)
        with pytest.raises(XCWireError):
            recv_frame(pair[1])
        pair[0].close()
        pair[1].close()

    def test_bad_version_fails_closed(self):
        pair = socket.socketpair()
        body = b"{}"
        pair[0].sendall(HEADER_STRUCT.pack(WIRE_MAGIC, 99, len(body)) + body)
        with pytest.raises(XCWireError):
            recv_frame(pair[1])
        pair[0].close()
        pair[1].close()

    def test_oversized_declared_length_fails_closed(self):
        pair = socket.socketpair()
        pair[0].sendall(HEADER_STRUCT.pack(WIRE_MAGIC, 1, BODY_BUDGET + 1))
        with pytest.raises(XCWireError):
            recv_frame(pair[1])
        pair[0].close()
        pair[1].close()

    def test_malformed_json_fails_closed(self):
        pair = socket.socketpair()
        body = b"{not json"
        pair[0].sendall(HEADER_STRUCT.pack(WIRE_MAGIC, 1, len(body)) + body)
        with pytest.raises(XCWireError):
            recv_frame(pair[1])
        pair[0].close()
        pair[1].close()

    def test_mid_frame_disconnect_fails_closed(self):
        pair = socket.socketpair()
        frame = encode_frame(_request())
        pair[0].sendall(frame[: len(frame) // 2])
        pair[0].close()
        with pytest.raises(XCWireError):
            recv_frame(pair[1])
        pair[1].close()


class TestValidateRequest:
    def test_report_request_is_valid(self):
        identity = validate_request(_request())
        assert identity["operation"] == "REPORT"

    def test_generate_requires_session_and_prompt(self):
        with pytest.raises(XCWireError):
            validate_request(_request("GENERATE", max_new_tokens=2))
        with pytest.raises(XCWireError):
            validate_request(
                _request("GENERATE", session_id=1, max_new_tokens=2, prompt_token_ids=[])
            )

    def test_generate_rejects_non_int_tokens(self):
        with pytest.raises(XCWireError):
            validate_request(
                _request(
                    "GENERATE",
                    session_id=1,
                    max_new_tokens=2,
                    prompt_token_ids=[1, "two"],
                )
            )

    def test_unknown_operation_rejected(self):
        with pytest.raises(XCWireError):
            validate_request(_request("EXFILLOL"))

    def test_wrong_protocol_rejected(self):
        body = _request()
        body["protocol"] = "some-other-protocol/9"
        with pytest.raises(XCWireError):
            validate_request(body)

    def test_missing_scope_rejected(self):
        body = _request()
        del body["scope_id"]
        with pytest.raises(XCWireError):
            validate_request(body)


class TestValidateResponse:
    def _ok_response(self, **extra):
        return {
            "kind": "response",
            "protocol": PROTOCOL_ID,
            "epoch_id": "remote-realization:abc123",
            "plan_digest": "sha256:" + "a" * 64,
            "operation": "REPORT",
            "ok": True,
            "result": {"anything": 1},
            **extra,
        }

    def test_ok_response_valid(self):
        validate_response(self._ok_response())

    def test_error_response_requires_error_string(self):
        with pytest.raises(XCWireError):
            validate_response(
                {"kind": "response", "ok": False}
            )

    def test_failed_response_must_not_carry_result(self):
        with pytest.raises(XCWireError):
            validate_response(
                {
                    "kind": "response",
                    "ok": False,
                    "error": "boom",
                    "result": {"sneaky": True},
                }
            )

    def test_generate_checksum_mismatch_fails_closed(self):
        response = self._ok_response(
            operation="GENERATE",
            result={"generated_token_ids": [9764]},
            result_checksum="sha256:" + "0" * 64,
        )
        with pytest.raises(XCWireError):
            validate_response(response)

    def test_generate_checksum_ok(self):
        result = {"generated_token_ids": [9764]}
        response = self._ok_response(
            operation="GENERATE", result=result, result_checksum=body_checksum(result)
        )
        validate_response(response)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
