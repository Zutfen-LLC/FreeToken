"""CPU-testable coverage for the R4 wire seam, fail-closed semantics, and
planner opacity.  Bounded; no GPU, no model, no real network beyond
loopback socketpair/pair-of-threads used for partial-IO and disconnect
tests.
"""

from __future__ import annotations

import json
import socket
import struct
import threading

import pytest

from freetoken.research import r4_wire as wire
from freetoken.research.r4_wire import (
    HEADER_STRUCT,
    WireError,
    encode_frame,
    payload_checksum,
    read_exact,
    recv_frame,
    send_exact,
    validate_request,
)

CONTRACT = {
    "dtype": "bfloat16",
    "layout": "plane-major-contiguous",
    "planes": 2,
    "row_width": 2048,
    "element_bytes": 2,
    "max_token_count": 64,
}


def _request(token_count: int = 1, **overrides) -> dict:
    header = {
        "kind": "request",
        "protocol": "inferswarm.r4.boundary-wire/1",
        "experiment_id": "exp-x",
        "session_id": 7,
        "op": "BOUNDARY",
        "operation": "decode",
        "position": 3,
        "token_count": token_count,
        "dtype": "bfloat16",
        "layout": "plane-major-contiguous",
        "payload_len": token_count * 2 * 2048 * 2,
    }
    header.update(overrides)
    return header


# 1. frame encode/decode round-trip -----------------------------------------


def test_frame_round_trip() -> None:
    payload = bytes(range(256)) * 8
    frame = encode_frame(_request(token_count=1, payload_len=len(payload)), payload)
    parent, child = socket.socketpair()
    try:
        send_exact(child, frame)
        header, received = recv_frame(
            parent,
            {"protocol": "inferswarm.r4.boundary-wire/1", "session_id": 7},
        )
        assert header["op"] == "BOUNDARY"
        assert bytes(received) == payload
    finally:
        parent.close()
        child.close()


def test_header_budget_enforced() -> None:
    big = _request()
    big["padding"] = "x" * (wire.HEADER_BUDGET + 1)
    with pytest.raises(WireError, match="budget"):
        encode_frame(big)


def test_unknown_kind_rejected() -> None:
    with pytest.raises(WireError, match="kind"):
        encode_frame(_request(kind="control-plane"))


# 2/3. partial socket reads and writes --------------------------------------


def test_partial_reads_and_writes() -> None:
    payload = bytes(8192)
    frame = encode_frame(_request(payload_len=len(payload)), payload)
    parent, child = socket.socketpair()
    # force small sends by slicing
    try:
        for start in range(0, len(frame), 997):
            send_exact(child, frame[start : start + 997])
        header, received = recv_frame(parent)
        assert bytes(received) == payload
    finally:
        parent.close()
        child.close()


# 4. invalid protocol version fails closed -----------------------------------


def _empty_frame() -> bytes:
    return encode_frame(_request(token_count=1, payload_len=0))


def test_invalid_version_fails_closed() -> None:
    frame = _empty_frame()
    corrupted = HEADER_STRUCT.pack(wire.WIRE_MAGIC, 99, 1, 0) + frame[HEADER_STRUCT.size :]
    parent, child = socket.socketpair()
    try:
        send_exact(child, corrupted)
        with pytest.raises(WireError, match="version"):
            recv_frame(parent)
    finally:
        parent.close()
        child.close()


def test_bad_magic_fails_closed() -> None:
    frame = _empty_frame()
    corrupted = b"XXXX" + frame[4:]
    parent, child = socket.socketpair()
    try:
        send_exact(child, corrupted)
        with pytest.raises(WireError, match="magic"):
            recv_frame(parent)
    finally:
        parent.close()
        child.close()


# 5/6. plan/session identity mismatch fails closed ---------------------------


def test_session_mismatch_fails_closed() -> None:
    frame = _empty_frame()
    parent, child = socket.socketpair()
    try:
        send_exact(child, frame)
        with pytest.raises(WireError, match="session_id mismatch"):
            recv_frame(parent, {"protocol": "inferswarm.r4.boundary-wire/1", "session_id": 8})
    finally:
        parent.close()
        child.close()


def test_experiment_mismatch_fails_closed() -> None:
    frame = _empty_frame()
    parent, child = socket.socketpair()
    try:
        send_exact(child, frame)
        with pytest.raises(WireError, match="experiment_id mismatch"):
            recv_frame(parent, {"protocol": "inferswarm.r4.boundary-wire/1", "experiment_id": "other"})
    finally:
        parent.close()
        child.close()


# 7. malformed payload length fails closed -----------------------------------


def test_malformed_payload_length_fails_closed() -> None:
    header = _request()
    header["payload_len"] = 2**30
    body = wire.canonical_header(header)
    prefix = HEADER_STRUCT.pack(wire.WIRE_MAGIC, wire.WIRE_VERSION, len(body), 0)
    parent, child = socket.socketpair()
    try:
        send_exact(child, prefix + body)
        with pytest.raises(WireError, match="budget"):
            recv_frame(parent)
    finally:
        parent.close()
        child.close()


def test_encode_rejects_len_disagreement() -> None:
    with pytest.raises(WireError, match="disagrees"):
        encode_frame(_request(payload_len=8192), bytes(16))


# 8. dtype/layout/token-count mismatch fails closed --------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"dtype": "float32"},
        {"layout": "row-interleaved"},
        {"token_count": 65},
        {"token_count": 0},
        {"payload_len": 4096},
    ],
)
def test_contract_mismatch_fails_closed(overrides) -> None:
    header = _request(**overrides)
    payload = bytes(header.get("payload_len", 8192))
    with pytest.raises(WireError):
        validate_request(header, contract=CONTRACT, payload=payload)


def test_contract_happy_path() -> None:
    payload = bytes(8192)
    assert validate_request(_request(), contract=CONTRACT, payload=payload) == 8192


# 9. checksum mismatch fails closed ------------------------------------------


def test_checksum_mismatch_fails_closed() -> None:
    header = _request(payload_sha256="sha256:" + "0" * 64)
    with pytest.raises(WireError, match="checksum"):
        validate_request(
            header, contract=CONTRACT, checksum=payload_checksum(bytes(8192))
        )


def test_checksum_good_passes() -> None:
    payload = bytes(8192)
    header = _request(payload_len=len(payload), payload_sha256=payload_checksum(payload))
    validate_request(header, contract=CONTRACT, checksum=payload_checksum(payload))


def test_diagnostic_requires_checksum() -> None:
    with pytest.raises(WireError, match="requires"):
        validate_request(_request(), contract=CONTRACT, checksum="anything")


# 10. disconnect mid-frame fails closed --------------------------------------


def test_disconnect_mid_frame_fails_closed() -> None:
    frame = encode_frame(_request(payload_len=8192), bytes(8192))
    parent, child = socket.socketpair()
    try:
        send_exact(child, frame[: len(frame) // 2])
        child.close()
        with pytest.raises(WireError, match="disconnect"):
            recv_frame(parent)
    finally:
        parent.close()


# 11. persistent connection reused across boundaries -------------------------


def test_persistent_connection_many_boundaries() -> None:
    payload = bytes(8192)
    parent, child = socket.socketpair()

    def server() -> None:
        count = 0
        try:
            while count < 10:
                header, received = recv_frame(child)
                reply = {
                    "kind": "response",
                    "protocol": header["protocol"],
                    "experiment_id": header["experiment_id"],
                    "session_id": header["session_id"],
                    "op": "TOKEN_RESULT",
                    "token_id": count,
                    "payload_len": 0,
                }
                send_exact(child, encode_frame(reply))
                count += 1
        except WireError:
            pass

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    try:
        tokens = []
        for _ in range(10):
            send_exact(parent, encode_frame(_request(payload_len=len(payload)), payload))
            header, _ = recv_frame(
                parent, {"protocol": "inferswarm.r4.boundary-wire/1", "session_id": 7}
            )
            tokens.append(header["token_id"])
        assert tokens == list(range(10))
    finally:
        parent.close()
        child.close()
        thread.join(timeout=2)


# 12. activation remains exact binary payload --------------------------------


def test_activation_binary_exactness() -> None:
    payload = bytes((i * 37 + 11) % 256 for i in range(8192))
    frame = encode_frame(_request(payload_len=len(payload)), payload)
    parent, child = socket.socketpair()
    try:
        send_exact(child, frame)
        _, received = recv_frame(parent)
        assert bytes(received) == payload
        assert payload_checksum(received) == payload_checksum(payload)
    finally:
        parent.close()
        child.close()


# 13/14. bounded accounting; clean mode excludes diagnostics ------------------


def test_response_accounting_bounded() -> None:
    reply = {
        "kind": "response",
        "protocol": "p",
        "experiment_id": "e",
        "session_id": 1,
        "op": "TOKEN_RESULT",
        "token_id": 42,
        "payload_len": 0,
    }
    frame = encode_frame(reply)
    assert len(frame) < wire.HEADER_BUDGET + HEADER_STRUCT.size + 256


def test_clean_mode_omits_logit_transfer() -> None:
    # The clean arm never sets capture_logits; full-logit payloads exist
    # only when the diagnostic flag reaches the wire header.
    clean_header = _request(payload_len=0, capture_logits=False, capture_state=False)
    encoded = encode_frame(clean_header).decode("latin-1")
    assert "full_logits" not in encoded
    diagnostic_header = _request(payload_len=0, capture_logits=True)
    assert "capture_logits" in json.dumps(diagnostic_header)


# 15. planner/evidence code free of R4 wire nouns -----------------------------


def test_generic_modules_have_no_r4_nouns() -> None:
    from freetoken.research import r2_local_split, r3_planner

    for module in (r2_local_split, r3_planner):
        source = open(module.__file__, encoding="utf-8").read()
        for noun in ("FTR4", "boundary-wire", "TCP_NODELAY", "token_id"):
            assert noun not in source, f"{module.__name__} leaks {noun!r}"


def test_wire_module_is_model_opaque() -> None:
    source = open(wire.__file__, encoding="utf-8").read()
    for noun in ("qwen", "Qwen", "expert", "logits"):
        assert noun not in source, f"wire module leaks model noun {noun!r}"


# 16. network candidate remains feasible/unranked -----------------------------


def test_network_candidate_feasible_unranked() -> None:
    from freetoken.research.r3_planner import freeze as freeze_generic
    from freetoken.research.r3_planner import plan as generic_plan

    from benchmarks.inferswarm_r4.r4_plan import (
        r4_network_problem,
        r4_resource_snapshot,
        validate_network_candidate,
    )

    hardware_a = {
        "gpus": [
            {
                "uuid": "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099",
                "name": "RTX 3060",
                "pci_bus_id": "00000000:02:00.0",
                "memory_total_bytes": "12884901888",
            }
        ],
        "memory": {"mem_total_kib": 131072000},
    }
    hardware_b = {
        "gpus": [
            {
                "uuid": "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176",
                "name": "RTX 3060",
                "pci_bus_id": "00000000:01:00.0",
                "memory_total_bytes": "12884901888",
            }
        ],
        "memory": {"mem_total_kib": 16777216},
    }
    problem = r4_network_problem(None)
    snapshot = r4_resource_snapshot(hardware_a, hardware_b)
    objective = {
        "schema": "inferswarm.r3.objective/1",
        "metric": "warm_decode_tok_s",
        "direction": "MAXIMIZE",
        "unit": "tok/s",
        "statistic": "median",
    }
    policy = {"schema": "inferswarm.r3.operator-policy/1", "rules": []}
    decision = generic_plan(
        problem,
        freeze_generic(snapshot),
        freeze_generic(policy),
        freeze_generic(objective),
        evidence_catalog=freeze_generic({"records": []}),
    )
    r4_plan_stub = {"digest": "sha256:stub"}
    authorization = validate_network_candidate(r4_plan_stub, decision)
    assert authorization["state"] == "FEASIBLE_UNRANKED"
    assert authorization["applicable_evidence_ids"] == []


# 17. boundary byte-count geometry --------------------------------------------


def test_boundary_geometry() -> None:
    decode = 1 * 2 * 2048 * 2
    prefill = 64 * 2 * 2048 * 2
    assert decode == 8192
    assert prefill == 524288
    assert validate_request(_request(token_count=64), contract=CONTRACT) == prefill


# 18. read_exact negative count ----------------------------------------------


def test_read_exact_negative() -> None:
    parent, child = socket.socketpair()
    try:
        with pytest.raises(WireError):
            read_exact(parent, -1)
    finally:
        parent.close()
        child.close()


# 19. wire responses never carry unbounded logit values -----------------------


def test_diagnostic_response_never_carries_full_logits() -> None:
    # The Node B service builds the bounded record only; assert the
    # response builder code does not reference full_logits at all.
    import inspect

    from benchmarks.inferswarm_r4 import node_b_service

    source = inspect.getsource(node_b_service)
    assert "full_logits" not in source, (
        "diagnostic responses must carry hash records, not full logit values"
    )


def test_clean_arm_run_experiment_has_no_diagnostic_payload() -> None:
    import inspect

    from benchmarks.inferswarm_r4 import run_experiment

    source = inspect.getsource(run_experiment)
    assert "full_logits" not in source


# 20. persistent connection reused across multiple sessions ------------------


def test_persistent_connection_across_sessions() -> None:
    parent, child = socket.socketpair()

    def server() -> None:
        session = None
        try:
            while True:
                header, payload = recv_frame(child)
                kind = header.get("kind")
                if kind == "hello":
                    session = header["session_id"]
                    send_exact(
                        child,
                        encode_frame(
                            {
                                "kind": "response",
                                "protocol": header["protocol"],
                                "experiment_id": header["experiment_id"],
                                "session_id": session,
                                "op": "HELLO_ACK",
                                "runtime_ready": True,
                                "payload_len": 0,
                            }
                        ),
                    )
                elif kind == "request":
                    if header.get("session_id") != session:
                        raise WireError("session changed without hello")
                    send_exact(
                        child,
                        encode_frame(
                            {
                                "kind": "response",
                                "protocol": header["protocol"],
                                "experiment_id": header["experiment_id"],
                                "session_id": session,
                                "op": "TOKEN_RESULT",
                                "token_id": 1,
                                "payload_len": 0,
                            }
                        ),
                    )
        except WireError:
            pass

    import threading

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    try:
        for session_id in (1000, 1001):
            header = {
                "kind": "hello",
                "protocol": "inferswarm.r4.boundary-wire/1",
                "experiment_id": "exp-x",
                "session_id": session_id,
                "payload_len": 0,
            }
            send_exact(parent, encode_frame(header))
            reply, _ = recv_frame(parent)
            assert reply["op"] == "HELLO_ACK"
            request = {
                "kind": "request",
                "protocol": "inferswarm.r4.boundary-wire/1",
                "experiment_id": "exp-x",
                "session_id": session_id,
                "op": "BOUNDARY",
                "payload_len": 0,
            }
            send_exact(parent, encode_frame(request))
            reply, _ = recv_frame(parent)
            assert reply["op"] == "TOKEN_RESULT"
    finally:
        parent.close()
        child.close()
        thread.join(timeout=2)
