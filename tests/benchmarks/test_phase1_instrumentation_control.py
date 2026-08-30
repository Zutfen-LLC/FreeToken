"""Instrumentation control-path tests for the P6 campaign runner (CPU-only).

The physical P6 Session-1 blocker this file pins down: the canonical B1/W1
post-block snapshot of ~204,800 retained complete-layer timing records was
MEASURED at 136.93 s, the frozen control budget was 60 s, and the server's
HTTP 504 escaped the runner as a raw ``urllib`` exception instead of entering
its fail-closed ServerError path — crashing SessionExecution after a fully
completed block.

What is exercised here, with the process/GPU/HTTP layers mocked and the REAL
``moe_instrumentation``/``_post_json`` control path live against a fake
``urllib.request.urlopen``:

* the frozen 300 s server-side operation budget and the strictly longer HTTP
  client budget reach BOTH reset and snapshot;
* a snapshot that logically needs more than the old 60 s budget completes the
  session normally (no sleeping: the fake refuses budgets below the measured
  physical duration);
* every expected HTTP control status (409/422/503/504) is decoded and converted
  to a structured fail-closed ServerError, and a session hit by one finalizes
  INCOMPLETE/INVALID instead of crashing;
* malformed HTTP error bodies and transport failures become bounded ServerErrors;
* a failed post-block snapshot preserves the completed W1 generations unchanged,
  represents every unexecuted later planned generation as not executed, writes
  the normal session summary, and produces no cross-arm ratio or verdict.
"""

from __future__ import annotations

import io
import json
import typing
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from inferswarm_phase1 import campaign as campaign_mod
from inferswarm_phase1.campaign import (
    MOE_INSTRUMENTATION_HTTP_GRACE_SECONDS,
    MOE_INSTRUMENTATION_TIMEOUT_SECONDS,
    CampaignDefinition,
    CampaignSettings,
    ServerError,
    SessionExecution,
)
from inferswarm_phase1.campaign_arms import (
    BASELINE_ARM_ID,
    CANDIDATE_ARM_ID,
    baseline_b1_arm,
    candidate_v2_arm,
    predeclared_kv_matched_arm,
)
from inferswarm_phase1.campaign_protocol import build_protocol

from .phase1_fakes import (
    INFERSWARM_SHA40,
    SHA40,
    install_clean_environment,
    install_mocked_server,
    moe_window_snapshot,
)

# Captured BEFORE any fixture patches the module attribute: the real control path.
_REAL_MOE_INSTRUMENTATION = campaign_mod.moe_instrumentation

# MEASURED canonical B1/W1 snapshot wall time on the crashed P6 build: the fake
# "slow snapshot" refuses any budget below it (60 s would have 504'd) and
# completes under the frozen 300 s budget — logically >60 s and <300 s without
# the test ever sleeping.
MEASURED_SLOW_SNAPSHOT_SECONDS = 136.93


class _FakeHTTPResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._raw = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> typing.Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _http_error(
    url: str, code: int, reason: str, body: bytes
) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, reason, None, io.BytesIO(body))


def _json_error_body(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


def install_control_transport(monkeypatch, behavior):
    """Serve POST /v1/moe/instrumentation from a fake ``urlopen``.

    ``behavior(body)`` returns ``("ok", payload)`` or
    ``("error", code, reason, raw_body)``. Every request seen is recorded with
    its decoded body (including the server-side operation budget the runner
    sent) and the HTTP timeout the client actually passed.
    """
    seen: list[dict] = []

    def fake_urlopen(request, timeout=None):
        url = request.full_url
        assert url.endswith("/v1/moe/instrumentation"), (
            f"unexpected request through the control transport: {url}"
        )
        body = json.loads(request.data.decode("utf-8"))
        seen.append(
            {
                "url": url,
                "operation": body["operation"],
                "sent_timeout": body.get("timeout"),
                "http_timeout": timeout,
            }
        )
        outcome = behavior(body)
        if outcome[0] == "ok":
            return _FakeHTTPResponse({"status": "ok", "payload": outcome[1]})
        _, code, reason, raw = outcome
        raise _http_error(url, code, reason, raw)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


def ok_behavior(body: dict):
    if body["operation"] == "reset":
        return ("ok", {"boundary": {"operation": "reset", "idle": True}})
    return ("ok", moe_window_snapshot())


def slow_snapshot_behavior(minimum_seconds: float = MEASURED_SLOW_SNAPSHOT_SECONDS):
    """A snapshot that needs more than the old 60 s budget but under 300 s.

    Under any budget smaller than the measured physical duration the server
    answers 504 exactly as the crashed P6 session saw; under the frozen budget
    the full payload returns. No sleeping happens anywhere.
    """

    def behavior(body: dict):
        if body["operation"] == "reset":
            return ("ok", {"boundary": {"operation": "reset", "idle": True}})
        if body["timeout"] < minimum_seconds:
            return (
                "error",
                504,
                "Gateway Timeout",
                _json_error_body({"status": "timeout", "request_id": "slow-snapshot"}),
            )
        return ("ok", moe_window_snapshot())

    return behavior


def _definition(tmp_path: Path, frozen: dict) -> CampaignDefinition:
    return CampaignDefinition(
        arms=[baseline_b1_arm(), candidate_v2_arm(), predeclared_kv_matched_arm()],
        protocol=build_protocol(
            warmups=None, repetitions=None, classes=None, dev_smoke=False
        ),
        settings=CampaignSettings(
            model_path=str(tmp_path / "model"),
            manifest_path=frozen["manifest"],
            model_revision=SHA40,
            placement_path=frozen["placement"],
            inferswarm_commit=INFERSWARM_SHA40,
            out_root=tmp_path / "runs",
            prerequisites_path=frozen["prerequisites"],
            echo_server_output=False,
        ),
        canonical=True,
    )


@pytest.fixture
def clean_environment(monkeypatch, tmp_path):
    return install_clean_environment(monkeypatch, tmp_path)


@pytest.fixture
def mock_server(monkeypatch, tmp_path):
    """A full mocked canonical session using the shared fakes verbatim, so the
    runner's own instrumentation call sites can be inspected."""
    frozen = install_clean_environment(monkeypatch, tmp_path)
    calls = install_mocked_server(monkeypatch)
    doc = SessionExecution(
        definition=_definition(tmp_path, frozen), session_number=1
    ).execute()
    assert doc["execution_status"] == "COMPLETE"
    return calls


def _run_session(monkeypatch, tmp_path, behavior, *, session_number: int = 1):
    """A canonical session with every boundary mocked EXCEPT the real
    instrumentation control path (``moe_instrumentation`` -> ``_post_json`` ->
    the fake ``urlopen`` transport)."""
    frozen = install_clean_environment(monkeypatch, tmp_path)
    calls = install_mocked_server(monkeypatch)
    monkeypatch.setattr(campaign_mod, "moe_instrumentation", _REAL_MOE_INSTRUMENTATION)
    seen = install_control_transport(monkeypatch, behavior)
    doc = SessionExecution(
        definition=_definition(tmp_path, frozen), session_number=session_number
    ).execute()
    return doc, calls, seen


# --- A: timeout propagation ---------------------------------------------------------------------


def test_frozen_300s_budget_reaches_both_reset_and_snapshot(monkeypatch):
    assert MOE_INSTRUMENTATION_TIMEOUT_SECONDS == 300.0
    assert MOE_INSTRUMENTATION_HTTP_GRACE_SECONDS == 5.0
    seen = install_control_transport(monkeypatch, ok_behavior)
    for operation in ("reset", "snapshot"):
        payload = campaign_mod.moe_instrumentation("http://127.0.0.1:9", operation)
        assert payload is not None
    assert [(s["operation"], s["sent_timeout"], s["http_timeout"]) for s in seen] == [
        ("reset", 300.0, 305.0),
        ("snapshot", 300.0, 305.0),
    ]
    # the HTTP client waits strictly longer than the server-side operation
    # budget, so a server-produced timeout response wins over a local socket
    # timeout
    for s in seen:
        assert s["http_timeout"] > s["sent_timeout"]


def test_canonical_session_uses_exactly_the_frozen_budget_for_every_window(
    tmp_path, monkeypatch
):
    doc, _calls, seen = _run_session(monkeypatch, tmp_path, ok_behavior)
    assert doc["execution_status"] == "COMPLETE"
    operations = [(s["operation"], s["sent_timeout"]) for s in seen]
    # 2 primary arms x 4 classes x (reset + snapshot), every request at 300 s
    assert operations.count(("reset", 300.0)) == 8
    assert operations.count(("snapshot", 300.0)) == 8
    assert all(s["http_timeout"] == 305.0 for s in seen)
    # the budget is recorded in the plan and provenance artifacts
    root = Path(doc["run_directory"])
    for artifact in ("plan.json", "provenance.json"):
        record = json.loads((root / artifact).read_text())["instrumentation_control"]
        assert record["operation_timeout_seconds"] == 300.0
        assert record["http_client_timeout_seconds"] == 305.0
        assert record["cli_override"] is None
        assert record["frozen_before_execution"] is True
    # distinct from the model-startup budget: --server-timeout is not reused
    settings_timeout = json.loads((root / "plan.json").read_text())
    assert "server_timeout" not in settings_timeout["instrumentation_control"]


def test_runner_call_sites_pass_the_frozen_budget_to_every_fake(mock_server):
    """Even with the instrumentation function fully faked, the runner's own
    call sites pass exactly the frozen constant for both operations."""
    assert mock_server["moe_timeouts"]
    assert all(
        timeout == MOE_INSTRUMENTATION_TIMEOUT_SECONDS
        for _operation, timeout in mock_server["moe_timeouts"]
    )
    assert {operation for operation, _ in mock_server["moe_timeouts"]} == {
        "reset",
        "snapshot",
    }


def test_no_cli_knob_can_move_the_canonical_instrumentation_budget(capsys):
    """The budget is a frozen campaign constant, not a canonical CLI knob."""
    from inferswarm_phase1 import campaign_cli

    parser = campaign_cli.build_parser()
    for argv in (
        ["--instrumentation-timeout", "10"],
        ["--moe-instrumentation-timeout", "10"],
        ["--moe-instrumentation-timeout-seconds", "10"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(
                [
                    "run-session",
                    "--session",
                    "1",
                    "--model",
                    "m",
                    "--manifest",
                    "w.json",
                    *argv,
                ]
            )
        assert excinfo.value.code == 2  # unrecognized argument: the knob does not exist
    # and CampaignSettings carries no such field either
    assert not any(
        "instrumentation_timeout" in name
        for name in CampaignSettings.__dataclass_fields__
    )


# --- B: a logically slow snapshot succeeds under the frozen budget ------------------------------


def test_snapshot_slower_than_the_old_60s_budget_completes_the_session(
    tmp_path, monkeypatch
):
    doc, _calls, seen = _run_session(monkeypatch, tmp_path, slow_snapshot_behavior())
    # the runner accepted every slow snapshot and continued; nothing was
    # recorded as a protocol deviation
    assert doc["execution_status"] == "COMPLETE"
    assert doc["validity"] == "VALID"
    assert doc["completion"]["observed_generations"] == 96
    assert doc["completion"]["failed_generations"] == 0
    assert not doc["stopped_early_reason"]
    assert not doc["campaign_invalidation_codes"]
    # every snapshot needed more than 60 s and got the frozen 300 s budget
    snapshots = [s for s in seen if s["operation"] == "snapshot"]
    assert snapshots and all(s["sent_timeout"] >= 300.0 for s in snapshots)
    assert all(
        s["sent_timeout"] > MEASURED_SLOW_SNAPSHOT_SECONDS - 1 for s in snapshots
    )


# --- C/D/E: expected HTTP control statuses fail closed ------------------------------------------


@pytest.mark.parametrize(
    "code,reason,payload",
    [
        (504, "Gateway Timeout", {"status": "timeout", "request_id": "req-504-1"}),
        (
            503,
            "Service Unavailable",
            {"status": "failed", "error": "failed to dispatch instrumentation: boom"},
        ),
        (409, "Conflict", {"status": "busy", "error": "engine is draining"}),
        (
            422,
            "Unprocessable Entity",
            {"status": "unsupported", "error": "timing is disabled"},
        ),
    ],
)
def test_expected_http_error_statuses_are_decoded_into_structured_server_errors(
    monkeypatch, code, reason, payload
):
    def behavior(body: dict):
        return ("error", code, reason, _json_error_body(payload))

    install_control_transport(monkeypatch, behavior)
    with pytest.raises(ServerError) as excinfo:
        campaign_mod.moe_instrumentation("http://127.0.0.1:9", "snapshot")
    message = str(excinfo.value)
    assert "snapshot" in message  # the operation
    assert str(code) in message  # the HTTP status
    assert payload["status"] in message  # the engine status
    for key in ("request_id", "error"):
        if key in payload:
            assert str(payload[key]) in message
    # and the engine's own document, not a urllib traceback, is what surfaced
    assert not isinstance(excinfo.value, urllib.error.HTTPError)


@pytest.mark.parametrize(
    "code,reason,payload",
    [
        (504, "Gateway Timeout", {"status": "timeout", "request_id": "req-504-1"}),
        (
            503,
            "Service Unavailable",
            {"status": "failed", "error": "failed to dispatch instrumentation: boom"},
        ),
        (409, "Conflict", {"status": "busy", "error": "engine is draining"}),
    ],
)
def test_http_control_failure_finalizes_the_session_fail_closed(
    tmp_path, monkeypatch, code, reason, payload
):
    def behavior(body: dict):
        if body["operation"] == "reset":
            return ("ok", {"boundary": {"operation": "reset", "idle": True}})
        return ("error", code, reason, _json_error_body(payload))

    doc, calls, _seen = _run_session(monkeypatch, tmp_path, behavior)
    # SessionExecution finalized INCOMPLETE/INVALID rather than crashing
    assert doc["execution_status"] == "INCOMPLETE"
    assert doc["validity"] == "INVALID"
    assert doc["completion"]["failed_generations"] == 84  # 96 - the finished W1 block
    assert "execution.server_failed" in doc["campaign_invalidation_codes"]
    # the structured outcome is visible in the stop record
    assert doc["stopped_early_reason"]
    assert str(code) in doc["stopped_early_reason"]
    assert payload["status"] in doc["stopped_early_reason"]
    # the failed control answer is preserved verbatim on the session record
    summary = json.loads(
        (Path(doc["run_directory"]) / "session-summary.json").read_text()
    )
    assert summary["execution_status"] == "INCOMPLETE"
    assert summary["validity"] == "INVALID"
    # never continued into another arm as though instrumentation succeeded
    assert [s["arm"] for s in calls["started"]] == [BASELINE_ARM_ID]
    assert ("reset", CANDIDATE_ARM_ID) not in calls["moe_ops"]


# --- F: malformed bodies and transport failures -------------------------------------------------


def test_malformed_http_error_body_is_a_bounded_server_error(monkeypatch):
    body = b"<html>Bad Gateway</html>" * 40  # 960 bytes of non-JSON

    def behavior(_body: dict):
        return ("error", 502, "Bad Gateway", body)

    install_control_transport(monkeypatch, behavior)
    with pytest.raises(ServerError) as excinfo:
        campaign_mod.moe_instrumentation("http://127.0.0.1:9", "snapshot")
    message = str(excinfo.value)
    assert "502" in message
    assert "non-JSON" in message
    # bounded diagnostic: the snippet is capped, never the whole body
    assert message.count("Bad Gateway") <= 9
    assert len(message) < 400
    assert not isinstance(excinfo.value, urllib.error.HTTPError)


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("connection refused"),
        TimeoutError("timed out"),
        ConnectionError("connection reset by peer"),
    ],
)
def test_transport_failures_become_server_error(monkeypatch, failure):
    def fake_urlopen(request, timeout=None):
        raise failure

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ServerError) as excinfo:
        campaign_mod.moe_instrumentation("http://127.0.0.1:9", "reset")
    assert "failed before a response arrived" in str(excinfo.value)
    assert repr(failure) in str(excinfo.value)


def test_malformed_http_error_body_fails_the_session_closed_without_crashing(
    tmp_path, monkeypatch
):
    def behavior(body: dict):
        if body["operation"] == "reset":
            return ("ok", {"boundary": {"operation": "reset", "idle": True}})
        return ("error", 502, "Bad Gateway", b"<html>Bad Gateway</html>" * 40)

    doc, calls, _seen = _run_session(monkeypatch, tmp_path, behavior)
    assert doc["execution_status"] == "INCOMPLETE"
    assert doc["validity"] == "INVALID"
    assert "502" in doc["stopped_early_reason"]
    assert [s["arm"] for s in calls["started"]] == [BASELINE_ARM_ID]


# --- G: block preservation around a failed post-block snapshot ----------------------------------


def test_failed_post_block_snapshot_preserves_completed_and_planned_generations(
    tmp_path, monkeypatch
):
    state = {"snapshots": 0}

    def behavior(body: dict):
        if body["operation"] == "reset":
            return ("ok", {"boundary": {"operation": "reset", "idle": True}})
        state["snapshots"] += 1
        if state["snapshots"] == 1:  # the baseline W1 post-block snapshot
            return (
                "error",
                504,
                "Gateway Timeout",
                _json_error_body({"status": "timeout", "request_id": "req-blocker"}),
            )
        return ("ok", moe_window_snapshot())

    doc, calls, seen = _run_session(monkeypatch, tmp_path, behavior)
    root = Path(doc["run_directory"])

    # the session finalized (no crash, no raw urllib escape)
    assert doc["execution_status"] == "INCOMPLETE"
    assert doc["validity"] == "INVALID"
    assert "instrumentation snapshot" in doc["stopped_early_reason"]
    assert "504" in doc["stopped_early_reason"]

    # the 12 completed W1 generations remain unchanged, in place
    w1 = [
        json.loads(line)
        for line in (root / BASELINE_ARM_ID / "W1.jsonl").read_text().splitlines()
    ]
    assert len(w1) == 12
    assert all(record["failed"] is False for record in w1)
    assert all(record["inter_token_ms"] for record in w1)

    # every unexecuted later planned generation is represented as not executed
    assert doc["completion"]["failed_generations"] == 84
    recorded = doc["execution_order"]["execution_indices_recorded"]
    assert recorded == list(range(96))
    candidate_w1 = [
        json.loads(line)
        for line in (root / CANDIDATE_ARM_ID / "W1.jsonl").read_text().splitlines()
    ]
    assert len(candidate_w1) == 12
    assert all(
        record["failed"] and "not executed" in record["error"]
        for record in candidate_w1
    )

    # no mechanism window was fabricated for the failed block
    assert not (root / BASELINE_ARM_ID / "block-mechanism-W1.json").exists()

    # the normal session summary and its artifact index exist
    summary_path = root / "session-summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["artifact_sha256"]
    assert summary["stopped_early_reason"] == doc["stopped_early_reason"]

    # no cross-arm ratio or verdict exists anywhere in the session output
    assert "no_verdict_note" in summary
    flattened = json.dumps(summary)
    assert "decode_tok_s_ratio" not in flattened
    assert "campaign_verdict" not in flattened

    # the candidate arm was never started: instrumentation did not "succeed"
    assert [s["arm"] for s in calls["started"]] == [BASELINE_ARM_ID]
    # and exactly one snapshot was attempted — no invisible retry
    assert state["snapshots"] == 1
    assert [s["operation"] for s in seen].count("snapshot") == 1
