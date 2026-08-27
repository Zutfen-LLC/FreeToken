"""Regression for benchmark server teardown ordering.

The Phase-0 harness starts ``ft serve`` in a new process group.  A whole-group SIGTERM as
the *first* shutdown action lets a worker die before Uvicorn's lifespan hook sets the
server's shutting-down flag, which makes the backend supervisor log a false crash after a
successful campaign.  Teardown must signal the frontend parent first; process-group signals
remain a cleanup/escalation backstop only after the parent has had a chance to run lifespan.
"""

from __future__ import annotations

import signal
from types import SimpleNamespace

from inferswarm_phase0 import client as client_mod


class _Proc:
    pid = 4242

    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.returncode = None

    def poll(self):
        return None

    def terminate(self) -> None:
        self.events.append("parent-SIGTERM")

    def wait(self, timeout=None):
        self.events.append(("wait", timeout))
        self.returncode = 0
        return 0


def test_stop_server_signals_frontend_before_process_group(monkeypatch):
    events: list[object] = []
    proc = _Proc(events)
    handle = client_mod.ServerHandle(
        proc=proc,
        origin="http://127.0.0.1:1",
        log_path="/tmp/not-used.log",
        command=["ft", "serve"],
    )

    monkeypatch.setattr(
        client_mod.os,
        "killpg",
        lambda pid, sig: events.append(("killpg", pid, sig)),
    )
    monkeypatch.setattr(client_mod.time, "sleep", lambda seconds: None)

    client_mod.stop_server(handle)

    assert events[0] == "parent-SIGTERM"
    group_signals = [e for e in events if isinstance(e, tuple) and e[0] == "killpg"]
    assert group_signals, "process-group cleanup backstop must remain"
    assert group_signals[0][2] == signal.SIGTERM
    assert events.index("parent-SIGTERM") < events.index(group_signals[0])
