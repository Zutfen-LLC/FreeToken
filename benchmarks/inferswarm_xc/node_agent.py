"""Node-local execution agent for the InferSwarm external-Coordinator proof
(inferswarm #67).

Runs on a compute Node (inferswarm01).  Owns exactly one Node-agent listen
socket; for each authorized REALIZE request it constructs the accepted
R5B-isolated R4 runtime (Block A resident on the local GPU plus the persistent
R4 boundary connection to the Node B service), executes authorized GENERATE
operations against it, and returns attributed results.  All realization,
CUDA initialization, model materialization, and backend-native execution stay
inside this process and this Node.

Fail-closed: any wire violation, identity mismatch (protocol/scope/epoch/plan/
operation/position), or internal inconsistency tears the connection down; the
agent never guesses.  Research-internal only: bounded per-proof process, not a
production daemon or public API.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from freetoken.research.xc_wire import (
    PROTOCOL_ID,
    XCWireError,
    body_checksum,
    encode_frame,
    identity_of,
    recv_frame,
    send_exact,
    validate_request,
)


class _AgentError(Exception):
    pass


class _Session:
    """One accepted connection = at most one authorized realization."""

    def __init__(self, server: "NodeAgent", conn: socket.socket) -> None:
        self._server = server
        self._conn = conn
        self._scope_id = server.scope_id
        self._runtime = None
        self._epoch_id: str | None = None
        self._generation: int | None = None
        self._realization_id: str | None = None
        self._plan_digest: str | None = None
        self._generate_sequence = 0
        self._closed = False
        self.audit: list[dict[str, Any]] = []

    def _reject(self, message: str) -> None:
        record = {
            "kind": "response",
            "protocol": PROTOCOL_ID,
            "scope_id": self._scope_id,
            "operation": "REJECT",
            "ok": False,
            "error": message,
            "rejected_at_ns": time.time_ns(),
        }
        try:
            send_exact(self._conn, encode_frame(record))
        except XCWireError:
            pass

    def _accept_response(
        self, request: Mapping[str, Any], operation: str, result: Any, **extra: Any
    ) -> None:
        response = {
            "kind": "response",
            "protocol": PROTOCOL_ID,
            "scope_id": self._scope_id,
            "session_id": request.get("session_id"),
            "position": request.get("position"),
            "epoch_id": self._epoch_id,
            "generation": self._generation,
            "realization_id": self._realization_id,
            "plan_digest": self._plan_digest,
            "operation": operation,
            "ok": True,
            "result": result,
            **extra,
        }
        if operation == "GENERATE":
            response["result_checksum"] = body_checksum(result)
        send_exact(self._conn, encode_frame(response))

    def _require_identity_match(self, request: Mapping[str, Any]) -> None:
        if request.get("protocol") != PROTOCOL_ID:
            raise _AgentError(f"unsupported protocol {request.get('protocol')!r}")
        if request.get("scope_id") != self._scope_id:
            raise _AgentError("scope identity mismatch")
        if self._runtime is None and request.get("operation") != "REALIZE":
            raise _AgentError("no authorized realization for this connection")
        if request.get("operation") != "REALIZE":
            if (
                request.get("epoch_id") != self._epoch_id
                or request.get("generation") != self._generation
                or request.get("realization_id") != self._realization_id
                or request.get("plan_digest") != self._plan_digest
            ):
                raise _AgentError(
                    "request identity does not match the authorized realization"
                )

    def _realize(self, request: Mapping[str, Any]) -> None:
        if self._runtime is not None:
            raise _AgentError("connection already holds an authorized realization")
        plan = request.get("execution_plan")
        if not isinstance(plan, dict) or not isinstance(plan.get("digest"), str):
            raise _AgentError("REALIZE lacks a frozen execution plan with a digest")
        epoch_id = request.get("epoch_id")
        generation = request.get("generation")
        realization_id = request.get("realization_id")
        if not isinstance(epoch_id, str) or not epoch_id:
            raise _AgentError("REALIZE lacks the Coordinator-authorized epoch identity")
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise _AgentError("REALIZE lacks the Coordinator-authorized generation")
        if not isinstance(realization_id, str) or not realization_id:
            raise _AgentError(
                "REALIZE lacks the Coordinator realization-attempt identity"
            )
        # Activation/epoch identity is Coordinator-owned authority, consumed
        # from the wire.  The agent must never derive it from the plan digest:
        # the same plan may legally be activated again in a later generation,
        # and every activation must carry distinct epoch/generation authority.
        plan_tag = plan["digest"].split(":")[-1][:12]
        if epoch_id.startswith("remote-realization:"):
            raise _AgentError(
                "refusing plan-derived epoch identity; activation authority "
                "must come from the Coordinator epoch/generation"
            )
        expected = f"research-generation-{generation}:{plan_tag}"
        if epoch_id != expected:
            raise _AgentError(
                "epoch identity does not bind the authorized generation and plan"
            )
        if self._server.authorization_retired(realization_id):
            raise _AgentError(
                "realization authorization is retired and can never be accepted"
            )
        self._plan_digest = plan["digest"]
        self._epoch_id = epoch_id
        self._generation = generation
        self._realization_id = realization_id
        started = time.time_ns()
        built = self._server.build_runtime(plan)
        # build_runtime returns the accepted RealizedStaticPlan wrapper for a
        # network realization; the agent drives its runtime facade directly.
        runtime = getattr(built, "runtime", built)
        try:
            self._accept_response(
                request,
                "REALIZE",
                {
                    "observation": getattr(runtime, "observation", None)
                    or getattr(built, "observation", {}),
                    "node_identity": self._server.node_identity(),
                    "realization_wall_ns": time.time_ns() - started,
                },
            )
        except BaseException:
            runtime.close()
            # A failed realization attempt is dead: never accept it later.
            self._server.retire_authorization(
                realization_id, reason="realization failed"
            )
            raise
        self._runtime = runtime
        self.audit.append(
            {
                "operation": "REALIZE",
                "epoch_id": self._epoch_id,
                "generation": self._generation,
                "realization_id": self._realization_id,
                "plan_digest": self._plan_digest,
                "authorized_at_ns": time.time_ns(),
            }
        )

    def _generate(self, request: Mapping[str, Any]) -> None:
        if self._runtime is None:
            raise _AgentError("GENERATE before an authorized realization")
        expected = self._generate_sequence + 1
        position = request.get("position")
        if position is not None and int(position) != expected:
            raise _AgentError(
                f"stale/reordered execution position {position} (expected {expected})"
            )
        self._generate_sequence = expected
        result = dict(
            self._runtime.generate(
                session_id=int(request["session_id"]),
                prompt_token_ids=[int(t) for t in request["prompt_token_ids"]],
                max_new_tokens=int(request["max_new_tokens"]),
            )
        )
        self._accept_response(request, "GENERATE", result)
        self.audit.append(
            {
                "operation": "GENERATE",
                "session_id": request.get("session_id"),
                "position": expected,
                "token_count": len(result.get("generated_token_ids", [])),
                "executed_at_ns": time.time_ns(),
            }
        )

    def _report(self, request: Mapping[str, Any]) -> None:
        if self._runtime is None:
            raise _AgentError("REPORT before an authorized realization")
        self._accept_response(request, "REPORT", dict(self._runtime.report()))

    def _close(self, request: Mapping[str, Any]) -> None:
        if self._runtime is None:
            raise _AgentError("CLOSE before an authorized realization")
        final_report = dict(self._runtime.report())
        reclamation = dict(getattr(self._runtime, "reclamation_report", {}) or {})
        self._runtime.close()
        self._runtime = None
        # The closed epoch's realization authorization is dead: the Node
        # agent must never accept it again, even for the same plan digest.
        self._server.retire_authorization(
            str(self._realization_id), reason="epoch closed"
        )
        self._accept_response(request, "CLOSE", final_report)
        self.audit.append(
            {
                "operation": "CLOSE",
                "epoch_id": self._epoch_id,
                "generation": self._generation,
                "realization_id": self._realization_id,
                "closed_at_ns": time.time_ns(),
                "reclamation": deepcopy(reclamation),
            }
        )
        self._closed = True

    def run(self) -> None:
        try:
            while not self._closed:
                try:
                    request = recv_frame(self._conn)
                except XCWireError:
                    break
                try:
                    validate_request(request)
                    self._require_identity_match(request)
                    operation = request["operation"]
                    if operation == "REALIZE":
                        self._realize(request)
                    elif operation == "GENERATE":
                        self._generate(request)
                    elif operation == "REPORT":
                        self._report(request)
                    elif operation == "CLOSE":
                        self._close(request)
                    else:  # pragma: no cover - validate_request already gates
                        raise _AgentError(f"unknown operation {operation!r}")
                except _AgentError as exc:
                    self._reject(str(exc))
                    self._server.audit(
                        {"accepted": False, "reason": str(exc), "request": identity_of(request)}
                    )
                    break
                except Exception as exc:  # realization/execution failure: fail closed
                    self._reject(f"{type(exc).__name__}: {exc}")
                    self._server.audit(
                        {
                            "accepted": False,
                            "reason": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                    break
        finally:
            if self._runtime is not None:
                try:
                    self._runtime.close()
                except Exception:
                    pass
                # Torn-down without CLOSE: the authorization is dead.
                if self._realization_id is not None:
                    self._server.retire_authorization(
                        self._realization_id, reason="session torn down"
                    )
            try:
                self._conn.close()
            except OSError:
                pass
            self._server.audit({"session_ended_at_ns": time.time_ns(), "operations": self.audit})


class NodeAgent:
    """Bounded per-proof Node-local execution agent."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.scope_id = args.scope_id
        self._audit_lock = threading.Lock()
        self._audit: list[dict[str, Any]] = []
        self._build_lock = threading.Lock()
        self._authorization_lock = threading.Lock()
        # Retired realization authorizations can never be accepted again,
        # even for the same plan digest or generation slot: a closed epoch's
        # correctness-bearing identity is dead forever.
        self._retired_authorizations: set[str] = set()

    def retire_authorization(self, realization_id: str, reason: str) -> None:
        with self._authorization_lock:
            self._retired_authorizations.add(realization_id)
        self.audit(
            {
                "authorization_retired": True,
                "realization_id": realization_id,
                "reason": reason,
                "retired_at_ns": time.time_ns(),
            }
        )

    def authorization_retired(self, realization_id: str) -> bool:
        with self._authorization_lock:
            return realization_id in self._retired_authorizations

    def audit(self, record: dict[str, Any]) -> None:
        with self._audit_lock:
            self._audit.append(record)

    def node_identity(self) -> dict[str, Any]:
        return {
            "hostname": socket.gethostname(),
            "role": "node-local-execution-agent",
            "scope_id": self.scope_id,
        }

    def build_runtime(self, plan: Mapping[str, Any]):
        """Construct the accepted R5B-isolated R4 runtime for the frozen plan."""
        from benchmarks.inferswarm_r5b.runtime import realize_isolated_network_plan

        with self._build_lock:
            return realize_isolated_network_plan(
                dict(plan),
                r4_plan=self._load_r4_plan(),
                model_path=self.args.model,
                peer_host=self.args.peer_host,
                peer_port=int(self.args.peer_port),
                diagnostic=bool(self.args.diagnostic),
            )

    def _load_r4_plan(self) -> dict[str, Any]:
        from benchmarks.inferswarm_r4.r4_plan import load_r4_plan

        return load_r4_plan(Path(self.args.participant_plan))

    def serve(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.args.listen_host, int(self.args.listen_port)))
        listener.listen(4)
        print(
            json.dumps(
                {
                    "node_agent_listening": True,
                    "host": self.args.listen_host,
                    "port": int(self.args.listen_port),
                    "scope_id": self.scope_id,
                }
            ),
            flush=True,
        )
        if self.args.ready_file:
            Path(self.args.ready_file).write_text(
                json.dumps({"port": int(self.args.listen_port), "scope_id": self.scope_id})
            )
        try:
            while True:
                conn, _peer = listener.accept()
                session = _Session(self, conn)
                thread = threading.Thread(
                    target=session.run, name="xc-node-agent-session", daemon=True
                )
                thread.start()
        finally:
            listener.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--scope-id", required=True)
    parser.add_argument("--participant-plan", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--peer-host", required=True)
    parser.add_argument("--peer-port", type=int, default=18485)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--ready-file")
    args = parser.parse_args(argv)
    agent = NodeAgent(args)
    agent.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
