"""Coordinator-side remote realization adapter for the external-Coordinator
proof (inferswarm #67).

Runs on the CPU-only Coordinator host.  Satisfies the accepted
``EpochServingController`` realizer contract (``realizer(execution_plan) ->
RealizedStaticPlan``) by speaking the research-internal ``xc_wire`` protocol
to a Node-local execution agent instead of constructing any local runtime.
Every correctness-bearing exchange carries and checks the immutable
session/epoch/plan/operation identity; any mismatch, malformed response, or
disconnect fails closed.

This module is deliberately free of torch, CUDA, model, and weight nouns.
"""

from __future__ import annotations

import socket
import threading
import time
from copy import deepcopy
from typing import Any, Callable, Mapping

from .r5a_serving import RealizedStaticPlan, reconcile_realization
from .xc_wire import (
    PROTOCOL_ID,
    XCWireError,
    body_checksum,
    encode_frame,
    identity_of,
    recv_frame,
    send_exact,
    validate_response,
)


class RemoteRealizationError(RuntimeError):
    """The remote realization/execution seam failed; the scope fails closed."""


class RemoteEpochRuntime:
    """RPC facade over one Node-agent realization of the frozen plan.

    Implements the ``EpochRuntime`` protocol (``generate``/``report``/``close``)
    the accepted epoch controller already consumes.  ``fail_resource`` is
    deliberately absent: this proof exercises no remote participant-loss arm.
    """

    def __init__(
        self,
        *,
        sock: socket.socket,
        scope_id: str,
        epoch_id: str,
        plan_digest: str,
        observation: Mapping[str, Any],
    ) -> None:
        self._sock = sock
        self._lock = threading.Lock()
        self._scope_id = scope_id
        self._epoch_id = epoch_id
        self._plan_digest = plan_digest
        self._operation_sequence = 0
        self._closed = False
        self._final_report: Mapping[str, Any] | None = None
        self.observation = deepcopy(dict(observation))
        self.reclamation_report: dict[str, Any] = {}

    def _request(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._closed and operation != "CLOSE":
            raise RemoteRealizationError("remote epoch runtime is closed")
        with self._lock:
            body = {
                "kind": "request",
                "protocol": PROTOCOL_ID,
                "scope_id": self._scope_id,
                "epoch_id": self._epoch_id,
                "plan_digest": self._plan_digest,
                "operation": operation,
                **payload,
            }
            try:
                send_exact(self._sock, encode_frame(body))
                response = recv_frame(self._sock)
            except XCWireError as exc:
                raise RemoteRealizationError(
                    f"node-agent exchange for {operation} failed: {exc}"
                ) from exc
        validate_response(response)
        if response.get("protocol") != PROTOCOL_ID:
            raise RemoteRealizationError("response names a different protocol")
        if response.get("epoch_id") != self._epoch_id or response.get(
            "plan_digest"
        ) != self._plan_digest:
            raise RemoteRealizationError(
                "response identity does not match the authorized epoch/plan"
            )
        if response.get("operation") != operation:
            raise RemoteRealizationError("response operation mismatch")
        if not response.get("ok"):
            raise RemoteRealizationError(
                f"node-agent {operation} failed: {response.get('error')}"
            )
        return response

    def generate(
        self,
        *,
        session_id: int,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        on_token: Callable[[int, int, Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        self._operation_sequence += 1
        response = self._request(
            "GENERATE",
            {
                "session_id": int(session_id),
                "position": self._operation_sequence,
                "prompt_token_ids": [int(t) for t in prompt_token_ids],
                "max_new_tokens": int(max_new_tokens),
            },
        )
        result = dict(response["result"])
        if result.get("plan_digest") != self._plan_digest:
            raise RemoteRealizationError(
                "remote runtime silently substituted a plan"
            )
        if response.get("result_checksum") != body_checksum(result):
            raise RemoteRealizationError("remote result checksum mismatch")
        if on_token is not None:
            boundaries = {
                item.get("generated_step"): item
                for item in result.get("boundaries", [])
                if "generated_step" in item
            }
            for step, token in enumerate(result.get("generated_token_ids", [])):
                on_token(step, int(token), deepcopy(boundaries.get(step, {})))
        return result

    def report(self) -> Mapping[str, Any]:
        if self._final_report is not None:
            return deepcopy(dict(self._final_report))
        return deepcopy(dict(self._request("REPORT", {})["result"]))

    def close(self) -> None:
        if self._closed:
            return
        try:
            response = self._request("CLOSE", {})
            self._final_report = deepcopy(dict(response["result"]))
            self.reclamation_report = {
                "kind": "remote-node-agent-epoch-close",
                "node_reclamation": deepcopy(
                    dict(response.get("reclamation") or {})
                ),
                "closed_at_ns": time.time_ns(),
            }
        finally:
            self._closed = True
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()


class RemoteNodeAgentConnection:
    """One bounded connection to one Node-local execution agent."""

    def __init__(self, *, host: str, port: int, scope_id: str) -> None:
        self._host = host
        self._port = int(port)
        self._scope_id = scope_id

    def realize(
        self, execution_plan: Mapping[str, Any], *, timeout: float = 900.0
    ) -> RemoteEpochRuntime:
        """Authorize one remote realization of the frozen Execution Plan."""
        plan = deepcopy(dict(execution_plan))
        digest = plan["digest"]
        try:
            sock = socket.create_connection((self._host, self._port), timeout=timeout)
        except OSError as exc:
            raise RemoteRealizationError(
                f"cannot reach node agent at {self._host}:{self._port}: {exc}"
            ) from exc
        epoch_id = f"remote-realization:{digest.split(':')[-1][:12]}"
        runtime = RemoteEpochRuntime(
            sock=sock,
            scope_id=self._scope_id,
            epoch_id=epoch_id,
            plan_digest=digest,
            observation={},
        )
        try:
            response = runtime._request(
                "REALIZE", {"execution_plan": plan}
            )
        except Exception:
            sock.close()
            raise
        runtime.observation = deepcopy(dict(response["result"]["observation"]))
        runtime.reclamation_report = {
            "kind": "remote-node-agent-realization",
            "node_identity": deepcopy(
                dict(response["result"].get("node_identity") or {})
            ),
        }
        return runtime


def make_remote_realizer(
    *, host: str, port: int, scope_id: str
) -> Callable[[Mapping[str, Any]], RealizedStaticPlan]:
    """Build an ``EpochServingController`` realizer over the remote seam."""

    def realizer(execution_plan: Mapping[str, Any]) -> RealizedStaticPlan:
        connection = RemoteNodeAgentConnection(
            host=host, port=port, scope_id=scope_id
        )
        runtime = connection.realize(execution_plan)
        # Coordinator-side reconciliation against the frozen plan runs here,
        # on the CPU-only control plane, exactly as in accepted R5A/R5B.
        reconcile_realization(execution_plan, runtime.observation)
        return RealizedStaticPlan(
            runtime=runtime, observation=runtime.observation
        )

    return realizer


__all__ = [
    "RemoteEpochRuntime",
    "RemoteNodeAgentConnection",
    "RemoteRealizationError",
    "make_remote_realizer",
    "identity_of",
]
