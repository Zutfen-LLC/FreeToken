"""Epoch-lifetime process wrapper around the accepted R4 resident runtime.

R4 was accepted with process-lifetime CUDA ownership.  R5B can activate R4 more
than once in one host serving session, so each immutable R4 epoch owns a fresh
Block-A process.  The child constructs and executes the unchanged accepted R4
runtime; this module only supplies lifecycle isolation and a control pipe.
"""

from __future__ import annotations

import multiprocessing
import traceback
from copy import deepcopy
from typing import Any, Mapping

from freetoken.research.r5a_serving import RealizedStaticPlan


def _send_error(connection, exc: BaseException) -> None:
    try:
        connection.send(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    except (BrokenPipeError, EOFError, OSError):
        pass


def _r4_epoch_process(
    connection,
    execution_plan: dict[str, Any],
    r4_plan: dict[str, Any],
    model_path: str,
    peer_host: str,
    peer_port: int,
    diagnostic: bool,
) -> None:
    runtime = None
    try:
        from benchmarks.inferswarm_r5a.runtime import realize_network_plan

        realized = realize_network_plan(
            execution_plan,
            r4_plan=r4_plan,
            model_path=model_path,
            peer_host=peer_host,
            peer_port=peer_port,
            diagnostic=diagnostic,
        )
        runtime = realized.runtime
        connection.send({"ok": True, "observation": dict(realized.observation)})
        while True:
            request = connection.recv()
            operation = request["operation"]
            if operation == "GENERATE":
                result = runtime.generate(
                    session_id=request["session_id"],
                    prompt_token_ids=request["prompt_token_ids"],
                    max_new_tokens=request["max_new_tokens"],
                )
                connection.send({"ok": True, "result": result})
            elif operation == "REPORT":
                connection.send({"ok": True, "result": runtime.report()})
            elif operation == "CLOSE":
                final_report = runtime.report()
                runtime.close()
                connection.send(
                    {
                        "ok": True,
                        "result": final_report,
                        "accepted_runtime_reclamation": deepcopy(
                            dict(runtime.reclamation_report)
                        ),
                    }
                )
                runtime = None
                return
            else:
                raise RuntimeError(f"unknown isolated R4 operation {operation!r}")
    except BaseException as exc:  # child must return exact fail-closed cause
        _send_error(connection, exc)
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                pass
        connection.close()


class EpochIsolatedR4Runtime:
    """RPC facade whose child exclusively owns one accepted R4 realization."""

    def __init__(
        self,
        *,
        execution_plan: dict[str, Any],
        r4_plan: dict[str, Any],
        model_path: str,
        peer_host: str,
        peer_port: int,
        diagnostic: bool,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            target=_r4_epoch_process,
            args=(
                child,
                execution_plan,
                r4_plan,
                model_path,
                peer_host,
                peer_port,
                diagnostic,
            ),
            name=f"inferswarm-r5b-r4-{execution_plan['digest'][-12:]}",
        )
        self._process.start()
        child.close()
        self._closed = False
        self._final_report: Mapping[str, Any] | None = None
        self.reclamation_report: dict[str, Any] = {}
        try:
            ready = self._receive(timeout=300)
        except Exception:
            self._connection.close()
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(10)
            raise
        self.observation = deepcopy(dict(ready["observation"]))

    def _receive(self, *, timeout: float = 300) -> dict[str, Any]:
        if not self._connection.poll(timeout):
            raise TimeoutError("isolated R4 epoch process did not respond")
        try:
            response = dict(self._connection.recv())
        except EOFError as exc:
            raise RuntimeError(
                f"isolated R4 epoch process exited with {self._process.exitcode}"
            ) from exc
        if not response.get("ok"):
            raise RuntimeError(
                response.get("error", "isolated R4 epoch process failed")
                + "\n"
                + response.get("traceback", "")
            )
        return response

    def _rpc(self, operation: str, **payload) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("isolated R4 epoch is closed")
        self._connection.send({"operation": operation, **payload})
        return self._receive()

    def generate(
        self,
        *,
        session_id: int,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        on_token=None,
    ) -> dict[str, Any]:
        result = dict(
            self._rpc(
                "GENERATE",
                session_id=session_id,
                prompt_token_ids=list(prompt_token_ids),
                max_new_tokens=max_new_tokens,
            )["result"]
        )
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
        return deepcopy(dict(self._rpc("REPORT")["result"]))

    def close(self) -> None:
        if self._closed:
            return
        response = self._rpc("CLOSE")
        self._final_report = deepcopy(dict(response["result"]))
        accepted = deepcopy(dict(response.get("accepted_runtime_reclamation", {})))
        self._closed = True
        self._connection.close()
        self._process.join(30)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(10)
        self.reclamation_report = {
            "kind": "r5b-epoch-process-isolation-around-accepted-r4-runtime",
            "accepted_runtime_reclamation": accepted,
            "process_pid": self._process.pid,
            "process_exit_code": self._process.exitcode,
            "process_stopped": not self._process.is_alive(),
        }


def realize_isolated_network_plan(
    execution_plan: dict[str, Any],
    *,
    r4_plan: dict[str, Any],
    model_path: str,
    peer_host: str,
    peer_port: int,
    diagnostic: bool,
) -> RealizedStaticPlan:
    runtime = EpochIsolatedR4Runtime(
        execution_plan=execution_plan,
        r4_plan=r4_plan,
        model_path=model_path,
        peer_host=peer_host,
        peer_port=peer_port,
        diagnostic=diagnostic,
    )
    return RealizedStaticPlan(runtime=runtime, observation=runtime.observation)


__all__ = ["EpochIsolatedR4Runtime", "realize_isolated_network_plan"]
