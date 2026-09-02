"""R6 two-stage dense local coordinator (compute-node side).

Drives stage A (embeddings + layers [0,24)) and stage B (layers [24,48) +
final norm + tied lm_head) through one process per GPU, with the boundary
crossing GPU-to-GPU over the accepted R4 wire (semantics identical to the
accepted R2/R4 boundary: 2-plane bf16, plane-major-contiguous, row width
3840).  Provides the ``EpochRuntime`` protocol (generate/report/close) so
the external-Coordinator serving path consumes it unchanged.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

from benchmarks.inferswarm_r6.stage_runtime import (
    BOUNDARY_PLANES,
    HIDDEN_SIZE,
    GemmaDenseStage,
)


class _StageProcessClient:
    """Control pipe to one spawn-isolated stage process (one assigned GPU)."""

    def __init__(self, context, *, role, adapter_data, model_path, gpu_index: int):
        import os

        parent, child = context.Pipe()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_index)}
        self.parent = parent
        self.process = context.Process(
            target=_stage_entry_env,
            args=(env,),
            kwargs={
                "role": role,
                "adapter_data": adapter_data,
                "model_path": model_path,
                "connection": child,
            },
        )
        self.process.start()
        self.ready = None

    def recv(self):
        return self.parent.recv()

    def send(self, message):
        self.parent.send(message)

    def request(self, message):
        self.send(message)
        response = self.recv()
        if isinstance(response, dict) and response.get("op") == "ERROR":
            raise RuntimeError(f"stage error: {response}")
        return response

    def shutdown(self):
        try:
            self.send({"op": "SHUTDOWN"})
            self.parent.recv()
        except (BrokenPipeError, EOFError, OSError):
            pass
        self.process.join(timeout=30)
        if self.process.is_alive():
            self.process.terminate()


def _stage_entry_env(env: dict, *, role, adapter_data, model_path, connection):
    import os

    os.environ.clear()
    os.environ.update(env)
    _stage_entry(role=role, adapter_data=adapter_data, model_path=model_path,
                 connection=connection)


def _stage_entry(*, role, adapter_data, model_path, connection):
    import traceback

    try:
        runtime = GemmaDenseStage(
            role=role, model_path=model_path, adapter_data=dict(adapter_data)
        )
        connection.send(
            {
                "op": "READY",
                "role": role,
                "runtime_report": runtime.report("P4_ready_for_resident_execution"),
            }
        )
        while True:
            message = connection.recv()
            op = message["op"]
            if op == "PREFILL":
                if role == "a":
                    hidden, _ = runtime.prefill(
                        message["token_ids"], None, message["position"]
                    )
                    connection.send(
                        {
                            "op": "BOUNDARY_PAYLOAD",
                            "hidden": hidden,
                            "compute_ns": 0,
                        }
                    )
                else:
                    token, logits = runtime.prefill(
                        None, message["hidden"], message["position"]
                    )
                    connection.send({"op": "TOKEN_RESULT", "token_id": token})
            elif op == "DECODE":
                if role == "a":
                    hidden, _ = runtime.decode(message["token_id"], message["position"])
                    connection.send({"op": "BOUNDARY_PAYLOAD", "hidden": hidden})
                else:
                    token, logits = runtime.decode(
                        message["hidden"], message["position"]
                    )
                    connection.send({"op": "TOKEN_RESULT", "token_id": token})
            elif op == "STATE_RECORD":
                connection.send(
                    {
                        "op": "STATE",
                        "records": runtime.logical_state_records(message["used_tokens"]),
                    }
                )
            elif op == "REPORT":
                connection.send({"op": "REPORT", "report": runtime.report()})
            elif op == "RESET":
                runtime.reset_session_state()
                connection.send({"op": "ACK"})
            elif op == "SHUTDOWN":
                connection.send({"op": "ACK", "report": runtime.report()})
                return
            else:
                raise RuntimeError(f"unknown stage op {op!r}")
    except BaseException as exc:
        try:
            connection.send(
                {
                    "op": "ERROR",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass


class GemmaTwoStageRuntime:
    """EpochRuntime facade over the two stage processes (one per GPU)."""

    def __init__(
        self,
        *,
        adapter_data_a: dict,
        adapter_data_b: dict,
        model_path: str,
        runtime_capacity_tokens: int = 256,
    ) -> None:
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        common = {
            "runtime_capacity_tokens": runtime_capacity_tokens,
            "declared_shared_state": adapter_data_a.get("declared_shared_state"),
        }
        self.stage_a = _StageProcessClient(
            context,
            role="a",
            adapter_data={**adapter_data_a, **common},
            model_path=model_path,
            gpu_index=0,
        )
        try:
            self.stage_b = _StageProcessClient(
                context,
                role="b",
                adapter_data={**adapter_data_b, **common},
                model_path=model_path,
                gpu_index=1,
            )
        except BaseException:
            self.stage_a.shutdown()
            raise
        self.ready = {
            "a": self.stage_a.recv(),
            "b": self.stage_b.recv(),
        }
        for role, ready in self.ready.items():
            if ready.get("op") == "ERROR":
                raise RuntimeError(f"stage {role} failed: {ready}")
        self._sessions: list[dict[str, Any]] = []
        self._closed = False
        self.reclamation_report: dict[str, Any] = {}

    # -- EpochRuntime protocol ------------------------------------------

    def generate(
        self, *, session_id: int, prompt_token_ids: list[int], max_new_tokens: int,
        on_token=None,
    ) -> dict[str, Any]:
        started = time.perf_counter_ns()
        self.stage_a.send({"op": "RESET"})
        self.stage_b.send({"op": "RESET"})
        self.stage_a.recv()
        self.stage_b.recv()
        # Prefill in chunks bounded by the frozen boundary geometry.
        chunk = 32
        position = 0
        token_id = None
        prefill_ns = 0
        total = len(prompt_token_ids)
        while position < total:
            count = min(chunk, total - position)
            t = time.perf_counter_ns()
            hidden, _ = (
                self.stage_a.request(
                    {
                        "op": "PREFILL",
                        "token_ids": prompt_token_ids[position : position + count],
                        "position": position,
                    }
                ).get("hidden"),
                None,
            )
            token_id = self.stage_b.request(
                {"op": "PREFILL", "hidden": hidden, "position": position}
            )["token_id"]
            prefill_ns += time.perf_counter_ns() - t
            position += count
        ttft_ns = time.perf_counter_ns() - started
        generated = []
        steps: list[dict] = []
        # NOTE: first generated token comes from the last prefill chunk.
        if token_id is not None:
            generated.append(int(token_id))
            steps.append(
                {"step": 0, "token_id": int(token_id), "boundary": "prefill-final"}
            )
            if on_token is not None:
                on_token(0, int(token_id), {"session_id": session_id, "position": 0})
        decode_start = time.perf_counter_ns()
        inter = []
        for step in range(1, max_new_tokens):
            t = time.perf_counter_ns()
            hidden, _ = (
                self.stage_a.request(
                    {
                        "op": "DECODE",
                        "token_id": generated[-1],
                        "position": position + step - 1,
                    }
                ).get("hidden"),
                None,
            )
            token_id = self.stage_b.request(
                {"op": "DECODE", "hidden": hidden, "position": position + step - 1}
            )["token_id"]
            inter.append(time.perf_counter_ns() - t)
            generated.append(int(token_id))
            if on_token is not None:
                on_token(
                    step, int(token_id), {"session_id": session_id, "position": step}
                )
        wall_ns = time.perf_counter_ns() - started
        session = {
            "session_id": session_id,
            "prompt_len": total,
            "generated_token_ids": generated,
            "ttft_ns": ttft_ns,
            "prefill_ns": prefill_ns,
            "decode_ns": time.perf_counter_ns() - decode_start,
            "inter_token_p50_ns": sorted(inter)[len(inter) // 2] if inter else None,
            "complete_request_wall_ns": wall_ns,
            "steps": steps,
        }
        self._sessions.append(deepcopy(session))
        return deepcopy(session)

    def report(self) -> dict[str, Any]:
        a = self.stage_a.request({"op": "REPORT"})
        b = self.stage_b.request({"op": "REPORT"})
        return {
            "stages": {"a": a["report"], "b": b["report"]},
            "sessions": deepcopy(self._sessions),
            "boundary_geometry": {
                "planes": BOUNDARY_PLANES,
                "row_width": HIDDEN_SIZE,
                "dtype": "bfloat16",
                "transport": "process-pipe-tensors",
            },
        }

    def close(self) -> None:
        if self._closed:
            return
        report = self.report()
        self.stage_a.shutdown()
        self.stage_b.shutdown()
        self.reclamation_report = {
            "stages_closed": ["a", "b"],
            "final_report_sessions": len(report["sessions"]),
        }
        self._closed = True


__all__ = ["GemmaTwoStageRuntime"]
