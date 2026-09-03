"""R6 N-stage dense chain coordinator (compute nodes).

Generalizes the accepted two-block A->B pattern to an ordered chain of
stages, each in its own process with exactly one assigned GPU (possibly on
different nodes via the R4 wire).  Stage roles: ``first`` (embeddings +
layers), any number of ``middle`` (layers only), ``last`` (layers + final
norm + tied lm_head).  Every boundary carries the same frozen geometry
(single-plane bf16 hidden state, plane-major-contiguous, row width 3840 —
one plane; Qwen's 2-plane boundary was a first-model artifact).

Provides the ``EpochRuntime`` protocol (generate/report/close) so the
external-Coordinator serving path consumes a dense chain unchanged.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

from benchmarks.inferswarm_r6.stage_runtime import (
    BOUNDARY_PLANES,
    HIDDEN_SIZE,
)


class StageClient:
    """Control pipe to one spawn-isolated local stage process."""

    def __init__(self, context, *, role, adapter_data, model_path, gpu_index: int):
        import os

        parent, child = context.Pipe()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_index)}
        self.parent = parent
        self.role = role
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

    def recv(self):
        return self.parent.recv()

    def send(self, message):
        self.parent.send(message)

    def request(self, message):
        self.send(message)
        response = self.recv()
        if isinstance(response, dict) and response.get("op") == "ERROR":
            raise RuntimeError(f"stage {self.role} error: {response}")
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
        import torch

        from freetoken.distributed import set_tp_info, try_get_tp_info
        from freetoken.layers.rotary import set_rope_device

        if try_get_tp_info() is None:
            set_tp_info(rank=0, size=1)
        set_rope_device(torch.device("cuda:0"))
        from benchmarks.inferswarm_r6.stage_runtime import GemmaDenseStage

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
                if role != "first" and message.get("hidden") is not None:
                    message["hidden"] = message["hidden"].to(
                        device="cuda:0", dtype=torch.bfloat16
                    )
                if role == "first":
                    hidden, _ = runtime.prefill(
                        message["token_ids"], None, message["position"]
                    )
                    connection.send({"op": "BOUNDARY_PAYLOAD",
                                     "hidden": hidden.cpu()})
                else:
                    out = runtime.prefill(None, message["hidden"], message["position"])
                    if role == "last":
                        connection.send({"op": "TOKEN_RESULT", "token_id": out[0]})
                    else:
                        connection.send(
                            {"op": "BOUNDARY_PAYLOAD", "hidden": out[0].cpu()}
                        )
            elif op == "DECODE":
                if role != "first" and message.get("hidden") is not None:
                    message["hidden"] = message["hidden"].to(
                        device="cuda:0", dtype=torch.bfloat16
                    )
                if role == "first":
                    hidden, _ = runtime.decode(
                        message["token_id"], message["position"]
                    )
                    connection.send({"op": "BOUNDARY_PAYLOAD",
                                     "hidden": hidden.cpu()})
                else:
                    out = runtime.decode(message["hidden"], message["position"])
                    if role == "last":
                        connection.send({"op": "TOKEN_RESULT", "token_id": out[0]})
                    else:
                        connection.send({"op": "BOUNDARY_PAYLOAD",
                                         "hidden": out[0].cpu()})
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


class GemmaStageChainRuntime:
    """EpochRuntime facade over an ordered dense stage chain."""

    def __init__(
        self,
        *,
        stage_specs: list[dict],
        model_path: str,
        runtime_capacity_tokens: int = 256,
    ) -> None:
        """``stage_specs``: ordered list of
        {"role": first|middle|last, "adapter_data": {...}, "gpu_index": int}
        (a remote last stage may be substituted by an R4 wire client with
        the same request/response shape — see r4_last_stage.py)."""
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        self.stages: list[Any] = []
        try:
            for spec in stage_specs:
                shared = spec["adapter_data"].get("declared_shared_state")
                # Tied-embedding shared state materializes ONLY on the stages
                # that own the embedding (first) and the tied lm_head (last);
                # a middle stage owns neither and must not fetch it.
                if spec["role"] == "middle":
                    shared = None
                common = {
                    "runtime_capacity_tokens": runtime_capacity_tokens,
                    "declared_shared_state": shared,
                }
                self.stages.append(
                    StageClient(
                        context,
                        role=spec["role"],
                        adapter_data={**spec["adapter_data"], **common},
                        model_path=model_path,
                        gpu_index=spec["gpu_index"],
                    )
                )
            self.ready = []
            for stage in self.stages:
                ready = stage.recv()
                if ready.get("op") == "ERROR":
                    raise RuntimeError(f"stage failed: {ready}")
                self.ready.append(ready)
        except BaseException:
            for stage in self.stages:
                stage.shutdown()
            raise
        if self.stages[0].role != "first" or self.stages[-1].role != "last":
            raise RuntimeError("chain must start with 'first' and end with 'last'")
        self._sessions: list[dict[str, Any]] = []
        self._closed = False
        self.reclamation_report: dict[str, Any] = {}

    def _chain_prefill(self, token_ids, position, count):
        hidden = None
        for index, stage in enumerate(self.stages):
            if index == 0:
                response = stage.request(
                    {
                        "op": "PREFILL",
                        "token_ids": token_ids,
                        "position": position,
                    }
                )
            else:
                response = stage.request(
                    {"op": "PREFILL", "hidden": hidden, "position": position}
                )
            if response.get("op") == "TOKEN_RESULT":
                return None, response["token_id"]
            hidden = response["hidden"]
        raise RuntimeError("chain ended without a last stage")

    def _chain_decode(self, token_id, position):
        hidden = None
        for index, stage in enumerate(self.stages):
            if index == 0:
                response = stage.request(
                    {"op": "DECODE", "token_id": token_id, "position": position}
                )
            else:
                response = stage.request(
                    {"op": "DECODE", "hidden": hidden, "position": position}
                )
            if response.get("op") == "TOKEN_RESULT":
                return response["token_id"]
            hidden = response["hidden"]
        raise RuntimeError("chain ended without a last stage")

    # -- EpochRuntime protocol ------------------------------------------

    def generate(
        self, *, session_id: int, prompt_token_ids: list[int], max_new_tokens: int,
        on_token=None,
    ) -> dict[str, Any]:
        started = time.perf_counter_ns()
        for stage in self.stages:
            if hasattr(stage, "request"):
                stage.request({"op": "RESET"})
            else:
                stage.send({"op": "RESET"})
                stage.recv()
        chunk = 64  # single-chunk canonical replays; see strategy PREFILL_CHUNK
        position = 0
        token_id = None
        total = len(prompt_token_ids)
        while position < total:
            count = min(chunk, total - position)
            _, token_id = self._chain_prefill(
                prompt_token_ids[position : position + count], position, count
            )
            position += count
        ttft_ns = time.perf_counter_ns() - started
        generated = []
        if token_id is not None:
            generated.append(int(token_id))
            if on_token is not None:
                on_token(0, int(token_id), {"session_id": session_id, "position": 0})
        decode_start = time.perf_counter_ns()
        inter = []
        for step in range(1, max_new_tokens):
            t = time.perf_counter_ns()
            next_id = self._chain_decode(generated[-1], position + step - 1)
            inter.append(time.perf_counter_ns() - t)
            generated.append(int(next_id))
            if on_token is not None:
                on_token(step, int(next_id),
                         {"session_id": session_id, "position": step})
        session = {
            "session_id": session_id,
            "prompt_len": total,
            "generated_token_ids": generated,
            "ttft_ns": ttft_ns,
            "decode_ns": time.perf_counter_ns() - decode_start,
            "inter_token_p50_ns": sorted(inter)[len(inter) // 2] if inter else None,
            "complete_request_wall_ns": time.perf_counter_ns() - started,
        }
        self._sessions.append(deepcopy(session))
        return deepcopy(session)

    def report(self) -> dict[str, Any]:
        reports = [stage.request({"op": "REPORT"})["report"] for stage in self.stages]
        return {
            "stages": reports,
            "sessions": deepcopy(self._sessions),
            "boundary_geometry": {
                "planes": BOUNDARY_PLANES,
                "row_width": HIDDEN_SIZE,
                "dtype": "bfloat16",
                "layout": "plane-major-contiguous",
            },
        }

    def close(self) -> None:
        if self._closed:
            return
        report = self.report()
        for stage in self.stages:
            stage.shutdown()
        self.reclamation_report = {
            "stages_closed": [s.role for s in self.stages],
            "final_report_sessions": len(report["sessions"]),
        }
        self._closed = True


__all__ = ["GemmaStageChainRuntime", "StageClient"]
