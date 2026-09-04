"""#76 local stage entry: stages 1-2 of the chain with per-case capture.

Reuses the accepted R6 stage-process protocol (benchmarks.inferswarm_r6.
stage_chain._stage_entry shape) with these evidence-side differences:

- ``CASE_ARM {out_dir, tag, gpu_uuid, after_layers}`` swaps in a fresh
  ``RowPruningSink`` for the next case (the R6 ARM_CAPTURE armed one sink
  for the whole connection);
- ``SAVE_CAPTURE {suffix}`` persists the current sink (same as R6);
- the #76 capture wrappers (o_proj checkpoints) are installed once at
  startup via ``arm_full_capture``; they read the CURRENT sink through
  ``runtime._capture_sink`` so per-case swapping just works.
"""

from __future__ import annotations

import os


def _stage_entry(*, role, adapter_data, model_path, connection):
    import traceback

    try:
        import torch

        from freetoken.distributed import set_tp_info, try_get_tp_info
        from freetoken.layers.rotary import set_rope_device

        if try_get_tp_info() is None:
            set_tp_info(rank=0, size=1)
        set_rope_device(torch.device("cuda:0"))

        from benchmarks.inferswarm_76.capture import (
            RowPruningSink,
            arm_full_capture,
        )
        from benchmarks.inferswarm_r6.stage_runtime import GemmaDenseStage

        runtime = GemmaDenseStage(
            role=role, model_path=model_path, adapter_data=dict(adapter_data)
        )
        sink: RowPruningSink | None = None
        capture_out_dir = None
        capture_tag = None

        def _arm(message) -> None:
            nonlocal sink, capture_out_dir, capture_tag
            capture_out_dir = message["out_dir"]
            capture_tag = message["tag"]
            sink = RowPruningSink(
                role=role, gpu_uuid=message.get("gpu_uuid")
            )
            runtime._capture_sink = sink
            runtime._capture_after_layers = frozenset(
                int(x) for x in message.get("after_layers", [])
            )

        # wrappers read runtime._capture_sink dynamically; arming with a
        # placeholder sink here only installs the bound methods once.
        runtime._capture_sink = RowPruningSink(role=role)
        arm_full_capture(runtime, runtime._capture_sink)
        runtime._capture_sink = None

        connection.send({
            "op": "READY",
            "role": role,
            "runtime_report": runtime.report("P4_ready_for_resident_execution"),
        })
        while True:
            message = connection.recv()
            op = message["op"]
            if op == "CASE_ARM":
                _arm(message)
                connection.send({"op": "ACK"})
            elif op == "SAVE_CAPTURE":
                if sink is None or capture_out_dir is None:
                    raise RuntimeError("SAVE_CAPTURE without CASE_ARM")
                manifest = sink.save(
                    capture_out_dir, f"{capture_tag}-{message['suffix']}"
                )
                connection.send({"op": "ACK", "manifest": manifest})
            elif op == "PREFILL":
                if role != "first" and message.get("hidden") is not None:
                    message["hidden"] = message["hidden"].to(
                        device="cuda:0", dtype=torch.bfloat16
                    )
                if message.get("capture_step") is not None:
                    runtime._capture_step = int(message["capture_step"])
                if role == "first":
                    hidden, _ = runtime.prefill(
                        message["token_ids"], None, message["position"]
                    )
                    if message.get("capture_step") is not None:
                        runtime._capture_step = None
                    connection.send({"op": "BOUNDARY_PAYLOAD",
                                     "hidden": hidden.cpu()})
                else:
                    out = runtime.prefill(
                        None, message["hidden"], message["position"]
                    )
                    if message.get("capture_step") is not None:
                        runtime._capture_step = None
                    connection.send({"op": "BOUNDARY_PAYLOAD",
                                     "hidden": out[0].cpu()})
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
            connection.send({
                "op": "ERROR",
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            })
        except (BrokenPipeError, EOFError, OSError):
            pass


class I76StageClient:
    """Control pipe to one spawn-isolated #76 stage process."""

    def __init__(self, context, *, role, adapter_data, model_path, gpu_index: int):
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
    os.environ.clear()
    os.environ.update(env)
    _stage_entry(role=role, adapter_data=adapter_data, model_path=model_path,
                 connection=connection)
