"""R6 node-agent runtime builder: the dense 3-stage chain as an EpochRuntime.

Runs INSIDE the node-agent process on inferswarm01: spawns stage 1 (GPU0)
and stage 2 (GPU1) locally, connects stage 3 (last) to the remote
last-stage service on inferswarm03 over the accepted R4 wire.  Presents
generate/report/close so the accepted xc node-agent drives it exactly as
it drove the R5B-isolated R4 runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODEL_PATH_DEFAULT = "/srv/models/gemma-r6"
LAST_STAGE_HOST = "10.0.0.219"
LAST_STAGE_PORT = 18485


class ChainEpochRuntime:
    """EpochRuntime facade over the dense chain (stages 1-2 local, 3 wire)."""

    def __init__(
        self,
        *,
        chain_plan_path: str,
        model_path: str = MODEL_PATH_DEFAULT,
        last_stage_host: str = LAST_STAGE_HOST,
        last_stage_port: int = LAST_STAGE_PORT,
    ) -> None:
        from benchmarks.inferswarm_r6.stage_chain import (
            GemmaStageChainRuntime,
            StageClient,
        )
        from benchmarks.inferswarm_r6.wire_client import RemoteLastStageClient

        import multiprocessing

        plan = json.loads(Path(chain_plan_path).read_text())
        self.plan_digest = plan["digest"]
        shared = plan.get("declared_shared_state")
        context = multiprocessing.get_context("spawn")

        runtime = self

        class _Chain(GemmaStageChainRuntime):
            def __init__(self):
                self.stages = []
                try:
                    for index, block in enumerate(plan["blocks"][:-1]):
                        self.stages.append(
                            StageClient(
                                context,
                                role="first" if index == 0 else "middle",
                                adapter_data={
                                    **block,
                                    "declared_shared_state": (
                                        shared if index == 0 else None
                                    ),
                                    "runtime_capacity_tokens": plan[
                                        "runtime_capacity_tokens"
                                    ],
                                },
                                model_path=model_path,
                                gpu_index=index,
                            )
                        )
                    self.stages.append(
                        RemoteLastStageClient(
                            host=last_stage_host,
                            port=last_stage_port,
                            experiment_id=plan["digest"],
                        )
                    )
                    self.ready = []
                    for stage in self.stages[:-1]:
                        ready = stage.recv()
                        if ready.get("op") == "ERROR":
                            raise RuntimeError(f"stage failed: {ready}")
                        self.ready.append(ready)
                    self.ready.append(
                        {"op": "READY", "role": "last", "remote": True}
                    )
                except BaseException:
                    for stage in self.stages:
                        stage.shutdown()
                    raise
                self._sessions = []
                self._closed = False
                self.reclamation_report = {}

        self._chain = _Chain()
        ready_reports = {
            ready.get("role"): ready.get("runtime_report", {})
            for ready in self._chain.ready
        }
        self.observation = {
            "schema": "inferswarm.r6.chain-realization-observation/1",
            "participant_plan_digest": plan["digest"],
            "stages": [
                {
                    "role": role,
                    "global_layer_ids": report.get("global_layer_ids"),
                    "fetched_bytes": report.get("fetched_bytes"),
                    "cuda_allocated_bytes": report.get("cuda_allocated_bytes"),
                }
                for role, report in ready_reports.items()
                if role in ("first", "middle")
            ]
            + [{"role": "last", "remote": True}],
        }

    # -- EpochRuntime protocol ------------------------------------------

    def generate(
        self, *, session_id: int, prompt_token_ids: list[int],
        max_new_tokens: int, on_token=None,
    ) -> dict[str, Any]:
        result = self._chain.generate(
            session_id=session_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            on_token=on_token,
        )
        result["plan_digest"] = self.plan_digest
        return result

    def report(self) -> dict[str, Any]:
        return self._chain.report()

    def close(self) -> None:
        self._chain.close()


def realize_dense_chain(
    execution_plan: dict[str, Any],
    *,
    chain_plan_path: str,
    model_path: str,
    last_stage_host: str,
    last_stage_port: int,
    **_ignored: Any,
) -> ChainEpochRuntime:
    """Signature-compatible with realize_isolated_network_plan so the
    node-agent's build_runtime can call either unchanged."""
    runtime = ChainEpochRuntime(
        chain_plan_path=chain_plan_path,
        model_path=model_path,
        last_stage_host=last_stage_host,
        last_stage_port=last_stage_port,
    )
    # Coordinator-side reconciliation compares these fields verbatim; echo
    # exactly what the frozen plan declares for the realized chain.
    for field in (
        "participants",
        "compute_units",
        "representations",
        "backend_choices",
        "state_placement",
        "state_authority",
        "semantic_boundaries",
    ):
        runtime.observation[field] = execution_plan[field]
    runtime.observation["plan_digest"] = execution_plan["digest"]
    return runtime


__all__ = ["ChainEpochRuntime", "realize_dense_chain"]
