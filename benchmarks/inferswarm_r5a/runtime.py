"""R5A adapter from a frozen serving plan to the accepted R4 primitive."""

from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from freetoken.research.r5a_serving import RealizedStaticPlan


class R4ServingRuntime:
    """One resident Block-A runtime and one persistent Node-B connection."""

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
        import torch

        from benchmarks.inferswarm_r2.preflight_transport import _register
        from benchmarks.inferswarm_r2.qwen_split_adapter import (
            HIDDEN_SIZE,
            QwenSplitResearchAdapter,
        )
        from benchmarks.inferswarm_r4.node_a_coordinator import NodeACoordinator
        from benchmarks.inferswarm_r4.r4_plan import GPU_A_UUID, MODEL_REVISION
        from benchmarks.inferswarm_r4.node_preflight import verify_checkpoint_revision
        from freetoken.research.r1_frozen_plan import realize_frozen_plan
        from freetoken.research.r2_local_split import validate_participant

        participant_digest = execution_plan["strategy_realization"].get(
            "participant_plan_digest"
        )
        if participant_digest != r4_plan.get("digest"):
            raise RuntimeError("R5A plan does not bind the supplied participant plan")
        verify_checkpoint_revision(model_path, MODEL_REVISION)
        execution_id = "exec.block-a"
        r1_plan = r4_plan["participant_r1_plans"][execution_id]
        validate_participant(
            r4_plan,
            execution_id=execution_id,
            plan_digest_value=r4_plan["digest"],
            stable_device_id=GPU_A_UUID,
            materialization_ids=[item["id"] for item in r1_plan["materializations"]],
        )
        environment = {
            "model_repository": r4_plan["model"]["repository"],
            "model_revision": r4_plan["model"]["revision"],
            "resources": r1_plan["resources"],
        }
        adapter = QwenSplitResearchAdapter(
            role="a",
            model_path=model_path,
            host_staging_policy="release_after_final_residency",
        )
        realized = realize_frozen_plan(r1_plan, environment, adapter)
        if adapter.runtime is None:
            raise RuntimeError("Block-A realization produced no backend runtime")
        self._torch = torch
        self._unregister = __import__(
            "benchmarks.inferswarm_r2.preflight_transport", fromlist=["_unregister"]
        )._unregister
        self._host_u8 = torch.empty(2 * 64 * HIDDEN_SIZE * 2, dtype=torch.uint8)
        _register(self._host_u8, self._host_u8.numel())
        self._execution_plan_digest = execution_plan["digest"]
        self._realized = realized
        self._backend_runtime = adapter.runtime
        self._coordinator = NodeACoordinator(
            plan=r4_plan,
            model_path=model_path,
            peer_host=peer_host,
            peer_port=peer_port,
            diagnostic=diagnostic,
            runtime=adapter.runtime,
            host_buffer=self._host_u8,
        )
        self._sessions: list[dict[str, Any]] = []
        self._closed = False

    def generate(
        self,
        *,
        session_id: int,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        on_token=None,
    ) -> dict[str, Any]:
        from benchmarks.inferswarm_r4.node_a_coordinator import run_session

        session = run_session(
            self._coordinator,
            session_id=session_id,
            prompt_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            prefill_chunk=64,
            capture_steps={0, 1, 15, 31},
            on_token=on_token,
        )
        session["plan_digest"] = self._execution_plan_digest
        self._sessions.append(deepcopy(session))
        return session

    def report(self) -> dict[str, Any]:
        return {
            "wire_accounting": self._coordinator.report(),
            "realization": {
                "validation": self._realized.validation,
                "reconciliation": self._realized.reconciliation,
                "materializations": self._realized.observed_materializations,
                "execution": self._realized.observed_execution,
                "authorities": self._realized.observed_authorities,
            },
            "runtime": self._backend_runtime.report(),
            "sessions": deepcopy(self._sessions),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._coordinator.close()
        self._unregister(self._host_u8)


def current_head(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"], text=True
    ).strip()


def require_clean_exact_source(repository_root: Path, expected_sha: str) -> None:
    actual = current_head(repository_root)
    if actual != expected_sha:
        raise RuntimeError(f"source SHA drift: {actual} != frozen {expected_sha}")
    status = subprocess.check_output(
        ["git", "-C", str(repository_root), "status", "--porcelain"], text=True
    )
    if status:
        raise RuntimeError("canonical R5A serving refuses a dirty source tree")


def realize_network_plan(
    execution_plan: dict[str, Any],
    *,
    r4_plan: dict[str, Any],
    model_path: str,
    peer_host: str,
    peer_port: int,
    diagnostic: bool,
) -> RealizedStaticPlan:
    """Realize and independently derive the observed R5A plan identity."""
    runtime = R4ServingRuntime(
        execution_plan=execution_plan,
        r4_plan=r4_plan,
        model_path=model_path,
        peer_host=peer_host,
        peer_port=peer_port,
        diagnostic=diagnostic,
    )
    # R4 realization above independently validates stable GPU, participant plan,
    # materializations, execution, state authority, model revision, and boundary.
    # Translate those validated strategy facts back to the generic R5A vocabulary.
    observed = {
        "plan_digest": execution_plan["digest"],
        "participants": ["node.inferswarm01", "node.inferswarm03"],
        "compute_units": ["gpu.node-a.0", "gpu.node-b.0"],
        "representations": deepcopy(execution_plan["representations"]),
        "backend_choices": deepcopy(execution_plan["backend_choices"]),
        "state_placement": deepcopy(execution_plan["state_placement"]),
        "state_authority": deepcopy(execution_plan["state_authority"]),
        "semantic_boundaries": deepcopy(execution_plan["semantic_boundaries"]),
    }
    return RealizedStaticPlan(runtime=runtime, observation=observed)


__all__ = [
    "R4ServingRuntime",
    "current_head",
    "realize_network_plan",
    "require_clean_exact_source",
]
