"""R5A adapter from a frozen serving plan to the accepted R4 primitive."""

from __future__ import annotations

import gc
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from freetoken.research.r5a_serving import RealizedStaticPlan


def require_current_local_split_devices(local_gate: dict[str, Any]) -> None:
    """Recheck frozen UUID/BDF/capacity before any R2 child materializes."""
    if local_gate.get("result") != "LOCAL_SPLIT_PREFLIGHT_PASSED":
        raise RuntimeError("local split lacks a successful frozen preflight")
    fields = "uuid,pci.bus_id,memory.total"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
    )
    current = {}
    for line in output.splitlines():
        uuid, bdf, total_mib = (item.strip() for item in line.split(","))
        current[uuid] = {
            "pci_bdf": bdf,
            "capacity_bytes": int(total_mib) * 1024 * 1024,
        }
    for expected in local_gate["vram_headroom"].values():
        actual = current.get(expected["uuid"])
        if actual is None:
            raise RuntimeError(f"frozen local GPU {expected['uuid']} is absent")
        if actual["pci_bdf"] != expected["pci_bdf"]:
            raise RuntimeError(f"frozen local GPU {expected['uuid']} BDF drifted")
        if actual["capacity_bytes"] != expected["capacity_bytes"]:
            raise RuntimeError(f"frozen local GPU {expected['uuid']} capacity drifted")
        if (
            expected["required_bytes"] + expected["reservation_bytes"]
            > actual["capacity_bytes"]
        ):
            raise RuntimeError(f"frozen local GPU {expected['uuid']} lost headroom")


class R2ServingRuntime:
    """Ordinary-serving adapter around the accepted local R2 coordinator."""

    def __init__(
        self,
        *,
        execution_plan: dict[str, Any],
        local_plan: dict[str, Any],
        local_plan_path: Path,
        model_path: str,
        diagnostic: bool,
        local_gate: dict[str, Any],
    ) -> None:
        from benchmarks.inferswarm_r2.coordinator import LocalSplitCoordinator
        from benchmarks.inferswarm_r4.node_preflight import verify_checkpoint_revision
        from freetoken.research.r2_local_split import plan_digest

        participant_digest = execution_plan["strategy_realization"].get(
            "participant_plan_digest"
        )
        calculated = f"sha256:{plan_digest(local_plan)}"
        if participant_digest != local_plan.get("digest") or calculated != participant_digest:
            raise RuntimeError("R5A plan does not bind an intact local participant plan")
        if local_gate.get("participant_plan_digest") != participant_digest:
            raise RuntimeError("local participant plan differs from frozen preflight")
        if not local_gate.get("representation_backend", {}).get("compatible"):
            raise RuntimeError("local representation/backend compatibility is not proven")
        verify_checkpoint_revision(model_path, local_plan["model"]["revision"])
        require_current_local_split_devices(local_gate)
        self._execution_plan_digest = execution_plan["digest"]
        self._coordinator = LocalSplitCoordinator(
            plan_path=str(local_plan_path),
            model_path=model_path,
            diagnostic=diagnostic,
            diagnostic_prefill_chunk=64,
            host_staging_policy="release_after_final_residency",
        )
        self._sessions: list[dict[str, Any]] = []
        self._closed = False
        self._final_reports: dict[str, Any] | None = None
        self._last_reports: dict[str, Any] | None = None
        self._failed_resources: list[dict[str, Any]] = []
        self.reclamation_report: dict[str, Any] = {}

    def generate(
        self,
        *,
        session_id: int,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        on_token=None,
    ) -> dict[str, Any]:
        session = self._coordinator.run_session(
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
        if self._final_reports is not None:
            participant_reports = deepcopy(self._final_reports)
        else:
            participant_reports = self._coordinator.reports()
            self._last_reports = deepcopy(participant_reports)
        return {
            "transport_accounting": {
                "kind": "registered-pinned-host-staging",
                "participant_plan_digest": self._coordinator.plan["digest"],
                "block_a_activation_bytes": participant_reports["a"]["activation_bytes"],
                "block_b_activation_bytes": participant_reports["b"]["activation_bytes"],
                "block_a_control_rx_bytes": participant_reports["a"]["control_rx_bytes"],
                "block_a_control_tx_bytes": participant_reports["a"]["control_tx_bytes"],
                "block_b_control_rx_bytes": participant_reports["b"]["control_rx_bytes"],
                "block_b_control_tx_bytes": participant_reports["b"]["control_tx_bytes"],
            },
            "ready": deepcopy(self._coordinator.ready),
            "participants": deepcopy(participant_reports),
            "sessions": deepcopy(self._sessions),
            "failed_resources": deepcopy(getattr(self, "_failed_resources", [])),
        }

    def fail_resource(self, resource_id: str) -> None:
        """R5B controlled real loss of the required local Block-B participant."""
        if resource_id != "gpu.node-a.1":
            raise RuntimeError(f"R2 runtime cannot fail unknown resource {resource_id}")
        process = self._coordinator.processes["b"]
        before = process.is_alive()
        if before:
            process.terminate()
            process.join(10)
        self._failed_resources.append(
            {
                "resource_id": resource_id,
                "participant_role": "b",
                "pid": process.pid,
                "alive_before": before,
                "alive_after": process.is_alive(),
                "exit_code": process.exitcode,
                "physical_execution_capability_lost": not process.is_alive(),
            }
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._last_reports is None and all(
            process.is_alive() for process in self._coordinator.processes.values()
        ):
            self._last_reports = self._coordinator.reports()
        self._final_reports = deepcopy(self._last_reports)
        self._closed = True
        self._coordinator.shutdown()
        processes = getattr(self._coordinator, "processes", {})
        self.reclamation_report = {
            "kind": "accepted-r2-child-process-reclamation",
            "participant_exit_codes": {
                role: process.exitcode
                for role, process in processes.items()
            },
            "all_participants_stopped": all(
                not process.is_alive() for process in processes.values()
            ),
        }


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
        self.reclamation_report: dict[str, Any] = {}

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
        before = {
            "allocated_bytes": int(self._torch.cuda.memory_allocated(0)),
            "reserved_bytes": int(self._torch.cuda.memory_reserved(0)),
        }
        self._coordinator.close()
        self._unregister(self._host_u8)
        # The accepted R4 service historically ended with its process. R5B
        # reuses it inside a longer-lived host process, so all owning references
        # must be dropped before the replacement can materialize on GPU A0.
        runtime = self._backend_runtime
        from freetoken import core as core_module

        if core_module._GLOBAL_CTX is not runtime.ctx:
            raise RuntimeError(
                "R5B cannot prove ownership of the process-global runtime context"
            )
        core_module._GLOBAL_CTX = None
        graph = getattr(getattr(runtime, "decode_graph", None), "graph", None)
        if graph is not None and hasattr(graph, "reset"):
            graph.reset()
        self._coordinator.runtime = None
        self._realized.adapter.runtime = None
        runtime.decode_graph = None
        runtime.cache = None
        runtime.ctx = None
        runtime.block = None
        runtime.loaded = None
        self._host_u8 = None
        self._coordinator = None
        self._backend_runtime = None
        self._realized = None
        gc.collect()
        self._torch.cuda.empty_cache()
        self._torch.cuda.synchronize(0)
        self.reclamation_report = {
            "kind": "accepted-r4-in-process-materialization-reclamation",
            "before": before,
            "after": {
                "allocated_bytes": int(self._torch.cuda.memory_allocated(0)),
                "reserved_bytes": int(self._torch.cuda.memory_reserved(0)),
            },
        }


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


def realize_local_split_plan(
    execution_plan: dict[str, Any],
    *,
    local_plan: dict[str, Any],
    local_plan_path: Path,
    model_path: str,
    diagnostic: bool,
    local_gate: dict[str, Any],
) -> RealizedStaticPlan:
    """Realize the selected same-node candidate through accepted R2."""
    runtime = R2ServingRuntime(
        execution_plan=execution_plan,
        local_plan=local_plan,
        local_plan_path=local_plan_path,
        model_path=model_path,
        diagnostic=diagnostic,
        local_gate=local_gate,
    )
    observed = {
        "plan_digest": execution_plan["digest"],
        "participants": ["node.inferswarm01"],
        "compute_units": ["gpu.node-a.0", "gpu.node-a.1"],
        "representations": deepcopy(execution_plan["representations"]),
        "backend_choices": deepcopy(execution_plan["backend_choices"]),
        "state_placement": deepcopy(execution_plan["state_placement"]),
        "state_authority": deepcopy(execution_plan["state_authority"]),
        "semantic_boundaries": deepcopy(execution_plan["semantic_boundaries"]),
    }
    return RealizedStaticPlan(runtime=runtime, observation=observed)


__all__ = [
    "R2ServingRuntime",
    "R4ServingRuntime",
    "current_head",
    "realize_local_split_plan",
    "realize_network_plan",
    "require_current_local_split_devices",
    "require_clean_exact_source",
]
