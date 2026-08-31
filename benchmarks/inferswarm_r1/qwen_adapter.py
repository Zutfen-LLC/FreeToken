"""Pinned model adapter for the R1 research plan; not a public strategy API."""

from __future__ import annotations

import gc
import weakref
from typing import ClassVar

import torch
from freetoken.research.n0_model_block import (
    ModelBlockSpec,
    load_selective_qwen35_block,
)

from benchmarks.inferswarm_p48.run_resident_block import (
    _checkpoint,
    _owned_host_roles,
    _populate_all_experts,
    _post_detach_load_sentinels,
    _restore_runtime_state,
    _run_block,
    _setup,
    _snapshot_runtime_state,
    _unique_tensor_bytes,
)


class QwenBlockAResearchAdapter:
    """One deliberately narrow adapter for the pinned Block A fixture."""

    _representations: ClassVar[dict[str, set[str]]] = {
        "state.block-a.non-routed": {"freetoken-native-device"},
        "state.block-a.routed": {
            "checkpoint-native-host",
            "freetoken-nvfp4-slot-banks",
        },
        "state.block-a.mutable-runtime": {"freetoken-block-runtime-device"},
        "state.block-a.replay-baseline": {"freetoken-runtime-snapshot-device"},
    }

    def __init__(self, *, model_path: str, fixture_path: str, repetitions: int = 4):
        self.model_path = model_path
        self.fixture_path = fixture_path
        self.repetitions = repetitions
        self.result = self.ctx = self.cache = self.block_state = self.replay_state = (
            None
        )
        self.fixture = None
        self.prefill = None
        self.source_refs = []
        self.source_bytes = 0
        self.released_owner_bytes = 0
        self.detach_report = {}
        self.checkpoints = []
        self.plan = None

    def supports_representation(self, logical_state_id, representation):
        return representation in self._representations.get(logical_state_id, set())

    def supports_execution(self, execution):
        return (
            execution.get("strategy_unit")
            == "qwen-block-a.same-backend-resident-decode-v1"
        )

    def begin(self, plan, environment):
        if not torch.cuda.is_available():
            raise RuntimeError("physical R1 realization requires CUDA hardware")
        if self.repetitions < 2:
            raise ValueError("R1 requires repeated steady-state decode")
        self.plan = plan
        self.environment = environment
        self.device = torch.device(environment["device"])
        torch.cuda.set_device(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.fixture = torch.load(
            self.fixture_path, map_location="cpu", weights_only=False
        )
        model = plan["model"]
        if (
            self.fixture.get("model") != model["repository"]
            or self.fixture.get("revision") != model["revision"]
        ):
            raise ValueError(
                "same-backend fixture model/revision does not match frozen plan"
            )
        self.checkpoints = [_checkpoint("process_baseline")]

    def _load_selective(self):
        if self.result is not None:
            return
        config = self.plan["adapter_data"]
        spec = ModelBlockSpec(**config["spec"])
        allowed = frozenset(config["allowed_tensor_keys"])
        self.result = load_selective_qwen35_block(
            self.model_path, spec, allowed, device=self.device
        )
        self.source_bytes = _unique_tensor_bytes(
            self.result.expert_banks, device_type="cpu"
        )
        self.source_refs = [
            weakref.ref(tensor)
            for per_layer in self.result.expert_banks.values()
            for tensor in per_layer
        ]
        self.checkpoints.append(
            _checkpoint(
                "after_selective_host_materialization",
                host_source_bytes=self.source_bytes,
                host_roles={
                    "same_backend_correctness_reference_tensors": _unique_tensor_bytes(
                        self.fixture, device_type="cpu"
                    )
                },
                block=self.result.block,
            )
        )

    def _setup_runtime(self):
        self._load_selective()
        if self.cache is None:
            self.ctx, self.cache, self.block_state = _setup(
                self.result, device=self.device
            )
            self.checkpoints.append(
                _checkpoint(
                    "after_accelerator_cache_allocation",
                    cache=self.cache,
                    host_roles={
                        "same_backend_correctness_reference_tensors": _unique_tensor_bytes(
                            self.fixture, device_type="cpu"
                        )
                    },
                    block=self.result.block,
                    state=self.block_state,
                )
            )

    def _memory_resource_id(self, kind):
        matches = [
            resource["id"]
            for node in self.environment["resources"]["nodes"]
            for resource in node["memory_resources"]
            if resource.get("kind") == kind
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"fixture requires exactly one observed {kind!r} Memory Resource"
            )
        return matches[0]

    def realize_materialization(self, item):
        state_id = item["logical_state_id"]
        self._load_selective()
        if item["role"] == "staging":
            observed_bytes = self.source_bytes
            if any(
                tensor.device.type != "cpu"
                for per_layer in self.result.expert_banks.values()
                for tensor in per_layer
            ):
                raise RuntimeError("staging bank was not physically host-resident")
            actual_representation = "checkpoint-native-host"
            actual_memory_resource_id = self._memory_resource_id("system-ram")
        elif state_id == "state.block-a.non-routed":
            observed_bytes = _unique_tensor_bytes(
                self.result.block.state_dict(), device_type="cuda"
            )
            actual_representation = "freetoken-native-device"
            actual_memory_resource_id = self._memory_resource_id("accelerator-vram")
        elif state_id == "state.block-a.routed":
            self._setup_runtime()
            # The strategy declares context establishment before resident-only
            # finalization. This is setup-time legality, not hot-path plan lookup.
            self.prefill = _run_block(
                self.result.block, self.ctx, self.fixture, "prefill", "a"
            )
            if not self.prefill["passed"]:
                raise RuntimeError("prefill correctness failed before final residency")
            self.replay_state = _snapshot_runtime_state(self.ctx)
            observed_bytes = _populate_all_experts(self.cache)
            if any(
                tensor.device.type != "cuda"
                for tensor in self.cache.bank_caches.values()
            ):
                raise RuntimeError("final bank was not physically accelerator-resident")
            actual_representation = "freetoken-nvfp4-slot-banks"
            actual_memory_resource_id = self._memory_resource_id("accelerator-vram")
            self.checkpoints.append(
                _checkpoint(
                    "after_complete_accelerator_population",
                    cache=self.cache,
                    host_roles={
                        "same_backend_correctness_reference_tensors": _unique_tensor_bytes(
                            self.fixture, device_type="cpu"
                        )
                    },
                    block=self.result.block,
                    state=self.block_state,
                )
            )
        elif state_id == "state.block-a.mutable-runtime":
            self._setup_runtime()
            observed_bytes = self.block_state["total_block_local_state_bytes"]
            actual_representation = "freetoken-block-runtime-device"
            actual_memory_resource_id = self._memory_resource_id("accelerator-vram")
        elif state_id == "state.block-a.replay-baseline":
            if self.replay_state is None:
                raise RuntimeError(
                    "runtime replay baseline requested before context establishment"
                )
            observed_bytes = _unique_tensor_bytes(self.replay_state, device_type="cuda")
            actual_representation = "freetoken-runtime-snapshot-device"
            actual_memory_resource_id = self._memory_resource_id("accelerator-vram")
        else:
            raise RuntimeError(f"adapter cannot realize state {state_id!r}")
        return {
            "actual_representation": actual_representation,
            "actual_memory_resource_id": actual_memory_resource_id,
            "observed_bytes": observed_bytes,
            "lifecycle_state": "live",
            "status": "PLANNED_AND_REALIZED",
        }

    def release_materialization(self, item):
        if (
            item["role"] != "staging"
            or item["logical_state_id"] != "state.block-a.routed"
        ):
            raise RuntimeError(
                f"adapter cannot release planned materialization {item['id']!r}"
            )
        self.detach_report = self.cache.detach_host_sources_for_full_residency()
        self.released_owner_bytes = self.result.release_expert_banks_after_residency(
            self.cache
        )
        gc.collect()
        torch.cuda.synchronize(self.device)
        dead = sum(ref() is None for ref in self.source_refs)
        if dead != len(self.source_refs):
            raise RuntimeError(
                f"{len(self.source_refs) - dead} released staging tensors remain live"
            )
        host_roles = _owned_host_roles(self.result, self.ctx, self.cache, self.fixture)
        self.checkpoints.append(
            _checkpoint(
                "after_plan_driven_staging_release",
                cache=self.cache,
                host_roles=host_roles,
                other_gpu_roles={
                    "correctness_replay_state_snapshot": _unique_tensor_bytes(
                        self.replay_state, device_type="cuda"
                    )
                },
                block=self.result.block,
                state=self.block_state,
            )
        )
        return {"observed_bytes": 0, "released_bytes": self.released_owner_bytes}

    def activate_execution(self, execution):
        if not self.cache.resident_only or self.cache.host_source_tensor_bytes():
            raise RuntimeError(
                "execution activation requires finalized accelerator residency"
            )
        return {
            "execution_id": execution["id"],
            "compute_unit_id": execution["compute_unit_id"],
            "strategy_unit": execution["strategy_unit"],
            "status": "ACTIVE_BACKEND_NATIVE",
        }

    def observe_authorities(self):
        return list(self.plan["authorities"])

    def run_repeated_decode_and_audit(self):
        runs = []
        with _post_detach_load_sentinels() as sentinels:
            for repetition in range(self.repetitions):
                _restore_runtime_state(self.ctx, self.replay_state)
                metrics = _run_block(
                    self.result.block, self.ctx, self.fixture, "decode", "a"
                )
                runs.append({"repetition": repetition, **metrics})
        torch.cuda.synchronize(self.device)
        gc.collect()
        host_roles = _owned_host_roles(self.result, self.ctx, self.cache, self.fixture)
        self.checkpoints.append(
            _checkpoint(
                "after_repeated_plan_realized_decode",
                cache=self.cache,
                host_roles=host_roles,
                other_gpu_roles={
                    "correctness_replay_state_snapshot": _unique_tensor_bytes(
                        self.replay_state, device_type="cuda"
                    )
                },
                block=self.result.block,
                state=self.block_state,
            )
        )
        dead = sum(ref() is None for ref in self.source_refs)
        passed = (
            all(run["passed"] for run in runs)
            and not any(sentinels.values())
            and self.cache.resident_source_access_attempts == 0
            and self.cache.host_source_tensor_bytes() == 0
            and dead == len(self.source_refs)
        )
        return {
            "prefill": self.prefill,
            "decode_runs": runs,
            "post_release_loader_sentinels": sentinels,
            "resident_source_access_attempts": self.cache.resident_source_access_attempts,
            "dead_staging_tensor_count": dead,
            "staging_tensor_count": len(self.source_refs),
            "passed": passed,
        }

    def memory_report(self):
        final_host_roles = _owned_host_roles(
            self.result, self.ctx, self.cache, self.fixture
        )
        persistent_required = (
            _unique_tensor_bytes(self.result.block.state_dict(), device_type="cuda")
            + self.cache.expert_bank_tensor_bytes()
            + self.block_state["total_block_local_state_bytes"]
            + _unique_tensor_bytes(self.replay_state, device_type="cuda")
        )
        return {
            "persistent_required_bytes": persistent_required,
            "persistent_optional_bytes": 0,
            "transient_upper_bound_bytes": self.source_bytes
            + self.result.fetched_bytes,
            "transient_upper_bound_status": "CALCULATED conservative upper bound; selective fetched bytes are cumulative",
            "persistent_host_evidence_bytes": sum(final_host_roles.values()),
            "persistent_host_evidence_roles": final_host_roles,
            "host_staging_bytes_before_release": self.source_bytes,
            "host_staging_bytes_after_repeated_execution": self.cache.host_source_tensor_bytes(),
            "unplanned_persistent_bytes": 0,
            "unexplained_persistent_host_mirror_bytes": 0,
        }
