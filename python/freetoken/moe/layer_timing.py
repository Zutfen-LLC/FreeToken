"""Opt-in complete routed-MoE layer timing for eager and CUDA-graph decode.

Timing boundaries are persistent device marker kernels backed by ``%globaltimer``.
The marker nodes survive CUDA graph capture and write into a bounded per-replay ring
using device-side cursors.  This avoids both per-layer synchronization and the stale /
overwritten-event ambiguity created when scheduler overlap launches replay N+1 before
the host drains replay N's ``copy_done_event``.

The schema is deliberately shared by the ordinary RAM-offload path and the narrow
InferSwarm candidate.  Operations that do not exist for a role are represented as
``not_applicable`` rather than numeric zeroes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

MOE_LAYER_TIMING_SCHEMA = "freetoken.moe-layer-timing/2"
TIMER_MECHANISM = "cuda_globaltimer_marker_kernel"
TIMER_UNIT = "nanoseconds"

MARKERS = (
    "complete_start",
    "local_start",
    "cache_service_end",
    "weight_fetch_end",
    "local_expert_end",
    "local_branch_end",
    "returned_route_contributions_h2d_start",
    "returned_route_contributions_h2d_end",
    "route_reconstruction_start",
    "route_reconstruction_end",
    "final_sum_reduce_start",
    "final_sum_reduce_end",
    "complete_end",
)
MARKER_ID = {name: i for i, name in enumerate(MARKERS)}

METRICS = (
    "total_route_selections",
    "local_unique_experts",
    "local_cache_misses",
    "local_fetched_experts",
    "host_to_gpu0_expert_weight_bytes",
)
METRIC_ID = {name: i for i, name in enumerate(METRICS)}


def _duration(
    value_ms: float | None, source: str, status: str = "valid"
) -> dict[str, Any]:
    return {"status": status, "value_ms": value_ms, "source": source}


def not_applicable(source: str = "not_applicable") -> dict[str, Any]:
    return _duration(None, source, "not_applicable")


def unavailable(source: str, reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "value_ms": None,
        "source": source,
        "reason": reason,
    }


def measured_bytes(value: int | None, status: str = "measured") -> dict[str, Any]:
    return {"status": status, "bytes": value}


def not_applicable_bytes() -> dict[str, Any]:
    return {"status": "not_applicable", "bytes": None}


@dataclass(frozen=True)
class DecodeStepContext:
    step: int
    batch_size: int
    padded_batch_size: int
    graph_replay: bool


class MoeLayerTiming:
    """Bounded device timing storage plus host annotations for remote-only facts."""

    def __init__(
        self,
        *,
        max_steps: int,
        num_layers: int,
        device: torch.device,
        bytes_per_identity: int,
        role: str,
        graph_requested: bool,
        remote_overlap_active: bool,
    ) -> None:
        if max_steps < 1:
            raise ValueError("MoE layer timing capacity must be positive")
        if role not in ("unspecified", "baseline", "candidate"):
            raise ValueError(f"invalid MoE layer timing role {role!r}")
        self.max_steps = int(max_steps)
        self.num_layers = int(num_layers)
        self.device = device
        self.bytes_per_identity = int(bytes_per_identity)
        self.role = role
        self.graph_requested = bool(graph_requested)
        self.graph_captured_batch_sizes: list[int] = []
        self.remote_overlap_active = bool(remote_overlap_active)
        # Last slot is an overflow sink. Captured graphs always have a valid address even
        # after the retained trace fills; snapshots exclude it.
        self.timestamps = torch.full(
            (self.max_steps + 1, self.num_layers, len(MARKERS)),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.metadata = torch.full(
            (self.max_steps + 1, self.num_layers, len(METRICS)),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.layer_steps = torch.zeros(
            self.num_layers, dtype=torch.int64, device=device
        )
        self._step_contexts: dict[int, DecodeStepContext] = {}
        self._annotations: dict[tuple[int, int], dict[str, Any]] = {}
        self.steps_observed = 0

    def set_graph_state(self, captured_batch_sizes: list[int]) -> None:
        self.graph_captured_batch_sizes = sorted(int(v) for v in captured_batch_sizes)

    def configuration_report(self) -> dict[str, Any]:
        return {
            "schema": MOE_LAYER_TIMING_SCHEMA,
            "enabled": True,
            "capacity_steps": self.max_steps,
            "role": self.role,
            "graph_requested": self.graph_requested,
            "graph_captured_batch_sizes": list(self.graph_captured_batch_sizes),
            "timer_mechanism": TIMER_MECHANISM,
            "timer_unit": TIMER_UNIT,
            "timer_unit_basis": "NVIDIA CUDA %globaltimer nanosecond counter",
            "host_sync_per_layer": False,
            "remote_overlap_active": self.remote_overlap_active,
        }

    def begin_decode_step(
        self,
        step: int,
        *,
        batch_size: int,
        padded_batch_size: int,
        graph_replay: bool,
    ) -> None:
        self.steps_observed = max(self.steps_observed, int(step) + 1)
        if step < self.max_steps:
            self._step_contexts[int(step)] = DecodeStepContext(
                step=int(step),
                batch_size=int(batch_size),
                padded_batch_size=int(padded_batch_size),
                graph_replay=bool(graph_replay),
            )

    def mark(self, layer_id: int, marker: str, *, begin_layer: bool = False) -> None:
        from .layer_timing_kernels import record_timestamp

        record_timestamp(
            self.timestamps,
            self.layer_steps,
            capacity=self.max_steps,
            num_layers=self.num_layers,
            num_markers=len(MARKERS),
            layer_id=int(layer_id),
            marker_id=MARKER_ID[marker],
            begin_layer=begin_layer,
        )

    def record_cache_metadata(self, layer_id: int, cache, *, total_routes: int) -> None:
        from .layer_timing_kernels import record_cache_metadata

        record_cache_metadata(
            self.metadata,
            self.layer_steps,
            cache.lru_stats[layer_id],
            cache.num_indices,
            capacity=self.max_steps,
            num_layers=self.num_layers,
            num_metrics=len(METRICS),
            layer_id=int(layer_id),
            total_routes=int(total_routes),
            bytes_per_identity=self.bytes_per_identity,
        )

    def annotate(self, step: int, layer_id: int, values: dict[str, Any]) -> None:
        if 0 <= step < self.max_steps:
            self._annotations[(int(step), int(layer_id))] = values

    @staticmethod
    def _marker_duration(
        row: list[int], start: str, end: str, *, applicable: bool = True
    ) -> dict[str, Any]:
        if not applicable:
            return not_applicable()
        a, b = row[MARKER_ID[start]], row[MARKER_ID[end]]
        if a < 0 or b < 0 or b < a:
            return unavailable(
                "cuda_globaltimer_marker_gpu0",
                f"missing or invalid marker interval {start}->{end}",
            )
        return _duration((b - a) / 1_000_000.0, "cuda_globaltimer_marker_gpu0")

    @staticmethod
    def _annotation_duration(
        annotation: dict[str, Any], name: str, *, applicable: bool
    ) -> dict[str, Any]:
        if not applicable:
            return not_applicable()
        value = annotation.get("durations", {}).get(name)
        if value is None:
            return unavailable("not_recorded", f"candidate did not record {name}")
        return value

    def _build_record(
        self,
        step: int,
        layer_id: int,
        marker_row: list[int],
        metric_row: list[int],
    ) -> dict[str, Any]:
        annotation = self._annotations.get((step, layer_id), {})
        identity = annotation.get("identity")
        remote_participated = bool(
            identity and identity.get("gpu1_owned_selections", 0)
        )
        local_participated = (
            bool(identity.get("gpu0_owned_selections", 0))
            if identity is not None
            else metric_row[METRIC_ID["total_route_selections"]] >= 0
        )
        if identity is None:
            total = metric_row[METRIC_ID["total_route_selections"]]
            identity = {
                "decode_step": step,
                "layer_id": layer_id,
                "total_route_selections": None if total < 0 else total,
                "gpu0_owned_selections": None if total < 0 else total,
                "gpu1_owned_selections": 0,
                "unique_gpu1_expert_identities": 0,
                "dispatch_count": 0,
            }

        local_start = "local_start"
        local_durations = {
            "route_cache_service_bookkeeping": self._marker_duration(
                marker_row,
                local_start,
                "cache_service_end",
                applicable=local_participated,
            ),
            "host_to_gpu0_expert_fetch_copy": self._marker_duration(
                marker_row,
                "cache_service_end",
                "weight_fetch_end",
                applicable=local_participated,
            ),
            "local_expert_execution": self._marker_duration(
                marker_row,
                "weight_fetch_end",
                "local_expert_end",
                applicable=local_participated,
            ),
            "complete_local_branch": self._marker_duration(
                marker_row,
                local_start,
                "local_branch_end",
                applicable=local_participated,
            ),
        }
        candidate = bool(annotation.get("candidate", False))
        remote_durations = {
            "classification_control_host_wait": self._annotation_duration(
                annotation, "classification_control_host_wait", applicable=candidate
            ),
            "gpu0_to_host_activation_routing": self._annotation_duration(
                annotation,
                "gpu0_to_host_activation_routing",
                applicable=remote_participated,
            ),
            "gpu0_to_host_staging_host_wait": self._annotation_duration(
                annotation,
                "gpu0_to_host_staging_host_wait",
                applicable=remote_participated,
            ),
            "host_remote_submit_control": self._annotation_duration(
                annotation, "host_remote_submit_control", applicable=remote_participated
            ),
        }
        gpu1_durations = {
            "host_to_gpu1_payload_h2d": self._annotation_duration(
                annotation, "host_to_gpu1_payload_h2d", applicable=remote_participated
            ),
            "gpu1_route_contribution_execution": self._annotation_duration(
                annotation,
                "gpu1_route_contribution_execution",
                applicable=remote_participated,
            ),
            "gpu1_to_host_route_contributions_d2h": self._annotation_duration(
                annotation,
                "gpu1_to_host_route_contributions_d2h",
                applicable=remote_participated,
            ),
            "complete_gpu1_branch": self._annotation_duration(
                annotation, "complete_gpu1_branch", applicable=remote_participated
            ),
        }
        join_durations = {
            "host_remote_join_wait": self._annotation_duration(
                annotation, "host_remote_join_wait", applicable=remote_participated
            ),
            "host_to_gpu0_returned_route_contributions": self._marker_duration(
                marker_row,
                "returned_route_contributions_h2d_start",
                "returned_route_contributions_h2d_end",
                applicable=remote_participated,
            ),
            "route_reconstruction": self._marker_duration(
                marker_row,
                "route_reconstruction_start",
                "route_reconstruction_end",
                applicable=remote_participated,
            ),
            "final_moe_sum_reduce": self._marker_duration(
                marker_row,
                "final_sum_reduce_start",
                "final_sum_reduce_end",
                applicable=remote_participated,
            ),
        }
        cache = {
            name: (None if metric_row[index] < 0 else metric_row[index])
            for name, index in METRIC_ID.items()
            if name != "total_route_selections"
        }
        transfer = annotation.get("transfer_bytes")
        if transfer is None:
            weight_bytes = cache["host_to_gpu0_expert_weight_bytes"]
            transfer = {
                "gpu0_to_host": {
                    "activation": not_applicable_bytes(),
                    "routing_weights": not_applicable_bytes(),
                    "routing_ids": not_applicable_bytes(),
                },
                "host_to_gpu1": {
                    "activation": not_applicable_bytes(),
                    "routing_weights": not_applicable_bytes(),
                    "routing_ids": not_applicable_bytes(),
                    "expert_weights": not_applicable_bytes(),
                },
                "gpu1_to_host": {
                    "returned_route_contributions": not_applicable_bytes()
                },
                "host_to_gpu0": {
                    "returned_route_contributions": not_applicable_bytes(),
                    "expert_weights": measured_bytes(weight_bytes),
                },
            }
        return {
            "identity": identity,
            "cache_service": cache,
            "transfer_bytes": transfer,
            "durations": {
                "complete_layer": self._marker_duration(
                    marker_row, "complete_start", "complete_end"
                ),
                "gpu0_branch": local_durations,
                "remote_dispatch_control": remote_durations,
                "gpu1_branch": gpu1_durations,
                "join_reconstruct_reduce": join_durations,
            },
            "step_context": (
                self._step_contexts[step].__dict__.copy()
                if step in self._step_contexts
                else {
                    "step": step,
                    "batch_size": None,
                    "padded_batch_size": None,
                    "graph_replay": None,
                }
            ),
        }

    def snapshot(self) -> dict[str, Any]:
        layer_steps = [int(v) for v in self.layer_steps.tolist()]
        observed_device = max(layer_steps, default=0)
        retained_steps = min(min(layer_steps, default=0), self.max_steps)
        timestamps = self.timestamps[:retained_steps].tolist()
        metadata = self.metadata[:retained_steps].tolist()
        # The graph-captured marker stores flashlib's cumulative per-layer LRU
        # counters.  Convert them to per-replay observations on the host only after
        # decode completion has made the bounded device records safe to read.
        for layer in range(self.num_layers):
            previous_active = 0
            previous_misses = 0
            for step in range(retained_steps):
                row = metadata[step][layer]
                active_index = METRIC_ID["local_unique_experts"]
                miss_index = METRIC_ID["local_cache_misses"]
                active_cumulative = row[active_index]
                miss_cumulative = row[miss_index]
                if active_cumulative >= 0:
                    row[active_index] = active_cumulative - previous_active
                    previous_active = active_cumulative
                if miss_cumulative >= 0:
                    row[miss_index] = miss_cumulative - previous_misses
                    previous_misses = miss_cumulative
        records = [
            self._build_record(
                step, layer, timestamps[step][layer], metadata[step][layer]
            )
            for step in range(retained_steps)
            for layer in range(self.num_layers)
        ]
        incomplete = len(set(layer_steps)) > 1
        truncated = observed_device > self.max_steps or incomplete
        complete_valid = bool(records) and all(
            record["durations"]["complete_layer"]["status"] == "valid"
            for record in records
        )
        component_valid = bool(records) and all(
            all(
                duration["status"] in ("valid", "not_applicable")
                for group in record["durations"].values()
                if isinstance(group, dict) and "status" not in group
                for duration in group.values()
            )
            for record in records
        )
        return {
            "schema": MOE_LAYER_TIMING_SCHEMA,
            "enabled": True,
            "capacity_steps": self.max_steps,
            "steps_observed": max(self.steps_observed, observed_device),
            "steps_retained": retained_steps,
            "records_retained": len(records),
            "truncated": truncated,
            "overflow_layer_calls": sum(
                max(0, n - self.max_steps) for n in layer_steps
            ),
            "incomplete_layer_alignment": incomplete,
            "layer_steps_observed": layer_steps,
            "role": self.role,
            "graph": {
                "requested": self.graph_requested,
                "captured_batch_sizes": self.graph_captured_batch_sizes,
                "active": bool(self.graph_captured_batch_sizes),
            },
            "timer": {
                "mechanism": TIMER_MECHANISM,
                "device": str(self.device),
                "unit": TIMER_UNIT,
                "unit_basis": "NVIDIA CUDA %globaltimer nanosecond counter",
                "conversion_to_ms": "(end_globaltimer - start_globaltimer) / 1_000_000",
                "calibration": (
                    "physical diagnostic compares the marker interval with a same-stream "
                    "CUDA-event interval; no cross-device clock comparison"
                ),
                "host_sync_per_layer": False,
                "commit_boundary": (
                    "bounded device records become host-visible only after the existing "
                    "decode output completion and idle instrumentation synchronization"
                ),
            },
            "validity": {
                "complete_layer_timing_valid": complete_valid,
                "component_timing_valid": component_valid,
            },
            "remote_overlap_active": self.remote_overlap_active,
            "records": records,
        }

    def reset(self) -> None:
        """Clear bounded measurement state without touching expert residency."""
        self.timestamps.fill_(-1)
        self.metadata.fill_(-1)
        self.layer_steps.zero_()
        self._step_contexts.clear()
        self._annotations.clear()
        self.steps_observed = 0


def absent_moe_layer_timing_report() -> dict[str, Any]:
    return {
        "schema": MOE_LAYER_TIMING_SCHEMA,
        "enabled": False,
        "capacity_steps": 0,
        "steps_observed": 0,
        "steps_retained": 0,
        "records_retained": 0,
        "truncated": False,
        "overflow_layer_calls": 0,
        "incomplete_layer_alignment": False,
        "layer_steps_observed": [],
        "role": "unspecified",
        "graph": None,
        "timer": None,
        "validity": {
            "complete_layer_timing_valid": False,
            "component_timing_valid": False,
        },
        "remote_overlap_active": False,
        "records": [],
    }
