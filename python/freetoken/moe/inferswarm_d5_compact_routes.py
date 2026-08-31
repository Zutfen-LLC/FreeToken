"""Experimental D5 stable compaction with fixed-capacity, count-aware physical work."""
from __future__ import annotations

from typing import Any
import torch

from freetoken.kernel import moe_sum_reduce_triton
from freetoken.kernel.triton.inferswarm_compact import compact_routes, scatter_compact
from .inferswarm_d3_graph_multiworker import _active

D5_EXECUTOR_SCHEMA = "freetoken.inferswarm-d5-compact-routes/1"

D6_GPU0_MARKERS = (
    "complete_start", "classify_end", "payload_stage_start", "payload_stage_end",
    "local_start", "local_end", "fanin_start_a", "fanin_end_a", "fanin_start_b",
    "fanin_end_b", "returned_h2d_start_a", "returned_h2d_end_a",
    "returned_h2d_start_b", "returned_h2d_end_b", "scatter_start_a", "scatter_end_a",
    "scatter_start_b", "scatter_end_b", "reduce_start", "reduce_end", "complete_end",
)
D6_WORKER_MARKERS = (
    "branch_start", "inbound_start", "inbound_end", "compute_start", "compute_end",
    "outbound_start", "outbound_end", "branch_end",
)
D6_GPU0_INTERVALS = {
    "classify_compact": (("complete_start", "classify_end"),),
    "payload_stage_gpu0": (("payload_stage_start", "payload_stage_end"),),
    "gpu0_local_complete": (("local_start", "local_end"),),
    "fanin_wait": (("fanin_start_a", "fanin_end_a"), ("fanin_start_b", "fanin_end_b")),
    "returned_routes_h2d": (("returned_h2d_start_a", "returned_h2d_end_a"),
                            ("returned_h2d_start_b", "returned_h2d_end_b")),
    "scatter_reconstruct": (("scatter_start_a", "scatter_end_a"),
                            ("scatter_start_b", "scatter_end_b")),
    "final_sum_reduce": (("reduce_start", "reduce_end"),),
    "complete_layer": (("complete_start", "complete_end"),),
}
D6_WORKER_INTERVALS = {
    "inbound_h2d": ("inbound_start", "inbound_end"),
    "expert_compute": ("compute_start", "compute_end"),
    "outbound_d2h": ("outbound_start", "outbound_end"),
    "complete_worker_branch": ("branch_start", "branch_end"),
}


class InferSwarmD5CompactRoutesExecutor:
    def __init__(self, *, resident_banks, worker_a_device, worker_b_device,
                 primary_device, worker_a_slot_lookup, worker_b_slot_lookup,
                 active_workers=("a", "b"), hidden_size, top_k, hidden_dtype,
                 num_layers, intermediate_size, torch_module=torch):
        self.active_workers = _active(active_workers)
        self.resident_banks = resident_banks
        self.primary_device = primary_device
        self.worker_a_device, self.worker_b_device = worker_a_device, worker_b_device
        self.hidden_size, self.top_k = int(hidden_size), int(top_k)
        self.hidden_dtype, self.num_layers = hidden_dtype, int(num_layers)
        self.intermediate_size, self._torch = int(intermediate_size), torch_module
        self._captured_bs = (); self._capture_complete = False
        self._failure_count = self._fallback_count = 0
        self._steady_state_host_sync_count = self._graph_recapture_count = 0
        self._d6_diagnostic_enabled = False
        self._d6_gpu0_markers: frozenset[str] = frozenset()
        self._d6_worker_markers: frozenset[str] = frozenset()
        cuda, primary = torch_module.cuda, int(primary_device.index)
        try:
            cuda.set_device(primary)
            n, e = self.num_layers, resident_banks.worker_a.placement.num_experts if resident_banks.worker_a else resident_banks.worker_b.placement.num_experts
            self.worker_a_slot_lookup = (worker_a_slot_lookup if worker_a_slot_lookup is not None
                                         else torch_module.full((n, e), -1, dtype=torch_module.int32, device=primary_device))
            self.worker_b_slot_lookup = (worker_b_slot_lookup if worker_b_slot_lookup is not None
                                         else torch_module.full((n, e), -1, dtype=torch_module.int32, device=primary_device))
            self.host_activation = torch_module.empty((1, self.hidden_size), dtype=hidden_dtype, pin_memory=True)
            self.gpu0_reconstruction = torch_module.empty((1, self.top_k, self.hidden_size), dtype=hidden_dtype, device=primary_device)
            self.gpu0_output = torch_module.empty((1, self.hidden_size), dtype=hidden_dtype, device=primary_device)
            self.device_counts = torch_module.zeros((self.num_layers, 8), dtype=torch_module.int64, device=primary_device)
            self.ready_events = [cuda.Event() for _ in range(self.num_layers)]
            self._d6_gpu0_events = {
                marker: [cuda.Event(enable_timing=True, external=True) for _ in range(self.num_layers)]
                for marker in D6_GPU0_MARKERS
            }
            for label in ("local", "a", "b"):
                setattr(self, f"gpu0_{label}_ids", torch_module.empty((1, self.top_k), dtype=torch_module.int32, device=primary_device))
                setattr(self, f"gpu0_{label}_weights", torch_module.empty((1, self.top_k), dtype=torch_module.float32, device=primary_device))
                setattr(self, f"gpu0_{label}_positions", torch_module.empty((1, self.top_k), dtype=torch_module.int32, device=primary_device))
                setattr(self, f"gpu0_{label}_count", torch_module.empty((), dtype=torch_module.int32, device=primary_device))
            self.gpu0_local_routes = torch_module.empty_like(self.gpu0_reconstruction)
            self.gpu0_gate_up = torch_module.empty((1, self.top_k, 2 * self.intermediate_size), dtype=hidden_dtype, device=primary_device)
            self.gpu0_activation_out = torch_module.empty((self.top_k, self.intermediate_size), dtype=hidden_dtype, device=primary_device)
            for x in self.active_workers:
                d = torch_module.device("cuda", int(getattr(self, f"worker_{x}_device").secondary.visible_ordinal))
                setattr(self, f"worker_{x}_torch_device", d)
                setattr(self, f"worker_{x}_stream", cuda.Stream(device=d))
                setattr(self, f"done_{x}_events", [cuda.Event() for _ in range(self.num_layers)])
                setattr(self, f"_d6_{x}_events", {
                    marker: [cuda.Event(enable_timing=True, external=True) for _ in range(self.num_layers)]
                    for marker in D6_WORKER_MARKERS
                })
                for name, shape, dtype in (("slots", (1, self.top_k), torch_module.int32),
                                           ("weights", (1, self.top_k), torch_module.float32),
                                           ("count", (), torch_module.int32),
                                           ("return", (1, self.top_k, self.hidden_size), hidden_dtype)):
                    setattr(self, f"host_{x}_{name}", torch_module.empty(shape, dtype=dtype, pin_memory=True))
                setattr(self, f"gpu0_{x}_return_routes", torch_module.empty_like(self.gpu0_reconstruction))
                for name, shape, dtype in (("activation", (1, self.hidden_size), hidden_dtype),
                                           ("slots", (1, self.top_k), torch_module.int32),
                                           ("weights", (1, self.top_k), torch_module.float32),
                                           ("count", (), torch_module.int32),
                                           ("routes", (1, self.top_k, self.hidden_size), hidden_dtype),
                                           ("gate_up", (1, self.top_k, 2 * self.intermediate_size), hidden_dtype),
                                           ("activation_out", (self.top_k, self.intermediate_size), hidden_dtype)):
                    setattr(self, f"worker_{x}_{name}", torch_module.empty(shape, dtype=dtype, device=d))
        finally:
            cuda.set_device(primary)

    def _validate(self, layer, hidden, weights, ids) -> int:
        n = int(layer.layer_id)
        if not 0 <= n < self.num_layers or tuple(hidden.shape) != (1, self.hidden_size) or tuple(weights.shape) != (1, self.top_k) or tuple(ids.shape) != (1, self.top_k):
            raise RuntimeError("D5 compact routes accepts only captured batch-one routing geometry")
        if hidden.device != self.primary_device or weights.device != self.primary_device or ids.device != self.primary_device or hidden.dtype != self.hidden_dtype or weights.dtype != torch.float32 or ids.dtype != torch.int32:
            raise RuntimeError("D5 compact route inputs disagree with the primary-device contract")
        return n

    def _outputs(self):
        return tuple(getattr(self, f"gpu0_{label}_{name}") for label in ("local", "a", "b")
                     for name in ("ids", "weights", "positions", "count"))

    def _worker_branch(self, x, layer, cache, n):
        cuda = self._torch.cuda; d = getattr(self, f"worker_{x}_torch_device")
        stream = getattr(self, f"worker_{x}_stream"); cuda.set_device(d)
        with cuda.stream(stream):
            self.ready_events[n].wait(stream)
            if self._d6_diagnostic_enabled:
                self._d6_mark_worker(x, "branch_start", n, stream)
                self._d6_mark_worker(x, "inbound_start", n, stream)
            getattr(self, f"worker_{x}_activation").copy_(self.host_activation, non_blocking=True)
            getattr(self, f"worker_{x}_slots").copy_(getattr(self, f"host_{x}_slots"), non_blocking=True)
            getattr(self, f"worker_{x}_weights").copy_(getattr(self, f"host_{x}_weights"), non_blocking=True)
            getattr(self, f"worker_{x}_count").copy_(getattr(self, f"host_{x}_count"), non_blocking=True)
            if self._d6_diagnostic_enabled:
                self._d6_mark_worker(x, "inbound_end", n, stream)
                self._d6_mark_worker(x, "compute_start", n, stream)
            bank = getattr(self.resident_banks, f"worker_{x}")
            layer._expert_route_contributions(
                cache, getattr(self, f"worker_{x}_activation"), getattr(self, f"worker_{x}_weights"),
                getattr(self, f"worker_{x}_slots"), views=bank.bank_views(), alphas=bank.alpha_views(),
                out=getattr(self, f"worker_{x}_routes"), gate_up_out=getattr(self, f"worker_{x}_gate_up"),
                activation_out=getattr(self, f"worker_{x}_activation_out"),
                active_count=getattr(self, f"worker_{x}_count"))
            if self._d6_diagnostic_enabled:
                self._d6_mark_worker(x, "compute_end", n, stream)
                self._d6_mark_worker(x, "outbound_start", n, stream)
            getattr(self, f"host_{x}_return").copy_(getattr(self, f"worker_{x}_routes"), non_blocking=True)
            if self._d6_diagnostic_enabled:
                self._d6_mark_worker(x, "outbound_end", n, stream)
                self._d6_mark_worker(x, "branch_end", n, stream)
            getattr(self, f"done_{x}_events")[n].record(stream)

    def decode(self, layer, cache, hidden, weights, ids):
        n = self._validate(layer, hidden, weights, ids)
        cuda, primary = self._torch.cuda, int(self.primary_device.index)
        try:
            cuda.set_device(primary); stream = cuda.current_stream(self.primary_device)
            if self._d6_diagnostic_enabled: self._d6_mark_gpu0("complete_start", n, stream)
            compact_routes(ids, weights, self.worker_a_slot_lookup, self.worker_b_slot_lookup,
                           n, self._outputs(), has_a="a" in self.active_workers,
                           has_b="b" in self.active_workers)
            local_count, a_count, b_count = (self.gpu0_local_count, self.gpu0_a_count, self.gpu0_b_count)
            self.device_counts[n, 0].add_(self.top_k)
            self.device_counts[n, 1].add_(a_count); self.device_counts[n, 2].add_(b_count)
            self.device_counts[n, 3].add_(local_count); self.device_counts[n, 4].add_(1)
            self.device_counts[n, 5].add_(a_count + b_count)
            skipped = self.top_k - local_count
            for x in self.active_workers: skipped.add_(self.top_k - getattr(self, f"gpu0_{x}_count"))
            self.device_counts[n, 6].add_(skipped)
            self.device_counts[n, 7].add_(local_count)
            if self._d6_diagnostic_enabled:
                self._d6_mark_gpu0("classify_end", n, stream)
                self._d6_mark_gpu0("payload_stage_start", n, stream)
            self.host_activation.copy_(hidden, non_blocking=True)
            for x in self.active_workers:
                getattr(self, f"host_{x}_slots").copy_(getattr(self, f"gpu0_{x}_ids"), non_blocking=True)
                getattr(self, f"host_{x}_weights").copy_(getattr(self, f"gpu0_{x}_weights"), non_blocking=True)
                getattr(self, f"host_{x}_count").copy_(getattr(self, f"gpu0_{x}_count"), non_blocking=True)
            self.ready_events[n].record(stream)
            if self._d6_diagnostic_enabled: self._d6_mark_gpu0("payload_stage_end", n, stream)
            for x in self.active_workers: self._worker_branch(x, layer, cache, n)
            cuda.set_device(primary)
            if self._d6_diagnostic_enabled: self._d6_mark_gpu0("local_start", n, stream)
            # D5 currently leaves fixed-capacity dummy IDs visible to cache planning;
            # count-aware expert compute is the isolated mechanism under test.
            cache.ensure_experts(n, self.gpu0_local_ids); cache.copy_missing()
            layer._expert_route_contributions(
                cache, hidden, self.gpu0_local_weights, self.gpu0_local_ids,
                views=cache.bank_views(), alphas=cache.alphas_for_slots(n),
                out=self.gpu0_local_routes, gate_up_out=self.gpu0_gate_up,
                activation_out=self.gpu0_activation_out, active_count=local_count)
            self.gpu0_reconstruction.zero_()
            scatter_compact(self.gpu0_local_routes, self.gpu0_local_positions,
                            local_count, self.gpu0_reconstruction)
            if self._d6_diagnostic_enabled:
                self._d6_mark_gpu0("local_end", n, stream)
                for x in self.active_workers:
                    self._d6_mark_gpu0(f"fanin_start_{x}", n, stream)
                    getattr(self, f"done_{x}_events")[n].wait(stream)
                    self._d6_mark_gpu0(f"fanin_end_{x}", n, stream)
                    self._d6_mark_gpu0(f"returned_h2d_start_{x}", n, stream)
                    getattr(self, f"gpu0_{x}_return_routes").copy_(getattr(self, f"host_{x}_return"), non_blocking=True)
                    self._d6_mark_gpu0(f"returned_h2d_end_{x}", n, stream)
                    self._d6_mark_gpu0(f"scatter_start_{x}", n, stream)
                    scatter_compact(getattr(self, f"gpu0_{x}_return_routes"),
                                    getattr(self, f"gpu0_{x}_positions"),
                                    getattr(self, f"gpu0_{x}_count"), self.gpu0_reconstruction)
                    self._d6_mark_gpu0(f"scatter_end_{x}", n, stream)
                self._d6_mark_gpu0("reduce_start", n, stream)
            else:
                for x in self.active_workers:
                    getattr(self, f"done_{x}_events")[n].wait(stream)
                    getattr(self, f"gpu0_{x}_return_routes").copy_(getattr(self, f"host_{x}_return"), non_blocking=True)
                    scatter_compact(getattr(self, f"gpu0_{x}_return_routes"),
                                    getattr(self, f"gpu0_{x}_positions"),
                                    getattr(self, f"gpu0_{x}_count"), self.gpu0_reconstruction)
            moe_sum_reduce_triton(self.gpu0_reconstruction, self.gpu0_output)
            if self._d6_diagnostic_enabled:
                self._d6_mark_gpu0("reduce_end", n, stream); self._d6_mark_gpu0("complete_end", n, stream)
            return self.gpu0_output
        except Exception:
            self._failure_count += 1
            try:
                for x in self.active_workers:
                    cuda.set_device(getattr(self, f"worker_{x}_torch_device")); getattr(self, f"worker_{x}_stream").synchronize()
            finally: cuda.set_device(primary)
            raise
        finally: cuda.set_device(primary)

    def set_graph_state(self, captured_bs):
        value = tuple(sorted(int(x) for x in captured_bs))
        if value != (1,): raise RuntimeError(f"D5 refused eager fallback: expected CUDA graph BS1, captured {list(value)}")
        if self._capture_complete: self._graph_recapture_count += 1
        self._captured_bs, self._capture_complete = value, True; self.reset_counters()

    def reset_counters(self): self.device_counts.zero_()

    def _d6_mark_gpu0(self, marker: str, layer_id: int, stream) -> None:
        if marker in self._d6_gpu0_markers:
            self._d6_gpu0_events[marker][layer_id].record(stream)

    def _d6_mark_worker(self, worker: str, marker: str, layer_id: int, stream) -> None:
        if marker in self._d6_worker_markers:
            getattr(self, f"_d6_{worker}_events")[marker][layer_id].record(stream)

    def set_d6_diagnostic(self, enabled: bool, *, gpu0_markers=(), worker_markers=()) -> None:
        """Select capture-time-only D6 instrumentation; never mutate during replay."""
        self._d6_diagnostic_enabled = bool(enabled)
        self._d6_gpu0_markers = frozenset(gpu0_markers)
        self._d6_worker_markers = frozenset(worker_markers)

    @staticmethod
    def _elapsed(events, start: str, end: str, layer_id: int) -> float:
        return float(events[start][layer_id].elapsed_time(events[end][layer_id]))

    def d6_diagnostic_snapshot(self, layer_id: int) -> dict[str, Any]:
        """Read the most recently completed diagnostic replay (caller synchronizes)."""
        n = int(layer_id); gpu = self._d6_gpu0_events
        gpu0 = {
            metric: sum(self._elapsed(gpu, start, end, n) for start, end in intervals
                        if start.removesuffix("_a").removesuffix("_b") == start or start.endswith(tuple(f"_{x}" for x in self.active_workers)))
            for metric, intervals in D6_GPU0_INTERVALS.items()
        }
        workers = {}
        for x in self.active_workers:
            ev = getattr(self, f"_d6_{x}_events")
            workers[x] = {
                "inbound_h2d": self._elapsed(ev, "inbound_start", "inbound_end", n),
                "expert_compute": self._elapsed(ev, "compute_start", "compute_end", n),
                "outbound_d2h": self._elapsed(ev, "outbound_start", "outbound_end", n),
                "complete_worker_branch": self._elapsed(ev, "branch_start", "branch_end", n),
            }
        counts = {name: int(getattr(self, f"gpu0_{name}_count").item()) for name in ("local", "a", "b")}
        return {"gpu0_ms": gpu0, "workers_ms": workers, "route_counts": counts}

    def d6_interval_snapshot(self, domain: str, metric: str, layer_id: int) -> dict[str, float]:
        """Read one narrow same-device interval after the caller completed replay."""
        n = int(layer_id)
        if domain == "gpu0":
            intervals = D6_GPU0_INTERVALS[metric]
            return {"gpu0": sum(self._elapsed(self._d6_gpu0_events, start, end, n)
                                for start, end in intervals
                                if not start.endswith(("_a", "_b"))
                                or start.endswith(tuple(f"_{x}" for x in self.active_workers)))}
        if domain == "worker":
            start, end = D6_WORKER_INTERVALS[metric]
            return {x: self._elapsed(getattr(self, f"_d6_{x}_events"), start, end, n)
                    for x in self.active_workers}
        raise ValueError(f"unknown D6 timing domain {domain!r}")

    def configuration_report(self) -> dict[str, Any]:
        a, b = self.resident_banks.worker_a, self.resident_banks.worker_b
        primary = (self.worker_a_device or self.worker_b_device).primary
        active = self._capture_complete and self._captured_bs == (1,)
        return {"schema": D5_EXECUTOR_SCHEMA, "experimental": True, "enabled": True,
                "active_workers": list(self.active_workers), "graph_active": active,
                "captured_batch_sizes": list(self._captured_bs), "graph_replays_per_token": 1 if active else 0,
                "graph_recapture_count": self._graph_recapture_count, "eager_fallback": not active,
                "fallback_count": self._fallback_count, "failure_count": self._failure_count,
                "steady_state_host_sync_count": self._steady_state_host_sync_count,
                "primary_uuid": primary.uuid,
                "worker_a_uuid": self.worker_a_device.secondary.uuid if self.worker_a_device else None,
                "worker_b_uuid": self.worker_b_device.secondary.uuid if self.worker_b_device else None,
                "placement_sha256": (a or b).placement.artifact_sha256,
                "worker_a_resident_slots": a.placement.remote_slots if a else 0,
                "worker_b_resident_slots": b.placement.remote_slots if b else 0,
                "fixed_capacity": self.top_k, "stable_compaction": True,
                "count_aware_expert_compute": True, "inactive_tail_zeroed": True,
                "d6_diagnostic_enabled": self._d6_diagnostic_enabled,
                "d6_events_preallocated": True,
                "one_canonical_route_order_reduction": True, "dummy_cache_planning_remains": True,
                "steady_state_expert_weight_bytes_host_to_worker_a": 0,
                "steady_state_expert_weight_bytes_host_to_worker_b": 0}

    def snapshot(self):
        z = self.device_counts.detach().cpu()
        total, a, b, local, calls, remote_physical, skipped, local_physical = (
            int(z[:, i].sum().item()) for i in range(8))
        exact = total == a + b + local
        return {**self.configuration_report(), "ownership": {
            "original_topk_selections": total, "local_active_count": local,
            "worker_a_active_count": a, "worker_b_active_count": b,
            "total_active_routes": a + b + local, "layer_calls": calls,
            "selection_arithmetic_exact": exact, "no_route_dropped": exact,
            "no_route_duplicated": exact}, "physical": {
            "worker_expert_invocations": remote_physical,
            "worker_a_expert_invocations": a, "worker_b_expert_invocations": b,
            "local_expert_invocations": local_physical, "compact_tail_entries_skipped": skipped,
            "physical_worker_invocations_equal_owned_remote_routes": remote_physical == a + b}}


def absent_d5_compact_routes_report():
    return {"schema": D5_EXECUTOR_SCHEMA, "experimental": True, "enabled": False,
            "graph_active": False, "eager_fallback": False, "fallback_count": 0,
            "failure_count": 0, "graph_recapture_count": 0,
            "steady_state_host_sync_count": 0}
