"""D6 graph-preserving count-aware returned-route transport."""
from __future__ import annotations

from typing import Any

import torch

from freetoken.kernel import moe_sum_reduce_triton
from freetoken.kernel.count_aware_transport import (
    pack_active_metadata, pack_active_routes, scatter_active_routes, unpack_active_metadata,
)
from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.kernel.triton.inferswarm_compact import compact_routes, scatter_compact
from .inferswarm_d5_compact_routes import InferSwarmD5CompactRoutesExecutor

D6_EXECUTOR_SCHEMA = "freetoken.inferswarm-d6-count-aware-transport/1"


def transport_byte_geometry(active_count: int, *, top_k: int, hidden_size: int,
                            element_size: int) -> dict[str, int | float]:
    if not 0 <= active_count <= top_k: raise ValueError("active_count outside transport capacity")
    activation = hidden_size * element_size; d5_metadata = top_k * 8 + 4
    d6_metadata = active_count * 8 + 4
    useful_return = active_count * hidden_size * element_size
    d5_return = top_k * hidden_size * element_size
    d6_total = activation + d6_metadata + 2 * useful_return
    d5_total = activation + d5_metadata + 2 * d5_return
    return {"activation_h2d": activation, "d5_metadata_h2d": d5_metadata,
            "d6_metadata_h2d": d6_metadata,
            "useful_return_per_leg": useful_return, "d5_return_per_leg": d5_return,
            "d6_actual_return_per_leg": useful_return, "d5_total_path": d5_total,
            "d6_total_path": d6_total, "bytes_saved": d5_total - d6_total,
            "transport_efficiency": d6_total / d5_total}


class InferSwarmD6CountAwareTransportExecutor(InferSwarmD5CompactRoutesExecutor):
    def __init__(self, *, d7_participation_diagnostics: bool = False, **kwargs):
        super().__init__(**kwargs)
        torch_module = self._torch
        self.d7_participation_diagnostics = bool(d7_participation_diagnostics)
        for x in self.active_workers:
            # cudaHostAllocMapped storage is directly written by the worker and read by
            # GPU0. Kernels gate every row with the device active count, so tail rows
            # generate no PCIe payload traffic and can remain stale safely.
            setattr(self, f"host_{x}_return", alloc_pinned_tensor(
                1, self.top_k, self.hidden_size, dtype=self.hidden_dtype))
            setattr(self, f"host_{x}_slots", alloc_pinned_tensor(1, self.top_k, dtype=torch.int32))
            setattr(self, f"host_{x}_weights", alloc_pinned_tensor(1, self.top_k, dtype=torch.float32))
        primary = self.primary_device
        self.transport_counts = torch_module.zeros(
            (self.num_layers, 2, 8), dtype=torch_module.int64, device=primary)
        self.active_route_histogram = torch_module.zeros(
            (self.num_layers, 2, self.top_k + 1), dtype=torch_module.int64, device=primary)
        self._histogram_values = torch_module.arange(
            self.top_k + 1, dtype=torch_module.int32, device=primary)
        if self.d7_participation_diagnostics:
            # Joint state is 0=zero, 1=A-only, 2=B-only, 3=A+B.  D6's marginal
            # per-worker histograms cannot reconstruct the A/B intersection.
            self.d7_participation_histogram = torch_module.zeros(
                (self.num_layers, 4), dtype=torch_module.int64, device=primary)
            self._d7_participation_values = torch_module.arange(
                4, dtype=torch_module.int32, device=primary)
        self._return_row_bytes = self.hidden_size * torch_module.empty(
            (), dtype=self.hidden_dtype).element_size()
        self._activation_bytes = self.hidden_size * torch_module.empty(
            (), dtype=self.hidden_dtype).element_size()
        self._metadata_capacity_bytes = self.top_k * 8 + 4

    def _worker_branch(self, x, layer, cache, n):
        cuda = self._torch.cuda; d = getattr(self, f"worker_{x}_torch_device")
        stream = getattr(self, f"worker_{x}_stream"); cuda.set_device(d)
        with cuda.stream(stream):
            self.ready_events[n].wait(stream)
            if self._d6_diagnostic_enabled:
                self._d6_mark_worker(x, "branch_start", n, stream)
                self._d6_mark_worker(x, "inbound_start", n, stream)
            getattr(self, f"worker_{x}_activation").copy_(self.host_activation, non_blocking=True)
            getattr(self, f"worker_{x}_count").copy_(getattr(self, f"host_{x}_count"), non_blocking=True)
            unpack_active_metadata(getattr(self, f"worker_{x}_slots"),
                                   getattr(self, f"worker_{x}_weights"),
                                   getattr(self, f"host_{x}_slots"),
                                   getattr(self, f"host_{x}_weights"),
                                   getattr(self, f"worker_{x}_count"))
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
            pack_active_routes(getattr(self, f"host_{x}_return"),
                               getattr(self, f"worker_{x}_routes"),
                               getattr(self, f"worker_{x}_count"))
            if self._d6_diagnostic_enabled:
                self._d6_mark_worker(x, "outbound_end", n, stream)
                self._d6_mark_worker(x, "branch_end", n, stream)
            getattr(self, f"done_{x}_events")[n].record(stream)

    def _record_transport(self, n: int) -> None:
        for index, x in enumerate(("a", "b")):
            if x not in self.active_workers: continue
            count = getattr(self, f"gpu0_{x}_count")
            row = self.transport_counts[n, index]
            row[0].add_(self._activation_bytes)
            row[1].add_(count * 8 + 4)
            row[2].add_(count * self._return_row_bytes)
            row[3].add_(count * self._return_row_bytes)
            row[4].add_(count * self._return_row_bytes)
            row[5].add_(4); row[6].add_(count == 0); row[7].add_(1)
            self.active_route_histogram[n, index].add_(self._histogram_values == count)

    def decode(self, layer, cache, hidden, weights, ids):
        n = self._validate(layer, hidden, weights, ids)
        cuda, primary = self._torch.cuda, int(self.primary_device.index)
        try:
            cuda.set_device(primary); stream = cuda.current_stream(self.primary_device)
            compact_routes(ids, weights, self.worker_a_slot_lookup, self.worker_b_slot_lookup,
                           n, self._outputs(), has_a="a" in self.active_workers,
                           has_b="b" in self.active_workers)
            local_count, a_count, b_count = self.gpu0_local_count, self.gpu0_a_count, self.gpu0_b_count
            self.device_counts[n, 0].add_(self.top_k)
            self.device_counts[n, 1].add_(a_count); self.device_counts[n, 2].add_(b_count)
            self.device_counts[n, 3].add_(local_count); self.device_counts[n, 4].add_(1)
            self.device_counts[n, 5].add_(a_count + b_count)
            skipped = self.top_k - local_count
            for x in self.active_workers: skipped.add_(self.top_k - getattr(self, f"gpu0_{x}_count"))
            self.device_counts[n, 6].add_(skipped); self.device_counts[n, 7].add_(local_count)
            self._record_transport(n)
            if self.d7_participation_diagnostics:
                state = (a_count > 0).to(self._torch.int32) + 2 * (b_count > 0).to(self._torch.int32)
                self.d7_participation_histogram[n].add_(self._d7_participation_values == state)
            self.host_activation.copy_(hidden, non_blocking=True)
            for x in self.active_workers:
                pack_active_metadata(getattr(self, f"host_{x}_slots"),
                                     getattr(self, f"host_{x}_weights"),
                                     getattr(self, f"gpu0_{x}_ids"),
                                     getattr(self, f"gpu0_{x}_weights"),
                                     getattr(self, f"gpu0_{x}_count"))
                getattr(self, f"host_{x}_count").copy_(getattr(self, f"gpu0_{x}_count"), non_blocking=True)
            self.ready_events[n].record(stream)
            for x in self.active_workers: self._worker_branch(x, layer, cache, n)
            cuda.set_device(primary)
            cache.ensure_experts(n, self.gpu0_local_ids); cache.copy_missing()
            layer._expert_route_contributions(
                cache, hidden, self.gpu0_local_weights, self.gpu0_local_ids,
                views=cache.bank_views(), alphas=cache.alphas_for_slots(n),
                out=self.gpu0_local_routes, gate_up_out=self.gpu0_gate_up,
                activation_out=self.gpu0_activation_out, active_count=local_count)
            self.gpu0_reconstruction.zero_()
            scatter_compact(self.gpu0_local_routes, self.gpu0_local_positions,
                            local_count, self.gpu0_reconstruction)
            for x in self.active_workers:
                getattr(self, f"done_{x}_events")[n].wait(stream)
                scatter_active_routes(self.gpu0_reconstruction, getattr(self, f"host_{x}_return"),
                                      getattr(self, f"gpu0_{x}_positions"),
                                      getattr(self, f"gpu0_{x}_count"))
            moe_sum_reduce_triton(self.gpu0_reconstruction, self.gpu0_output)
            return self.gpu0_output
        except Exception:
            self._failure_count += 1
            try:
                for x in self.active_workers:
                    cuda.set_device(getattr(self, f"worker_{x}_torch_device"))
                    getattr(self, f"worker_{x}_stream").synchronize()
            finally: cuda.set_device(primary)
            raise
        finally: cuda.set_device(primary)

    def reset_counters(self):
        super().reset_counters()
        if hasattr(self, "transport_counts"): self.transport_counts.zero_()
        if hasattr(self, "active_route_histogram"): self.active_route_histogram.zero_()
        if hasattr(self, "d7_participation_histogram"): self.d7_participation_histogram.zero_()

    def configuration_report(self) -> dict[str, Any]:
        base = super().configuration_report()
        return {**base, "schema": D6_EXECUTOR_SCHEMA,
                "count_aware_return_transport": True,
                "return_transport_mechanism": "device_packed_mapped_host_zero_copy",
                "count_aware_inbound_metadata": True,
                "d7_participation_diagnostics": self.d7_participation_diagnostics,
                "zero_route_return_payload_bytes": 0,
                "d5_fixed_return_bytes_per_leg": self.top_k * self._return_row_bytes,
                "steady_state_expert_weight_bytes_host_to_worker_a": 0,
                "steady_state_expert_weight_bytes_host_to_worker_b": 0}

    def snapshot(self):
        base = super().snapshot(); counts = self.transport_counts.detach().cpu()
        histogram = self.active_route_histogram.detach().cpu()
        workers = {}
        for index, x in enumerate(("a", "b")):
            if x not in self.active_workers: continue
            row = [int(counts[:, index, field].sum().item()) for field in range(8)]
            hist = [int(histogram[:, index, count].sum().item()) for count in range(self.top_k + 1)]
            workers[x] = {"activation_bytes_h2d": row[0], "route_metadata_bytes_h2d": row[1],
                          "useful_returned_contribution_bytes": row[2],
                          "actual_returned_bytes_d2h": row[3], "actual_returned_bytes_h2d_gpu0": row[4],
                          "fixed_overhead_bytes": row[5], "zero_route_layers": row[6],
                          "layer_calls": row[7], "active_route_histogram": hist}
        d5_return = sum(v["layer_calls"] for v in workers.values()) * self.top_k * self._return_row_bytes * 2
        actual_return = sum(v["actual_returned_bytes_d2h"] + v["actual_returned_bytes_h2d_gpu0"]
                            for v in workers.values())
        participation = None
        if self.d7_participation_diagnostics:
            joint = self.d7_participation_histogram.detach().cpu()
            values = [int(joint[:, state].sum().item()) for state in range(4)]
            participation = {"state_order": ["zero", "a_only", "b_only", "both"],
                             "counts": dict(zip(("zero", "a_only", "b_only", "both"), values, strict=True)),
                             "total_layer_events": sum(values),
                             "mean_remote_workers_active": (values[1] + values[2] + 2 * values[3]) / sum(values) if sum(values) else 0.0}
        return {**base, "d7_participation": participation, "transport": {"workers": workers,
                "total_worker_transport_bytes": sum(v["activation_bytes_h2d"] + v["route_metadata_bytes_h2d"]
                                                     + v["actual_returned_bytes_d2h"]
                                                     + v["actual_returned_bytes_h2d_gpu0"] for v in workers.values()),
                "returned_bytes_saved_vs_d5": d5_return - actual_return,
                "d5_fixed_return_bytes": d5_return, "actual_return_bytes": actual_return}}


def absent_d6_count_aware_transport_report():
    return {"schema": D6_EXECUTOR_SCHEMA, "experimental": True, "enabled": False,
            "graph_active": False, "eager_fallback": False, "fallback_count": 0,
            "failure_count": 0, "graph_recapture_count": 0, "steady_state_host_sync_count": 0}
