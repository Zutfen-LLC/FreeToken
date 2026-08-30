"""Experimental D3 captured three-way InferSwarm route executor.

This is intentionally separate from D2.  It has no eager execution mode: the
whole-model CUDA graph owns GPU0, worker-A and worker-B nodes.
"""
from __future__ import annotations

from typing import Any
import torch

from freetoken.kernel import moe_sum_reduce_triton

D3_EXECUTOR_SCHEMA = "freetoken.inferswarm-d3-graph-multiworker/1"
D3_TOPOLOGY = "unified_three_device_whole_model_graph_independent_ab_fanout"
D3_DEPENDENCY = "cuda_capture_internal_gpu0_ready_ab_independent_done_fanin"
D3_FANOUT_SHAPE = "CONCURRENT_BOUNDED_TWO_WORKER"


def build_d3_route_lookups(placement, primary_device: torch.device):
    """Build GPU0 slot tables and prove the frozen logical ownership partition."""
    nlayer, nexpert = placement.worker_a.num_layers, placement.worker_a.num_experts
    a = torch.full((nlayer, nexpert), -1, dtype=torch.int32, device=primary_device)
    b = torch.full_like(a, -1)
    for layer in placement.worker_a.per_layer:
        if layer.expert_ids:
            a[layer.layer_id, torch.tensor(layer.expert_ids, device=primary_device)] = torch.tensor(layer.remote_slots, dtype=torch.int32, device=primary_device)
    for layer in placement.worker_b.per_layer:
        if layer.expert_ids:
            b[layer.layer_id, torch.tensor(layer.expert_ids, device=primary_device)] = torch.tensor(layer.remote_slots, dtype=torch.int32, device=primary_device)
    # Startup proof intentionally uses host placement metadata, never decode data.
    a_ids, b_ids, local_ids = set(placement.worker_a.flat_ids_in_rank_order), set(placement.worker_b.flat_ids_in_rank_order), set(placement.local_remainder)
    if a_ids & b_ids or len(a_ids | b_ids) != 6000 or len(local_ids) != 4240:
        raise ValueError("D3 frozen worker ownership/disjointness arithmetic failed")
    if (a_ids | b_ids) & local_ids or a_ids | b_ids | local_ids != set(range(nlayer * nexpert)):
        raise ValueError("D3 frozen placement does not resolve every identity exactly once")
    return a, b


def build_d3_local_fallback_ids(placement, primary_device: torch.device) -> torch.Tensor:
    """Return one *local-remainder* identity per layer; fail closed if absent."""
    nlayer, nexpert = placement.worker_a.num_layers, placement.worker_a.num_experts
    local = set(placement.local_remainder)
    values = []
    for layer in range(nlayer):
        fallback = next((expert for expert in range(nexpert) if layer * nexpert + expert in local), None)
        if fallback is None:
            raise ValueError(f"D3 layer {layer} has no GPU0 local-remainder fallback identity")
        values.append(fallback)
    return torch.tensor(values, dtype=torch.int32, device=primary_device)


def validate_d3_runtime(config, cache, banks, workers) -> None:
    if bool(getattr(config, "inferswarm_remote_decode", False)) or bool(getattr(config, "inferswarm_experimental_d2_graph_remote", False)):
        raise ValueError("D3 is mutually exclusive with canonical remote decode and D2")
    if int(getattr(getattr(config, "tp_info", None), "size", -1)) != 1 or int(getattr(config, "cuda_graph_max_bs", -1) or -1) != 1 or int(getattr(config, "max_running_req", -1)) != 1:
        raise ValueError("D3 requires TP1, CUDA graph BS1, and max running requests 1")
    if getattr(config, "moe_backend", None) != "offload" or getattr(cache, "decode_target", None) != "gpu" or getattr(cache, "cpu_layer_ids", frozenset()):
        raise ValueError("D3 requires GPU decode with offload and zero CPU MoE layers")
    if getattr(cache, "quant_format", None) != "nvfp4":
        raise ValueError("D3 requires native NVFP4/Triton")
    if workers[0].secondary.visible_ordinal == workers[1].secondary.visible_ordinal:
        raise ValueError("D3 worker devices must be distinct")
    for bank, worker in ((banks.worker_a, workers[0]), (banks.worker_b, workers[1])):
        if bank.report.secondary_visible_ordinal != worker.secondary.visible_ordinal:
            raise ValueError("D3 resident bank device binding disagreement")


class InferSwarmD3GraphMultiworkerExecutor:
    """Fixed [1, top_k] three-owner route executor captured into the CUDA graph."""
    def __init__(self, *, resident_banks, worker_a_device, worker_b_device, primary_device, worker_a_slot_lookup, worker_b_slot_lookup, local_fallback_ids, hidden_size, top_k, hidden_dtype, num_layers, intermediate_size, torch_module=torch):
        self.resident_banks, self.worker_a_device, self.worker_b_device = resident_banks, worker_a_device, worker_b_device
        self.primary_device = primary_device
        self.worker_a_torch_device = torch.device("cuda", int(worker_a_device.secondary.visible_ordinal))
        self.worker_b_torch_device = torch.device("cuda", int(worker_b_device.secondary.visible_ordinal))
        self.worker_a_slot_lookup, self.worker_b_slot_lookup, self.local_fallback_ids = worker_a_slot_lookup, worker_b_slot_lookup, local_fallback_ids
        self.hidden_size, self.top_k, self.hidden_dtype, self.num_layers, self.intermediate_size = int(hidden_size), int(top_k), hidden_dtype, int(num_layers), int(intermediate_size)
        self._torch, self._captured_bs, self._capture_complete, self._failure_count, self._steady_state_host_sync_count, self._graph_recapture_count = torch_module, (), False, 0, 0, 0
        cuda, primary = torch_module.cuda, int(primary_device.index)
        try:
            cuda.set_device(primary)
            # Host activation is immutable until both workers have consumed it.
            self.host_activation = torch_module.empty((1, self.hidden_size), dtype=hidden_dtype, pin_memory=True)
            for prefix in ("a", "b"):
                setattr(self, f"host_{prefix}_slots", torch_module.empty((1, self.top_k), dtype=torch_module.int32, pin_memory=True))
                setattr(self, f"host_{prefix}_weights", torch_module.empty((1, self.top_k), dtype=torch_module.float32, pin_memory=True))
                setattr(self, f"host_{prefix}_return", torch_module.empty((1, self.top_k, self.hidden_size), dtype=hidden_dtype, pin_memory=True))
            for prefix in ("a", "b"):
                setattr(self, f"gpu0_{prefix}_slots", torch_module.empty((1, self.top_k), dtype=torch_module.int32, device=primary_device))
                setattr(self, f"gpu0_{prefix}_weights", torch_module.empty((1, self.top_k), dtype=torch_module.float32, device=primary_device))
                setattr(self, f"gpu0_{prefix}_return_routes", torch_module.empty((1, self.top_k, self.hidden_size), dtype=hidden_dtype, device=primary_device))
            self.gpu0_local_ids = torch_module.empty((1, self.top_k), dtype=torch_module.int32, device=primary_device)
            self.gpu0_local_weights = torch_module.empty((1, self.top_k), dtype=torch_module.float32, device=primary_device)
            self.gpu0_local_routes = torch_module.empty((1, self.top_k, self.hidden_size), dtype=hidden_dtype, device=primary_device)
            self.gpu0_reconstruction = torch_module.empty_like(self.gpu0_local_routes)
            self.gpu0_final_routes = torch_module.empty_like(self.gpu0_local_routes)
            self.gpu0_output = torch_module.empty((1, self.hidden_size), dtype=hidden_dtype, device=primary_device)
            self.gpu0_gate_up = torch_module.empty((1, self.top_k, 2 * self.intermediate_size), dtype=hidden_dtype, device=primary_device)
            self.gpu0_activation_out = torch_module.empty((self.top_k, self.intermediate_size), dtype=hidden_dtype, device=primary_device)
            self.zero_weight = torch_module.zeros((), dtype=torch_module.float32, device=primary_device)
            self.device_counts = torch_module.zeros((self.num_layers, 5), dtype=torch_module.int64, device=primary_device)
            self.ready_events = [cuda.Event() for _ in range(self.num_layers)]
            self.done_a_events = [cuda.Event() for _ in range(self.num_layers)]
            self.done_b_events = [cuda.Event() for _ in range(self.num_layers)]
            # Bounded, startup-only timing surface.  The later physical primitive may
            # attach device-local markers here; cross-device absolute clocks are never
            # implied by these labels.
            self.timing_hooks = {
                "gpu0_local_branch": ("gpu0_local_start", "gpu0_local_end"),
                "gpu0_join_wait": ("gpu0_join_start", "gpu0_join_end"),
                "gpu0_return_a_h2d": ("gpu0_return_a_start", "gpu0_return_a_end"),
                "gpu0_return_b_h2d": ("gpu0_return_b_start", "gpu0_return_b_end"),
                "gpu0_reconstruction_reduction": ("gpu0_reconstruct_start", "gpu0_reduce_end"),
                "worker_a_branch": ("worker_a_local_start", "worker_a_local_end"),
                "worker_b_branch": ("worker_b_local_start", "worker_b_local_end"),
            }
            for prefix, device in (("a", self.worker_a_torch_device), ("b", self.worker_b_torch_device)):
                cuda.set_device(device)
                setattr(self, f"worker_{prefix}_stream", cuda.Stream(device=device))
                setattr(self, f"worker_{prefix}_activation", torch_module.empty((1, self.hidden_size), dtype=hidden_dtype, device=device))
                setattr(self, f"worker_{prefix}_slots", torch_module.empty((1, self.top_k), dtype=torch_module.int32, device=device))
                setattr(self, f"worker_{prefix}_weights", torch_module.empty((1, self.top_k), dtype=torch_module.float32, device=device))
                setattr(self, f"worker_{prefix}_routes", torch_module.empty((1, self.top_k, self.hidden_size), dtype=hidden_dtype, device=device))
                setattr(self, f"worker_{prefix}_gate_up", torch_module.empty((1, self.top_k, 2 * self.intermediate_size), dtype=hidden_dtype, device=device))
                setattr(self, f"worker_{prefix}_activation_out", torch_module.empty((self.top_k, self.intermediate_size), dtype=hidden_dtype, device=device))
        finally:
            cuda.set_device(primary)

    def _validate(self, layer, hidden, weights, ids):
        layer_id = int(layer.layer_id)
        if not 0 <= layer_id < self.num_layers or tuple(hidden.shape) != (1, self.hidden_size) or tuple(weights.shape) != (1, self.top_k) or tuple(ids.shape) != (1, self.top_k):
            raise RuntimeError("D3 accepts only captured batch-one routing geometry")
        if hidden.device != self.primary_device or weights.device != self.primary_device or ids.device != self.primary_device or hidden.dtype != self.hidden_dtype or weights.dtype != torch.float32 or ids.dtype != torch.int32:
            raise RuntimeError("D3 inputs disagree with the captured primary-device contract")
        return layer_id

    def _worker_branch(self, prefix, layer, cache, layer_id):
        cuda = self._torch.cuda; device = getattr(self, f"worker_{prefix}_torch_device"); stream = getattr(self, f"worker_{prefix}_stream")
        cuda.set_device(device)
        with cuda.stream(stream):
            self.ready_events[layer_id].wait(stream)
            getattr(self, f"worker_{prefix}_activation").copy_(self.host_activation, non_blocking=True)
            getattr(self, f"worker_{prefix}_slots").copy_(getattr(self, f"host_{prefix}_slots"), non_blocking=True)
            getattr(self, f"worker_{prefix}_weights").copy_(getattr(self, f"host_{prefix}_weights"), non_blocking=True)
            bank = getattr(self.resident_banks, f"worker_{prefix}")
            layer._expert_route_contributions(cache, getattr(self, f"worker_{prefix}_activation"), getattr(self, f"worker_{prefix}_weights"), getattr(self, f"worker_{prefix}_slots"), views=bank.bank_views(), alphas=bank.alpha_views(), out=getattr(self, f"worker_{prefix}_routes"), gate_up_out=getattr(self, f"worker_{prefix}_gate_up"), activation_out=getattr(self, f"worker_{prefix}_activation_out"))
            getattr(self, f"host_{prefix}_return").copy_(getattr(self, f"worker_{prefix}_routes"), non_blocking=True)
            getattr(self, f"done_{prefix}_events")[layer_id].record(stream)

    def decode(self, layer, cache, hidden_states, topk_weights, topk_ids):
        layer_id = self._validate(layer, hidden_states, topk_weights, topk_ids)
        cuda, primary = self._torch.cuda, int(self.primary_device.index)
        try:
            cuda.set_device(primary); primary_stream = cuda.current_stream(self.primary_device)
            a_slots = self.worker_a_slot_lookup[layer_id][topk_ids.long()]; b_slots = self.worker_b_slot_lookup[layer_id][topk_ids.long()]
            self.gpu0_a_slots.copy_(a_slots); self.gpu0_b_slots.copy_(b_slots)
            a_mask, b_mask = self.gpu0_a_slots >= 0, self.gpu0_b_slots >= 0
            # Startup construction proves lookup overlap impossible.  Do not turn this
            # device predicate into a Python value during captured decode.
            local_mask = ~(a_mask | b_mask)
            torch.where(a_mask, topk_weights, self.zero_weight, out=self.gpu0_a_weights); torch.where(b_mask, topk_weights, self.zero_weight, out=self.gpu0_b_weights)
            self.gpu0_a_slots.clamp_min_(0); self.gpu0_b_slots.clamp_min_(0)
            torch.where(local_mask, topk_ids, self.local_fallback_ids[layer_id], out=self.gpu0_local_ids); torch.where(local_mask, topk_weights, self.zero_weight, out=self.gpu0_local_weights)
            self.device_counts[layer_id, 0].add_(self.top_k); self.device_counts[layer_id, 1].add_(a_mask.sum()); self.device_counts[layer_id, 2].add_(b_mask.sum()); self.device_counts[layer_id, 3].add_(local_mask.sum()); self.device_counts[layer_id, 4].add_(1)
            self.host_activation.copy_(hidden_states, non_blocking=True)
            for prefix in ("a", "b"):
                getattr(self, f"host_{prefix}_slots").copy_(getattr(self, f"gpu0_{prefix}_slots"), non_blocking=True); getattr(self, f"host_{prefix}_weights").copy_(getattr(self, f"gpu0_{prefix}_weights"), non_blocking=True)
            self.ready_events[layer_id].record(primary_stream)
            self._worker_branch("a", layer, cache, layer_id); self._worker_branch("b", layer, cache, layer_id)
            cuda.set_device(primary)
            cache.ensure_experts(layer_id, self.gpu0_local_ids); cache.copy_missing()
            layer._expert_route_contributions(cache, hidden_states, self.gpu0_local_weights, self.gpu0_local_ids, views=cache.bank_views(), alphas=cache.alphas_for_slots(layer_id), out=self.gpu0_local_routes, gate_up_out=self.gpu0_gate_up, activation_out=self.gpu0_activation_out)
            self.done_a_events[layer_id].wait(primary_stream); self.done_b_events[layer_id].wait(primary_stream)
            self.gpu0_a_return_routes.copy_(self.host_a_return, non_blocking=True); self.gpu0_b_return_routes.copy_(self.host_b_return, non_blocking=True)
            torch.add(self.gpu0_local_routes, self.gpu0_a_return_routes, out=self.gpu0_reconstruction); torch.add(self.gpu0_reconstruction, self.gpu0_b_return_routes, out=self.gpu0_final_routes)
            moe_sum_reduce_triton(self.gpu0_final_routes, self.gpu0_output)
            return self.gpu0_output
        except Exception:
            self._failure_count += 1
            try:
                for device, stream in ((self.worker_a_torch_device, self.worker_a_stream), (self.worker_b_torch_device, self.worker_b_stream)):
                    cuda.set_device(device); stream.synchronize()
            finally: cuda.set_device(primary)
            raise
        finally: cuda.set_device(primary)

    def set_graph_state(self, captured_bs):
        captured = tuple(sorted(int(bs) for bs in captured_bs))
        if captured != (1,): raise RuntimeError(f"D3 refused silent eager fallback: expected exactly CUDA graph BS1, captured {list(captured)}")
        if self._capture_complete: self._graph_recapture_count += 1
        self._captured_bs, self._capture_complete = captured, True; self.reset_counters()
    def reset_counters(self): self.device_counts.zero_()
    def configuration_report(self) -> dict[str, Any]:
        a, b = self.worker_a_device.secondary, self.worker_b_device.secondary; primary = self.worker_a_device.primary; active = self._capture_complete and self._captured_bs == (1,)
        return {"schema": D3_EXECUTOR_SCHEMA, "experimental": True, "enabled": True, "graph_active": active, "graph_topology": D3_TOPOLOGY, "captured_batch_sizes": list(self._captured_bs), "graph_replays_per_token": 1 if active else 0, "graph_recapture_count": self._graph_recapture_count, "eager_fallback": not active, "cross_device_dependency": D3_DEPENDENCY, "fanout_shape": D3_FANOUT_SHAPE, "primary_uuid": primary.uuid, "worker_a_uuid": a.uuid, "worker_b_uuid": b.uuid, "corrected_placement_sha256": self.resident_banks.worker_a.placement.artifact_sha256, "worker_a_resident_slots": self.resident_banks.worker_a.placement.remote_slots, "worker_b_resident_slots": self.resident_banks.worker_b.placement.remote_slots, "worker_a_startup_resident_bytes": self.resident_banks.worker_a.report.total_live_resident_bytes, "worker_b_startup_resident_bytes": self.resident_banks.worker_b.report.total_live_resident_bytes, "steady_state_expert_weight_bytes_host_to_worker_a": 0, "steady_state_expert_weight_bytes_host_to_worker_b": 0, "steady_state_host_sync_count": self._steady_state_host_sync_count, "fallback_count": 0, "failure_count": self._failure_count, "fixed_allocations": True, "stable_tensor_addresses": True, "reconstruction_method": "elementwise_local_plus_a_plus_b_then_one_canonical_route_sum", "timing_instrumentation": {"bounded": True, "worker_a_local_clock": True, "worker_b_local_clock": True, "gpu0_local_clock": True, "hooks": self.timing_hooks}}
    def snapshot(self):
        counts = self.device_counts.detach().cpu(); total, a, b, local, calls = (int(counts[:, i].sum().item()) for i in range(5)); exact = total == a + b + local
        return {**self.configuration_report(), "ownership": {"total_router_selections": total, "executed_on_worker_a": a, "executed_on_worker_b": b, "executed_on_gpu0_local": local, "layer_calls": calls, "selection_arithmetic_exact": exact, "no_route_dropped": exact, "no_route_duplicated": exact, "worker_ab_disjoint": True, "per_layer": [{"layer_id": i, "total_router_selections": int(row[0].item()), "executed_on_worker_a": int(row[1].item()), "executed_on_worker_b": int(row[2].item()), "executed_on_gpu0_local": int(row[3].item()), "layer_calls": int(row[4].item()), "selection_arithmetic_exact": int(row[0].item()) == int(row[1].item()) + int(row[2].item()) + int(row[3].item())} for i, row in enumerate(counts)]}}

def absent_d3_graph_multiworker_report():
    return {"schema": D3_EXECUTOR_SCHEMA, "experimental": True, "enabled": False, "graph_active": False, "graph_topology": None, "captured_batch_sizes": [], "graph_replays_per_token": 0, "eager_fallback": False, "steady_state_host_sync_count": 0}
