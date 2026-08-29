from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from freetoken.engine.engine import Engine
from freetoken.layers.moe import OffloadMoELayer
from freetoken.moe.inferswarm_remote_decode import (
    HostStagedRemoteTransport,
    InferSwarmRemoteDecodeExecutor,
    TransferByteCounters,
    absent_remote_decode_report,
    build_remote_slot_lookup,
    validate_remote_decode_runtime,
)


@pytest.fixture(autouse=True)
def _cpu_moe_sum_reduce(monkeypatch):
    """The executor unit fixtures are CPU-only; production coverage uses Triton below."""
    import freetoken.moe.inferswarm_remote_decode as remote_decode

    def reduce_routes(routes, out):
        out.copy_(routes.sum(dim=1))

    monkeypatch.setattr(remote_decode, "moe_sum_reduce_triton", reduce_routes)


class _Placement:
    num_layers = 2
    num_experts = 3
    remote_slots = 3
    bytes_per_slot = 10
    artifact_sha256 = "a" * 64
    identities_in_rank_order = (
        SimpleNamespace(layer_id=1, expert_id=1, remote_slot=0),
        SimpleNamespace(layer_id=0, expert_id=0, remote_slot=1),
        SimpleNamespace(layer_id=1, expert_id=2, remote_slot=2),
    )


class _Resident:
    placement = _Placement()
    report = SimpleNamespace(
        layout=SimpleNamespace(
            quant_format="nvfp4",
            nvfp4_backend="triton",
            bank_layout="native_modelopt_nvfp4",
            bank_schema=("bank",),
        ),
        total_live_resident_bytes=18,
        expert_bank_tensor_bytes=18,
    )

    def bank_views(self):
        return (torch.zeros(3, 1),)

    def alpha_views(self):
        return None


def _secondary():
    return SimpleNamespace(
        primary=SimpleNamespace(uuid="GPU-primary", visible_ordinal=0),
        secondary=SimpleNamespace(uuid="GPU-secondary", visible_ordinal=1),
    )


class _Cache:
    def __init__(self, order=None):
        self.ensure_calls = []
        self.copy_calls = 0
        self.routing_records = []
        self.views = (torch.zeros(8, 1),)
        self.layer_timing = None
        self.order = order

    def ensure_experts(self, layer_id, ids, *, record_routing=True):
        if self.order is not None:
            self.order.append("local_ensure")
        raw = ids.clone()
        self.ensure_calls.append((layer_id, raw, record_routing))
        if record_routing:
            self.record_decode_routing(layer_id, raw)
        ids.add_(100)

    def copy_missing(self):
        self.copy_calls += 1

    def record_decode_routing(self, layer_id, ids):
        self.routing_records.append((layer_id, ids.clone()))

    def bank_views(self):
        return self.views

    def alphas_for_slots(self, _layer_id):
        return None


class _Layer:
    def __init__(self, layer_id, order=None):
        self.layer_id = layer_id
        self.local_gemm_calls = 0
        self.route_calls = []
        self.order = order

    def _expert_gemm(
        self,
        cache,
        hidden,
        weights,
        slots,
        *,
        views,
        n,
        alphas,
        is_prefill,
    ):
        assert views is cache.views
        assert n is None and alphas is None and is_prefill is False
        routes = self._expert_route_contributions(
            cache,
            hidden,
            weights,
            slots,
            views=views,
            alphas=alphas,
        )
        return routes.sum(dim=1)

    def _expert_route_contributions(
        self,
        cache,
        hidden,
        weights,
        slots,
        *,
        views,
        alphas,
        out=None,
    ):
        assert views is cache.views
        assert alphas is None
        self.local_gemm_calls += 1
        if self.order is not None:
            self.order.append("local_gemm")
        experts = slots.float() - 100
        routes = (weights * (experts + 1)).unsqueeze(-1).expand(
            *weights.shape, hidden.shape[1]
        ).to(hidden.dtype)
        self.route_calls.append((weights.clone(), slots.clone(), routes.clone()))
        if out is not None:
            out.copy_(routes)
            return out
        return routes.contiguous()


class _Transport:
    # P2 slot -> raw expert for the layer used by each test.
    def __init__(self, slot_to_expert, fail=False, fail_finish=False, order=None):
        self.slot_to_expert = slot_to_expert
        self.fail = fail
        self.fail_finish = fail_finish
        self.calls = []
        self.order = order
        self.transfer_bytes = TransferByteCounters()
        self.drains = 0
        self.returned_routes = []

    def submit(self, layer, cache, hidden, weights, slots):
        if self.order is not None:
            self.order.append("remote_submit")
        self.calls.append(
            (layer.layer_id, hidden.clone(), weights.clone(), slots.clone())
        )
        if self.fail:
            raise RuntimeError("injected remote submit failure")
        activation = hidden.numel() * hidden.element_size()
        weight_bytes = weights.numel() * weights.element_size()
        id_bytes = slots.numel() * slots.element_size()
        self.transfer_bytes.gpu0_to_host_activation += activation
        self.transfer_bytes.gpu0_to_host_routing_weights += weight_bytes
        self.transfer_bytes.gpu0_to_host_routing_ids += id_bytes
        self.transfer_bytes.host_to_gpu1_activation += activation
        self.transfer_bytes.host_to_gpu1_routing_weights += weight_bytes
        self.transfer_bytes.host_to_gpu1_routing_ids += id_bytes
        transfer = {
            "gpu0_to_host": {
                "activation": activation,
                "routing_weights": weight_bytes,
                "routing_ids": id_bytes,
            },
            "host_to_gpu1": {
                "activation": activation,
                "routing_weights": weight_bytes,
                "routing_ids": id_bytes,
                "expert_weights": 0,
            },
            "gpu1_to_host": {
                "returned_route_contributions": activation * weights.shape[1]
            },
            "host_to_gpu0": {
                "returned_route_contributions": activation * weights.shape[1]
            },
        }
        return SimpleNamespace(
            slot_index=0,
            generation=len(self.calls),
            tokens=hidden.shape[0],
            completion_event=None,
            completion_recorded=True,
            finished=False,
            released=False,
            timing_values={},
            transfer_bytes=transfer,
            payload=(hidden.clone(), weights.clone(), slots.clone()),
        )

    def finish(self, pending, **_kwargs):
        if self.order is not None:
            self.order.append("remote_finish")
        if self.fail_finish:
            raise RuntimeError("injected remote completion failure")
        pending.finished = True
        hidden, weights, slots = pending.payload
        expert = torch.zeros_like(slots, dtype=torch.float32)
        for slot, raw in self.slot_to_expert.items():
            expert = torch.where(slots == slot, expert.new_tensor(float(raw)), expert)
        routes = (weights * (expert + 1)).unsqueeze(-1).expand(
            *weights.shape, hidden.shape[1]
        )
        activation = hidden.numel() * hidden.element_size()
        route_bytes = activation * weights.shape[1]
        self.transfer_bytes.gpu1_to_host_returned_route_contributions += route_bytes
        self.transfer_bytes.host_to_gpu0_returned_route_contributions += route_bytes
        routes = routes.to(hidden.dtype).contiguous()
        self.returned_routes.append(routes.clone())
        return routes

    def drain(self, pending):
        self.drains += 1
        pending.finished = True
        pending.released = True

    def release(self, pending):
        if self.order is not None:
            self.order.append("remote_release")
        pending.released = True

    def reset_counters(self):
        self.transfer_bytes.reset()

    def report(self):
        return {"mode": "fake_host_staged"}


def _executor(layer_id=0, *, fail=False, fail_finish=False, mode="overlap", order=None):
    route_lookup = torch.tensor([[-1, -1, -1], [-1, -1, -1]], dtype=torch.int32)
    route_lookup[0, 0] = 1
    route_lookup[1, 1] = 0
    route_lookup[1, 2] = 2
    slot_to_expert = {1: 0} if layer_id == 0 else {0: 1, 2: 2}
    transport = _Transport(
        slot_to_expert, fail=fail, fail_finish=fail_finish, order=order
    )
    executor = InferSwarmRemoteDecodeExecutor(
        resident_bank=_Resident(),
        secondary_device=_secondary(),
        primary_device=torch.device("cpu"),
        transport=transport,
        route_lookup=route_lookup,
        mode=mode,
    )
    executor.begin_decode_step(0)
    return executor, transport


def _reference(raw_ids, weights, hidden):
    scalar = (weights * (raw_ids.float() + 1)).sum(dim=1, keepdim=True)
    return scalar.expand_as(hidden).to(hidden.dtype)


def test_route_lookup_comes_from_p2_deterministic_remote_slots():
    lookup = build_remote_slot_lookup(_Placement(), torch.device("cpu"))
    assert lookup.dtype == torch.int32
    assert lookup.tolist() == [[1, -1, -1], [-1, 0, 2]]


@pytest.mark.parametrize("invalid_id", [-1, _Placement.num_experts])
def test_invalid_raw_expert_id_fails_before_classification_or_service(invalid_id):
    executor, transport = _executor()
    cache, layer = _Cache(), _Layer(0)
    counters_before = executor.counters.aggregate()

    with pytest.raises(RuntimeError, match="invalid P4 raw expert routing"):
        executor.decode(
            layer,
            cache,
            torch.zeros(1, 2),
            torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            torch.tensor([[1, invalid_id]], dtype=torch.int32),
        )

    assert transport.calls == []
    assert cache.ensure_calls == []
    assert cache.copy_calls == 0
    assert cache.routing_records == []
    assert layer.local_gemm_calls == 0
    assert executor.counters.aggregate() == counters_before


def test_local_only_uses_ordinary_cache_service_and_no_remote_dispatch():
    executor, transport = _executor()
    cache, layer = _Cache(), _Layer(0)
    hidden = torch.zeros(1, 2)
    raw = torch.tensor([[1, 2]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    out = executor.decode(layer, cache, hidden, weights, raw)
    torch.testing.assert_close(out, _reference(torch.tensor([[1, 2]]), weights, hidden))
    assert len(transport.calls) == 0
    assert cache.ensure_calls[0][1].tolist() == [[1, 2]]
    assert executor.snapshot()["aggregate"]["executed_on_gpu0"] == 2


def test_remote_only_dispatches_once_skips_gpu0_cache_and_local_gemm():
    executor, transport = _executor()
    cache, layer = _Cache(), _Layer(0)
    hidden = torch.zeros(1, 2)
    raw = torch.tensor([[0, 0]], dtype=torch.int32)
    original = raw.clone()
    weights = torch.tensor([[0.4, 0.6]], dtype=torch.float32)
    out = executor.decode(layer, cache, hidden, weights, raw)
    torch.testing.assert_close(out, _reference(original, weights, hidden))
    assert raw.equal(original)
    assert len(transport.calls) == 1
    assert cache.ensure_calls == [] and cache.copy_calls == 0
    assert layer.local_gemm_calls == 0
    assert executor.snapshot()["aggregate"]["executed_on_gpu0"] == 0


def test_mixed_partition_preserves_route_positions_and_reduces_once(monkeypatch):
    import freetoken.moe.inferswarm_remote_decode as remote_decode

    reduced = []

    def record_reduce(routes, out):
        reduced.append(routes.clone())
        out.copy_(routes.sum(dim=1))

    monkeypatch.setattr(remote_decode, "moe_sum_reduce_triton", record_reduce)
    executor, transport = _executor()
    cache, layer = _Cache(), _Layer(0)
    hidden = torch.zeros(1, 3)
    raw = torch.tensor([[0, 2]], dtype=torch.int32)
    original = raw.clone()
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    out = executor.decode(layer, cache, hidden, weights, raw)
    torch.testing.assert_close(out, _reference(original, weights, hidden))
    assert raw.equal(original)
    assert cache.ensure_calls[0][1].tolist() == [[2, 2]]
    assert cache.ensure_calls[0][2] is False
    assert all(0 not in call[1].tolist() for call in cache.ensure_calls)
    assert cache.routing_records[0][1].tolist() == [[0, 2]]
    _, _, remote_weights, remote_slots = transport.calls[0]
    assert remote_weights.tolist() == [[0.25, 0.0]]
    assert remote_slots.tolist() == [
        [1, 0]
    ]  # slot 0 is the valid zero-weight placeholder
    local_weights, _, local_routes = layer.route_calls[0]
    assert local_weights.tolist() == [[0.0, 0.75]]
    assert local_routes[:, 0].count_nonzero() == 0
    assert transport.returned_routes[0][:, 1].count_nonzero() == 0
    assert len(reduced) == 1
    assert reduced[0][:, :, 0].tolist() == [[0.25, 2.25]]
    snap = executor.snapshot()
    assert snap["aggregate"]["total_router_selections"] == 2
    assert snap["aggregate"]["selected_for_gpu1"] == 1
    assert snap["aggregate"]["executed_on_gpu1"] == 1
    assert snap["aggregate"]["executed_on_gpu0"] == 1
    assert snap["aggregate"]["combine_operations"] == 1
    assert snap["aggregate"]["route_reconstruction_operations"] == 1
    assert snap["aggregate"]["final_sum_reductions"] == 1
    assert snap["ownership"]["successful_selection_arithmetic_exact"] is True


def test_fixed_shape_placeholder_preserves_exact_multi_local_identity_set():
    executor, _transport = _executor()
    cache, layer = _Cache(), _Layer(0)
    raw = torch.tensor([[0, 2], [1, 0]], dtype=torch.int32)
    weights = torch.full((2, 2), 0.5)
    executor.decode(layer, cache, torch.zeros(2, 2), weights, raw)
    serviced = cache.ensure_calls[0][1]
    assert set(serviced.reshape(-1).tolist()) == {1, 2}
    assert 0 not in serviced


def test_multiple_gpu1_experts_and_multiple_tokens_still_use_one_dispatch():
    executor, transport = _executor(layer_id=1)
    cache, layer = _Cache(), _Layer(1)
    hidden = torch.zeros(2, 2)
    raw = torch.tensor([[1, 2], [2, 0]], dtype=torch.int32)
    original = raw.clone()
    weights = torch.tensor([[0.5, 0.5], [0.75, 0.25]], dtype=torch.float32)
    out = executor.decode(layer, cache, hidden, weights, raw)
    torch.testing.assert_close(out, _reference(original, weights, hidden))
    assert len(transport.calls) == 1
    assert cache.ensure_calls[0][1].tolist() == [[0, 0], [0, 0]]
    assert transport.calls[0][3].tolist() == [[0, 2], [2, 0]]
    snap = executor.snapshot()["aggregate"]
    assert snap["remote_dispatches"] == 1
    assert snap["executed_on_gpu1"] == 3
    assert snap["executed_on_gpu0"] == 1


def test_overlap_submits_before_local_service_and_joins_after_local_branch():
    order = []
    executor, transport = _executor(order=order, mode="overlap")
    cache, layer = _Cache(order), _Layer(0, order)
    out = executor.decode(
        layer,
        cache,
        torch.zeros(1, 2),
        torch.tensor([[0.25, 0.75]]),
        torch.tensor([[0, 2]], dtype=torch.int32),
    )
    torch.testing.assert_close(out, torch.full((1, 2), 2.5))
    assert order == [
        "remote_submit",
        "local_ensure",
        "local_gemm",
        "remote_finish",
        "remote_release",
    ]
    assert transport.drains == 0


def test_serialized_diagnostic_joins_before_local_service_with_same_result_and_bytes():
    overlap, _ = _executor(mode="overlap")
    serialized_order = []
    serialized, _ = _executor(mode="serialized", order=serialized_order)
    hidden = torch.zeros(1, 2)
    weights = torch.tensor([[0.25, 0.75]])
    ids = torch.tensor([[0, 2]], dtype=torch.int32)
    out_overlap = overlap.decode(_Layer(0), _Cache(), hidden, weights, ids.clone())
    out_serialized = serialized.decode(
        _Layer(0, serialized_order),
        _Cache(serialized_order),
        hidden,
        weights,
        ids.clone(),
    )
    torch.testing.assert_close(out_overlap, out_serialized)
    assert serialized_order == [
        "remote_submit",
        "remote_finish",
        "local_ensure",
        "local_gemm",
        "remote_release",
    ]
    assert overlap.snapshot()["aggregate"] == serialized.snapshot()["aggregate"]
    assert (
        overlap.snapshot()["steady_state_transfer_bytes"]
        == serialized.snapshot()["steady_state_transfer_bytes"]
    )


def test_remote_completion_failure_is_explicit_and_drains_no_fallback():
    executor, transport = _executor(fail_finish=True)
    with pytest.raises(RuntimeError, match="remote completion failure"):
        executor.decode(
            _Layer(0),
            _Cache(),
            torch.zeros(1, 2),
            torch.tensor([[0.5, 0.5]]),
            torch.tensor([[0, 2]], dtype=torch.int32),
        )
    snap = executor.snapshot()["aggregate"]
    assert snap["explicit_failure"] == 1
    assert snap["fallback_elsewhere"] == 0
    assert transport.drains == 1


def test_local_failure_drains_pending_remote_before_surfacing():
    order = []
    executor, transport = _executor(order=order)

    class FailLocal(_Layer):
        def _expert_route_contributions(self, cache, *args, views, **kwargs):
            if views is cache.views:
                raise RuntimeError("injected local failure")
            return super()._expert_route_contributions(
                cache, *args, views=views, **kwargs
            )

    with pytest.raises(RuntimeError, match="injected local failure"):
        executor.decode(
            FailLocal(0, order),
            _Cache(order),
            torch.zeros(1, 2),
            torch.tensor([[0.5, 0.5]]),
            torch.tensor([[0, 2]], dtype=torch.int32),
        )
    assert order == ["remote_submit", "local_ensure"]
    assert transport.drains == 1
    snap = executor.snapshot()["aggregate"]
    assert snap["explicit_failure"] == 1
    assert snap["fallback_elsewhere"] == 0


def test_serialized_local_failure_after_remote_success_still_fails_f6():
    executor, transport = _executor(mode="serialized")

    class FailLocal(_Layer):
        def _expert_route_contributions(self, cache, *args, views, **kwargs):
            if views is cache.views:
                raise RuntimeError("injected serialized local failure")
            return super()._expert_route_contributions(
                cache, *args, views=views, **kwargs
            )

    with pytest.raises(RuntimeError, match="serialized local failure"):
        executor.decode(
            FailLocal(0),
            _Cache(),
            torch.zeros(1, 2),
            torch.tensor([[0.5, 0.5]]),
            torch.tensor([[0, 2]], dtype=torch.int32),
        )
    snapshot = executor.snapshot()
    assert transport.drains == 1
    assert snapshot["aggregate"]["executed_on_gpu1"] == 1
    assert snapshot["aggregate"]["explicit_failure"] == 1
    assert snapshot["gates"]["F6"]["passed"] is False


def test_remote_failure_is_explicit_and_has_zero_local_fallback():
    executor, _transport = _executor(fail=True)
    cache, layer = _Cache(), _Layer(0)
    with pytest.raises(RuntimeError, match="injected remote submit failure"):
        executor.decode(
            layer,
            cache,
            torch.zeros(1, 2),
            torch.tensor([[0.5, 0.5]]),
            torch.tensor([[0, 2]], dtype=torch.int32),
        )
    assert cache.ensure_calls == []
    snap = executor.snapshot()["aggregate"]
    assert snap["explicit_failure"] == 1
    assert snap["fallback_elsewhere"] == 0
    assert snap["executed_on_gpu0"] == 0


def test_snapshot_reset_preserves_route_map_and_resident_storage():
    executor, _transport = _executor()
    cache, layer = _Cache(), _Layer(0)
    executor.decode(
        layer,
        cache,
        torch.zeros(1, 2),
        torch.tensor([[0.5, 0.5]]),
        torch.tensor([[0, 2]], dtype=torch.int32),
    )
    lookup = executor.route_lookup
    resident = executor.resident_bank
    executor.reset()
    assert executor.route_lookup is lookup
    assert executor.resident_bank is resident
    assert executor.snapshot()["aggregate"]["total_router_selections"] == 0
    assert cache.copy_calls == 1  # local cache residency was not reset/rebuilt


def test_engine_idle_reset_clears_p3_counters_and_preserves_both_residencies():
    calls = []
    cache = SimpleNamespace(
        instrumentation_snapshot=lambda: {"cache_residency": "warm"},
        reset_instrumentation=lambda: calls.append("cache_reset"),
        cache_size=8,
        expert_bank_tensor_bytes=lambda: 80,
    )
    resident = SimpleNamespace(
        report=SimpleNamespace(as_dict=lambda: {"resident_slots": 3})
    )
    remote = SimpleNamespace(
        snapshot=lambda **_kwargs: {"aggregate": {"remote_dispatches": 4}},
        reset=lambda: calls.append("remote_reset"),
    )
    correctness = SimpleNamespace(
        snapshot=lambda: {"enabled": True, "records": [{"uid": 7}]},
        reset=lambda: calls.append("correctness_reset"),
    )
    engine = SimpleNamespace(
        moe_offload_cache=cache,
        inferswarm_resident_bank=resident,
        inferswarm_remote_decode=remote,
        inferswarm_correctness_diagnostics=correctness,
        moe_layer_timing=None,
        _moe_measurement_step=3,
    )
    payload = Engine.moe_instrumentation(engine, "reset")
    assert payload["cache_residency"] == "warm"
    assert payload["inferswarm_resident_bank"]["resident_slots"] == 3
    assert payload["inferswarm_remote_decode"]["aggregate"]["remote_dispatches"] == 4
    assert payload["inferswarm_correctness_diagnostics"]["records"] == [{"uid": 7}]
    assert calls == ["cache_reset", "remote_reset", "correctness_reset"]
    assert engine.moe_offload_cache is cache
    assert engine.inferswarm_resident_bank is resident


@pytest.mark.parametrize(
    "config_change,cache_change,layout_change,match",
    [
        ({"tp_info": SimpleNamespace(size=2)}, {}, {}, "parallel size 1"),
        ({"cuda_graph_max_bs": 1}, {}, {}, "cuda-graph-max-bs 0"),
        ({"moe_backend": "hybrid"}, {}, {}, "moe-backend offload"),
        ({}, {"decode_target": "cpu"}, {}, "GPU decode target"),
        ({}, {"decode_target": "hybrid"}, {}, "GPU decode target"),
        ({}, {"cpu_layer_ids": frozenset({1})}, {}, "zero CPU MoE layers"),
        ({}, {"quant_format": "nvfp4_marlin"}, {}, "native nvfp4"),
        ({}, {}, {"bank_layout": "nvfp4_marlin"}, "native NVFP4/Triton"),
    ],
)
def test_runtime_shape_refusals(config_change, cache_change, layout_change, match):
    config = SimpleNamespace(
        tp_info=SimpleNamespace(size=1),
        cuda_graph_max_bs=0,
        moe_backend="offload",
    )
    cache = SimpleNamespace(
        decode_target="gpu",
        cpu_layer_ids=frozenset(),
        quant_format="nvfp4",
        bank_schema=("bank",),
        bank_views=lambda: (torch.zeros(8, 1),),
    )
    resident = _Resident()
    config.__dict__.update(config_change)
    cache.__dict__.update(cache_change)
    old = resident.report.layout
    resident.report = SimpleNamespace(
        layout=SimpleNamespace(**{**old.__dict__, **layout_change}),
        total_live_resident_bytes=18,
    )
    try:
        with pytest.raises(ValueError, match=match):
            validate_remote_decode_runtime(config, cache, resident, _secondary())
    finally:
        resident.report = SimpleNamespace(layout=old, total_live_resident_bytes=18)


def test_p3_source_has_no_implicit_cuda_one_and_prefill_has_no_remote_branch():
    source = inspect.getsource(HostStagedRemoteTransport)
    assert '"cuda:1"' not in source and "'cuda:1'" not in source
    assert "inferswarm" not in inspect.getsource(OffloadMoELayer._prefill_routed)
    assert absent_remote_decode_report()["aggregate"]["prefill_remote_dispatches"] == 0


def test_offload_native_nvfp4_route_contribution_seam_accepts_none_alphas(monkeypatch):
    import freetoken.layers.moe as moe_layers
    import freetoken.moe.fused_nvfp4 as fused_nvfp4

    monkeypatch.setattr(moe_layers, "get_tp_info", lambda: SimpleNamespace(size=1))
    layer = OffloadMoELayer(
        layer_id=0,
        num_experts=3,
        top_k=2,
        hidden_size=16,
        intermediate_size=16,
        activation="silu",
    )
    hidden = torch.zeros(1, 16)
    weights = torch.tensor([[0.25, 0.75]])
    ids = torch.tensor([[0, 1]], dtype=torch.int32)
    views = (
        torch.empty(3, 32, 8, dtype=torch.uint8),
        torch.empty(3, 32, 1, dtype=torch.float8_e4m3fn),
        torch.empty(3, 32, dtype=torch.float16),
        torch.empty(3, 16, 8, dtype=torch.uint8),
        torch.empty(3, 16, 1, dtype=torch.float8_e4m3fn),
        torch.empty(3, 16, dtype=torch.float16),
    )
    persistent = torch.empty(1, 2, 16)
    calls = []

    def route_kernel(*args, out, **_kwargs):
        calls.append(args)
        return out.fill_(7)

    monkeypatch.setattr(
        fused_nvfp4,
        "fused_experts_decode_nvfp4_marlin_route_contributions",
        route_kernel,
    )
    cache = SimpleNamespace(
        quant_format="nvfp4",
        bank_schema=(
            "gate_up_packed",
            "gate_up_scale",
            "gate_up_global",
            "down_packed",
            "down_scale",
            "down_global",
        ),
    )
    returned = layer._expert_route_contributions(
        cache,
        hidden,
        weights,
        ids,
        views=views,
        alphas=None,
        out=persistent,
    )
    assert returned is persistent
    assert all(
        actual is expected
        for actual, expected in zip(calls[0][:8], (hidden, *views, weights), strict=True)
    )
    assert calls[0][8] is ids

    cache.quant_format = "fp8_block"
    with pytest.raises(RuntimeError, match="native NVFP4/Triton"):
        layer._expert_route_contributions(
            cache,
            hidden,
            weights,
            ids,
            views=views,
            alphas=None,
        )

    cache.quant_format = "nvfp4"
    with pytest.raises(RuntimeError, match="six-bank layout"):
        layer._expert_route_contributions(
            cache,
            hidden,
            weights,
            ids,
            views=views[:4],
            alphas=None,
        )


class _FakeCuda:
    def __init__(self):
        self.current = 0
        self.streams = {0: _FakeStream(), 1: _FakeStream()}

    def set_device(self, ordinal):
        self.current = int(ordinal)

    def current_device(self):
        return self.current

    def synchronize(self, _ordinal):
        pass

    def memory_allocated(self, _ordinal):
        return 0

    def memory_reserved(self, _ordinal):
        return 0

    def Stream(self, device=None):
        return _FakeStream()

    def Event(self, enable_timing=False):
        return _FakeEvent()

    def current_stream(self, _device=None):
        return self.streams[self.current]

    def stream(self, _stream):
        return _FakeContext()


class _FakeContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FakeEvent:
    def record(self, _stream=None):
        pass

    def synchronize(self):
        pass

    def query(self):
        return True

    def elapsed_time(self, _other):
        return 1.0


class _FakeStream:
    def __init__(self):
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1


class _BlockingReturnEvent(_FakeEvent):
    def __init__(self):
        self.synchronize_calls = 0

    def query(self):
        return False

    def synchronize(self):
        self.synchronize_calls += 1


def _cpu_transport_for_device_restore():
    transport = object.__new__(HostStagedRemoteTransport)
    transport.primary_device = torch.device("cpu")
    transport.secondary_device = torch.device("cpu")
    transport.primary_ordinal = 0
    transport.secondary_ordinal = 1
    transport.max_tokens = 2
    transport.hidden_size = 2
    transport.top_k = 2
    transport.hidden_dtype = torch.float32
    transport.resident_bank = _Resident()
    transport._torch = SimpleNamespace(cuda=_FakeCuda())
    transport.timing_enabled = False
    transport.transfer_bytes = TransferByteCounters()
    transport._next_slot = 0
    transport._buffer_reuse_waits = 0
    transport.stream = _FakeStream()
    transport._gpu1_allocated_before_init = 0
    transport._gpu1_allocated_after_init = 0
    transport._gpu1_reserved_before_init = 0
    transport._gpu1_reserved_after_init = 0

    def slot():
        return SimpleNamespace(
            host_activation=torch.empty(2, 2),
            host_slots=torch.empty(2, 2, dtype=torch.int32),
            host_weights=torch.empty(2, 2),
            host_route_contributions=torch.empty(2, 2, 2),
            gpu1_activation=torch.empty(2, 2),
            gpu1_slots=torch.empty(2, 2, dtype=torch.int32),
            gpu1_weights=torch.empty(2, 2),
            gpu1_route_contributions=torch.empty(2, 2, 2),
            gpu0_route_contributions=torch.empty(2, 2, 2),
            stage_ready_event=_FakeEvent(),
            completion_event=_FakeEvent(),
            return_consumed_event=_FakeEvent(),
            timing_events={},
            generation=0,
            inflight=False,
            return_pending=False,
        )

    transport._slots = [slot(), slot()]
    return transport


def test_transport_initialization_failure_restores_primary():
    cuda = _FakeCuda()

    def fail_empty(*_args, **_kwargs):
        raise RuntimeError("pinned allocation failed")

    fake_torch = SimpleNamespace(cuda=cuda, empty=fail_empty)
    with pytest.raises(RuntimeError, match="pinned allocation failed"):
        HostStagedRemoteTransport(
            primary_device=torch.device("cuda", 0),
            secondary_device=torch.device("cuda", 1),
            max_tokens=1,
            hidden_size=2,
            top_k=2,
            hidden_dtype=torch.float32,
            resident_bank=_Resident(),
            torch_module=fake_torch,
        )
    assert cuda.current_device() == 0


@pytest.mark.parametrize("fail", [False, True])
def test_transport_restores_primary_after_success_and_kernel_failure(fail):
    transport = _cpu_transport_for_device_restore()

    class Layer:
        def _expert_route_contributions(
            self, _cache, hidden, _weights, slots, *, out, **_kwargs
        ):
            if fail:
                raise RuntimeError("kernel failed")
            return out.copy_((hidden + 1).unsqueeze(1).expand(-1, slots.shape[1], -1))

    args = (
        Layer(),
        SimpleNamespace(),
        torch.zeros(1, 2),
        torch.tensor([[0.5, 0.5]]),
        torch.tensor([[0, 1]], dtype=torch.int32),
    )
    if fail:
        with pytest.raises(RuntimeError, match="kernel failed"):
            transport.execute(*args)
    else:
        torch.testing.assert_close(transport.execute(*args), torch.ones(1, 2, 2))
    assert transport._torch.cuda.current_device() == 0


def test_transport_repeated_calls_wait_before_reusing_return_buffers():
    transport = _cpu_transport_for_device_restore()
    events = [_BlockingReturnEvent(), _BlockingReturnEvent()]
    for slot, event in zip(transport._slots, events, strict=True):
        slot.return_consumed_event = event

    class Layer:
        def _expert_route_contributions(
            self, _cache, hidden, _weights, slots, *, out, **_kwargs
        ):
            return out.copy_((hidden + 1).unsqueeze(1).expand(-1, slots.shape[1], -1))

    args = (
        Layer(),
        SimpleNamespace(),
        torch.zeros(1, 2),
        torch.tensor([[0.5, 0.5]]),
        torch.tensor([[0, 1]], dtype=torch.int32),
    )
    for _ in range(8):
        torch.testing.assert_close(transport.execute(*args), torch.ones(1, 2, 2))
    assert transport._buffer_reuse_waits == 6
    assert [event.synchronize_calls for event in events] == [3, 3]
    assert all(not slot.inflight for slot in transport._slots)


def test_transport_buffers_and_accounting_use_full_route_geometry():
    transport = _cpu_transport_for_device_restore()

    class Layer:
        def _expert_route_contributions(
            self, _cache, hidden, _weights, slots, *, out, **_kwargs
        ):
            routes = (hidden + 1).unsqueeze(1).expand(-1, slots.shape[1], -1)
            return out.copy_(routes)

    routes = transport.execute(
        Layer(),
        SimpleNamespace(),
        torch.zeros(1, 2),
        torch.tensor([[0.5, 0.5]]),
        torch.tensor([[0, 1]], dtype=torch.int32),
    )
    assert routes.shape == (1, 2, 2)
    report = transport.report()
    capacity = report["payload_capacity_bytes_per_slot"]
    assert capacity == {
        "activation": 16,
        "routing_ids": 16,
        "routing_weights": 16,
        "returned_route_contributions": 32,
    }
    assert report["pinned_host_staging_bytes"] == 160
    assert report["gpu1_persistent_payload_bytes"] == 160
    assert report["gpu0_persistent_return_bytes"] == 64
    transfer = report["steady_state_transfer_bytes"]
    assert transfer["gpu1_to_host"]["returned_route_contributions"] == 16
    assert transfer["host_to_gpu0"]["returned_route_contributions"] == 16


def test_returned_route_copy_failure_releases_slot_and_restores_primary():
    transport = _cpu_transport_for_device_restore()

    class Layer:
        def _expert_route_contributions(
            self, _cache, hidden, _weights, slots, *, out, **_kwargs
        ):
            return out.copy_((hidden + 1).unsqueeze(1).expand(-1, slots.shape[1], -1))

    class FailCopy:
        def __getitem__(self, _key):
            return self

        def copy_(self, _source, non_blocking=False):
            del non_blocking
            raise RuntimeError("injected returned-route copy failure")

    transport._slots[0].gpu0_route_contributions = FailCopy()
    pending = transport.submit(
        Layer(),
        SimpleNamespace(),
        torch.zeros(1, 2),
        torch.tensor([[0.5, 0.5]]),
        torch.tensor([[0, 1]], dtype=torch.int32),
    )
    with pytest.raises(RuntimeError, match="returned-route copy failure"):
        transport.finish(pending)
    assert transport._slots[0].inflight is False
    assert transport._torch.cuda.current_device() == 0


def test_post_return_copy_failure_drains_primary_before_generation_reuse():
    transport = _cpu_transport_for_device_restore()

    class Layer:
        def _expert_route_contributions(
            self, _cache, hidden, _weights, slots, *, out, **_kwargs
        ):
            return out.copy_((hidden + 1).unsqueeze(1).expand(-1, slots.shape[1], -1))

    args = (
        Layer(),
        SimpleNamespace(),
        torch.zeros(1, 2),
        torch.tensor([[0.5, 0.5]]),
        torch.tensor([[0, 1]], dtype=torch.int32),
    )
    pending = transport.submit(*args)
    old_generation = pending.generation

    def fail_after_copy():
        raise RuntimeError("injected post-return-copy timing failure")

    with pytest.raises(RuntimeError, match="post-return-copy timing failure"):
        transport.finish(pending, after_return_copy=fail_after_copy)

    primary_stream = transport._torch.cuda.streams[transport.primary_ordinal]
    assert primary_stream.synchronize_calls == 1
    assert transport._slots[0].inflight is False
    assert pending.finished is False and pending.released is False
    assert transport._torch.cuda.current_device() == transport.primary_ordinal

    # Slot 1 is next, then slot 0 is safely reused at a new generation. The failed
    # handle cannot finish or release that reused storage.
    transport.execute(*args)
    transport.execute(*args)
    assert transport._slots[0].generation == old_generation + 1
    with pytest.raises(RuntimeError, match="stale generation"):
        transport.finish(pending)
    with pytest.raises(RuntimeError, match="stale generation"):
        transport.release(pending)
    assert all(not slot.inflight for slot in transport._slots)
    assert transport._torch.cuda.current_device() == transport.primary_ordinal


def test_post_return_copy_failure_is_explicit_and_cannot_pass_f6():
    executor, _ = _executor(mode="overlap")
    transport = _cpu_transport_for_device_restore()
    executor.transport = transport
    cache = _Cache()

    class Timing:
        def mark(self, _layer_id, marker, *, begin_layer=False):
            del begin_layer
            if marker == "returned_route_contributions_h2d_end":
                raise RuntimeError("injected post-return-copy timing failure")

        def record_cache_metadata(self, *_args, **_kwargs):
            pass

        def annotate(self, *_args, **_kwargs):
            pass

    class Layer(_Layer):
        def _expert_route_contributions(
            self, cache, hidden, weights, slots, *, views, alphas, out=None
        ):
            if views is cache.views:
                return super()._expert_route_contributions(
                    cache,
                    hidden,
                    weights,
                    slots,
                    views=views,
                    alphas=alphas,
                    out=out,
                )
            routes = (hidden + 1).unsqueeze(1).expand(-1, slots.shape[1], -1)
            if out is not None:
                return out.copy_(routes)
            return routes.contiguous()

    cache.layer_timing = Timing()
    with pytest.raises(RuntimeError, match="post-return-copy timing failure"):
        executor.decode(
            Layer(0),
            cache,
            torch.zeros(1, 2),
            torch.tensor([[0.5, 0.5]]),
            torch.tensor([[0, 2]], dtype=torch.int32),
        )
    snapshot = executor.snapshot()
    aggregate = snapshot["aggregate"]
    assert aggregate["selected_for_gpu1"] == 1
    assert aggregate["executed_on_gpu1"] == 0
    assert aggregate["explicit_failure"] == 1
    assert aggregate["fallback_elsewhere"] == 0
    assert aggregate["combine_operations"] == 0
    assert snapshot["gates"]["F6"]["passed"] is False
    assert all(not slot.inflight for slot in transport._slots)
    assert transport._torch.cuda.current_device() == transport.primary_ordinal
