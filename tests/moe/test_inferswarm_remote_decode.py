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
    absent_remote_decode_report,
    build_remote_slot_lookup,
    validate_remote_decode_runtime,
)


class _Placement:
    num_layers = 2
    num_experts = 3
    remote_slots = 3
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
    def __init__(self):
        self.ensure_calls = []
        self.copy_calls = 0
        self.routing_records = []
        self.views = (torch.zeros(8, 1),)

    def ensure_experts(self, layer_id, ids, *, record_routing=True):
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
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.local_gemm_calls = 0

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
        self.local_gemm_calls += 1
        experts = slots.float() - 100
        scalar = (weights * (experts + 1)).sum(dim=1, keepdim=True)
        return scalar.expand_as(hidden).to(hidden.dtype)


class _Transport:
    # P2 slot -> raw expert for the layer used by each test.
    def __init__(self, slot_to_expert, fail=False):
        self.slot_to_expert = slot_to_expert
        self.fail = fail
        self.calls = []

    def execute(self, layer, cache, hidden, weights, slots):
        self.calls.append(
            (layer.layer_id, hidden.clone(), weights.clone(), slots.clone())
        )
        if self.fail:
            raise RuntimeError("injected remote failure")
        expert = torch.zeros_like(slots, dtype=torch.float32)
        for slot, raw in self.slot_to_expert.items():
            expert = torch.where(slots == slot, expert.new_tensor(float(raw)), expert)
        scalar = (weights * (expert + 1)).sum(dim=1, keepdim=True)
        return scalar.expand_as(hidden).to(hidden.dtype)

    def report(self):
        return {"mode": "fake_host_staged"}


def _executor(layer_id=0, *, fail=False):
    route_lookup = torch.tensor([[-1, -1, -1], [-1, -1, -1]], dtype=torch.int32)
    route_lookup[0, 0] = 1
    route_lookup[1, 1] = 0
    route_lookup[1, 2] = 2
    slot_to_expert = {1: 0} if layer_id == 0 else {0: 1, 2: 2}
    transport = _Transport(slot_to_expert, fail=fail)
    executor = InferSwarmRemoteDecodeExecutor(
        resident_bank=_Resident(),
        secondary_device=_secondary(),
        primary_device=torch.device("cpu"),
        transport=transport,
        route_lookup=route_lookup,
    )
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

    with pytest.raises(RuntimeError, match="invalid P3 raw expert routing"):
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


def test_mixed_partition_excludes_remote_ids_and_combines_once():
    executor, transport = _executor()
    cache, layer = _Cache(), _Layer(0)
    hidden = torch.zeros(1, 3)
    raw = torch.tensor([[0, 2]], dtype=torch.int32)
    original = raw.clone()
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    out = executor.decode(layer, cache, hidden, weights, raw)
    torch.testing.assert_close(out, _reference(original, weights, hidden))
    assert raw.equal(original)
    assert cache.ensure_calls[0][1].tolist() == [2]
    assert cache.ensure_calls[0][2] is False
    assert all(0 not in call[1].tolist() for call in cache.ensure_calls)
    assert cache.routing_records[0][1].tolist() == [[0, 2]]
    _, _, remote_weights, remote_slots = transport.calls[0]
    assert remote_weights.tolist() == [[0.25, 0.0]]
    assert remote_slots.tolist() == [
        [1, 0]
    ]  # slot 0 is the valid zero-weight placeholder
    snap = executor.snapshot()
    assert snap["aggregate"]["total_router_selections"] == 2
    assert snap["aggregate"]["selected_for_gpu1"] == 1
    assert snap["aggregate"]["executed_on_gpu1"] == 1
    assert snap["aggregate"]["executed_on_gpu0"] == 1
    assert snap["aggregate"]["combine_operations"] == 1
    assert snap["ownership"]["successful_selection_arithmetic_exact"] is True


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
    assert cache.ensure_calls[0][1].tolist() == [0]
    assert transport.calls[0][3].tolist() == [[0, 2], [2, 0]]
    snap = executor.snapshot()["aggregate"]
    assert snap["remote_dispatches"] == 1
    assert snap["executed_on_gpu1"] == 3
    assert snap["executed_on_gpu0"] == 1


def test_remote_failure_is_explicit_and_has_zero_local_fallback():
    executor, _transport = _executor(fail=True)
    cache, layer = _Cache(), _Layer(0)
    with pytest.raises(RuntimeError, match="injected remote failure"):
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
    )
    resident = SimpleNamespace(
        report=SimpleNamespace(as_dict=lambda: {"resident_slots": 3})
    )
    remote = SimpleNamespace(
        snapshot=lambda: {"aggregate": {"remote_dispatches": 4}},
        reset=lambda: calls.append("remote_reset"),
    )
    engine = SimpleNamespace(
        moe_offload_cache=cache,
        inferswarm_resident_bank=resident,
        inferswarm_remote_decode=remote,
    )
    payload = Engine.moe_instrumentation(engine, "reset")
    assert payload["cache_residency"] == "warm"
    assert payload["inferswarm_resident_bank"]["resident_slots"] == 3
    assert payload["inferswarm_remote_decode"]["aggregate"]["remote_dispatches"] == 4
    assert calls == ["cache_reset", "remote_reset"]
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


class _FakeCuda:
    def __init__(self):
        self.current = 0

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
    transport.host_activation = torch.empty(2, 2)
    transport.host_slots = torch.empty(2, 2, dtype=torch.int32)
    transport.host_weights = torch.empty(2, 2)
    transport.host_partial = torch.empty(2, 2)
    transport.gpu1_activation = torch.empty(2, 2)
    transport.gpu1_slots = torch.empty(2, 2, dtype=torch.int32)
    transport.gpu1_weights = torch.empty(2, 2)
    transport.gpu0_partial = torch.empty(2, 2)
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
        def _expert_gemm(self, _cache, hidden, *_args, **_kwargs):
            if fail:
                raise RuntimeError("kernel failed")
            return hidden + 1

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
        torch.testing.assert_close(transport.execute(*args), torch.ones(1, 2))
    assert transport._torch.cuda.current_device() == 0
