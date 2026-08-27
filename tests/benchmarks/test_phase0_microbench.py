"""The two GPU microbenchmarks: what they measure, what they refuse to claim.

CUDA is not available in CI, so the kernels themselves are not run. What is tested is
everything that decides whether the resulting number means what it says: the geometry
contract, the GPU binding, the ``top_k = 1`` single-expert measurement, and the refusal to
divide a grouped step by ``top_k`` and call the quotient a latency.
"""

from __future__ import annotations

import sys
import types

import pytest

from .fakes import FAKE_UUID

from inferswarm_phase0 import device_bandwidth as dbw
from inferswarm_phase0 import expert_microbench as micro
from inferswarm_phase0 import gpu as gpu_mod

OTHER_UUID = "GPU-99999999-8888-7777-6666-555555555555"


# --- geometry ---------------------------------------------------------------------------------

def test_valid_qwen_geometry_is_accepted():
    assert micro.validate_geometry(2048, 512, 8, 32) is None


@pytest.mark.parametrize(
    "args,expected",
    [
        ((2047, 512, 8, 32), "multiple of 16"),
        ((2048, 500, 8, 32), "multiple of 16"),
        ((2048, 512, 0, 32), "top_k must be at least 1"),
        ((2048, 512, 8, 4), "cannot hold top_k"),
        ((0, 512, 8, 32), "must be positive"),
    ],
)
def test_bad_geometry_is_refused_with_a_reason(args, expected):
    assert expected in micro.validate_geometry(*args)


def test_the_per_expert_weight_bytes_follow_the_native_bank_layout():
    """gate_up [2I, H/2] + [2I, H/16] + [2I], down [H, I/2] + [H, I/16] + [H]."""
    hidden, inter = 2048, 512
    expected = (
        2 * inter * (hidden // 2 + hidden // 16 + 2)
        + hidden * (inter // 2 + inter // 16 + 2)
    )
    assert micro.per_expert_weight_bytes(hidden, inter) == expected


# --- what the two measurements are, and are not -------------------------------------------------

def test_the_single_expert_measurement_is_top_k_one():
    assert "top_k=1" in micro.SINGLE_EXPERT_MEASUREMENT
    assert "ONE routed expert per call" in micro.SINGLE_EXPERT_MEASUREMENT


def test_the_grouped_measurement_refuses_the_per_expert_division():
    assert "NOT divided by" in micro.GROUPED_MEASUREMENT
    assert "not single-expert latency" in micro.GROUPED_MEASUREMENT


# --- GPU binding, with the hardware mocked -----------------------------------------------------

class _FakeDevice:
    def __init__(self, index=0):
        self.index = index
        self.type = "cuda"


def _fake_torch(available=True):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        get_device_capability=lambda device: (8, 6),
        synchronize=lambda device=None: None,
        empty_cache=lambda: None,
        Event=lambda enable_timing=False: types.SimpleNamespace(
            record=lambda: None, synchronize=lambda: None, elapsed_time=lambda other: 4.0
        ),
    )
    return torch


def _mock_kernels(monkeypatch, timings):
    """Run the microbenchmark end to end with the CUDA work replaced by fixed timings."""
    torch = _fake_torch()
    for name in ("bfloat16", "float16", "float32", "int32", "uint8", "float8_e4m3fn"):
        setattr(torch, name, name)
    torch.randn = lambda *shape, **kw: _FakeTensor(1)
    torch.arange = lambda *a, **kw: _FakeTensor(1)
    torch.full = lambda *a, **kw: _FakeTensor(1)
    monkeypatch.setitem(sys.modules, "torch", torch)

    perf = types.ModuleType("freetoken.benchmark.perf")
    perf.perf_cuda = lambda fn, repetitions=0, cuda_graph_repetitions=None: 0.0
    monkeypatch.setitem(sys.modules, "freetoken.benchmark.perf", perf)
    fused = types.ModuleType("freetoken.moe.fused_nvfp4")
    fused.fused_experts_decode_nvfp4_marlin = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "freetoken.moe.fused_nvfp4", fused)

    monkeypatch.setattr(micro, "_synthetic_banks", lambda *a, **kw: {})
    monkeypatch.setattr(
        micro, "bind_torch_device",
        lambda selector: (
            _FakeDevice(0),
            {"index": 0, "name": "NVIDIA GeForce RTX 3060", "uuid": FAKE_UUID},
            {"matches": True, "bound_uuid": FAKE_UUID},
        ),
    )
    supplied = iter(timings)
    monkeypatch.setattr(
        micro, "_time_kernel",
        lambda torch_, perf_cuda, fn, *, warmup, repetitions, device: next(supplied),
    )


def test_the_single_expert_number_comes_from_a_top_k_one_call(monkeypatch):
    _mock_kernels(monkeypatch, [0.5, 2.0])
    result = micro.measure_single_expert_nvfp4(gpu=FAKE_UUID, top_k=8, repetitions=4, warmup=1)
    assert result["single_expert"]["top_k"] == 1
    assert result["single_expert"]["latency_ms"] == 0.5


def test_the_grouped_step_is_never_divided_into_a_per_expert_latency(monkeypatch):
    """The old implementation reported step_ms / top_k as per_expert_ms. Expert work in a
    grouped call executes concurrently, so that quotient is an amortized throughput-like
    quantity the hardware never produced."""
    _mock_kernels(monkeypatch, [0.5, 2.0])
    result = micro.measure_single_expert_nvfp4(gpu=FAKE_UUID, top_k=8, repetitions=4, warmup=1)
    grouped = result["grouped_topk"]
    assert grouped["top_k"] == 8
    assert grouped["step_ms"] == 2.0
    assert "grouped" in grouped["diagnostic_kind"]
    # 2.0 / 8 == 0.25 appears nowhere, under any name
    assert not any(k.startswith("per_expert_ms") and not k.endswith("absent") for k in grouped)
    assert 0.25 not in [v for v in grouped.values() if isinstance(v, float)]
    assert "not single-expert latency" in grouped["per_expert_ms_deliberately_absent"]


def test_the_grouped_diagnostic_can_be_skipped(monkeypatch):
    _mock_kernels(monkeypatch, [0.5])
    result = micro.measure_single_expert_nvfp4(
        gpu=FAKE_UUID, top_k=8, repetitions=4, warmup=1, include_grouped=False
    )
    assert result["single_expert"]["latency_ms"] == 0.5
    assert result["grouped_topk"]["value"] is None


def test_the_microbenchmark_binds_to_the_requested_gpu(monkeypatch):
    """It must not construct torch.device("cuda") and hope: on a multi-GPU box that
    benchmarks device 0 while labelling the result as another card."""
    seen = {}
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())

    # This test is about binding, not kernel importability. Force the first production-kernel
    # import to fail deterministically. On a fully provisioned CUDA host, letting the real
    # modules import while fake torch is installed can cache a production module bound to the
    # fake and poison unrelated tests later in the same pytest process.
    fused_before = sys.modules.get("freetoken.moe.fused_nvfp4")
    monkeypatch.setitem(sys.modules, "freetoken.benchmark.perf", None)

    def fake_bind(selector):
        seen["selector"] = selector
        return _FakeDevice(1), {"index": 1, "name": "RTX 3060", "uuid": FAKE_UUID}, {
            "matches": True, "bound_cuda_index": 1, "bound_uuid": FAKE_UUID,
        }

    monkeypatch.setattr(micro, "bind_torch_device", fake_bind)
    # The perf import is deliberately unavailable, so the run stops right after binding --
    # exactly the step under test -- without importing a real kernel module under fake torch.
    result = micro.measure_single_expert_nvfp4(gpu=FAKE_UUID)
    assert seen["selector"] == FAKE_UUID
    assert "not importable" in result["unavailable"]
    assert sys.modules.get("freetoken.moe.fused_nvfp4") is fused_before


def test_the_microbenchmark_refuses_rather_than_misattributing_the_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())

    def refuse(selector):
        raise gpu_mod.GpuBindError(
            f"--gpu {selector!r} resolved to {FAKE_UUID}, but this process bound {OTHER_UUID}"
        )

    monkeypatch.setattr(micro, "bind_torch_device", refuse)
    result = micro.measure_single_expert_nvfp4(gpu=FAKE_UUID)
    assert result["value"] is None
    assert OTHER_UUID in result["unavailable"]


def test_no_cuda_is_an_explicit_null_with_a_reason(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False))
    result = micro.measure_single_expert_nvfp4(gpu=FAKE_UUID)
    assert result == {"value": None, "unavailable": "no CUDA device available"}


def test_bad_geometry_is_rejected_before_any_binding(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())

    def explode(selector):
        raise AssertionError("binding must not be attempted for an invalid geometry")

    monkeypatch.setattr(micro, "bind_torch_device", explode)
    result = micro.measure_single_expert_nvfp4(hidden=2047, gpu=FAKE_UUID)
    assert "multiple of 16" in result["unavailable"]


# --- device (VRAM) memory bandwidth ---------------------------------------------------------------

def test_the_method_states_its_byte_accounting():
    assert "2 x buffer_bytes" in dbw.METHOD
    assert "one read + one write" in dbw.METHOD


def test_the_default_working_set_is_far_beyond_any_consumer_l2():
    # The RTX 3060's L2 is 3 MiB; a cache-resident buffer would report an L2 number and
    # call it VRAM bandwidth.
    assert dbw.DEFAULT_BUFFER_BYTES >= 256 << 20


def test_no_cuda_is_an_explicit_null(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False))
    assert dbw.measure_device_memory_bandwidth(gpu=FAKE_UUID)["value"] is None


def test_a_bind_failure_is_an_explicit_null(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())

    def refuse(selector):
        raise gpu_mod.GpuBindError("bound the wrong card")

    monkeypatch.setattr(dbw, "bind_torch_device", refuse)
    result = dbw.measure_device_memory_bandwidth(gpu=FAKE_UUID)
    assert result["unavailable"] == "bound the wrong card"


class _FakeTensor:
    """Enough tensor surface for the harness's own plumbing; the kernels are mocked out."""

    def __init__(self, elements=1):
        self._elements = elements

    def numel(self):
        return self._elements

    def element_size(self):
        return 2

    def fill_(self, value):
        return self

    def copy_(self, other):
        return self

    def view(self, *shape):
        return self

    def __truediv__(self, other):
        return self

    def __mod__(self, other):
        return self


def test_a_measured_run_is_labelled_and_keeps_every_repetition(monkeypatch):
    torch = _fake_torch()
    torch.float16 = "float16"
    torch.empty = lambda elements, dtype=None, device=None: _FakeTensor(elements)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        dbw, "bind_torch_device",
        lambda selector: (
            _FakeDevice(0),
            {"index": 0, "name": "NVIDIA GeForce RTX 3060", "uuid": FAKE_UUID,
             "total_bytes": 12 << 30},
            {"matches": True, "bound_uuid": FAKE_UUID},
        ),
    )
    result = dbw.measure_device_memory_bandwidth(
        gpu=FAKE_UUID, buffer_bytes=1 << 20, repetitions=6, warmup=2
    )
    assert result["label"] == "MEASURED"
    assert result["diagnostic_only"] is True
    assert "never extrapolated into an end-to-end inference claim" in result["diagnostic_note"]
    assert result["device"]["uuid"] == FAKE_UUID
    # raw repetitions, not one opaque number
    assert len(result["per_repetition_ms"]) == 6
    assert len(result["per_repetition_gbs_read_write"]) == 6
    assert result["summary_gbs_read_write"]["n"] == 6
    # both byte conventions are reported, named, so neither has to be reverse-engineered
    assert result["per_repetition_gbs_read_only"][0] == pytest.approx(
        result["per_repetition_gbs_read_write"][0] / 2
    )
    assert result["geometry"]["bytes_moved_per_repetition"] == 2 * (1 << 20)
    assert result["geometry"]["byte_accounting"] == "read + write (2 x buffer_bytes)"


def test_non_positive_timings_yield_no_bandwidth(monkeypatch):
    torch = _fake_torch()
    torch.float16 = "float16"
    torch.empty = lambda elements, dtype=None, device=None: _FakeTensor(elements)
    torch.cuda.Event = lambda enable_timing=False: types.SimpleNamespace(
        record=lambda: None, synchronize=lambda: None, elapsed_time=lambda other: 0.0
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        dbw, "bind_torch_device",
        lambda selector: (_FakeDevice(0), {"uuid": FAKE_UUID}, {"matches": True}),
    )
    result = dbw.measure_device_memory_bandwidth(
        gpu=FAKE_UUID, buffer_bytes=1 << 20, repetitions=3, warmup=1
    )
    assert result["value"] is None
    assert "non-positive" in result["unavailable"]
