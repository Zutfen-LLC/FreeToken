"""The resolved-configuration report (``/v1/instrumentation``'s runtime_config).

CPU-only: the report is built from a fake engine's attributes, which is exactly what it
does in production -- it *reads back* what the engine resolved rather than re-deriving it,
so a fake engine with the same shape is a faithful subject.

What matters here is that the report never reports a flag as if it had chosen something.
The NVFP4 cases below are the source's own three resolution paths:
``expert_banks._nvfp4_banks`` skips ``select_nvfp4_backend`` entirely when banks load with
``decode_target="cpu"``, which covers ``--moe-backend cpu``/``hybrid`` AND an offload run
whose layers ``_auto_cpu_layers`` locked onto the CPU executor.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.engine.runtime_report import build_runtime_report, unavailable
from freetoken.moe.offload_cache import MARLIN_MAX_CACHE_SIZE


def _model_config(**kw):
    base = dict(
        expert_quant="nvfp4", moe_weight_format=None, num_moe_layers=40, num_experts=256,
        num_experts_per_tok=8, hidden_act="silu", is_moe=True,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _config(**kw):
    base = dict(
        moe_backend="offload", nvfp4_backend="auto", moe_cache_size=992, moe_cache_rate=None,
        moe_cache_auto=True, kv_reserve_tokens=8192, moe_cache_policy="lru",
        moe_cpu_threads=0, moe_cpu_layers=None, moe_hybrid_max_fetch=-1,
        moe_prefill_overlap=True, moe_prefill_hit_d2d=False, moe_collect_stats=False,
        attention_backend="auto", page_size=1, memory_ratio=0.9, max_running_req=1,
        max_seq_len=8192, cuda_graph_max_bs=1, cuda_graph_bs=[1], expert_load="auto",
        dtype="torch.bfloat16", model_config=_model_config(),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _cache(quant_format="nvfp4", decode_target="gpu", cache_size=992):
    return SimpleNamespace(
        quant_format=quant_format, decode_target=decode_target, cache_size=cache_size,
        bank_caches={}, hybrid_max_fetch=1,
    )


_DEFAULT = object()


def _engine(config=None, cache=_DEFAULT, **kw):
    base = dict(
        config=config or _config(),
        moe_offload_cache=_cache() if cache is _DEFAULT else cache,
        moe_resolution={
            "moe_backend_requested": "auto", "cpu_layer_ids": [],
            "auto_cpu_layers_fired": False, "auto_cpu_layer_ids": [], "split_residency": False,
        },
        moe_cache_auto_plan=None,
        num_pages=4096,
        graph_runner=SimpleNamespace(graph_map={1: object()}),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_report_separates_requested_from_resolved_moe_backend():
    engine = _engine(config=_config(moe_backend="offload"))  # already resolved in place
    report = build_runtime_report(engine)
    assert report["moe"]["backend_requested"] == "auto"
    assert report["moe"]["backend_resolved"] == "offload"


def test_triton_resolution_is_read_from_the_bank_layout():
    report = build_runtime_report(_engine(cache=_cache("nvfp4", "gpu")))
    assert report["nvfp4"] == {
        "requested": "auto", "resolved": "triton", "inert": False,
        "expert_quant_format": "nvfp4", "decode_target": "gpu",
    }


@pytest.mark.parametrize("quant_format,expected", [("nvfp4_marlin", "marlin"), ("nvfp4_b12x", "b12x")])
def test_repacked_layouts_name_their_backend(quant_format, expected):
    report = build_runtime_report(_engine(cache=_cache(quant_format, "gpu")))
    assert report["nvfp4"]["resolved"] == expected
    assert report["nvfp4"]["inert"] is False


@pytest.mark.parametrize("decode_target", ["cpu", "hybrid"])
def test_cpu_side_decode_records_the_flag_as_inert(decode_target):
    """--moe-backend cpu/hybrid load banks with decode_target=cpu, so the loader keeps the
    native ModelOpt layout and never calls select_nvfp4_backend."""
    report = build_runtime_report(
        _engine(config=_config(nvfp4_backend="triton"), cache=_cache("nvfp4", decode_target))
    )
    assert report["nvfp4"]["inert"] is True
    assert report["nvfp4"]["resolved"].startswith("not selected")
    assert "native nvfp4 layout" in report["nvfp4"]["resolved"]


def test_auto_cpu_layers_makes_the_flag_inert_on_an_offload_run():
    """_auto_cpu_layers locking layers flips the process to the native bank layout, which
    is the third resolution path criteria section 2.1 requires recording."""
    engine = _engine(
        config=_config(moe_backend="offload", nvfp4_backend="auto"),
        cache=_cache("nvfp4", "cpu"),
        moe_resolution={
            "moe_backend_requested": "offload", "cpu_layer_ids": [0, 1, 38, 39],
            "auto_cpu_layers_fired": True, "auto_cpu_layer_ids": [0, 1, 38, 39],
            "split_residency": True,
        },
    )
    report = build_runtime_report(engine)
    assert report["moe"]["auto_cpu_layers_fired"] is True
    assert report["moe"]["cpu_layers_resolved"] == [0, 1, 38, 39]
    assert report["nvfp4"]["inert"] is True


def test_marlin_cap_applicability_and_binding_come_from_the_plan():
    plan = {"resolved_slots": MARLIN_MAX_CACHE_SIZE, "uncapped_slots": 3000}
    report = build_runtime_report(
        _engine(cache=_cache("nvfp4_marlin", "gpu", MARLIN_MAX_CACHE_SIZE),
                moe_cache_auto_plan=plan)
    )
    cap = report["marlin_cache_cap"]
    assert cap["limit_slots"] == 992
    assert cap["applicable"] is True and cap["bound"] is True
    assert cap["slots_without_cap"] == 3000


def test_marlin_cap_applicable_but_not_binding_when_vram_ran_out_first():
    plan = {"resolved_slots": 400, "uncapped_slots": 400}
    report = build_runtime_report(
        _engine(cache=_cache("nvfp4_marlin", "gpu", 400), moe_cache_auto_plan=plan)
    )
    assert report["marlin_cache_cap"]["applicable"] is True
    assert report["marlin_cache_cap"]["bound"] is False


def test_marlin_cap_not_applicable_on_a_triton_layout():
    report = build_runtime_report(_engine(cache=_cache("nvfp4", "gpu")))
    cap = report["marlin_cache_cap"]
    assert cap["applicable"] is False and cap["bound"] is False
    assert "nvfp4_marlin" in cap["reason"]


def test_marlin_cap_binding_is_unknowable_for_an_explicit_cache_size():
    """With --moe-cache-size the user chose the number, so the cap can only have rejected
    it -- and the server would not have started. Say "unknown", never guess."""
    report = build_runtime_report(
        _engine(cache=_cache("nvfp4_marlin", "gpu", 500), moe_cache_auto_plan=None)
    )
    assert report["marlin_cache_cap"]["bound"] is None
    assert "explicitly" in report["marlin_cache_cap"]["unavailable"]


def test_resident_expert_coverage_is_reported_as_a_placement_fact():
    report = build_runtime_report(_engine(cache=_cache("nvfp4", "gpu", 1024)))
    # 1024 / (40 * 256)
    assert report["cache"]["resident_expert_coverage"] == pytest.approx(0.1)
    assert report["cache"]["total_expert_slots"] == 10240


def test_capture_happened_is_reported_not_inferred_from_the_flag():
    captured = build_runtime_report(_engine())
    assert captured["runtime"]["cuda_graph_capture_happened"] is True
    assert captured["runtime"]["cuda_graph_captured_bs"] == [1]
    none = build_runtime_report(_engine(graph_runner=SimpleNamespace(graph_map={})))
    assert none["runtime"]["cuda_graph_capture_happened"] is False


def test_missing_cache_is_an_explicit_null_with_a_reason():
    report = build_runtime_report(_engine(cache=None))
    assert report["nvfp4"]["resolved"] is None
    assert report["nvfp4"]["unavailable"]
    assert report["cache"]["resolved_slots"] is None
    assert report["cache"]["unavailable_resolved_slots"]


def test_the_local_model_path_is_never_included():
    """A host-specific absolute path must not ride along in a record that may be copied
    into a committed result directory."""
    report = build_runtime_report(_engine())
    assert report["model"]["model_path"] is None


def test_a_broken_engine_yields_an_error_report_rather_than_raising():
    """This runs on the readiness path; it must never keep a model from serving."""
    report = build_runtime_report(object())
    assert report["schema"].startswith("freetoken.runtime_report/")
    assert "error" in report


def test_unavailable_helper_shape():
    assert unavailable("because") == {"value": None, "unavailable": "because"}
