"""Machine-readable snapshot of the engine's **resolved** runtime configuration.

Why this exists: `--moe-backend auto`, `--moe-cache-auto`, `--nvfp4-backend auto` and
`--moe-cpu-layers` (unset) are all resolved inside the engine, and several of them can
resolve to something the flag text does not name -- or be inert on the path that actually
executes. A benchmark record that quotes the flags is therefore not a configuration
record. This module reads the resolution *back off the live engine* (the loaded bank
layout, the constructed cache, the resolved layer sets) instead of re-deriving it from
policy, so the report cannot disagree with what the process is running.

It is built once on the readiness path and shipped to the frontend with the rest of the
readiness metadata; `GET /v1/instrumentation` serves it. Every getter is best-effort:
anything that cannot be read becomes an explicit ``{"value": None, "unavailable": "<why>"}``
rather than a missing key, because a provenance record with silent holes is worse than one
that says what it could not see.
"""

from __future__ import annotations

from typing import Any, Dict

from freetoken.moe import is_offload_moe_backend
from freetoken.moe.offload_cache import MARLIN_MAX_CACHE_SIZE
from freetoken.utils import init_logger

logger = init_logger(__name__)

SCHEMA = "freetoken.runtime_report/1"

# quant_format tag written by the NVFP4 bank loader -> the expert-GEMM backend that tag
# implies. ``expert_banks._nvfp4_banks`` writes "nvfp4" for both the forced-Triton layout
# and the native (CPU-readable) ModelOpt layout; which of the two it is depends on the
# cache's decode_target, so the mapping below is only consulted for the repacked tags.
_QUANT_FORMAT_BACKEND = {"nvfp4_marlin": "marlin", "nvfp4_b12x": "b12x"}


def unavailable(reason: str) -> Dict[str, Any]:
    """The explicit-null form. Never omit a field; say why it is missing."""
    return {"value": None, "unavailable": reason}


def _resolved_nvfp4(cache, requested: str) -> Dict[str, Any]:
    """Resolved NVFP4 expert-GEMM backend, and whether ``--nvfp4-backend`` was inert.

    Read off the loaded banks, not off ``select_nvfp4_backend``: the loader skips backend
    selection entirely when the banks are loaded for CPU-side decode (``decode_target ==
    "cpu"``, which covers ``--moe-backend cpu``/``hybrid`` *and* an offload run whose MoE
    layers were locked onto the CPU executor). In that case the flag never reached a
    kernel choice, and this says so rather than echoing the flag back as if it had.
    """
    if cache is None:
        return {
            "requested": requested,
            "resolved": None,
            "inert": None,
            "unavailable": "no offload MoE cache (non-MoE model, or a resident-expert backend)",
        }
    quant_format = str(getattr(cache, "quant_format", "") or "")
    decode_target = str(getattr(cache, "decode_target", "") or "")
    if not quant_format.startswith("nvfp4"):
        return {
            "requested": requested,
            "resolved": None,
            "inert": True,
            "expert_quant_format": quant_format or None,
            "note": "checkpoint experts are not NVFP4; the flag selects nothing",
        }
    # decode_target cpu/hybrid => banks were loaded in the native ModelOpt layout and
    # select_nvfp4_backend was never called (expert_banks._nvfp4_banks). The executing
    # GPU expert kernels are the native-layout Triton ones either way.
    native_cpu_layout = decode_target in ("cpu", "hybrid")
    if native_cpu_layout:
        return {
            "requested": requested,
            "resolved": "not selected - native nvfp4 layout, Triton kernels",
            "inert": True,
            "expert_quant_format": quant_format,
            "decode_target": decode_target,
            "note": (
                "banks loaded with decode_target=cpu, so the loader kept the native ModelOpt layout and skipped backend selection"
            ),
        }
    resolved = _QUANT_FORMAT_BACKEND.get(quant_format, "triton")
    return {
        "requested": requested,
        "resolved": resolved,
        "inert": False,
        "expert_quant_format": quant_format,
        "decode_target": decode_target,
    }


def _marlin_cap(cache, plan: Dict[str, Any] | None) -> Dict[str, Any]:
    """Whether the 992-slot Marlin cache cap applies, and whether it actually bound.

    "Applies" is a property of the loaded bank layout. "Bound" is only knowable for an
    auto-sized cache, where the engine records the same plan re-solved without the cap
    (see ``Engine._resolve_auto_moe_cache_size``); with an explicit ``--moe-cache-size``
    the user chose the number, so the cap can only have *rejected* it, never silently
    clamped it -- and the run would have failed instead of reaching this report.
    """
    quant_format = (
        str(getattr(cache, "quant_format", "") or "") if cache is not None else ""
    )
    applicable = quant_format == "nvfp4_marlin"
    out: Dict[str, Any] = {
        "limit_slots": MARLIN_MAX_CACHE_SIZE,
        "applicable": applicable,
    }
    if not applicable:
        out["bound"] = False
        out["reason"] = (
            f"expert bank layout is {quant_format or 'unknown'}, not nvfp4_marlin"
        )
        return out
    if plan is None or plan.get("uncapped_slots") is None:
        out["bound"] = None
        out["unavailable"] = (
            "cache size was set explicitly (--moe-cache-size/--moe-cache-rate); the cap rejects an over-large size rather than clamping it"
        )
        return out
    resolved = int(plan.get("resolved_slots") or 0)
    uncapped = int(plan["uncapped_slots"])
    out["bound"] = uncapped > resolved
    out["resolved_slots"] = resolved
    out["slots_without_cap"] = uncapped
    return out


def _cache_block(engine, cache) -> Dict[str, Any]:
    config = engine.config
    plan = getattr(engine, "moe_cache_auto_plan", None)
    block: Dict[str, Any] = {
        "policy_requested": (
            "auto"
            if config.moe_cache_auto
            else ("rate" if config.moe_cache_rate is not None else "size")
        ),
        "moe_cache_rate_flag": config.moe_cache_rate,
        "kv_reserve_tokens": config.kv_reserve_tokens,
        "moe_cache_policy": config.moe_cache_policy,
        "resolved_slots": int(getattr(cache, "cache_size", 0) or 0)
        if cache is not None
        else None,
        "auto_plan": plan,
    }
    if cache is None:
        block["resolved_slots"] = None
        block["unavailable_resolved_slots"] = (
            "no offload MoE cache on this configuration"
        )
        block["resolved_bytes"] = None
        return block
    try:
        banks = getattr(cache, "bank_caches", None) or {}
        per_slot = int(sum(t[0].numel() * t.element_size() for t in banks.values()))
        block["bytes_per_slot"] = per_slot or None
        block["resolved_bytes"] = per_slot * int(cache.cache_size) if per_slot else None
    except Exception as e:  # noqa: BLE001 -- provenance must never break readiness
        block["bytes_per_slot"] = None
        block["resolved_bytes"] = None
        block["unavailable_resolved_bytes"] = f"could not measure bank caches: {e!r}"
    mc = config.model_config
    total_experts = int(getattr(mc, "num_moe_layers", 0)) * int(
        getattr(mc, "num_experts", 0)
    )
    block["total_expert_slots"] = total_experts or None
    if total_experts and cache.cache_size:
        # Placement fact only. Coverage is not hit rate and is not throughput
        # (InferSwarm phase-1 criteria section 13); it is reported, never cited.
        block["resident_expert_coverage"] = round(
            int(cache.cache_size) / total_experts, 6
        )
    else:
        block["resident_expert_coverage"] = None
    return block


def _moe_block(engine, cache) -> Dict[str, Any]:
    config = engine.config
    resolution = getattr(engine, "moe_resolution", None) or {}
    requested = resolution.get("moe_backend_requested", config.moe_backend)
    block: Dict[str, Any] = {
        "backend_requested": requested,
        # config.moe_backend is mutated in place by _adjust_config when `auto` resolves,
        # so the live value IS the resolved one.
        "backend_resolved": config.moe_backend,
        "decode_target": getattr(cache, "decode_target", None)
        if cache is not None
        else None,
        "cpu_threads": config.moe_cpu_threads,
        "cpu_layers_flag": config.moe_cpu_layers,
        "hybrid_max_fetch_flag": config.moe_hybrid_max_fetch,
        "hybrid_max_fetch_resolved": (
            getattr(cache, "hybrid_max_fetch", None) if cache is not None else None
        ),
        # The bandwidth-matched split `--moe-hybrid-max-fetch auto` resolved to, read off
        # the constructed cache. 0.0 means the fixed-cap fallback applies (no usable
        # `ft bench bw` profile), which is a different configuration from a benched split
        # and must not be reported as one.
        "hybrid_fetch_fraction_resolved": (
            getattr(cache, "hybrid_fetch_fraction", None) if cache is not None else None
        ),
        "prefill_overlap_resolved": config.moe_prefill_overlap,
        "prefill_hit_d2d": config.moe_prefill_hit_d2d,
        "collect_stats": config.moe_collect_stats,
        "trace_max_steps": int(getattr(config, "moe_trace_max_steps", 0) or 0),
        "trace_enabled": bool(getattr(config, "moe_trace_max_steps", 0) or 0),
        "layer_timing_max_steps": int(
            getattr(config, "moe_layer_timing_max_steps", 0) or 0
        ),
        "layer_timing_role": getattr(config, "moe_layer_timing_role", "unspecified"),
    }
    block.update(
        {
            "cpu_layers_resolved": resolution.get("cpu_layer_ids"),
            "auto_cpu_layers_fired": resolution.get("auto_cpu_layers_fired"),
            "auto_cpu_layers_ids": resolution.get("auto_cpu_layer_ids"),
            "split_residency": resolution.get("split_residency"),
        }
    )
    if not resolution:
        block["unavailable_cpu_layers_resolved"] = (
            "engine did not record a MoE resolution (no offload-family backend)"
        )
    return block


def _model_block(engine) -> Dict[str, Any]:
    config = engine.config
    mc = config.model_config
    return {
        "model_path": None,  # deliberately withheld: a local path is host-specific
        "expert_quant": getattr(mc, "expert_quant", None),
        "moe_weight_format": getattr(mc, "moe_weight_format", None),
        "num_moe_layers": getattr(mc, "num_moe_layers", None),
        "num_experts": getattr(mc, "num_experts", None),
        "top_k": getattr(mc, "num_experts_per_tok", None),
        "hidden_act": getattr(mc, "hidden_act", None),
        "is_moe": bool(getattr(mc, "is_moe", False)),
        "dtype": str(config.dtype),
    }


def _runtime_block(engine) -> Dict[str, Any]:
    config = engine.config
    block: Dict[str, Any] = {
        "attention_backend": config.attention_backend,
        "page_size": config.page_size,
        "memory_ratio": config.memory_ratio,
        "max_running_req": config.max_running_req,
        "max_seq_len": config.max_seq_len,
        "cuda_graph_max_bs": config.cuda_graph_max_bs,
        "cuda_graph_bs": list(config.cuda_graph_bs or []) or None,
        "num_pages": int(getattr(engine, "num_pages", 0) or 0) or None,
        "expert_load": config.expert_load,
        # Both live on SchedulerConfig, not EngineConfig, and both are held-constant values
        # the criteria (section 2.3) require as RESOLVED values rather than as flag text:
        # --max-prefill-length is the chunked-prefill chunk cap (dest max_extend_tokens),
        # and --cache-type is rewritten by _adjust_config for SWA/GDN models, so the flag
        # the user passed is frequently not what the cache manager was built with.
        "max_prefill_length_resolved": getattr(config, "max_extend_tokens", None),
        "cache_type_resolved": getattr(config, "cache_type", None),
    }
    # Whether capture ACTUALLY happened (criteria section 2.3), not whether it was asked for:
    # GraphRunner.graph_map holds one entry per captured batch size and stays empty when
    # capture is disabled or bailed out.
    graph_map = getattr(getattr(engine, "graph_runner", None), "graph_map", None)
    if graph_map is None:
        block["cuda_graph_captured_bs"] = None
        block["cuda_graph_capture_happened"] = None
        block["unavailable_cuda_graph_captured_bs"] = (
            "graph runner has no graph_map (capture never reached init_capture_graph)"
        )
    else:
        block["cuda_graph_captured_bs"] = sorted(int(b) for b in graph_map)
        block["cuda_graph_capture_happened"] = bool(graph_map)
    return block


def build_runtime_report(engine) -> Dict[str, Any]:
    """Resolved-configuration snapshot for ``engine``. Never raises."""
    try:
        config = engine.config
        cache = getattr(engine, "moe_offload_cache", None)
        from freetoken.moe.inferswarm_secondary import absent_secondary_device_report
        from freetoken.moe.inferswarm_resident_bank import absent_resident_bank_report
        from freetoken.moe.inferswarm_remote_decode import (
            absent_remote_decode_configuration_report,
        )
        from freetoken.moe.inferswarm_d2_graph_remote import (
            absent_d2_graph_remote_report,
        )
        from freetoken.moe.inferswarm_d3_graph_multiworker import absent_d3_graph_multiworker_report
        from freetoken.moe.inferswarm_d5_compact_routes import absent_d5_compact_routes_report
        from freetoken.moe.layer_timing import MOE_LAYER_TIMING_SCHEMA

        secondary = getattr(engine, "inferswarm_secondary_device", None)
        resident = getattr(engine, "inferswarm_resident_bank", None)
        remote_decode = getattr(engine, "inferswarm_remote_decode", None)
        d2_remote = getattr(engine, "inferswarm_d2_graph_remote", None)
        d3_remote = getattr(engine, "inferswarm_d3_graph_multiworker", None)
        d3_banks = getattr(engine, "inferswarm_d3_resident_banks", None)
        d5_compact = getattr(engine, "inferswarm_d5_compact_routes", None)
        layer_timing = getattr(engine, "moe_layer_timing", None)
        report: Dict[str, Any] = {
            "schema": SCHEMA,
            "offload_family": is_offload_moe_backend(config.moe_backend),
            "model": _model_block(engine),
            "moe": _moe_block(engine, cache),
            "nvfp4": _resolved_nvfp4(cache, config.nvfp4_backend),
            "cache": _cache_block(engine, cache),
            "marlin_cache_cap": _marlin_cap(
                cache, getattr(engine, "moe_cache_auto_plan", None)
            ),
            "runtime": _runtime_block(engine),
            # Downstream experimental schema: P1 discovery/provenance only.  This is not an
            # upstream-stable FreeToken API and does not imply remote model execution.
            "inferswarm_secondary_device": (
                secondary.as_dict()
                if secondary is not None
                else absent_secondary_device_report()
            ),
            # P2 startup residency/storage provenance. Runtime execution truth is kept in
            # the separate P3 block so static storage cannot contradict live counters.
            "inferswarm_resident_bank": (
                resident.report.as_dict()
                if resident is not None
                else absent_resident_bank_report()
            ),
            "inferswarm_remote_decode": (
                remote_decode.configuration_report()
                if remote_decode is not None
                else absent_remote_decode_configuration_report()
            ),
            "inferswarm_d2_graph_remote": (
                d2_remote.configuration_report()
                if d2_remote is not None
                else absent_d2_graph_remote_report()
            ),
            "inferswarm_d3_graph_multiworker": (
                d3_remote.configuration_report() if d3_remote is not None else absent_d3_graph_multiworker_report()
            ),
            "inferswarm_d5_resident_loader": (
                d3_banks.loader_profile if d3_banks is not None else None
            ),
            "inferswarm_d5_compact_routes": (
                d5_compact.configuration_report() if d5_compact is not None else absent_d5_compact_routes_report()
            ),
            "moe_layer_timing": (
                layer_timing.configuration_report()
                if layer_timing is not None
                else {
                    "schema": MOE_LAYER_TIMING_SCHEMA,
                    "enabled": False,
                    "capacity_steps": 0,
                }
            ),
        }
        return report
    except Exception as e:  # noqa: BLE001 -- runs on the readiness path; never block startup
        logger.warning(f"could not build the runtime report: {e!r}")
        return {"schema": SCHEMA, "error": repr(e)}
