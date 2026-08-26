"""Shared campaign fakes for the Phase-0 harness tests.

They produce a *fully valid* campaign: proven GPU, fresh and matching ``ft bench bw``
profile, complete resolved configuration, attributable prefill. Every test that exercises a
failure mode starts from that and breaks exactly one thing, which is what keeps "this is
what makes a campaign invalid" readable -- and means a newly required field is covered by
the existing parametrized tests rather than needing a new hand-built config.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict

SHA40 = "0" * 40
FAKE_UUID = "GPU-11111111-2222-3333-4444-555555555555"

# Every path in runner.REQUIRED_RUNTIME_FIELDS, populated. Tests delete or null one field at
# a time rather than hand-building a config, so a field added to the required list makes the
# "missing field invalidates" test cover it automatically.
FULL_RUNTIME_CONFIG: Dict[str, Any] = {
    "schema": "freetoken.runtime_report/1",
    "model": {"expert_quant": "nvfp4", "num_experts": 256, "num_moe_layers": 40},
    "moe": {
        "backend_requested": "offload",
        "backend_resolved": "offload",
        "cpu_threads": 12,
        "cpu_layers_flag": "0",
        "cpu_layers_resolved": [],
        "auto_cpu_layers_fired": False,
        "hybrid_max_fetch_resolved": 1,
        "hybrid_fetch_fraction_resolved": 0.0,
    },
    "nvfp4": {"requested": "auto", "resolved": "triton", "inert": False},
    "cache": {
        "policy_requested": "auto",
        "resolved_slots": 900,
        "kv_reserve_tokens": 8192,
        "resolved_bytes": 1 << 32,
    },
    "marlin_cache_cap": {"applicable": False, "bound": False},
    "runtime": {
        "attention_backend": "auto",
        "page_size": 1,
        "memory_ratio": 0.9,
        "max_running_req": 1,
        "max_seq_len": 40960,
        "num_pages": 20000,
        "cuda_graph_max_bs": 1,
        "cuda_graph_capture_happened": True,
        "max_prefill_length_resolved": 8192,
        "cache_type_resolved": "radix",
    },
}


def runtime_config(**overrides: Dict[str, Any]) -> Dict[str, Any]:
    """A complete resolved configuration, with per-block overrides merged one level deep."""
    config = copy.deepcopy(FULL_RUNTIME_CONFIG)
    for block, values in overrides.items():
        if values is None:
            config.pop(block, None)
        else:
            config.setdefault(block, {}).update(values)
    return config


def instrumentation_doc(
    config: Dict[str, Any] | None = None, prefill: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    return {
        "schema": "freetoken.instrumentation/1",
        "runtime_config": runtime_config() if config is None else config,
        "prefill": prefill or {"enabled": True, "observed": 0, "records": []},
    }


def write_manifest(tmp_path, classes=("W1", "W2", "W3", "W4"), canonical=True, sampling=None):
    """A manifest on disk, hash-pinned exactly as ``load_manifest`` requires."""
    from inferswarm_phase0.manifest import CLASS_SPECS, sha256_text

    entries = []
    for c in classes:
        content = f"prompt for {c}"
        entries.append({
            "class_id": c,
            "content": content,
            "content_sha256": sha256_text(content),
            "output_tokens": CLASS_SPECS[c].output_tokens,
            "ignore_eos": True,
            "sampling": dict(sampling or {"temperature": 0.0, "top_p": 1.0, "top_k": -1}),
            "seed": None,
            "chat_template_kwargs": {},
            "role": "user",
        })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema": "inferswarm.phase0.workload-manifest/1",
        "manifest_id": "test-manifest",
        "canonical": canonical,
        "workloads": entries,
    }))
    return path




def good_bench_bw_record(gpu_uuid: str, path: str = "/cache/freetoken/benchbw/gpu.json") -> Dict[str, Any]:
    """A successful ``ft bench bw`` run whose profile is readable and names this GPU."""
    contents = {
        "gpu": {"index": 0, "name": "NVIDIA GeForce RTX 3060", "uuid": gpu_uuid},
        "dtypes": {"nvfp4": "hybrid"},
        "dtype_kernels": {
            "nvfp4": {
                "cpu_moe_gbs": 30.0,
                "pcie_gather_gbs": 10.0,
                "cpu_moe_overlap_gbs": 20.0,
                "pcie_gather_overlap_gbs": 10.0,
                "recommended": "hybrid",
            }
        },
        "ceilings": {"cpu_stream_read_gbs": 40.0, "pcie_linear_h2d_gbs": 12.0},
    }
    return {
        "command": ["python", "-m", "freetoken.cli", "bench", "bw", "--dtype", "nvfp4",
                    "--gpu", gpu_uuid],
        "dtype": "nvfp4",
        "gpu_selector_used": gpu_uuid,
        "gpu_resolved_uuid": gpu_uuid,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:05:00+00:00",
        "ok": True,
        "returncode": 0,
        "stdout_tail": f"FTBENCH_OUT {path}\n",
        "stderr_tail": "",
        "profile_path_source": "FTBENCH_OUT",
        "profile": {
            "path": path,
            "sha256": "f" * 64,
            "bytes": len(json.dumps(contents)),
            "contents": contents,
            "profile_gpu": contents["gpu"],
            "gpu_matches": True,
            "nvfp4_calibration": {
                "dtype": "nvfp4",
                "usable": True,
                "backend_recommendation": "hybrid",
                "hybrid_fetch_fraction": 1.0 / 3.0,
            },
        },
    }
