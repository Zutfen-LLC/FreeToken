"""Shared fakes for the Phase-1 campaign tests.

Mirrors the Phase-0 ``fakes.py`` discipline: a fully valid canonical campaign — proven
GPUs, canonical manifest/placement identities (pinned to synthetic fixtures whose
digests stand in for the frozen ones), complete runtime reports, attributable
prefill. A test that exercises a failure mode breaks exactly one thing.

The frozen canonical manifest SHA and placement SHA are content-addressed artifacts
that live in the InferSwarm repository; the tests pin them to locally-built fixtures
by patching the module constants, so the mismatch-detection logic is exercised
against digests the tests control.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from inferswarm_phase1.campaign_arms import (
    GPU0_UUID,
    GPU1_UUID,
)

SHA40 = "4" * 40
INFERSWARM_SHA40 = "5" * 40

# Runtime-report blocks mirroring the engine's explicit-absence shapes.
ABSENT_SECONDARY = {
    "configured": False,
    "requested_secondary_spec": None,
    "validation_passed": None,
    "primary": None,
    "secondary": None,
    "peer_access": None,
    "transport_classification": None,
}
ABSENT_RESIDENT = {
    "placement_configured": False,
    "resident_bank_loaded": False,
    "resident_slots": 0,
}
ABSENT_REMOTE = {
    "enabled": False,
    "execution_mode": None,
    "transport": None,
    "placement_sha256": None,
    "primary": None,
    "secondary": None,
}

GPU1_BYTES_PER_BANK = 9_662_902_272 // 2  # two bank tensors summing to the frozen total

# The placement SHA the candidate runtime fixtures echo; updated by
# ``freeze_frozen_identities`` so runtime validation agrees with the frozen artifact.
CURRENT_PLACEMENT_SHA = "2f62bb84" + "0" * 56


def baseline_runtime_config(**overrides: Any) -> dict[str, Any]:
    """A complete, contract-satisfying B1 runtime report."""
    config: dict[str, Any] = {
        "schema": "freetoken.runtime_report/1",
        "model": {"expert_quant": "nvfp4", "num_experts": 256, "num_moe_layers": 40},
        "moe": {
            "backend_requested": "offload",
            "backend_resolved": "offload",
            "decode_target": "gpu",
            "cpu_threads": 12,
            "cpu_layers_flag": "0",
            "cpu_layers_resolved": [],
            "auto_cpu_layers_fired": False,
        },
        "nvfp4": {"requested": "auto", "resolved": "triton", "inert": False},
        "cache": {
            "policy_requested": "auto",
            "resolved_slots": 3774,
            "kv_reserve_tokens": 17075,
            "resolved_bytes": 1 << 32,
        },
        "runtime": {
            "attention_backend": "auto",
            "page_size": 1,
            "memory_ratio": 0.85,
            "max_running_req": 1,
            "max_seq_len": 40960,
            "num_pages": 17075,
            "cuda_graph_max_bs": 1,
            "cuda_graph_capture_happened": True,
            "max_prefill_length_resolved": 8192,
            "cache_type_resolved": "radix",
        },
        "inferswarm_secondary_device": dict(ABSENT_SECONDARY),
        "inferswarm_resident_bank": dict(ABSENT_RESIDENT),
        "inferswarm_remote_decode": dict(ABSENT_REMOTE),
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


def candidate_runtime_config(**overrides: Any) -> dict[str, Any]:
    """A complete, contract-satisfying candidate runtime report."""
    config: dict[str, Any] = {
        "schema": "freetoken.runtime_report/1",
        "model": {"expert_quant": "nvfp4", "num_experts": 256, "num_moe_layers": 40},
        "moe": {
            "backend_requested": "offload",
            "backend_resolved": "offload",
            "decode_target": "gpu",
            "cpu_threads": 12,
            "cpu_layers_flag": "0",
            "cpu_layers_resolved": [],
            "auto_cpu_layers_fired": False,
        },
        "nvfp4": {"requested": "triton", "resolved": "triton", "inert": False},
        "cache": {
            "policy_requested": "size",
            "resolved_slots": 3774,
            "kv_reserve_tokens": 17075,
            "resolved_bytes": 1 << 32,
        },
        "runtime": {
            "attention_backend": "auto",
            "page_size": 1,
            "memory_ratio": 0.85,
            "max_running_req": 1,
            "max_seq_len": 40960,
            "num_pages": 17075,
            "cuda_graph_max_bs": 0,
            "cuda_graph_capture_happened": False,
            "max_prefill_length_resolved": 8192,
            "cache_type_resolved": "radix",
        },
        "inferswarm_secondary_device": {
            "configured": True,
            "requested_secondary_spec": GPU1_UUID,
            "validation_passed": True,
            "primary": {"uuid": GPU0_UUID, "visible_cuda_ordinal": 0},
            "secondary": {"uuid": GPU1_UUID, "visible_cuda_ordinal": 1},
            "peer_access": {"primary_to_secondary": False, "secondary_to_primary": False},
            "transport_classification": "host_staged",
        },
        "inferswarm_resident_bank": {
            "placement_configured": True,
            "resident_bank_loaded": True,
            "resident_slots": 5442,
            "banks": [
                {"name": "w1", "total_resident_bytes": GPU1_BYTES_PER_BANK},
                {"name": "w2", "total_resident_bytes": GPU1_BYTES_PER_BANK},
            ],
            "artifact": {"sha256": CURRENT_PLACEMENT_SHA},
        },
        "inferswarm_remote_decode": {
            "enabled": True,
            "execution_mode": "overlap",
            "transport": "host_staged",
            "placement_sha256": CURRENT_PLACEMENT_SHA,
            "primary": {"uuid": GPU0_UUID},
            "secondary": {"uuid": GPU1_UUID},
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


def moe_window_snapshot(
    *, prefill_dispatches: int = 0, fallback: int = 0
) -> dict[str, Any]:
    return {
        "boundary": {"operation": "snapshot", "idle": True},
        "inferswarm_resident_bank": {"placement_configured": True},
        "inferswarm_remote_decode": {
            "enabled": True,
            "aggregate": {
                "selected_for_gpu1": 8000,
                "executed_on_gpu1": 8000,
                "executed_on_gpu0": 40000,
                "total_router_selections": 48000,
                "explicit_failure": 0,
                "fallback_elsewhere": fallback,
                "prefill_remote_dispatches": prefill_dispatches,
            },
            "ownership": {
                "successful_selection_arithmetic_exact": fallback == 0,
                "selected_accounted_exactly": fallback == 0,
            },
        },
        "moe_layer_timing": {"enabled": True, "records": []},
    }


def write_canonical_manifest(tmp_path: Path, *, canonical: bool = True) -> Path:
    """A structurally canonical W1-W4 manifest whose digest the tests pin as frozen."""
    from inferswarm_phase0.manifest import CLASS_SPECS, sha256_text

    entries = []
    for c in ("W1", "W2", "W3", "W4"):
        content = f"phase1 campaign fixture prompt for {c}"
        entries.append(
            {
                "class_id": c,
                "content": content,
                "content_sha256": sha256_text(content),
                "output_tokens": CLASS_SPECS[c].output_tokens,
                "ignore_eos": True,
                "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
                "seed": None,
                "chat_template_kwargs": {},
                "role": "user",
            }
        )
    raw = json.dumps(
        {
            "schema": "inferswarm.phase0.workload-manifest/1",
            "manifest_id": "phase1-test-canonical",
            "canonical": canonical,
            "workloads": entries,
        },
        indent=2,
    ).encode("utf-8")
    path = tmp_path / "manifest.json"
    path.write_bytes(raw)
    return path


def write_placement_artifact(tmp_path: Path, *, manifest_sha: str | None = None) -> Path:
    """A structurally valid frozen placement artifact (digest patched in as frozen).

    ``manifest_sha`` pins the same workload-manifest digest the campaign freezes, so
    the cross-artifact consistency check passes exactly as the published artifacts do.
    """
    doc = {
        "schema": "inferswarm.phase1.placement/1",
        "policy_id": "phase1-qwen36-placement-v2",
        "status": "FROZEN_BEFORE_PHASE1_PERFORMANCE",
        "canonical_remote_placement": "coverage_constrained_complement_5442",
        "source": {
            "model_repository": "nvidia/Qwen3.6-35B-A3B-NVFP4",
            "model_revision": SHA40,
            "workload_manifest_sha256": manifest_sha or hashlib.sha256(b"manifest").hexdigest(),
        },
        "geometry": {"num_moe_layers": 40, "num_experts_per_layer": 256},
        "budget": {
            "bytes_per_slot": 1_775_616,
            "remote_budget_bytes": 9_663_676_416,
            "remote_slots": 5442,
            "remote_resident_bytes": 9_662_902_272,
            "gpu0_primary_proxy_slots": 3774,
        },
    }
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def write_prerequisites(tmp_path: Path) -> Path:
    path = tmp_path / "prerequisites.json"
    path.write_text(
        json.dumps(
            {
                "correctness_reference_v2_artifact_sha256": "a" * 64,
                "candidate_c3_artifact_sha256": "b" * 64,
                "p2_p3_p4_requalification_artifact_sha256": "c" * 64,
                "freetoken_runtime_commit": SHA40,
            }
        ),
        encoding="utf-8",
    )
    return path


def freeze_frozen_identities(monkeypatch, tmp_path: Path) -> dict[str, str]:
    """Pin the frozen manifest/placement identities to the local fixtures.

    Production pins content-addressed artifacts published in the InferSwarm
    repository; the tests pin locally-built equivalents so every mismatch path is
    exercisable without the sibling checkout. The frozen production values themselves
    are asserted unchanged in the dedicated identity tests.
    """
    from inferswarm_phase1 import campaign as campaign_mod
    from inferswarm_phase1 import campaign_arms as arms_mod

    manifest = write_canonical_manifest(tmp_path)
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    placement = write_placement_artifact(tmp_path, manifest_sha=manifest_sha)
    placement_sha = hashlib.sha256(placement.read_bytes()).hexdigest()
    monkeypatch.setattr(campaign_mod, "CANONICAL_MANIFEST_ID", "phase1-test-canonical")
    monkeypatch.setattr(campaign_mod, "CANONICAL_MANIFEST_SHA256", manifest_sha)
    monkeypatch.setattr(campaign_mod, "CANONICAL_PLACEMENT_SHA256", placement_sha)
    monkeypatch.setattr(arms_mod, "CANONICAL_PLACEMENT_SHA256", placement_sha)
    # The candidate runtime fixtures must echo whatever placement SHA is frozen now.
    # The test package can be imported under two names (``tests.benchmarks.*`` via a
    # namespace package and ``benchmarks.*`` via pytest's rootdir prepend), so every
    # loaded instance of this module is pinned, not just one.
    import sys

    for module_name in list(sys.modules):
        if module_name.endswith("phase1_fakes"):
            monkeypatch.setattr(sys.modules[module_name], "CURRENT_PLACEMENT_SHA", placement_sha)
    return {"manifest": str(manifest), "placement": str(placement), "placement_sha": placement_sha}


def install_clean_environment(monkeypatch, tmp_path: Path) -> dict[str, Any]:
    """The complete canonical-campaign test environment: frozen identities, clean
    provenance, proven GPUs, quiet host — everything a valid session needs, patched
    once so every test breaks exactly one thing."""
    from inferswarm_phase0 import gpu as gpu_mod
    from inferswarm_phase0 import provenance as prov
    from inferswarm_phase1 import campaign as campaign_mod

    frozen = freeze_frozen_identities(monkeypatch, tmp_path)
    frozen["prerequisites"] = str(write_prerequisites(tmp_path))
    monkeypatch.setattr(
        prov,
        "git_commit",
        lambda repo_dir: {"value": "a" * 40, "dirty": False, "dirty_paths": []},
    )
    monkeypatch.setattr(
        gpu_mod,
        "_resolve_uuids",
        lambda selector: (selector,) if str(selector).startswith("GPU-") else (),
    )
    monkeypatch.setattr(gpu_mod, "_smi_index_for", lambda uuid: 0)
    monkeypatch.setattr(
        gpu_mod,
        "engine_gpus",
        lambda origin: [{
            "index": 0,
            "uuid": "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55",
            "name": "NVIDIA GeForce RTX 3060",
            "total_bytes": 12 << 30,
        }],
    )
    monkeypatch.setattr(
        prov,
        "gpu_provenance",
        lambda selector=None, resolved_uuid=None: {
            "gpus": [{"index": "0", "uuid": GPU0_UUID,
                      "name": "NVIDIA GeForce RTX 3060", "memory.total": "12288 MiB",
                      "selected": True}],
            "topology": "GPU0\tX\tPIX\tGPU1",
            "topology_p2p": "GPU0 to GPU1 : Not Supported",
            "selected": {"requested": selector, "resolved_uuid": resolved_uuid},
        },
    )
    monkeypatch.setattr(
        prov,
        "host_provenance",
        lambda: {"os": "Linux 6.0", "cpu_model": "Test CPU", "ram_total_bytes": 64 << 30},
    )
    monkeypatch.setattr(prov, "_torch_versions", lambda: {"torch": "2.11.0"})
    monkeypatch.setattr(campaign_mod, "_hostname", lambda: "test-host")
    monkeypatch.setattr(campaign_mod, "thermal_observation", lambda: {"observed_at": "now", "gpus": []})
    monkeypatch.setattr(campaign_mod, "gpu_memory_used_bytes", lambda uuid: 8 << 20)
    monkeypatch.setattr(campaign_mod, "BETWEEN_ARM_SETTLE_SECONDS", 0.0)
    return frozen


def install_mocked_server(monkeypatch):
    """Replace every process/HTTP boundary of a session. Defaults are fully valid.

    Shared by the session-run tests and the verdict-firewall tests. Defaults are
    built lazily per fetch so the runtime fixtures echo whatever frozen identities
    the environment installed.
    """
    from inferswarm_phase1 import campaign as campaign_mod
    from inferswarm_phase1.campaign import GenerationError, ServerError  # noqa: F401

    calls = {
        "started": [], "stopped": 0, "generations": [], "moe_ops": [],
        "instrumentation": [], "order": [],
    }
    runtime_by_arm = {}

    def default_runtime_for(arm):
        if arm == "candidate_v2":
            return candidate_runtime_config()
        return baseline_runtime_config()

    def arm_of(command):
        if "--inferswarm-remote-decode" in command:
            return "candidate_v2"
        if "--num-tokens" in command:
            return "baseline_b1_kv_matched"
        return "baseline_b1"

    def fake_start_server(command, origin, log_path, **kwargs):
        arm = arm_of(command)
        calls["started"].append({"arm": arm, "command": list(command)})
        calls["order"].append(("serve", arm))
        Path(log_path).write_text("fake server log\n")
        return _FakeHandle()

    def fake_stop_server(handle):
        calls["stopped"] += 1

    def fake_fetch_instrumentation(origin, limit=8):
        arm = calls["started"][-1]["arm"] if calls["started"] else "baseline_b1"
        calls["instrumentation"].append(origin)
        return {
            "schema": "freetoken.instrumentation/1",
            "runtime_config": runtime_by_arm.setdefault(arm, default_runtime_for(arm)),
            "prefill": {"enabled": True, "observed": 0, "records": []},
        }

    def fake_measure(origin, body, *, prefill_seq_floor=0, store_text=False, **kwargs):
        calls["generations"].append(dict(body))
        class_id = body["messages"][0]["content"].rsplit(" ", 1)[-1]
        prompt_tokens = {"W1": 1800, "W2": 900, "W3": 16000, "W4": 128}.get(class_id, 900)
        n = len(calls["generations"])
        return {
            "ttft_ms": 100.0 + n,
            "wall_total_ms": 5000.0 + n,
            "decode_window_s": 10.0,
            "decode_steps": body["max_tokens"] - 1,
            "decode_tok_s": 20.0 + n * 0.01,
            "ms_per_token_mean": 50.0,
            "inter_token_ms": [50.0, 51.0, 52.0],
            "inter_token_ms_p50": 51.0,
            "inter_token_ms_p95": 52.0,
            "inter_token_ms_max": 52.0,
            "token_events": 3,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": body["max_tokens"],
            "requested_max_tokens": body["max_tokens"],
            "completion_matches_request": True,
            "output_sha256": "deadbeef",
            "output_chars": 8,
            "output_preview": "hello",
            "output_text": "hello" if store_text else None,
            "output_text_stored": bool(store_text),
            "response_id": "chatcmpl-1",
            "request_uid": 1,
            "vram_bytes": 11 << 30,
            "prefill": {"gpu_ms": 40.0, "new_tokens": prompt_tokens,
                        "prefill_tok_s": prompt_tokens / 0.04},
            "prefill_status": {"ok": True, "code": "ok", "reason": None, "attribution": "uid"},
        }

    def fake_moe_instrumentation(origin, operation, *, timeout=60.0):
        arm = calls["started"][-1]["arm"] if calls["started"] else "baseline_b1"
        calls["moe_ops"].append((operation, arm))
        calls["order"].append((f"moe:{operation}", arm))
        if operation == "reset":
            return {"boundary": {"operation": "reset", "idle": True}}
        return moe_window_snapshot()

    monkeypatch.setattr(campaign_mod, "start_server", fake_start_server)
    monkeypatch.setattr(campaign_mod, "stop_server", fake_stop_server)
    monkeypatch.setattr(campaign_mod, "fetch_instrumentation", fake_fetch_instrumentation)
    monkeypatch.setattr(campaign_mod, "measure_generation", fake_measure)
    monkeypatch.setattr(campaign_mod, "prefill_seq_floor", lambda origin: 0)
    monkeypatch.setattr(campaign_mod, "moe_instrumentation", fake_moe_instrumentation)
    monkeypatch.setattr(campaign_mod, "_model_id", lambda origin: "qwen-test")
    calls["runtime_by_arm"] = runtime_by_arm
    return calls


class _FakeHandle:
    def __init__(self):
        self.proc = None
