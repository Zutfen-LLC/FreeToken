"""Arm-definition tests: exact flags, no leakage, exact UUIDs, exact differences."""

from __future__ import annotations

import dataclasses

import pytest
from inferswarm_phase1.campaign_arms import (
    BASELINE_ARM_ID,
    CANDIDATE_ARM_ID,
    EXPECTED_GPU1_EXPERT_BYTES,
    EXPECTED_GPU1_SLOTS,
    GPU0_UUID,
    GPU1_UUID,
    ArmDefinitionError,
    baseline_b1_arm,
    candidate_v2_arm,
    compare_primary_arms,
    flags_to_dict,
    kv_matched_arm,
    validate_arm_definitions,
)


def test_baseline_flags_are_exactly_the_frozen_b1_identity():
    flags = baseline_b1_arm().flags()
    assert flags == [
        "--gpu", GPU0_UUID,
        "--moe-backend", "offload",
        "--moe-cache-auto",
        "--nvfp4-backend", "auto",
        "--kv-reserve-tokens", "17075",
        "--memory-ratio", "0.85",
        "--max-running-requests", "1",
        "--cuda-graph-max-bs", "1",
        "--sampling-defaults", "none",
        "--moe-layer-timing-role", "baseline",
    ]


def test_candidate_flags_are_exactly_the_landed_candidate():
    flags = candidate_v2_arm().flags()
    assert flags == [
        "--gpu", GPU0_UUID,
        "--moe-backend", "offload",
        "--moe-cpu-layers", "0",
        "--nvfp4-backend", "triton",
        "--moe-cache-size", "3774",
        "--kv-reserve-tokens", "17075",
        "--num-tokens", "17075",
        "--memory-ratio", "0.85",
        "--cuda-graph-max-bs", "0",
        "--max-running-requests", "1",
        "--sampling-defaults", "none",
        "--inferswarm-secondary-gpu", GPU1_UUID,
        "--inferswarm-placement", "<placement-path>",
        "--inferswarm-remote-decode",
        "--moe-layer-timing-role", "candidate",
    ]


def test_no_inferswarm_flag_leaks_into_the_baseline():
    baseline_flags = {f for f in baseline_b1_arm().flags() if f.startswith("--")}
    assert not any(flag.startswith("--inferswarm-") for flag in baseline_flags)
    assert validate_arm_definitions([baseline_b1_arm(), candidate_v2_arm()]) == []


def test_candidate_uses_the_exact_frozen_gpu_uuids():
    flags = candidate_v2_arm().flags()
    assert flags[flags.index("--gpu") + 1] == GPU0_UUID
    assert flags[flags.index("--inferswarm-secondary-gpu") + 1] == GPU1_UUID


def test_leaking_a_secondary_gpu_into_the_baseline_is_refused():
    drifted = dataclasses.replace(
        baseline_b1_arm(),
        config_flags=(
            *baseline_b1_arm().config_flags,
            "--inferswarm-secondary-gpu",
            GPU1_UUID,
        ),
    )
    reasons = validate_arm_definitions([drifted, candidate_v2_arm()])
    assert any("leaks the InferSwarm treatment" in r for r in reasons)


def test_candidate_without_placement_is_refused():
    candidate = candidate_v2_arm()
    flags = [
        f
        for i, f in enumerate(candidate.config_flags)
        if f != "--inferswarm-placement" and candidate.config_flags[i - 1] != "--inferswarm-placement"
    ]
    stripped = dataclasses.replace(candidate, config_flags=tuple(flags))
    reasons = validate_arm_definitions([baseline_b1_arm(), stripped])
    assert any("--inferswarm-placement is required" in r for r in reasons)


def test_candidate_with_wrong_secondary_uuid_is_refused():
    candidate = dataclasses.replace(
        candidate_v2_arm(),
        config_flags=(
            "--gpu", GPU0_UUID,
            "--moe-backend", "offload",
            "--inferswarm-secondary-gpu", "GPU-00000000-0000-0000-0000-000000000000",
            "--inferswarm-placement", "<placement-path>",
            "--inferswarm-remote-decode",
        ),
    )
    reasons = validate_arm_definitions([baseline_b1_arm(), candidate])
    assert any("must be the frozen physical UUID" in r for r in reasons)


def test_the_c3_full_logit_recorder_is_forbidden_in_every_arm():
    poisoned = dataclasses.replace(
        baseline_b1_arm(),
        config_flags=(*baseline_b1_arm().config_flags, "--inferswarm-correctness-diagnostics"),
    )
    reasons = validate_arm_definitions([poisoned, candidate_v2_arm()])
    assert any("performance-incompatible" in r for r in reasons)


def test_the_supplementary_kv_arm_can_never_become_primary():
    kv = kv_matched_arm(17075)
    assert kv.role == "supplementary"
    assert validate_arm_definitions([baseline_b1_arm(), candidate_v2_arm(), kv]) == []
    usurper = dataclasses.replace(kv, role="primary")
    reasons = validate_arm_definitions([baseline_b1_arm(), candidate_v2_arm(), usurper])
    assert any("can never be primary" in r for r in reasons)


def test_kv_matched_arm_differs_from_b1_only_in_num_tokens():
    kv = kv_matched_arm(23456)
    kv_flags = flags_to_dict(kv.flags())
    b1_flags = flags_to_dict(baseline_b1_arm().flags())
    assert kv_flags["--num-tokens"] == "23456"
    assert b1_flags.get("--num-tokens") is None
    differing = {
        k for k in set(kv_flags) | set(b1_flags) if kv_flags.get(k) != b1_flags.get(k)
    }
    assert differing == {"--num-tokens"}


def test_kv_matched_arm_requires_a_positive_capacity():
    with pytest.raises(ArmDefinitionError):
        kv_matched_arm(0)


# --- the machine-readable comparison ------------------------------------------------------


def test_comparison_declares_exactly_the_intended_differences():
    comparison = compare_primary_arms(baseline_b1_arm(), candidate_v2_arm())
    assert comparison["held_equal_all"] is True
    assert comparison["undeclared_differences"] == []
    buckets = comparison["intended_differences"]
    assert set(buckets) == {
        "inferswarm_remote_execution",
        "fixed_candidate_expert_placement_cache",
        "cuda_graph_state",
        "candidate_kv_capacity_pin",
        "moe_layer_timing_role_label",
    }
    # every InferSwarm treatment flag is named under the remote-execution bucket
    remote_flags = {e["flag"] for e in buckets["inferswarm_remote_execution"]}
    assert remote_flags == {
        "--inferswarm-secondary-gpu", "--inferswarm-placement", "--inferswarm-remote-decode"
    }
    assert buckets["cuda_graph_state"][0]["baseline"] == "1"
    assert buckets["cuda_graph_state"][0]["candidate"] == "0"


def test_held_constants_cover_gpu0_memory_ratio_kv_reserve_batch_and_sampling():
    comparison = compare_primary_arms(baseline_b1_arm(), candidate_v2_arm())
    held = comparison["held_equal"]
    assert held["--gpu"] == {
        "equal": True, "baseline": GPU0_UUID, "candidate": GPU0_UUID,
    }
    assert held["--memory-ratio"]["baseline"] == "0.85"
    assert held["--kv-reserve-tokens"]["baseline"] == "17075"
    assert held["--max-running-requests"]["baseline"] == "1"
    assert held["--sampling-defaults"]["baseline"] == "none"


def test_a_single_undeclared_difference_survives_validation_as_a_failure():
    drifted = dataclasses.replace(
        candidate_v2_arm(),
        config_flags=(
            *candidate_v2_arm().config_flags[:4],
            *candidate_v2_arm().config_flags[6:],  # drop "--moe-cpu-layers 0"
            "--attention-backend",
            "flashinfer",
        ),
    )
    comparison = compare_primary_arms(baseline_b1_arm(), drifted)
    assert any(
        d["flag"] == "--attention-backend" for d in comparison["undeclared_differences"]
    )


def test_mutating_a_held_constant_breaks_held_equal():
    drifted = dataclasses.replace(
        candidate_v2_arm(),
        config_flags=(
            "--gpu", GPU0_UUID,
            "--moe-backend", "offload",
            "--moe-cpu-layers", "0",
            "--nvfp4-backend", "triton",
            "--moe-cache-size", "3774",
            "--kv-reserve-tokens", "17075",
            "--num-tokens", "17075",
            "--memory-ratio", "0.9",  # baseline says 0.85
            "--cuda-graph-max-bs", "0",
            "--max-running-requests", "1",
            "--sampling-defaults", "none",
            "--inferswarm-secondary-gpu", GPU1_UUID,
            "--inferswarm-placement", "<placement-path>",
            "--inferswarm-remote-decode",
            "--moe-layer-timing-role", "candidate",
        ),
    )
    comparison = compare_primary_arms(baseline_b1_arm(), drifted)
    assert comparison["held_equal"]["--memory-ratio"]["equal"] is False
    assert comparison["held_equal_all"] is False


def test_a_flag_declared_in_two_buckets_is_a_definition_error(monkeypatch):
    import inferswarm_phase1.campaign_arms as arms

    duplicated = arms.INTENDED_DIFFERENCE_BUCKETS + (("duplicated", ("--num-tokens",)),)
    monkeypatch.setattr(arms, "INTENDED_DIFFERENCE_BUCKETS", duplicated)
    with pytest.raises(ArmDefinitionError):
        compare_primary_arms(baseline_b1_arm(), candidate_v2_arm())


def test_frozen_gpu1_expectations_match_the_placement_contract():
    assert EXPECTED_GPU1_SLOTS == 5442
    assert EXPECTED_GPU1_EXPERT_BYTES == 9_662_902_272


def test_arm_ids_are_stable():
    assert baseline_b1_arm().id == BASELINE_ARM_ID
    assert candidate_v2_arm().id == CANDIDATE_ARM_ID
