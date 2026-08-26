"""The B1-B5 command matrix is checked against the criteria document, not against itself.

Expectations here are transcribed from InferSwarm
``docs/phase1-poc-success-criteria.md`` section 2.1 (the sweep table) and section 2.4 (the
correctness reference), which is an independent source: if someone edits the arm
definitions, these tests disagree with the document rather than agreeing with the new code.
"""

from __future__ import annotations

import pytest

from inferswarm_phase0.baselines import (
    BASELINE_ARMS,
    BASELINE_ARMS_BY_ID,
    MARLIN_MAX_CACHE_SIZE,
    correctness_reference_arm,
    validate_cache_floor,
)
from inferswarm_phase0.runner import ServeSettings, bench_bw_command, serve_command

# criteria section 2.1, verbatim: "Configuration (as passed, not as defaulted)"
CRITERIA_TABLE = {
    "B1": ["--moe-backend", "offload", "--moe-cache-auto", "--nvfp4-backend", "auto"],
    "B2": ["--moe-backend", "hybrid", "--moe-cache-auto", "--nvfp4-backend", "triton"],
    "B3": ["--moe-backend", "auto", "--moe-cache-auto", "--nvfp4-backend", "auto"],
    "B4": ["--moe-backend", "offload", "--moe-cache-auto", "--nvfp4-backend", "triton"],
    "B5": ["--moe-backend", "cpu", "--moe-cache-auto", "--nvfp4-backend", "triton"],
}


def _settings(**kwargs) -> ServeSettings:
    base = dict(
        model_path="/models/qwen",
        model_repository="nvidia/Qwen3.6-35B-A3B-NVFP4",
        model_revision="0" * 40,
        python_executable="python",
    )
    base.update(kwargs)
    return ServeSettings(**base)


def test_sweep_covers_exactly_b1_to_b5():
    assert [arm.id for arm in BASELINE_ARMS] == ["B1", "B2", "B3", "B4", "B5"]
    assert all(arm.role == "performance" for arm in BASELINE_ARMS)


@pytest.mark.parametrize("arm_id,expected", sorted(CRITERIA_TABLE.items()))
def test_arm_flags_match_the_criteria_table(arm_id, expected):
    assert BASELINE_ARMS_BY_ID[arm_id].moe_flags() == expected


@pytest.mark.parametrize("arm_id", sorted(CRITERIA_TABLE))
def test_every_arm_states_nvfp4_backend_explicitly(arm_id):
    """EngineConfig.nvfp4_backend defaults to "triton", not "auto": an arm that left the
    flag off would silently run Triton and duplicate B4."""
    flags = BASELINE_ARMS_BY_ID[arm_id].moe_flags()
    assert "--nvfp4-backend" in flags
    assert flags[flags.index("--nvfp4-backend") + 1] in ("auto", "triton", "marlin", "flashinfer")


@pytest.mark.parametrize("arm_id", sorted(CRITERIA_TABLE))
def test_every_arm_states_the_cache_policy_explicitly(arm_id):
    """--moe-cache-auto is applied by the CLI when no sizing flag is given, not by the
    dataclass default; a baseline must not depend on that."""
    assert "--moe-cache-auto" in BASELINE_ARMS_BY_ID[arm_id].moe_flags()


def test_only_b2_requires_a_fresh_bench_bw_profile():
    assert [arm.id for arm in BASELINE_ARMS if arm.requires_bench_bw] == ["B2"]
    cmd = bench_bw_command(_settings(gpu="GPU-abc"), "nvfp4")
    assert cmd[-4:] == ["--dtype", "nvfp4", "--gpu", "GPU-abc"]


def test_b1_and_b4_are_representable_as_an_equivalent_pair():
    """They differ only in --nvfp4-backend, so a rig where `auto` resolves to triton makes
    them the same resolved configuration. That collapse is a result, not an error: nothing
    in the harness forces them apart."""
    b1, b4 = BASELINE_ARMS_BY_ID["B1"], BASELINE_ARMS_BY_ID["B4"]
    assert b1.moe_backend == b4.moe_backend == "offload"
    assert (b1.nvfp4_backend, b4.nvfp4_backend) == ("auto", "triton")
    differing = [
        (a, b) for a, b in zip(b1.moe_flags(), b4.moe_flags()) if a != b
    ]
    assert differing == [("auto", "triton")]
    # And nothing anywhere asks for marlin to be installed or forced.
    assert "marlin" not in " ".join(b1.moe_flags() + b4.moe_flags())


def test_b2_and_b5_document_the_inert_nvfp4_flag():
    """Both load banks with decode_target="cpu", so the loader keeps the native layout and
    never calls select_nvfp4_backend. The arm must say so rather than implying the flag
    picked the executing kernel."""
    for arm_id in ("B2", "B5"):
        assert "INERT" in BASELINE_ARMS_BY_ID[arm_id].notes


def test_serve_command_pins_every_held_constant_value():
    cmd = serve_command(BASELINE_ARMS_BY_ID["B1"], _settings(gpu="GPU-1", kv_reserve_tokens=8192), 9001)
    joined = " ".join(cmd)
    for flag in (
        "--moe-backend offload",
        "--moe-cache-auto",
        "--nvfp4-backend auto",
        "--max-running-requests 1",
        "--cuda-graph-max-bs 1",
        "--memory-ratio 0.9",
        "--sampling-defaults none",
        "--kv-reserve-tokens 8192",
        "--gpu GPU-1",
        "--port 9001",
    ):
        assert flag in joined, flag


def test_serve_command_is_stable_for_the_same_inputs():
    """The recorded command line is a reproduction recipe, so it must not depend on dict
    ordering or anything else that varies run to run."""
    settings = _settings(gpu="GPU-1")
    a = serve_command(BASELINE_ARMS_BY_ID["B3"], settings, 7000)
    b = serve_command(BASELINE_ARMS_BY_ID["B3"], settings, 7000)
    assert a == b


def test_no_arm_can_shrink_the_expert_cache():
    """Criteria section 3 rule 1 prohibits lowering --moe-cache-size/--moe-cache-rate below
    the auto-resolved value on a performance arm."""
    for arm in BASELINE_ARMS:
        flags = serve_command(arm, _settings(), 1)
        assert "--moe-cache-size" not in flags
        assert "--moe-cache-rate" not in flags


def test_prefill_instrumentation_is_enabled_identically_for_every_arm():
    env = _settings().env_overrides()
    assert env == {"FREETOKEN_INSTRUMENT_PREFILL": "1"}


# --- CORRECTNESS_REFERENCE (criteria section 2.4) ---------------------------------------

def test_correctness_reference_matches_the_declared_configuration():
    arm = correctness_reference_arm("triton", 512)
    assert arm.role == "correctness"
    assert arm.moe_flags() == [
        "--moe-backend", "offload",
        "--moe-cpu-layers", "0",
        "--moe-cache-size", "512",
        "--nvfp4-backend", "triton",
    ]
    cmd = " ".join(serve_command(arm, _settings(), 1))
    # --sampling-defaults none: framework defaults -> greedy (criteria section 5.3)
    assert "--sampling-defaults none" in cmd
    # never auto-sized: a fixed cache keeps the reference stable run to run
    assert "--moe-cache-auto" not in cmd


def test_correctness_reference_rejects_an_unresolved_nvfp4_backend():
    with pytest.raises(ValueError, match="explicit resolved NVFP4 backend"):
        correctness_reference_arm("auto", 512)


def test_correctness_reference_respects_the_marlin_slot_cap():
    correctness_reference_arm("marlin", MARLIN_MAX_CACHE_SIZE)  # the cap itself is fine
    with pytest.raises(ValueError, match="marlin slot cap"):
        correctness_reference_arm("marlin", MARLIN_MAX_CACHE_SIZE + 1)
    # the cap is a marlin property; triton carries none
    correctness_reference_arm("triton", MARLIN_MAX_CACHE_SIZE * 4)


def test_correctness_reference_requires_a_fixed_cache_size():
    with pytest.raises(ValueError, match="fixed --moe-cache-size"):
        correctness_reference_arm("triton", 0)


def test_cache_floor_matches_the_engine_requirement():
    validate_cache_floor(256, 256)
    with pytest.raises(ValueError, match="num_experts"):
        validate_cache_floor(255, 256)
