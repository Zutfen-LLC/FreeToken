"""The Phase-0 baseline configuration matrix, as declared -- not as defaulted.

Source of truth: InferSwarm ``docs/phase1-poc-success-criteria.md`` section 2.1 (the B1-B5
sweep) and section 2.4 (``CORRECTNESS_REFERENCE``). Both are reproduced here as data, and
every value the criteria fix is passed on the command line explicitly.

Why "explicitly" is load-bearing, from FreeToken's own source:

* ``EngineConfig.nvfp4_backend`` defaults to ``"triton"``
  (``python/freetoken/engine/config.py``), *not* to ``auto``. A row that only said "NVFP4
  backend auto" in prose would silently run Triton and duplicate B4.
* ``--moe-cache-auto`` is applied by the CLI when no sizing flag is given
  (``python/freetoken/server/args.py``), not by the dataclass default. A baseline must be
  reproducible from the command line, never from a default that may change.

This module builds command lines. It does not decide anything: which backend ``auto``
resolves to, whether ``--nvfp4-backend`` was inert, and whether the Marlin slot cap bound
are all read back off the running engine (``/v1/instrumentation``), because a second
implementation of that policy here could disagree with the one that executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

# vLLM's Marlin grouped-GEMM caps padded experts at 1024, so the offload slot cache is
# capped at 992 (``python/freetoken/moe/offload_cache.py``: MARLIN_MAX_CACHE_SIZE).
# Mirrored here only to validate a CORRECTNESS_REFERENCE cache size before a server is
# started; whether the cap actually *bound* is read from the engine, never assumed.
MARLIN_MAX_CACHE_SIZE = 992


@dataclass(frozen=True)
class Arm:
    """One configuration arm of the sweep.

    ``moe_flags`` is the exact, ordered flag list the criteria declare for this arm.
    ``role`` separates the two things Phase 0 produces, which must never be conflated:
    ``performance`` arms feed the ``CANONICAL_PERFORMANCE_BASELINE`` selection (section
    2.2), while the single ``correctness`` arm is ``CORRECTNESS_REFERENCE`` (section 2.4),
    fixed in advance and never chosen by speed.
    """

    id: str
    role: str  # "performance" | "correctness"
    moe_backend: str
    nvfp4_backend: str
    description: str
    # Extra flags this arm fixes beyond --moe-backend/--nvfp4-backend.
    extra_flags: Sequence[str] = field(default_factory=tuple)
    # B2's hybrid split is resolved from a `ft bench bw` profile; the criteria require a
    # fresh one for this GPU + expert format before the arm runs.
    requires_bench_bw: bool = False
    # Whether this arm's engine can READ the bandwidth profile. Wider than
    # ``requires_bench_bw``: B3's ``--moe-backend auto`` consults the same profile to decide
    # whether to upgrade offload to hybrid (``engine._adjust_config`` ->
    # ``bench_profile.load_backend_recommendation``), so a stale profile changes what B3
    # resolves to even though the criteria table names the refresh only under B2. Both
    # therefore have to run after the session-level refresh, in either traversal order.
    consumes_bench_bw: bool = False
    notes: str = ""

    def moe_flags(self) -> List[str]:
        """The arm's configuration flags, in the order the criteria state them."""
        return [
            "--moe-backend", self.moe_backend,
            *self.extra_flags,
            "--nvfp4-backend", self.nvfp4_backend,
        ]


# --- Section 2.1: the B1-B5 sweep -------------------------------------------------------
#
# B1/B4 are a declared pair and are ALLOWED to collapse: if `--nvfp4-backend auto` resolves
# to triton on the actual rig (vLLM absent, Marlin donor symbols unusable, ...), B1 and B4
# are the same resolved configuration and are reported as two observations of it, with the
# reason `auto` declined Marlin. Marlin is never installed or forced to make them differ.
BASELINE_ARMS: tuple[Arm, ...] = (
    Arm(
        id="B1",
        role="performance",
        moe_backend="offload",
        nvfp4_backend="auto",
        extra_flags=("--moe-cache-auto",),
        description="offload + hardware-selected NVFP4 backend",
        notes=(
            "Measures what select_nvfp4_backend actually chooses on this rig. On sm_86 that "
            "is marlin if and only if vLLM's Marlin donor symbols are usable here."
        ),
    ),
    Arm(
        id="B2",
        role="performance",
        moe_backend="hybrid",
        nvfp4_backend="triton",
        extra_flags=("--moe-cache-auto",),
        requires_bench_bw=True,
        consumes_bench_bw=True,
        description="hybrid CPU/GPU decode + Triton NVFP4",
        notes=(
            "hybrid loads expert banks with decode_target=cpu, so the loader keeps the native "
            "ModelOpt layout and never calls select_nvfp4_backend: --nvfp4-backend is INERT "
            "for the expert path here. triton is passed because the native-layout Triton "
            "kernels are what executes. Needs a fresh `ft bench bw` profile for this GPU and "
            "expert format (it sets the per-step fetch split)."
        ),
    ),
    Arm(
        id="B3",
        role="performance",
        moe_backend="auto",
        nvfp4_backend="auto",
        extra_flags=("--moe-cache-auto",),
        consumes_bench_bw=True,
        description="auto MoE backend + hardware-selected NVFP4 backend",
        notes=(
            "The resolved MoE backend is recorded from the engine, and must coincide with B1 "
            "or B2 (criteria section 2.1). `auto` READS the same `ft bench bw` profile B2's "
            "fetch split comes from -- load_backend_recommendation upgrades the offload "
            "default to hybrid when the benched CPU/PCIe ratio clears the threshold -- so a "
            "stale profile changes what this arm resolves to."
        ),
    ),
    Arm(
        id="B4",
        role="performance",
        moe_backend="offload",
        nvfp4_backend="triton",
        extra_flags=("--moe-cache-auto",),
        description="offload + forced Triton NVFP4",
        notes=(
            "The B1 pair: forces Triton to find out whether a larger resident cache beats the "
            "faster-but-992-slot-capped Marlin path. Equivalence with B1 is a result, not an "
            "error."
        ),
    ),
    Arm(
        id="B5",
        role="performance",
        moe_backend="cpu",
        nvfp4_backend="triton",
        extra_flags=("--moe-cache-auto",),
        description="CPU MoE executor + Triton NVFP4",
        notes=(
            "Like hybrid, banks load with decode_target=cpu, so --nvfp4-backend is INERT for "
            "the expert path; decode experts run in the CPU executor's own dequant-in-GEMV "
            "kernels, which are not a GPU NVFP4 kernel at all."
        ),
    ),
)

BASELINE_ARMS_BY_ID = {arm.id: arm for arm in BASELINE_ARMS}


def correctness_reference_arm(nvfp4_backend: str, moe_cache_size: int) -> Arm:
    """Build ``CORRECTNESS_REFERENCE`` (criteria section 2.4).

    Both parameters are REQUIRED and have no default on purpose:

    * ``nvfp4_backend`` must be the value the candidate's remote expert GEMM resolves to,
      recorded as a resolved value -- pinning the flag pins the kernel family, which is
      what makes C1's 2e-3 backend-vs-backend tolerance mean anything. There is no
      defensible default for "whatever the candidate does".
    * ``moe_cache_size`` is fixed rather than auto so the reference is stable run to run
      instead of tracking whatever ``--moe-cache-auto`` resolves to on the day.

    This arm is never selected by speed and is never quoted as a performance comparator.
    """
    if nvfp4_backend not in ("marlin", "flashinfer", "triton"):
        raise ValueError(
            f"CORRECTNESS_REFERENCE needs an explicit resolved NVFP4 backend "
            f"(marlin|flashinfer|triton), got {nvfp4_backend!r}. 'auto' is not a "
            "configuration record (criteria section 2.3)."
        )
    if moe_cache_size <= 0:
        raise ValueError(
            f"CORRECTNESS_REFERENCE needs a fixed --moe-cache-size > 0, got {moe_cache_size}"
        )
    if nvfp4_backend == "marlin" and moe_cache_size > MARLIN_MAX_CACHE_SIZE:
        raise ValueError(
            f"--moe-cache-size {moe_cache_size} exceeds the marlin slot cap "
            f"{MARLIN_MAX_CACHE_SIZE}; the server would refuse to start (criteria section 2.4)"
        )
    return Arm(
        id="CORRECTNESS_REFERENCE",
        role="correctness",
        moe_backend="offload",
        nvfp4_backend=nvfp4_backend,
        extra_flags=("--moe-cpu-layers", "0", "--moe-cache-size", str(moe_cache_size)),
        description="fixed single-device GPU reference (criteria section 2.4)",
        notes=(
            "--moe-cpu-layers 0 is load-bearing: it forces every MoE layer onto the GPU "
            "offload path AND suppresses _auto_cpu_layers, which fires only when "
            "moe_cpu_layers is None. Without it a quota-capped host could silently move part "
            "of the reference onto the CPU executor and change what 'the same kernel' means."
        ),
    )


def validate_cache_floor(moe_cache_size: int, num_experts: int) -> None:
    """The offload cache needs at least ``num_experts`` slots (engine._require_offload_cache_size)."""
    if moe_cache_size < num_experts:
        raise ValueError(
            f"--moe-cache-size {moe_cache_size} < num_experts {num_experts}; the engine "
            "refuses this (criteria section 2.4 requires cache_size >= num_experts)"
        )
