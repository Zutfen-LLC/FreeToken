"""Issue #117 Arm-C SWA allocation remediation lifecycle proofs (InferSwarm
#166) — CPU-only, no model/GPU execution.

Proves the Phase-1 remediation invariant on the real runtime seam:

> If execution can perform an SWA KV operation that references
> prior-prefix state, its live sequence/session owns a valid non-sentinel
> full-to-SWA mapping established by the owning lifecycle before that
> operation executes.

The owning lifecycle is ``GemmaDenseStage`` itself (the standalone R6 stage
runtime): every ``prefill``/``decode`` entry point calls
``_ensure_swa_session_mapping`` BEFORE ``_prepare`` (whose
``prepare_metadata`` translates the full page-table range through the
mapping — the first prefix-consuming SWA operation of the chunk), the
mapping persists across chunks/decode via an incremental frontier, and
``reset_session_state`` releases all ownership wholesale.

Established against the accepted #157 facts (InferSwarm
arm-c-chunk2-diagnosis-157): the standalone path never reached the
scheduler's alloc_swa lifecycle; earliest varying boundary
``L0_kv_slice_post_write``; the one-variable ``alloc_swa`` intervention
stabilized both anchors. This suite proves the PRODUCTION lifecycle reaches
the same underlying allocation semantic (pool.alloc_swa over exactly the
referenced positions) before that boundary, with reset/reuse, teardown,
exhaustion, and non-SWA/no-op semantics — all deterministic on CPU.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "python"), str(REPO / "benchmarks")):
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmarks.inferswarm_r6.stage_runtime import GemmaDenseStage  # noqa: E402

STAGE_RUNTIME = REPO / "benchmarks/inferswarm_r6/stage_runtime.py"


# ---------------------------------------------------------------------------
# Fixtures: a real HybridSWAKVCache on CPU + a minimal stage stub
# ---------------------------------------------------------------------------


def _specs():
    from freetoken.models.config import KVCacheGroupSpec

    return (
        KVCacheGroupSpec(name="full", layer_ids=(1,), num_kv_heads=1, head_dim=8, sliding_window=None),
        KVCacheGroupSpec(name="swa", layer_ids=(0,), num_kv_heads=1, head_dim=8, sliding_window=4),
    )


def _patch_tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.hybrid_swa_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def _stage_with_pool(monkeypatch, *, num_full=256, num_swa=257, swa=True):
    """A GemmaDenseStage stub whose ctx.kv_cache is a REAL CPU pool (SWA)
    or a real non-SWA-shaped stub (swa_paged absent/False)."""
    _patch_tp(monkeypatch)
    if swa:
        from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

        pool = HybridSWAKVCache(
            groups=_specs(),
            num_layers=2,
            num_full_pages=num_full,
            page_size=1,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            num_swa_tokens=num_swa,
        )
    else:
        pool = SimpleNamespace(swa_paged=False)
    stage = object.__new__(GemmaDenseStage)
    stage.ctx = SimpleNamespace(kv_cache=pool)
    stage.device = torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    return stage, pool


def _mapping_valid(pool, count):
    """The Phase-1 invariant checker: positions [0, count) all own distinct
    live (non-sentinel) swa slots."""
    m = pool.full_to_swa_index_mapping[:count].to(torch.int64)
    if count == 0:
        return True
    return bool((m > 0).all()) and int(m.max()) < int(pool.swa_num_tokens) and len(set(m.tolist())) == count


# ---------------------------------------------------------------------------
# 1. Mapping-before-prefix: the ordering proof on the REAL entry points
# ---------------------------------------------------------------------------


class _StopBeforeForward(Exception):
    pass


def _capture_prepare_stage(monkeypatch, *, role="first"):
    """A stub whose _prepare records the mapping coverage AT THE MOMENT the
    prefix-consuming metadata would be built, then halts execution before
    any forward — proving allocation strictly precedes the first SWA KV
    operation inside the real prefill()/decode() code."""
    stage, pool = _stage_with_pool(monkeypatch)
    stage.role = role
    observed = []

    def _prepare(*, start, token_count, phase):
        observed.append(
            {
                "start": start,
                "token_count": token_count,
                "phase": phase,
                "coverage_valid": _mapping_valid(pool, start + token_count),
                "frontier": stage._swa_session_allocated,
            }
        )
        raise _StopBeforeForward()

    stage._prepare = _prepare
    return stage, pool, observed


def test_prefill_first_chunk_mapping_valid_before_prepare(monkeypatch):
    stage, pool, observed = _capture_prepare_stage(monkeypatch)
    with pytest.raises(_StopBeforeForward):
        stage.prefill(list(range(53)), None, 0)
    assert observed[0]["coverage_valid"] is True
    assert observed[0]["frontier"] == 53


def test_prefill_second_chunk_prefix_mapping_valid_before_prepare(monkeypatch):
    """The #157 failing shape: 64-row chunk then a nonzero-prefix chunk.
    At the moment chunk 2's metadata is built, the WHOLE referenced range
    [0, 67) must already own live slots (prefix gathers included)."""
    stage, pool, observed = _capture_prepare_stage(monkeypatch)
    # chunk 1 executed its allocation (halted at prepare, after ensure)
    with pytest.raises(_StopBeforeForward):
        stage.prefill(list(range(64)), None, 0)
    with pytest.raises(_StopBeforeForward):
        stage.prefill(list(range(3)), None, 64)
    assert observed[-1]["start"] == 64 and observed[-1]["token_count"] == 3
    assert observed[-1]["coverage_valid"] is True
    assert observed[-1]["frontier"] == 67


def test_prefill_receiver_role_mapping_valid_before_prepare(monkeypatch):
    """middle/last stages receive hidden rows; the same ordering proof via
    the received-hidden branch of prefill. The middle stage processed chunk
    1 first (chain order), so its frontier covers [0, 64) before chunk 2."""
    stage, pool, observed = _capture_prepare_stage(monkeypatch, role="middle")
    with pytest.raises(_StopBeforeForward):
        stage.prefill(None, torch.zeros(64, 8, dtype=torch.bfloat16), 0)
    with pytest.raises(_StopBeforeForward):
        stage.prefill(None, torch.zeros(3, 8, dtype=torch.bfloat16), 64)
    assert observed[-1]["start"] == 64 and observed[-1]["token_count"] == 3
    assert observed[-1]["coverage_valid"] is True
    assert observed[-1]["frontier"] == 67


def test_receiver_role_cannot_skip_chunk_one(monkeypatch):
    """Negative control: a receiver stage that somehow starts at a nonzero
    position (missed chunk 1) fails CLOSED — the gap check — rather than
    executing a prefix read through the sentinel mapping."""
    stage, pool, observed = _capture_prepare_stage(monkeypatch, role="middle")
    with pytest.raises(RuntimeError, match="gap"):
        stage.prefill(None, torch.zeros(3, 8, dtype=torch.bfloat16), 64)
    assert observed == []  # never reached prepare (no SWA op executed)


def test_decode_mapping_valid_before_prepare(monkeypatch):
    """Decode transition: position 66 after a 67-token prefix — the step
    reads [0, 66] and stores [66, 67); coverage valid at prepare time."""
    stage, pool, observed = _capture_prepare_stage(monkeypatch)
    stage._ensure_swa_session_mapping(67, 0)  # prefix established
    with pytest.raises(_StopBeforeForward):
        stage.decode(7, 66)
    assert observed[-1]["phase"] == "decode"
    assert observed[-1]["coverage_valid"] is True


def test_decode_receiver_role_mapping_valid_before_prepare(monkeypatch):
    stage, pool, observed = _capture_prepare_stage(monkeypatch, role="last")
    stage._ensure_swa_session_mapping(67, 0)
    hidden = torch.zeros(1, 8, dtype=torch.bfloat16)
    with pytest.raises(_StopBeforeForward):
        stage.decode(hidden, 66)
    assert observed[-1]["coverage_valid"] is True


def test_negative_control_delayed_allocation_is_caught(monkeypatch):
    """Negative control: with allocation REMOVED (the pre-#166 state), the
    same prepare-time checker must observe INVALID coverage — proving the
    checker can fail and the ordering proof is not vacuous."""
    stage, pool, observed = _capture_prepare_stage(monkeypatch)
    real = stage._ensure_swa_session_mapping
    stage._ensure_swa_session_mapping = lambda *a, **k: None  # allocation skipped
    with pytest.raises(_StopBeforeForward):
        stage.prefill(list(range(64)), None, 0)
    assert observed[0]["coverage_valid"] is False
    stage._ensure_swa_session_mapping = real


def test_negative_control_source_ordering_is_statically_pinned():
    """Every entry point allocates BEFORE _prepare, statically (so a future
    reorder cannot silently pass the runtime tests by stubbing)."""
    source = STAGE_RUNTIME.read_text()
    for entry, tail in (
        ("def prefill", "def decode"),
        ("def decode", "def logical_state_records"),
    ):
        body = source.split(entry, 1)[1].split(tail, 1)[0]
        ensures = [m.start() for m in re.finditer(r"self\._ensure_swa_session_mapping\(", body)]
        prepares = [m.start() for m in re.finditer(r"self\._prepare\(", body)]
        assert ensures, f"{entry}: no ensure call"
        assert prepares, f"{entry}: no prepare call"
        assert len(ensures) == len(prepares), entry
        for e, p in zip(ensures, prepares):
            assert e < p, f"{entry}: ensure must precede prepare"


# ---------------------------------------------------------------------------
# 2. Multi-chunk persistence / exactly-once / decode transition
# ---------------------------------------------------------------------------


def test_multi_chunk_preserves_one_mapping(monkeypatch):
    """first chunk (prefix_len==0) -> nonzero-prefix chunk -> later legal
    chunk: ONE mapping, slot identity stable, no re-allocation."""
    stage, pool = _stage_with_pool(monkeypatch)
    stage.reset_session_state()
    stage._ensure_swa_session_mapping(64, 0)
    before = pool.full_to_swa_index_mapping[:64].clone()
    available_after_chunk1 = pool.swa_available_size()

    stage._ensure_swa_session_mapping(3, 64)
    assert stage._swa_session_allocated == 67
    assert pool.swa_available_size() == available_after_chunk1 - 3
    assert torch.equal(pool.full_to_swa_index_mapping[:64], before)  # persisted

    stage._ensure_swa_session_mapping(13, 67)
    assert stage._swa_session_allocated == 80
    assert torch.equal(pool.full_to_swa_index_mapping[:64], before)
    assert _mapping_valid(pool, 80)


def test_allocation_is_exactly_once_per_position(monkeypatch):
    stage, pool = _stage_with_pool(monkeypatch)
    stage.reset_session_state()
    stage._ensure_swa_session_mapping(64, 0)
    stage._ensure_swa_session_mapping(3, 64)
    free = pool.swa_available_size()
    live = int((pool.full_to_swa_index_mapping[: pool.full_num_tokens] > 0).sum())
    assert live == 67
    # idempotent re-execution of an already-owned range allocates nothing
    stage._ensure_swa_session_mapping(64, 0)
    stage._ensure_swa_session_mapping(3, 64)
    assert pool.swa_available_size() == free
    assert int((pool.full_to_swa_index_mapping[: pool.full_num_tokens] > 0).sum()) == 67
    # each position's slot is DISTINCT (never aliased)
    slots = pool.full_to_swa_index_mapping[:67].tolist()
    assert len(set(slots)) == 67 and all(s > 0 for s in slots)


def test_decode_transition_extends_frontier_by_one(monkeypatch):
    stage, pool = _stage_with_pool(monkeypatch)
    stage.reset_session_state()
    stage._ensure_swa_session_mapping(67, 0)
    for position in (67, 68, 69):
        stage._ensure_swa_session_mapping(1, position)
    assert stage._swa_session_allocated == 70
    assert _mapping_valid(pool, 70)


def test_negative_control_all_alias_slot0_is_undetectable_no_more(monkeypatch):
    """The pre-#166 failure signature (every SWA op resolves to swa slot 0
    through the all-sentinel mapping) must FAIL the validity checker."""
    stage, pool = _stage_with_pool(monkeypatch)
    stage.reset_session_state()
    # no allocation: mapping is all-zero sentinel (the accepted defect state)
    assert not _mapping_valid(pool, 1)
    assert not _mapping_valid(pool, 64)


# ---------------------------------------------------------------------------
# 3. Reset / reuse / teardown / ownership conservation
# ---------------------------------------------------------------------------


def test_reset_releases_all_ownership(monkeypatch):
    stage, pool = _stage_with_pool(monkeypatch)
    stage.reset_session_state()
    stage._ensure_swa_session_mapping(67, 0)
    assert stage.swa_session_ownership_report()["live_mapped_slots"] == 67
    stage.reset_session_state()
    report = stage.swa_session_ownership_report()
    assert report["allocated"] == 0
    assert report["live_mapped_slots"] == 0
    assert report["available"] == pool.swa_num_tokens - 1  # full free-list back


def test_reset_does_not_leave_stale_mapping(monkeypatch):
    """Negative-control target: after reset the dense mapping is ALL
    sentinel — a stale entry surviving reset would be caught."""
    stage, pool = _stage_with_pool(monkeypatch)
    stage.reset_session_state()
    stage._ensure_swa_session_mapping(67, 0)
    stage.reset_session_state()
    m = pool.full_to_swa_index_mapping[: pool.full_num_tokens]
    assert int((m > 0).sum()) == 0
    # a fresh session's allocation cannot inherit the old slots
    stage._ensure_swa_session_mapping(64, 0)
    assert _mapping_valid(pool, 64)


def test_reuse_after_reset_gets_fresh_valid_lifecycle(monkeypatch):
    """Session A (67 tokens) then session B (53 tokens): B's mapping is
    valid over exactly its range and the free-list is conserved."""
    stage, pool = _stage_with_pool(monkeypatch)
    cap = pool.swa_num_tokens - 1
    stage.reset_session_state()
    stage._ensure_swa_session_mapping(64, 0)
    stage._ensure_swa_session_mapping(3, 64)
    stage.reset_session_state()
    stage._ensure_swa_session_mapping(53, 0)
    assert _mapping_valid(pool, 53)
    # ownership conservation (leak detector): free + live == capacity
    live = int((pool.full_to_swa_index_mapping[: pool.full_num_tokens] > 0).sum())
    assert live + pool.swa_available_size() == cap


def test_skipped_reset_is_observable_not_silent(monkeypatch):
    """If a driver skips reset between sessions, the ownership report must
    expose the stale frontier (>0 at the new session's start) — never a
    silently fresh-looking state."""
    stage, pool = _stage_with_pool(monkeypatch)
    stage.reset_session_state()
    stage._ensure_swa_session_mapping(67, 0)
    # driver bug: no reset; the report still shows the old ownership
    assert stage.swa_session_ownership_report()["allocated"] == 67


def test_double_reset_is_safe_by_contract(monkeypatch):
    stage, pool = _stage_with_pool(monkeypatch)
    stage.reset_session_state()
    stage._ensure_swa_session_mapping(64, 0)
    stage.reset_session_state()
    stage.reset_session_state()  # double teardown
    report = stage.swa_session_ownership_report()
    assert report["allocated"] == 0 and report["live_mapped_slots"] == 0
    stage._ensure_swa_session_mapping(64, 0)  # and the pool still works
    assert _mapping_valid(pool, 64)


def test_ownership_conservation_across_a_full_session(monkeypatch):
    """Teardown-leak negative-control target: across chunked prefill +
    decode + reset, free+live never deviates from capacity."""
    stage, pool = _stage_with_pool(monkeypatch)
    cap = pool.swa_num_tokens - 1

    def conserved():
        live = int((pool.full_to_swa_index_mapping[: pool.full_num_tokens] > 0).sum())
        return live + pool.swa_available_size() == cap

    stage.reset_session_state()
    stage._ensure_swa_session_mapping(64, 0)
    assert conserved()
    stage._ensure_swa_session_mapping(3, 64)
    assert conserved()
    for position in range(67, 80):
        stage._ensure_swa_session_mapping(1, position)
        assert conserved()
    stage.reset_session_state()
    assert conserved() and pool.swa_available_size() == cap


# ---------------------------------------------------------------------------
# 4. Fail-closed semantics
# ---------------------------------------------------------------------------


def test_exhaustion_fails_closed(monkeypatch):
    stage, pool = _stage_with_pool(monkeypatch, num_swa=5)  # 4 allocatable
    stage.reset_session_state()
    with pytest.raises(RuntimeError, match="fail-closed"):
        stage._ensure_swa_session_mapping(10, 0)
    # no partial state: frontier unmoved, nothing mapped, nothing consumed
    assert stage._swa_session_allocated == 0
    assert int((pool.full_to_swa_index_mapping > 0).sum()) == 0
    assert pool.swa_available_size() == 4


def test_exhaustion_mid_session_preserves_owned_range(monkeypatch):
    stage, pool = _stage_with_pool(monkeypatch, num_swa=66)  # 65 allocatable
    stage.reset_session_state()
    stage._ensure_swa_session_mapping(64, 0)
    with pytest.raises(RuntimeError):
        stage._ensure_swa_session_mapping(3, 64)  # needs 67, 1 short
    assert stage._swa_session_allocated == 64
    assert _mapping_valid(pool, 64)  # previously owned range still valid


def test_position_gap_fails_closed(monkeypatch):
    stage, pool = _stage_with_pool(monkeypatch)
    stage.reset_session_state()
    stage._ensure_swa_session_mapping(64, 0)
    with pytest.raises(RuntimeError, match="gap"):
        stage._ensure_swa_session_mapping(1, 70)  # 68..70 never owned
    assert stage._swa_session_allocated == 64


def test_full_capacity_session_fits_the_pool(monkeypatch):
    """The +1 sentinel sizing: a session using EVERY position of the
    runtime capacity owns a valid mapping — not one slot short."""
    stage, pool = _stage_with_pool(monkeypatch, num_full=256, num_swa=257)
    stage.reset_session_state()
    stage._ensure_swa_session_mapping(256, 0)
    assert _mapping_valid(pool, 256)
    stage._ensure_swa_session_mapping(1, 256) if pool.swa_available_size() else None


# ---------------------------------------------------------------------------
# 5. Non-SWA paths unaffected / single-chunk unchanged
# ---------------------------------------------------------------------------


def test_non_swa_pool_is_a_complete_no_op(monkeypatch):
    stage, pool = _stage_with_pool(monkeypatch, swa=False)
    stage.reset_session_state()
    stage._ensure_swa_session_mapping(64, 0)  # must not raise or allocate
    stage._ensure_swa_session_mapping(3, 64)
    assert stage.swa_session_ownership_report() == {"swa_paged": False, "allocated": 0}


def test_non_swa_execution_allocates_no_swa_state(monkeypatch):
    """A non-SWA pool exposes no alloc_swa surface at all; the lifecycle
    must not touch it (getattr-gated)."""
    stage, pool = _stage_with_pool(monkeypatch, swa=False)
    assert not hasattr(pool, "alloc_swa")
    stage.reset_session_state()
    stage._ensure_swa_session_mapping(64, 0)  # no AttributeError path


def test_single_chunk_behavior_semantically_unchanged(monkeypatch):
    """A <=capacity unit remains ONE chunk with exactly its own positions
    allocated — the accepted single-chunk execution contract (#153)."""
    from freetoken.research.prefill_partition import plan_prefill_partitions

    assert plan_prefill_partitions(53, 64) == [(0, 53)]
    stage, pool = _stage_with_pool(monkeypatch)
    stage.reset_session_state()
    stage._ensure_swa_session_mapping(53, 0)
    assert stage._swa_session_allocated == 53
    assert _mapping_valid(pool, 53)


def test_failing_population_partition_unchanged(monkeypatch):
    """65/66/67 stay 64+remainder (frozen 64-row contract untouched by the
    ownership lifecycle)."""
    from freetoken.research.prefill_partition import plan_prefill_partitions

    for rows in (65, 66, 67):
        assert plan_prefill_partitions(rows, 64) == [(0, 64), (64, rows - 64)]


# ---------------------------------------------------------------------------
# 6. Ordinary/control paths share the lifecycle (no bypass seams)
# ---------------------------------------------------------------------------


def _driver_sources():
    return {
        "stage_chain.py": REPO / "benchmarks/inferswarm_r6/stage_chain.py",
        "two_stage.py": REPO / "benchmarks/inferswarm_r6/two_stage.py",
        "last_stage_service.py": REPO / "benchmarks/inferswarm_r6/last_stage_service.py",
        "single_gpu_control.py": REPO / "benchmarks/inferswarm_r6/single_gpu_control.py",
        "r6_localization/run_single_arm.py": REPO / "benchmarks/inferswarm_r6_localization/run_single_arm.py",
        "r6_localization/run_stage1_diag.py": REPO / "benchmarks/inferswarm_r6_localization/run_stage1_diag.py",
        "issue97/last_stage_service.py": REPO / "benchmarks/inferswarm_97/last_stage_service.py",
        "issue97/reference_runner.py": REPO / "benchmarks/inferswarm_97/reference_runner.py",
        "issue110/last_stage_service.py": REPO / "benchmarks/inferswarm_110/last_stage_service.py",
    }


def test_every_r6_driver_goes_through_the_runtime_lifecycle():
    """No driver manipulates the swa mapping itself: the only FreeToken
    modules reaching the pool's allocator are the scheduler's CacheManager
    and stage_runtime's lifecycle. Direct (control/diagnostic) and ordinary
    (chain/service) paths therefore consume IDENTICAL semantics."""
    for name, path in _driver_sources().items():
        source = path.read_text()
        assert "alloc_swa" not in source, name
        assert "_init_swa_paged_state" not in source, name
        assert "full_to_swa" not in source, name
        assert "_ensure_swa_session_mapping" not in source, name


def test_chain_generate_resets_before_prefill():
    """The ordinary chain path opens each session with RESET (ownership
    release) before any PREFILL — statically pinned."""
    source = (REPO / "benchmarks/inferswarm_r6/stage_chain.py").read_text()
    body = source.split("def generate", 1)[1].split("def report", 1)[0]
    reset_pos = body.find('"op": "RESET"')
    prefill_pos = body.find("self._chain_prefill(")
    assert 0 <= reset_pos < prefill_pos


def test_alloc_swa_call_sites_are_exactly_the_two_lifecycle_owners():
    """Mechanical census over CALL SITES (AST, not text): pool.alloc_swa /
    swa_pool.alloc_swa is invoked only by CacheManager (scheduler) and
    stage_runtime (standalone lifecycle) — no parallel allocator seam."""
    callers = set()
    for root in (REPO / "python", REPO / "benchmarks"):
        for path in root.rglob("*.py"):
            if path.name == "hybrid_swa_pool.py":
                continue  # the definition (self-recursion-free)
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "alloc_swa"
                ):
                    callers.add(str(path.relative_to(REPO)))
    assert callers == {
        "python/freetoken/scheduler/cache.py",
        "benchmarks/inferswarm_r6/stage_runtime.py",
    }, callers


# ---------------------------------------------------------------------------
# 7. Scheduler-parity: the same underlying allocation semantic
# ---------------------------------------------------------------------------


def test_standalone_reaches_the_scheduler_allocation_semantic(monkeypatch):
    """Causal applicability (CPU-side half): given the same pool state, the
    standalone lifecycle's alloc_swa call produces EXACTLY the mapping the
    accepted #157 intervention produced — alloc_swa(arange(used_slots)) —
    and what CacheManager.allocate_paged produces per chunk (contiguous
    ascending positions, whole-range coverage)."""
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    _patch_tp(monkeypatch)
    reference = HybridSWAKVCache(
        groups=_specs(), num_layers=2, num_full_pages=256, page_size=1,
        dtype=torch.bfloat16, device=torch.device("cpu"), num_swa_tokens=257,
    )
    # the accepted #157 intervention shape: one alloc over arange(used)
    reference.alloc_swa(torch.arange(67, dtype=torch.int64))

    stage, pool = _stage_with_pool(monkeypatch)
    stage.reset_session_state()
    stage._ensure_swa_session_mapping(64, 0)
    stage._ensure_swa_session_mapping(3, 64)

    assert torch.equal(
        pool.full_to_swa_index_mapping[:67],
        reference.full_to_swa_index_mapping[:67],
    )


def test_scheduler_chunk_granularity_parity(monkeypatch):
    """The incremental frontier matches CacheManager.allocate_paged's
    per-chunk granularity: allocate exactly the new positions of each
    chunk, before the chunk runs, whole pages/positions exactly once."""
    source = (REPO / "python/freetoken/scheduler/cache.py").read_text()
    body = source.split("def allocate_paged", 1)[1].split("def cache_req", 1)[0]
    assert "self.swa_pool.alloc_swa(allocated)" in body  # atomic with pages
    stage, pool = _stage_with_pool(monkeypatch)
    stage.reset_session_state()
    # chunk sequence 64 + 3 + decode: allocation is per-chunk incremental
    stage._ensure_swa_session_mapping(64, 0)
    assert stage._swa_session_allocated == 64
    stage._ensure_swa_session_mapping(3, 64)
    assert stage._swa_session_allocated == 67


# ---------------------------------------------------------------------------
# 8. No-case-tuning / frozen-contract audit of the changed file
# ---------------------------------------------------------------------------


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


def test_no_case_tuning_in_stage_runtime():
    """The remediated runtime carries no case IDs, regime nouns, divergent
    token ids, holdout namespaces, or model-name special cases in its
    selection logic (docstrings stripped; #166 references are comments)."""
    tree = _strip_docstrings(ast.parse(STAGE_RUNTIME.read_text()))
    dumped = ast.dump(tree)
    for token in ("c109", "regime", "h109", "h95", "issue-117-case", "c74"):
        assert token not in dumped.lower(), token


def test_no_case_conditionals_gate_the_lifecycle():
    """The lifecycle branches only on pool capability (swa_paged), position
    arithmetic, and capacity — never on any case/session identifier."""
    tree = _strip_docstrings(ast.parse(STAGE_RUNTIME.read_text()))
    inside = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_ensure_swa_session_mapping", "_reset_swa_session_ownership", "_swa_pool",
        ):
            inside.append(node)
    assert inside
    for fn in inside:
        dump = ast.dump(fn)
        for bad in ("session_id", "case", "uid", "prompt"):
            assert bad not in dump.lower(), (fn.name, bad)


def test_frozen_constants_untouched():
    """The 64-row boundary contract, geometry, and boundary-plane constants
    are unchanged by the remediation."""
    source = STAGE_RUNTIME.read_text()
    assert "BOUNDARY_PLANES = 1" in source
    assert "HIDDEN_SIZE = 3840" in source
    strategy = (REPO / "benchmarks/inferswarm_r6/strategy.py").read_text()
    assert "PREFILL_CHUNK = 64" in strategy


def test_class_level_default_exists_for_stub_construction():
    """The ownership frontier has a class-level default so every instance
    (including object.__new__ stubs) carries the seam — regression guard
    for the _softcap_mode class-of-bugs."""
    assert GemmaDenseStage._swa_session_allocated == 0
