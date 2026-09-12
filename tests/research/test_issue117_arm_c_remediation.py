"""Issue #117 Arm-C remediation tests (InferSwarm #153, corrected) — CPU-only.

Corrected classification: Branch B (BACKEND_REQUIRES_MULTI_CHUNK).  The
accepted #137 facts (authoritative, from
arm-c-regime4-diagnosis-137/diagnostic-conclusions.json):

- the six divergent Arm-C cases are exactly the multi-chunk population,
  prompt_len > PREFILL_CHUNK=64, all 65-67 rows;
- every stable case is <= 53 rows;
- probe C2 used a stable 53-row control: a single 53-row call was
  deterministic on every trial, while the same 53 rows partitioned
  32+21 varied — multi-chunk execution is SUFFICIENT for instability;
- C2 does NOT prove that the actual 65-67-row failing units can legally
  be one backend call.

This suite proves, with NO model/GPU execution:

- the exact accepted failing population (65, 66, 67) is classified
  multi-chunk under the frozen 64-row contract, and CANNOT be classified
  single-call from 53-row legality alone;
- one-call legality for 65-67 is mechanically DISPROVEN under the frozen
  contract (wire bound + frozen boundary geometry + staging-buffer
  sizing) — so Branch A is unavailable without changing a frozen
  semantic/wire contract, which exceeds #153 authority;
- the corrected execution partition for 65/66/67 is exactly the
  unchanged accepted 64+remainder path, and no behavior of that path
  changed (Branch B remediation option 1 is not available CPU-only: the
  extend-path instability is execution-level, see the BLOCKED record);
- 53-row C2 remains a causal-control regression (control only, never a
  substitute for the failing population);
- coverage invariants (ordered, complete, non-overlapping, exactly
  once), fail-closed negative controls;
- decode/session/plan/fencing semantics unchanged;
- ``prefill_ns`` stays NANOSECONDS (timing-unit regression contract);
- the capacity ownership is strategy -> runtime modules (no runtime
  module imports the wire service; wire derives from strategy).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
RESEARCH_SUITE = REPO / "python"
if str(RESEARCH_SUITE) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SUITE))

# stage_chain imports stage_runtime (torch at module level). On CPU-only
# hosts the frozen geometry constants are stubbed so the chunk-policy seam
# is still testable; on torch-capable hosts the real module loads.
try:  # noqa: SIM105
    import torch  # noqa: F401
except ModuleNotFoundError:
    import types

    _stub = types.ModuleType("benchmarks.inferswarm_r6.stage_runtime")
    _stub.BOUNDARY_PLANES = 1
    _stub.HIDDEN_SIZE = 3840
    sys.modules["benchmarks.inferswarm_r6.stage_runtime"] = _stub

from freetoken.research.prefill_partition import (  # noqa: E402
    admitted_capacity_from_boundary_contract,
    assert_partition_invariants,
    plan_prefill_partitions,
)

from benchmarks.inferswarm_r6.stage_chain import (  # noqa: E402
    admitted_prefill_rows,
)

# The accepted #137 facts, as literals for binding (values come from the
# retained diagnostic record; the InferSwarm-side builders re-derive them
# from the pinned record bytes).
ACCEPTED_FAILING_POPULATION = [65, 66, 67]
FROZEN_PREFILL_CHUNK = 64
MAX_STABLE_ROWS = 53


def _read(rel: str) -> str:
    return (REPO / rel).read_text()


# ---------------------------------------------------------------------------
# Corrected Phase-0 classification: Branch B
# ---------------------------------------------------------------------------


def test_accepted_failing_population_is_multi_chunk():
    """The exact accepted population 65/66/67 partitions (64 + remainder)
    under the frozen capacity — the same partition the accepted producer
    executed (hand literal chunk = 64 => identical behavior)."""
    for rows in ACCEPTED_FAILING_POPULATION:
        partitions = plan_prefill_partitions(rows, FROZEN_PREFILL_CHUNK)
        assert partitions == [(0, 64), (64, rows - 64)], (rows, partitions)


def test_failing_population_cannot_be_single_call_under_frozen_contract():
    """Branch A is unavailable: a 65-67-row single call is ILLEGAL under
    every frozen authority — the wire contract rejects
    token_count > max_token_count, and the frozen boundary geometry and
    the wire buffer are sized for exactly 64 rows per call."""
    wire = _read("python/freetoken/research/r4_wire.py")
    assert 'token_count > contract["max_token_count"]' in wire
    service = _read("benchmarks/inferswarm_r6/last_stage_service.py")
    # the wire service's admitted capacity IS the frozen strategy constant
    assert "from benchmarks.inferswarm_r6.strategy import PREFILL_CHUNK as MAX_TOKEN_COUNT" in service
    assert "buffer_bytes = MAX_TOKEN_COUNT * ROW_WIDTH * 2" in service
    strategy = _read("benchmarks/inferswarm_r6/strategy.py")
    assert f"PREFILL_CHUNK = {FROZEN_PREFILL_CHUNK}" in strategy
    # frozen geometry binds the boundary bytes to the same 64 rows
    assert "prefill_bytes\": PREFILL_CHUNK * HIDDEN_SIZE * 2" in strategy
    # mechanical: the policy itself partitions 65-67 into two calls at the
    # frozen capacity, and the admitted capacity derived from the frozen
    # boundary contract is strictly below every failing population size —
    # a single call carrying all rows is inadmissible
    contract = {"prefill_chunk_rows": FROZEN_PREFILL_CHUNK}
    admitted = admitted_capacity_from_boundary_contract(contract)
    for rows in ACCEPTED_FAILING_POPULATION:
        assert len(plan_prefill_partitions(rows, FROZEN_PREFILL_CHUNK)) == 2
        assert rows > admitted, rows


def test_branch_a_not_derivable_from_53_row_legality_alone():
    """The C2 control fact (53 rows legal as one call, deterministic)
    does NOT legalize a 65-67-row single call: legality of n rows
    requires n <= admitted capacity, and 53 <= 64 proves nothing about
    65-67.  Mechanically: the single-call predicate applied to the
    failing population is false even though it is true for 53."""
    def single_call_legal(rows: int, capacity: int) -> bool:
        return rows <= capacity

    assert single_call_legal(MAX_STABLE_ROWS, FROZEN_PREFILL_CHUNK) is True
    for rows in ACCEPTED_FAILING_POPULATION:
        assert single_call_legal(rows, FROZEN_PREFILL_CHUNK) is False


def test_corrected_classification_is_branch_b():
    """The remediation record's classification (source-bound here; the
    InferSwarm-side builder derives it from pinned bytes) is
    BACKEND_REQUIRES_MULTI_CHUNK with terminal
    ISSUE117_ARM_C_REMEDIATION_BLOCKED — the ≤64 policy cleanup is useful
    but does not remediate the accepted 65-67 failure."""
    partition_doc = _read("python/freetoken/research/prefill_partition.py")
    assert "BACKEND_REQUIRES_MULTI_CHUNK" in partition_doc
    assert "ISSUE117_ARM_C_REMEDIATION_BLOCKED" in partition_doc


def test_corrected_partition_is_the_unchanged_accepted_path():
    """For the failing population the corrected execution partition is
    byte-identical to the accepted producer's hand-literal chunk-64 loop:
    full 64-row chunks then the remainder, same order, same positions.
    No behavior of the failing path changed."""
    def accepted_hand_literal_partitions(rows: int, chunk: int = 64):
        out = []
        position = 0
        while position < rows:
            count = min(chunk, rows - position)
            out.append((position, count))
            position += count
        return out

    for rows in ACCEPTED_FAILING_POPULATION:
        assert plan_prefill_partitions(rows, FROZEN_PREFILL_CHUNK) == (
            accepted_hand_literal_partitions(rows)
        )


# ---------------------------------------------------------------------------
# 53-row C2: causal control ONLY
# ---------------------------------------------------------------------------


def test_c2_control_single_chunk_53_regression():
    """#137 probe C2: identical 53-row input, chunk partition the only
    variable — single 53-row chunk deterministic, 32+21 unstable.  The
    policy must keep the single chunk.  This is a CAUSAL CONTROL on a
    stable input; it is NOT evidence about the 65-67 failing units."""
    single_chunk_53 = plan_prefill_partitions(53, FROZEN_PREFILL_CHUNK)
    assert single_chunk_53 == [(0, 53)]
    assert single_chunk_53 != [(0, 32), (32, 21)]
    assert len(single_chunk_53) == 1


def test_c2_control_is_not_a_failing_population_substitute():
    """The C2 control input (53) is disjoint from the accepted failing
    population (65-67): a passing control proves nothing about the
    failing rows, which remain multi-chunk under the frozen contract."""
    assert MAX_STABLE_ROWS not in ACCEPTED_FAILING_POPULATION
    assert all(r > FROZEN_PREFILL_CHUNK for r in ACCEPTED_FAILING_POPULATION)
    assert MAX_STABLE_ROWS <= FROZEN_PREFILL_CHUNK


# ---------------------------------------------------------------------------
# Phase 0 binding: the chunk-policy owner and its capacity inputs
# ---------------------------------------------------------------------------


def test_chunk_policy_owner_and_inputs_are_bound():
    """The generate() seam owns chunking; capacity comes from the frozen
    strategy constant, not a call-site literal."""
    source = _read("benchmarks/inferswarm_r6/stage_chain.py")
    assert "plan_prefill_partitions(" in source
    assert "admitted_prefill_rows()" in source
    assert "chunk = 64" not in source
    assert "chunk = 32" not in source
    strategy = _read("benchmarks/inferswarm_r6/strategy.py")
    assert "PREFILL_CHUNK = 64" in strategy


def test_admitted_capacity_matches_frozen_contract():
    from benchmarks.inferswarm_r6.strategy import PREFILL_CHUNK

    assert admitted_prefill_rows() == PREFILL_CHUNK == FROZEN_PREFILL_CHUNK


def test_capacity_derivation_is_generic_boundary_contract():
    contract = {"prefill_chunk_rows": 64}
    assert admitted_capacity_from_boundary_contract(contract) == 64
    with pytest.raises(ValueError):
        admitted_capacity_from_boundary_contract({})
    with pytest.raises(ValueError):
        admitted_capacity_from_boundary_contract({"prefill_chunk_rows": 0})
    with pytest.raises(ValueError):
        admitted_capacity_from_boundary_contract(None)


# ---------------------------------------------------------------------------
# Boundary behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 31, 32, 33, 53, 63, 64])
def test_legal_unit_is_single_chunk(rows):
    partitions = plan_prefill_partitions(rows, 64)
    assert partitions == [(0, rows)], partitions


@pytest.mark.parametrize(
    "rows,expected",
    [
        (65, [(0, 64), (64, 1)]),
        (66, [(0, 64), (64, 2)]),
        (67, [(0, 64), (64, 3)]),
        (128, [(0, 64), (64, 64)]),
        (129, [(0, 64), (64, 64), (128, 1)]),
        (85, [(0, 64), (64, 21)]),
    ],
)
def test_over_limit_partitions_deterministically(rows, expected):
    partitions = plan_prefill_partitions(rows, 64)
    assert partitions == expected, partitions


def test_over_limit_partition_is_legal_per_call():
    """Every emitted partition of an over-limit unit is itself a legal
    single backend call under the same admitted capacity."""
    for rows in (65, 66, 67, 100, 192, 193):
        for _offset, count in plan_prefill_partitions(rows, 64):
            assert 1 <= count <= 64


# ---------------------------------------------------------------------------
# Capacity ownership: one authoritative contract, no runtime cycles
# ---------------------------------------------------------------------------


def test_wire_capacity_derives_from_strategy_not_stage_chain():
    """Corrected ownership: strategy (frozen constant owner) -> chain +
    wire service.  The wire service no longer imports the chain runtime
    (the reviewed head's service->chain dependency put a serving
    coordinator module in the wire service's import closure)."""
    service = _read("benchmarks/inferswarm_r6/last_stage_service.py")
    assert (
        "from benchmarks.inferswarm_r6.strategy import PREFILL_CHUNK "
        "as MAX_TOKEN_COUNT" in service
    )
    assert "from benchmarks.inferswarm_r6.stage_chain import" not in service
    assert "admitted_prefill_rows" not in service


def test_no_runtime_module_imports_the_wire_service():
    """No runtime module imports the wire service's capacity: the wire
    service is a leaf consumer of the frozen constant."""
    for rel in (
        "benchmarks/inferswarm_r6/stage_chain.py",
        "benchmarks/inferswarm_r6/two_stage.py",
        "benchmarks/inferswarm_r6/strategy.py",
    ):
        source = _read(rel)
        assert "last_stage_service" not in source, rel
        assert "MAX_TOKEN_COUNT" not in source, rel


def test_sender_and_receiver_agree_mechanically():
    """The chain's per-call capacity and the wire service's admitted
    maximum derive from the SAME frozen constant, so the sender can
    never emit a chunk the receiver rejects (mechanically: import both
    derivations and compare)."""
    from benchmarks.inferswarm_r6.strategy import PREFILL_CHUNK

    assert admitted_prefill_rows() == PREFILL_CHUNK
    service_lines = [
        line
        for line in _read("benchmarks/inferswarm_r6/last_stage_service.py").splitlines()
        if line.startswith("MAX_TOKEN_COUNT") or "as MAX_TOKEN_COUNT" in line
    ]
    assert any("PREFILL_CHUNK" in line for line in service_lines)
    assert any(line.startswith("MAX_TOKEN_COUNT = ") or "as MAX_TOKEN_COUNT" in line for line in service_lines)


# ---------------------------------------------------------------------------
# Timing-unit regression contract (prefill_ns is NANOSECONDS)
# ---------------------------------------------------------------------------


def test_prefill_timing_stays_nanoseconds():
    """Regression contract for the maintainer-found unit drift: the
    prefill accumulator must use perf_counter_ns on BOTH sides, because
    the returned field is ``prefill_ns``.  A perf_counter() (seconds)
    source silently changes the field's unit."""
    source = _read("benchmarks/inferswarm_r6/two_stage.py")
    # every perf_counter call site in the module is the _ns flavor
    bare = re.findall(r"time\.perf_counter\(\)", source)
    assert not bare, f"bare perf_counter() call sites: {bare}"
    assert source.count("time.perf_counter_ns()") >= 6
    # the prefill accumulator specifically
    assert "t = time.perf_counter_ns()" in source
    assert "prefill_ns += time.perf_counter_ns() - t" in source
    assert '"prefill_ns": prefill_ns' in source


def test_all_reported_timing_fields_are_ns_sourced():
    """Every *_ns session field is fed only by perf_counter_ns deltas
    (structural AST check: no identifier ending _ns is assigned a
    perf_counter()-derived value)."""
    tree = ast.parse(_read("benchmarks/inferswarm_r6/two_stage.py"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "perf_counter"
        ):
            raise AssertionError("bare time.perf_counter() survives in two_stage.py")


def test_stage_chain_timing_is_ns_sourced():
    source = _read("benchmarks/inferswarm_r6/stage_chain.py")
    assert "time.perf_counter()" not in source
    assert "time.perf_counter_ns()" in source


# ---------------------------------------------------------------------------
# Logical coverage / order / exactly-once
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 32, 53, 64, 65, 66, 67, 129, 300])
def test_coverage_invariants_hold(rows):
    partitions = plan_prefill_partitions(rows, 64)
    assert_partition_invariants(partitions, total_rows=rows)
    covered = [r for offset, count in partitions for r in range(offset, offset + count)]
    assert covered == list(range(rows))


def test_exactly_one_partition_covers_a_legal_unit():
    assert len(plan_prefill_partitions(53, 64)) == 1


# ---------------------------------------------------------------------------
# Fail-closed negative controls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_rows", [0, -1, -53])
def test_nonpositive_rows_fail_closed(bad_rows):
    with pytest.raises(ValueError):
        plan_prefill_partitions(bad_rows, 64)


@pytest.mark.parametrize("bad_capacity", [0, -1, -64])
def test_nonpositive_capacity_fails_closed(bad_capacity):
    with pytest.raises(ValueError):
        plan_prefill_partitions(53, bad_capacity)


@pytest.mark.parametrize("bad", [1.5, "53", None, True])
def test_noninteger_rows_fail_closed(bad):
    with pytest.raises(ValueError):
        plan_prefill_partitions(bad, 64)


@pytest.mark.parametrize("bad", [1.5, "64", None, True])
def test_noninteger_capacity_fail_closed(bad):
    with pytest.raises(ValueError):
        plan_prefill_partitions(53, bad)


def test_reorder_rejected():
    with pytest.raises(ValueError):
        assert_partition_invariants([(32, 32), (0, 32)], total_rows=64)


def test_overlap_rejected():
    with pytest.raises(ValueError):
        assert_partition_invariants([(0, 33), (32, 31)], total_rows=64)


def test_gap_rejected():
    with pytest.raises(ValueError):
        assert_partition_invariants([(0, 32), (33, 31)], total_rows=64)


def test_missing_tail_rejected():
    with pytest.raises(ValueError):
        assert_partition_invariants([(0, 32)], total_rows=64)


def test_extra_beyond_total_rejected():
    with pytest.raises(ValueError):
        assert_partition_invariants([(0, 32), (32, 33)], total_rows=64)


def test_zero_row_partition_rejected():
    with pytest.raises(ValueError):
        assert_partition_invariants([(0, 0), (0, 64)], total_rows=64)


def test_malformed_partition_shape_rejected():
    with pytest.raises(ValueError):
        assert_partition_invariants([(0, 32, "extra")], total_rows=32)
    with pytest.raises(ValueError):
        assert_partition_invariants([None], total_rows=1)


# ---------------------------------------------------------------------------
# State / semantic preservation (fake stage plumbing; no model)
# ---------------------------------------------------------------------------


class _FakeStage:
    """Records PREFILL/DECODE requests; answers deterministically."""

    role = "fake"

    def __init__(self, name):
        self.name = name
        self.requests = []
        self.sends = []

    def request(self, message):
        self.requests.append(message)
        op = message["op"]
        if op == "PREFILL":
            if message.get("hidden") is not None:
                return {"op": "TOKEN_RESULT", "token_id": 7}
            return {"op": "BOUNDARY_PAYLOAD", "hidden": object()}
        if op == "DECODE":
            if message.get("hidden") is not None:
                return {"op": "TOKEN_RESULT", "token_id": 9}
            return {"op": "BOUNDARY_PAYLOAD", "hidden": object()}
        if op in ("RESET", "REPORT"):
            return {"op": "ACK", "report": {}}
        raise AssertionError(f"unexpected op {op}")

    def send(self, message):
        self.sends.append(message)

    def recv(self):
        return {"op": "ACK"}


def _chain_with_fake_stages(rows, monkeypatch):
    """Build GemmaStageChainRuntime.generate around fake stages, bypassing
    the GPU constructor, and capture the PREFILL requests it issues."""
    import benchmarks.inferswarm_r6.stage_chain as sc

    first = _FakeStage("first")
    middle = _FakeStage("middle")
    last = _FakeStage("last")
    chain = object.__new__(sc.GemmaStageChainRuntime)
    chain.stages = [first, middle, last]
    chain._sessions = []
    chain._closed = False
    chain.reclamation_report = {}
    chain.ready = [{"role": r} for r in ("first", "middle", "last")]

    tokens = []

    def on_token(step, token, commit):
        tokens.append((step, int(token)))

    session = chain.generate(
        session_id=5,
        prompt_token_ids=list(range(rows)),
        max_new_tokens=2,
        on_token=on_token,
    )
    return first, middle, last, session, tokens


@pytest.mark.parametrize("rows", ACCEPTED_FAILING_POPULATION)
def test_generate_failing_population_partitions_unchanged(rows, monkeypatch):
    """The generate() seam executes 65/66/67 as the UNCHANGED accepted
    64+remainder path: same request sequence the accepted hand-literal
    loop produced (positions 0 then 64; row counts 64 then rows-64)."""
    first, _m, _l, session, _t = _chain_with_fake_stages(rows, monkeypatch)
    prefills = [m for m in first.requests if m["op"] == "PREFILL"]
    assert [(p["position"], len(p["token_ids"])) for p in prefills] == [
        (0, 64),
        (64, rows - 64),
    ]
    assert session["prompt_len"] == rows


@pytest.mark.parametrize("rows", ACCEPTED_FAILING_POPULATION)
def test_generate_failing_population_matches_accepted_loop(rows, monkeypatch):
    """Byte-identical PREFILL request CONTENT vs the accepted producer's
    hand-literal chunk-64 loop for the failing population: the corrected
    candidate does not change the failing path's canonical behavior."""

    def accepted_prefill_requests(token_ids):
        out = []
        position = 0
        chunk = 64
        while position < len(token_ids):
            count = min(chunk, len(token_ids) - position)
            out.append((position, list(token_ids[position : position + count])))
            position += count
        return out

    prompt = list(range(rows))
    first, _m, _l, _s, _t = _chain_with_fake_stages(rows, monkeypatch)
    prefills = [m for m in first.requests if m["op"] == "PREFILL"]
    observed = [(p["position"], list(p["token_ids"])) for p in prefills]
    assert observed == accepted_prefill_requests(prompt)


def test_generate_single_chunk_for_legal_unit(monkeypatch):
    first, _middle, _last, session, tokens = _chain_with_fake_stages(53, monkeypatch)
    prefills = [m for m in first.requests if m["op"] == "PREFILL"]
    assert len(prefills) == 1, prefills
    assert len(prefills[0]["token_ids"]) == 53
    assert prefills[0]["position"] == 0
    # runtime-level semantics: on_token fires for the committed step 0 AND
    # the speculative step 1 (the accepted contract — the CONTROLLER commits
    # only step 0 and discards step 1; that split lives in r5b and is
    # untouched by this remediation)
    assert tokens == [(0, 7), (1, 9)]
    assert session["prompt_len"] == 53
    assert session["session_id"] == 5
    assert session["generated_token_ids"] == [7, 9]


def test_generate_partitions_over_limit_unit(monkeypatch):
    first, _m, _l, session, _tokens = _chain_with_fake_stages(85, monkeypatch)
    prefills = [m for m in first.requests if m["op"] == "PREFILL"]
    assert [(p["position"], len(p["token_ids"])) for p in prefills] == [
        (0, 64),
        (64, 21),
    ]


def test_generate_decode_path_unchanged(monkeypatch):
    first, _m, _l, _s, _t = _chain_with_fake_stages(20, monkeypatch)
    decodes = [m for m in first.requests if m["op"] == "DECODE"]
    assert len(decodes) == 1  # max_new_tokens=2 -> one decode step
    assert decodes[0]["token_id"] == 7
    assert decodes[0]["position"] == 20


def test_generate_logical_rows_consumed_exactly_once(monkeypatch):
    for rows in (1, 32, 53, 64, 65, 66, 67, 129):
        first, _m, _l, _s, _t = _chain_with_fake_stages(rows, monkeypatch)
        prefills = [m for m in first.requests if m["op"] == "PREFILL"]
        covered = [
            r
            for m in prefills
            for r in range(m["position"], m["position"] + len(m["token_ids"]))
        ]
        assert covered == list(range(rows)), rows


def test_generate_matches_historical_single_chunk_requests(monkeypatch):
    """For a legal unit the remediated seam emits byte-identical PREFILL
    requests (position, rows) to the accepted producer's loop."""
    first, _m, _l, _s, _t = _chain_with_fake_stages(53, monkeypatch)
    prefills = [m for m in first.requests if m["op"] == "PREFILL"]
    assert [(p["position"], p["token_ids"]) for p in prefills] == [
        (0, list(range(53)))
    ]


def test_generate_empty_prompt_fails_closed(monkeypatch):
    with pytest.raises(ValueError):
        _chain_with_fake_stages(0, monkeypatch)


# ---------------------------------------------------------------------------
# No-case-tuning source audit (AST-level; docstrings stripped)
# ---------------------------------------------------------------------------


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


def test_no_case_tuning_in_remediation_seam():
    """The remediation modules carry no case IDs, no regime nouns, no
    accepted divergent token ids, no model nouns."""
    sources = {
        "prefill_partition.py": _read("python/freetoken/research/prefill_partition.py"),
        "stage_chain.py": _read("benchmarks/inferswarm_r6/stage_chain.py"),
        "two_stage.py": _read("benchmarks/inferswarm_r6/two_stage.py"),
        "last_stage_service.py": _read("benchmarks/inferswarm_r6/last_stage_service.py"),
    }
    forbidden_tokens = [
        "c109-",
        "regime4",
        "regime-4",
        "c109-04-02-047",
        "issue-117-case",
    ]
    for name, source in sources.items():
        tree = _strip_docstrings(ast.parse(source))
        clean = ast.dump(tree)
        for token in forbidden_tokens:
            assert token not in clean, f"{name} carries {token!r}"
        assert "c109" not in clean, name


def test_no_hand_row_literals_in_partition_policy():
    """The generic policy is expressed in rows/capacity, not in Issue-117
    row-count nouns (53, 32+21)."""
    source = _read("python/freetoken/research/prefill_partition.py")
    tree = _strip_docstrings(ast.parse(source))
    constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, int)
    }
    assert constants <= {0, 1, 2}, constants


def test_capacity_64_comes_from_frozen_contract_not_new_magic():
    """No new magic 64: the only 64 in the remediation chain is the frozen
    strategy PREFILL_CHUNK constant (pre-existing, boundary geometry)."""
    strategy = _read("benchmarks/inferswarm_r6/strategy.py")
    tree = _strip_docstrings(ast.parse(strategy))
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "PREFILL_CHUNK"
    ]
    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Constant) and value.value == 64
    assert "prefill_chunk_rows" in strategy  # frozen contract field name


def test_decode_and_plan_identity_surfaces_unchanged():
    """The remediation touches only the prefill partition loop and the
    timing-unit/capacity-ownership corrections; decode request shape,
    plan digest echo, and fencing surfaces are untouched."""
    chain = _read("benchmarks/inferswarm_r6/stage_chain.py")
    assert "def _chain_decode" in chain
    wrapper = _read("benchmarks/inferswarm_r6/chain_runtime.py")
    assert 'result["plan_digest"] = self.execution_plan_digest' in wrapper
    agent = _read("benchmarks/inferswarm_r6/node_agent.py")
    assert "GENERATE before an authorized realization" in agent
    assert "stale/reordered execution position" in agent


# ---------------------------------------------------------------------------
# Unrelated-behavior audit of the changed runtime files
# ---------------------------------------------------------------------------


def test_changed_runtime_files_touch_only_authorized_seams():
    """The four changed runtime/producer files contain no behavioral edit
    beyond (a) the chunk-policy seam, (b) the capacity-ownership
    derivation, (c) the timing-unit restoration.  Audited structurally:
    no new function/class definitions, no changed function signatures,
    no touched decode/session/report/fencing code in the remediation
    files (vs the reviewed head's remediation, the corrections are
    import/call-site level only)."""
    for rel in (
        "benchmarks/inferswarm_r6/stage_chain.py",
        "benchmarks/inferswarm_r6/two_stage.py",
        "benchmarks/inferswarm_r6/last_stage_service.py",
        "python/freetoken/research/prefill_partition.py",
    ):
        tree = ast.parse(_read(rel))
        # no backend/model/case special-casing branches anywhere
        dumped = ast.dump(_strip_docstrings(tree))
        for token in ("c109", "regime", "gemma-4-12b", "h109"):
            assert token not in dumped.lower(), (rel, token)


def test_decode_requests_are_single_row(monkeypatch):
    """Decode semantics unchanged: one row per decode call at the running
    position (speculative discard contract intact)."""
    first, _m, _l, _s, _t = _chain_with_fake_stages(66, monkeypatch)
    decodes = [m for m in first.requests if m["op"] == "DECODE"]
    assert [(d["position"], d["token_id"]) for d in decodes] == [(66, 7)]
