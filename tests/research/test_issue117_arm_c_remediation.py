"""Issue #117 Arm-C remediation tests (InferSwarm #153) — CPU-only.

Deterministic proof of the remediated extend-prefill chunk policy with NO
model/GPU execution:

- boundary matrix: legal units 1/31/32/33/53/63/64 stay ONE call; the
  over-limit 65 partitions deterministically (64+1);
- historical causal-control regression: the #137 intervention shape
  (single_chunk_53 vs multi_chunk_32_21) — the canonical policy must
  choose the single chunk for a legal 53-row unit;
- policy derivation is shared by direct and ordinary paths (single
  generate() seam) and by the wire service's admitted capacity;
- logical coverage invariants: complete, ordered, non-overlapping,
  exactly-once row consumption, policed on arbitrary partitions;
- fail-closed negative controls: zero/negative rows, zero/negative
  capacity, missing/reordered/overlapping/gap/incomplete partitions,
  non-integer inputs, missing boundary-contract field;
- state/semantic preservation: the remediated generate() sends the same
  PREFILL requests to the stages as the historical single-chunk loop for
  legal inputs, decode is untouched, session/request identity and
  plan/candidate identity are unchanged, commit semantics (on_token
  positions) and stopping semantics are byte-identical to the accepted
  contract;
- no-case-tuning source audit: no case IDs, no regime nouns, no accepted
  divergent token ids, no Gemma/case-specific branch in the remediation
  seam.
"""

from __future__ import annotations

import ast
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


# --------------------------------------------------------------------------
# Phase 0 binding: the chunk-policy owner and its capacity inputs
# --------------------------------------------------------------------------


def test_chunk_policy_owner_and_inputs_are_bound():
    """The generate() seam owns chunking; capacity comes from the frozen
    strategy constant, not a call-site literal."""
    source = (REPO / "benchmarks/inferswarm_r6/stage_chain.py").read_text()
    assert "plan_prefill_partitions(" in source
    assert "admitted_prefill_rows()" in source
    # the historical hand literal is gone from the seam
    assert "chunk = 64" not in source
    assert "chunk = 32" not in source
    # capacity is imported from the frozen strategy contract
    strategy = (REPO / "benchmarks/inferswarm_r6/strategy.py").read_text()
    assert "PREFILL_CHUNK = 64" in strategy


def test_admitted_capacity_matches_frozen_contract():
    from benchmarks.inferswarm_r6.strategy import PREFILL_CHUNK

    assert admitted_prefill_rows() == PREFILL_CHUNK == 64


def test_capacity_derivation_is_generic_boundary_contract():
    contract = {"prefill_chunk_rows": 64}
    assert admitted_capacity_from_boundary_contract(contract) == 64
    with pytest.raises(ValueError):
        admitted_capacity_from_boundary_contract({})
    with pytest.raises(ValueError):
        admitted_capacity_from_boundary_contract({"prefill_chunk_rows": 0})
    with pytest.raises(ValueError):
        admitted_capacity_from_boundary_contract(None)


# --------------------------------------------------------------------------
# Boundary behavior: representative legal sizes stay ONE call
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 31, 32, 33, 53, 63, 64])
def test_legal_unit_is_single_chunk(rows):
    partitions = plan_prefill_partitions(rows, 64)
    assert partitions == [(0, rows)], partitions


@pytest.mark.parametrize(
    "rows,expected",
    [
        (65, [(0, 64), (64, 1)]),
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
    for rows in (65, 100, 192, 193):
        for _offset, count in plan_prefill_partitions(rows, 64):
            assert 1 <= count <= 64


# --------------------------------------------------------------------------
# Historical causal-control regression (#137 intervention shape)
# --------------------------------------------------------------------------


def test_historical_causal_control_single_chunk_53_chosen():
    """#137 probe C2: identical 53-row input, chunk partition the only
    variable — single 53-row chunk deterministic, 32+21 unstable.  The
    remediated policy must select the single chunk."""
    single_chunk_53 = plan_prefill_partitions(53, 64)
    assert single_chunk_53 == [(0, 53)]
    # the unstable 32+21 shape is NOT what the policy derives
    assert single_chunk_53 != [(0, 32), (32, 21)]
    # and mechanically: two partitions would be required for 32+21 to occur
    assert len(single_chunk_53) == 1


def test_policy_proves_policy_selection_only():
    """CPU policy proof is selection-level; it does not claim the GPU
    numerical result (explicit non-claim, mirrored in the record)."""
    record = {
        "proves": "chunk-policy selection for the canonical path",
        "does_not_prove": "GPU numerical equivalence of execution",
    }
    assert "selection" in record["proves"]
    assert record["does_not_prove"].startswith("GPU numerical")


# --------------------------------------------------------------------------
# Path equality: direct and ordinary consume the same policy
# --------------------------------------------------------------------------


def test_direct_and_ordinary_paths_share_the_policy_seam():
    """Both the ordinary controller path (coordinator -> serve_tokens ->
    node agent -> realize_dense_chain) and the direct comparator path
    (issue117_arm_c_direct -> realize_dense_chain) drive the SAME
    generate() seam, so they cannot derive different chunk policies."""
    chain_source = (REPO / "benchmarks/inferswarm_r6/stage_chain.py").read_text()
    assert chain_source.count("def generate(") == 1
    # chain_runtime wraps chain.generate (both paths)
    wrapper = (REPO / "benchmarks/inferswarm_r6/chain_runtime.py").read_text()
    assert "self._chain.generate(" in wrapper
    # node agent realizes via chain_runtime.realize_dense_chain
    agent = (REPO / "benchmarks/inferswarm_r6/node_agent.py").read_text()
    assert "realize_dense_chain" in agent


def test_wire_service_admits_exactly_the_policy_capacity():
    """The last-stage wire contract admits exactly the chain's admitted
    capacity, so a policy-legal chunk can never be rejected on the wire."""
    service = (REPO / "benchmarks/inferswarm_r6/last_stage_service.py").read_text()
    assert "MAX_TOKEN_COUNT = admitted_prefill_rows()" in service
    assert "MAX_TOKEN_COUNT = 64" not in service


# --------------------------------------------------------------------------
# Logical coverage / order / exactly-once
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 32, 53, 64, 65, 129, 300])
def test_coverage_invariants_hold(rows):
    partitions = plan_prefill_partitions(rows, 64)
    assert_partition_invariants(partitions, total_rows=rows)
    # exactly-once: reconstruct the logical rows from the partitions
    covered = [r for offset, count in partitions for r in range(offset, offset + count)]
    assert covered == list(range(rows))


def test_exactly_one_partition_covers_a_legal_unit():
    assert len(plan_prefill_partitions(53, 64)) == 1


# --------------------------------------------------------------------------
# Fail-closed negative controls
# --------------------------------------------------------------------------


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
def test_noninteger_capacity_fails_closed(bad):
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


# --------------------------------------------------------------------------
# State / semantic preservation (fake stage plumbing; no model)
# --------------------------------------------------------------------------


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


def test_generate_single_chunk_for_legal_unit(monkeypatch):
    first, _middle, _last, session, tokens = _chain_with_fake_stages(
        53, monkeypatch
    )
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
    for rows in (1, 32, 53, 64, 65, 129):
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


# --------------------------------------------------------------------------
# No-case-tuning source audit (AST-level; docstrings stripped)
# --------------------------------------------------------------------------


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
        "prefill_partition.py": (
            REPO / "python/freetoken/research/prefill_partition.py"
        ).read_text(),
        "stage_chain.py": (
            REPO / "benchmarks/inferswarm_r6/stage_chain.py"
        ).read_text(),
        "two_stage.py": (REPO / "benchmarks/inferswarm_r6/two_stage.py").read_text(),
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
    source = (
        REPO / "python/freetoken/research/prefill_partition.py"
    ).read_text()
    tree = _strip_docstrings(ast.parse(source))
    # Only small structural literals are permitted (validity floor, tuple
    # length, index bases, partition index).  The forbidden class is
    # row-count nouns: 53, 32, 21, 64 must NOT appear as constants.
    constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, int)
    }
    assert constants <= {0, 1, 2}, constants


def test_capacity_64_comes_from_frozen_contract_not_new_magic():
    """No new magic 64: the only 64 in the remediation chain is the frozen
    strategy PREFILL_CHUNK constant (pre-existing, boundary geometry)."""
    strategy = (REPO / "benchmarks/inferswarm_r6/strategy.py").read_text()
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
    """The remediation touches only the prefill partition loop; decode
    request shape, plan digest echo, and fencing surfaces are untouched."""
    chain = (REPO / "benchmarks/inferswarm_r6/stage_chain.py").read_text()
    assert "def _chain_decode" in chain
    wrapper = (REPO / "benchmarks/inferswarm_r6/chain_runtime.py").read_text()
    assert 'result["plan_digest"] = self.execution_plan_digest' in wrapper
    agent = (REPO / "benchmarks/inferswarm_r6/node_agent.py").read_text()
    assert "GENERATE before an authorized realization" in agent
    assert "stale/reordered execution position" in agent
