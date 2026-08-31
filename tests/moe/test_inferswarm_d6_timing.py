from __future__ import annotations

import ast
import inspect
import textwrap

from freetoken.moe.inferswarm_d5_compact_routes import (
    D6_GPU0_MARKERS,
    D6_WORKER_MARKERS,
    InferSwarmD5CompactRoutesExecutor,
)


def _tree(method):
    return ast.parse(textwrap.dedent(inspect.getsource(method)))


def test_d6_events_cover_frozen_component_boundaries():
    assert set(D6_GPU0_MARKERS) == {
        "complete_start", "classify_end", "payload_stage_start", "payload_stage_end",
        "local_start", "local_end", "fanin_start", "fanin_end", "returned_h2d_start",
        "returned_h2d_end", "scatter_start", "scatter_end", "reduce_start", "reduce_end",
        "complete_end",
    }
    assert set(D6_WORKER_MARKERS) == {
        "branch_start", "inbound_start", "inbound_end", "compute_start", "compute_end",
        "outbound_start", "outbound_end", "branch_end",
    }


def test_d6_events_are_constructed_only_during_executor_initialization():
    init = inspect.getsource(InferSwarmD5CompactRoutesExecutor.__init__)
    assert "Event(enable_timing=True)" in init
    assert "Event(" not in inspect.getsource(InferSwarmD5CompactRoutesExecutor.decode)
    assert "Event(" not in inspect.getsource(InferSwarmD5CompactRoutesExecutor._worker_branch)


def test_d6_decode_has_no_host_value_extraction_or_synchronization():
    tree = _tree(InferSwarmD5CompactRoutesExecutor.decode)
    # D5 retains worker synchronization only in its exception-cleanup path.
    forbidden = {"item", "cpu", "tolist", "elapsed_time"}
    assert not [node.attr for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and node.attr in forbidden]
    source = inspect.getsource(InferSwarmD5CompactRoutesExecutor.decode)
    assert source.index("except Exception:") < source.index(".synchronize()")


def test_d6_snapshot_uses_only_event_pairs_from_one_event_dictionary():
    source = inspect.getsource(InferSwarmD5CompactRoutesExecutor.d6_diagnostic_snapshot)
    assert "self._d6_gpu0_events" in source
    assert "self._d6_" in source
    assert "elapsed_time" not in source  # centralized same-dictionary helper
    helper = inspect.getsource(InferSwarmD5CompactRoutesExecutor._elapsed)
    assert "events[start]" in helper and "events[end]" in helper


def test_instrumentation_off_path_remains_explicit():
    tree = _tree(InferSwarmD5CompactRoutesExecutor.decode)
    assert any(isinstance(node, ast.If) for node in ast.walk(tree))
    assert "else:" in inspect.getsource(InferSwarmD5CompactRoutesExecutor.decode)
