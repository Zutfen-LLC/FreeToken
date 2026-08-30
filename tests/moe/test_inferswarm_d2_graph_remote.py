from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import torch

from freetoken.moe.inferswarm_d2_graph_remote import (
    D2_DEPENDENCY,
    D2_TOPOLOGY,
    FANOUT_SHAPE,
    InferSwarmD2GraphRemoteExecutor,
    absent_d2_graph_remote_report,
    build_local_fallback_ids,
)


def test_local_fallback_is_owned_by_gpu0_for_every_layer():
    placement = SimpleNamespace(
        num_experts=6,
        per_layer=(
            SimpleNamespace(layer_id=0, expert_ids=(0, 2, 4)),
            SimpleNamespace(layer_id=1, expert_ids=(1, 3, 5)),
        ),
    )
    fallback = build_local_fallback_ids(placement, torch.device("cpu"))
    assert fallback.tolist() == [1, 0]
    for layer, expert in zip(placement.per_layer, fallback.tolist(), strict=True):
        assert expert not in layer.expert_ids


def test_absent_report_never_claims_graph_or_eager_fallback():
    report = absent_d2_graph_remote_report()
    assert report["enabled"] is False
    assert report["gpu0_graph_active"] is False
    assert report["gpu1_graph_active"] is False
    assert report["eager_gpu0_fallback"] is False
    assert report["steady_state_host_sync_count"] == 0


def test_success_path_has_no_host_sync_and_failure_cleanup_does():
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(InferSwarmD2GraphRemoteExecutor.decode))
    )
    sync_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"synchronize", "item", "query"}
    ]
    assert len(sync_calls) == 1
    except_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]
    assert any(sync_calls[0] in set(ast.walk(handler)) for handler in except_nodes)


def test_graph_provenance_constants_are_explicit():
    assert D2_TOPOLOGY == "unified_multidevice_whole_model_graph"
    assert D2_DEPENDENCY == "cuda_capture_internal_cross_device_event_fork_join"
    assert FANOUT_SHAPE == "CONCURRENT_BOUNDED"


def test_graph_state_refuses_silent_eager_fallback():
    executor = object.__new__(InferSwarmD2GraphRemoteExecutor)
    executor._capture_complete = False
    executor._graph_recapture_count = 0
    executor.device_counts = torch.zeros((1, 4), dtype=torch.int64)
    for captured in ([], [2], [1, 2]):
        try:
            executor.set_graph_state(captured)
        except RuntimeError as exc:
            assert "silent eager fallback" in str(exc)
        else:
            raise AssertionError(f"unexpectedly accepted {captured}")
    executor.set_graph_state([1])
    assert executor._capture_complete is True
    assert executor._captured_bs == (1,)
