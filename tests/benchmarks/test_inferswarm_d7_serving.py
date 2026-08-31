from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from inferswarm_d7.serving_screen import classify
from freetoken.moe.inferswarm_d6_count_aware_transport import InferSwarmD6CountAwareTransportExecutor


def test_frozen_d7_classification_boundaries():
    assert classify(1.08, .90, .50) == "D7_FANIN_SPARSE_STRONG"
    assert classify(1.03, .85, .49) == "D7_FANIN_SPARSE_PROMISING"
    assert classify(1.03, .849, 1.0) == "D7_FANIN_SPARSE_PARTIAL"
    assert classify(1.029, .99, 1.0) == "D7_FANIN_SPARSE_NOT_SUPPORTED"
    assert classify(.969, .99, 1.0) == "D7_FANIN_SPARSE_HARMFUL"


def test_d7_joint_diagnostic_preserves_no_host_read_or_sync_in_decode():
    tree = ast.parse(textwrap.dedent(inspect.getsource(InferSwarmD6CountAwareTransportExecutor.decode)))
    assert not [node.attr for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and node.attr in {"item", "cpu", "tolist", "elapsed_time"}]
    parameter = inspect.signature(InferSwarmD6CountAwareTransportExecutor.__init__).parameters[
        "d7_participation_diagnostics"]
    assert parameter.default is False
