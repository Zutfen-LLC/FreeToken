import pytest

from freetoken.research.r5a_serving import checkpoint_identity_from_gate


def test_r5a_reads_checkpoint_identity_from_retained_r4_gate_schema():
    identity = {"revision": "frozen", "identical_across_nodes": True}
    gate = {
        "result": "ALL_PREFLIGHT_CHECKS_PASSED",
        "checks": {"checkpoint_identity": identity},
    }
    assert checkpoint_identity_from_gate(gate) == identity


def test_r5a_refuses_missing_or_unsuccessful_checkpoint_gate():
    with pytest.raises(ValueError, match="unsuccessful"):
        checkpoint_identity_from_gate({"result": "FAILED", "checks": {}})
    with pytest.raises(ValueError, match="lacks checkpoint"):
        checkpoint_identity_from_gate(
            {"result": "ALL_PREFLIGHT_CHECKS_PASSED", "checks": {}}
        )
