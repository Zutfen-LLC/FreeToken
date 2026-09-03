"""R6 result-composer negative controls and positive fixture.

The composer (docs/inferswarm_r6/compose_result.py) must fail closed on:
missing stage-3 evidence, fetched-byte mismatch, nonzero unexplained host
mirror, unexpected checkpoint keys, missing logit comparator, logit
threshold violation, and NaN/Inf in distributed logits — and must
reproduce all canonical checks from retained/synthetic evidence in the
positive case.

Each control copies the REAL retained evidence directory into a tmp dir
(cheap: a few MB), applies exactly one mutation, and asserts the
composer exits nonzero with the expected failed check.  The current
retained evidence itself carries one honest failure (the frozen
secondary comparator threshold); the positive fixture patches the
reference top-32 values to the distributed rows to prove the composer
can emit a full PASS when — and only when — the evidence genuinely
passes.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / "docs/inferswarm_r6"
COMPOSER = EVIDENCE / "compose_result.py"

pytestmark = pytest.mark.skipif(
    not (EVIDENCE / "lifecycle/last-stage-final-report.json").exists(),
    reason="retained canonical evidence not present in this checkout",
)


def load_composer():
    spec = importlib.util.spec_from_file_location("r6_compose_result", COMPOSER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def evidence_copy(tmp_path):
    target = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, target)
    return target


def run_composer(evidence_dir: Path) -> dict:
    module = load_composer()
    old_argv = sys.argv
    sys.argv = ["compose_result.py", str(evidence_dir)]
    try:
        code = module.main()
    finally:
        sys.argv = old_argv
    result = json.loads((evidence_dir / "result.json").read_text())
    result["_exit_code"] = code
    return result


def mutate_last_stage_report(evidence_dir: Path, mutate) -> None:
    path = evidence_dir / "lifecycle/last-stage-final-report.json"
    report = json.loads(path.read_text())
    mutate(report)
    path.write_text(json.dumps(report))


# --------------------------------------------------------------------------
# negative controls
# --------------------------------------------------------------------------

def test_missing_stage3_report_fails(evidence_copy):
    (evidence_copy / "lifecycle/last-stage-final-report.json").unlink()
    result = run_composer(evidence_copy)
    assert result["_exit_code"] == 1
    assert "stage_evidence_present_and_producer_bound" in result["failed_checks"]
    assert "stage_fetch_matches_frozen_plan" in result["failed_checks"]


def test_fetched_byte_mismatch_fails(evidence_copy):
    def bump(report):
        report["runtime"]["fetched_bytes"] += 1
    mutate_last_stage_report(evidence_copy, bump)
    result = run_composer(evidence_copy)
    assert result["_exit_code"] == 1
    assert "stage_fetch_matches_frozen_plan" in result["failed_checks"]


def test_nonzero_unexplained_host_mirror_fails(evidence_copy):
    def mirror(report):
        report["runtime"]["unexplained_persistent_host_mirror_bytes"] = 4096
    mutate_last_stage_report(evidence_copy, mirror)
    result = run_composer(evidence_copy)
    assert result["_exit_code"] == 1
    assert "no_unexplained_host_mirror" in result["failed_checks"]


def test_unexpected_checkpoint_key_fails(evidence_copy):
    def extra_key(report):
        report["runtime"]["unexpected_checkpoint_keys"] = ["model.bogus.weight"]
    mutate_last_stage_report(evidence_copy, extra_key)
    result = run_composer(evidence_copy)
    assert result["_exit_code"] == 1
    assert "stage_selective_load_clean" in result["failed_checks"]


def test_missing_logit_comparator_fails(evidence_copy):
    (evidence_copy / "lifecycle/distributed-logits-0-1-7.f32.bin").unlink()
    result = run_composer(evidence_copy)
    assert result["_exit_code"] == 1
    assert "secondary_logit_comparator_retained" in result["failed_checks"]


def test_logit_threshold_violation_fails(evidence_copy):
    """The retained canonical evidence HONESTLY violates the frozen 0.25
    threshold (aggregate 0.515625); the composer must report it."""
    result = run_composer(evidence_copy)
    assert result["_exit_code"] == 1
    assert "secondary_logit_comparator_threshold" in result["failed_checks"]
    comparator = result["detail"]["secondary_comparator"]
    assert comparator["aggregate_max_absdiff"] >= comparator["threshold"]
    assert result["verdict"] == "R6_DENSE_ARCHITECTURE_FALSIFICATION_FAIL"


def test_nan_inf_in_distributed_logits_fails(evidence_copy):
    """NaN outside the compared domain keeps absdiffs finite but must
    still fail the frozen NaN/Inf policy."""
    module = load_composer()
    blob_path = evidence_copy / "lifecycle/distributed-logits-0-1-7.f32.bin"
    blob = bytearray(blob_path.read_bytes())
    # poison one value in step 0's row far from any top-32 index
    struct.pack_into("<f", blob, 4 * 262000, float("nan"))
    blob_path.write_bytes(bytes(blob))
    result = run_composer(evidence_copy)
    assert result["_exit_code"] == 1
    comparator = result["detail"]["secondary_comparator"]
    assert comparator["nan_inf_count"] >= 1
    assert comparator["passes"] is False
    assert "secondary_logit_comparator_threshold" in result["failed_checks"]


def test_producer_substitution_in_last_stage_report_fails(evidence_copy):
    """A last-stage report NOT bound to the canonical producer must be
    rejected even if every number inside it is perfect."""
    def swap_producer(report):
        report["producer_freetoken_sha"] = "0" * 40
    mutate_last_stage_report(evidence_copy, swap_producer)
    result = run_composer(evidence_copy)
    assert result["_exit_code"] == 1
    assert "stage_evidence_present_and_producer_bound" in result["failed_checks"]


# --------------------------------------------------------------------------
# positive fixture
# --------------------------------------------------------------------------

def test_positive_fixture_all_checks_pass(evidence_copy):
    """Patch the retained reference top-32 values to equal the retained
    distributed rows (synthetic passing comparator): the composer must
    then derive 28/28 PASS with distinct provenance identities — proving
    the FAIL verdict above comes from the comparator evidence, not from a
    composer that can never pass."""
    reference_path = evidence_copy / "reference-generation.json"
    reference = json.loads(reference_path.read_text())
    blob = (evidence_copy / "lifecycle/distributed-logits-0-1-7.f32.bin").read_bytes()
    rows = {}
    for i, step in enumerate(("0", "1", "7")):
        rows[step] = list(
            struct.unpack_from("<262144f", blob, i * 262144 * 4)
        )
    for step, record in reference["step_top32_logits"].items():
        record["top_values"] = [
            rows[step][index] for index in record["top_indices"]
        ]
    reference_path.write_text(json.dumps(reference))
    result = run_composer(evidence_copy)
    assert result["_exit_code"] == 0
    assert result["failed_checks"] == []
    assert len(result["checks"]) >= 28
    assert result["verdict"] == "R6_DENSE_ARCHITECTURE_FALSIFICATION_PASS_CANDIDATE"
    provenance = result["provenance"]
    # distinct producers are never flattened
    assert provenance["canonical_physical_producer"] == (
        "44d6c94e4fd2ee967451cc959f930883ca3f4a25"
    )
    assert provenance["canonical_last_stage_final_report_producer"] == (
        "44d6c94e4fd2ee967451cc959f930883ca3f4a25"
    )
    assert provenance["secondary_comparator_arm_producer"] not in (None, "44d6c94e4fd2ee967451cc959f930883ca3f4a25")
    assert provenance["secondary_comparator_arm_mode"] == (
        "EXPLICIT_OVERRIDE_EVIDENCE_ARM"
    )


def test_composer_never_passes_check_by_constant(evidence_copy):
    """Structural: no check is assigned a bare True literal (regression
    guard for the removed placeholder checks).  Comparisons whose RESULT
    is assigned (e.g. ``x is True``) are fine; only constant passing is
    rejected."""
    import re

    source = COMPOSER.read_text()
    pattern = re.compile(r"checks\[[^\]]+\]\s*=\s*True\s*$")
    for number, line in enumerate(source.splitlines(), start=1):
        stripped = line.split("#")[0].rstrip()
        assert not pattern.match(stripped), (
            f"constant-passing check at line {number}: {stripped}"
        )
