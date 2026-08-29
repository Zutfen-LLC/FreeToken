"""The verdict firewall: the P5 package cannot emit a Phase-1 performance verdict.

The runner's complete output vocabulary is execution/validity states and per-arm
descriptive statistics. The decision vocabulary — and every cross-arm ratio that
would feed it — belongs to the P6 analysis of completed artifacts. These tests make
that a mechanical property of the code and its artifacts, not a promise.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from inferswarm_phase1 import campaign as campaign_mod
from inferswarm_phase1.campaign import (
    CampaignDefinition,
    CampaignSettings,
    SessionExecution,
)
from inferswarm_phase1.campaign_arms import baseline_b1_arm, candidate_v2_arm
from inferswarm_phase1.campaign_protocol import build_protocol

from .phase1_fakes import (
    INFERSWARM_SHA40,
    SHA40,
    install_clean_environment,
    install_mocked_server,
)

# The performance-decision vocabulary of the success criteria, and the quantities
# that feed it. None of these may appear as a key, value, or computed field anywhere
# in the P5 package's source or artifacts. (Prose that says the runner does NOT
# produce a campaign outcome is fine; the words themselves are not.)
FORBIDDEN_TERMS = (
    "GO",
    "ITERATE",
    "NO-GO",
    "NOGO",
    "R_agg",
    "R_c",
    "speedup",
    "speed_up",
)

_PACKAGE_DIR = Path(campaign_mod.__file__).parent
_CAMPAIGN_SOURCES = sorted(_PACKAGE_DIR.glob("campaign*.py"))

_FORBIDDEN_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(t).replace(r"\-", "-") for t in FORBIDDEN_TERMS) + r")\b"
)


def test_no_campaign_source_file_contains_the_decision_vocabulary():
    offenders = {}
    for path in _CAMPAIGN_SOURCES:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if _FORBIDDEN_PATTERN.search(line):
                offenders[f"{path.name}:{lineno}"] = line.strip()
    assert not offenders, offenders


def test_no_validity_state_is_a_decision_word():
    from inferswarm_phase1.campaign_validity import (
        EXECUTION_COMPLETE,
        EXECUTION_INCOMPLETE,
        VALIDITY_STATES,
    )

    assert set(VALIDITY_STATES) == {"VALID", "INVALID", "NON_CANONICAL_DEV_SMOKE"}
    assert EXECUTION_COMPLETE == "COMPLETE"
    assert EXECUTION_INCOMPLETE == "INCOMPLETE"


def test_the_cli_can_only_plan_validate_or_run_sessions():
    import argparse

    from inferswarm_phase1.campaign_cli import build_parser

    parser = build_parser()
    subparsers = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    assert set(subparsers.choices) == {"plan", "validate", "run-session"}


@pytest.fixture
def valid_session_document(monkeypatch, tmp_path):
    """One complete, valid canonical session executed against full fakes."""
    frozen = install_clean_environment(monkeypatch, tmp_path)
    settings = CampaignSettings(
        model_path=str(tmp_path / "model"),
        manifest_path=frozen["manifest"],
        model_revision=SHA40,
        placement_path=frozen["placement"],
        inferswarm_commit=INFERSWARM_SHA40,
        out_root=tmp_path / "runs",
        prerequisites_path=frozen["prerequisites"],
        echo_server_output=False,
    )
    definition = CampaignDefinition(
        arms=[baseline_b1_arm(), candidate_v2_arm()],
        protocol=build_protocol(warmups=None, repetitions=None, classes=None, dev_smoke=False),
        settings=settings,
        canonical=True,
    )
    install_mocked_server(monkeypatch)
    doc = SessionExecution(definition=definition, session_number=1).execute()
    assert doc["execution_status"] == "COMPLETE"
    assert doc["validity"] == "VALID"
    return doc


def test_no_artifact_of_a_complete_session_contains_a_decision_term(valid_session_document):
    root = Path(valid_session_document["run_directory"])
    offenders = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if _FORBIDDEN_PATTERN.search(line):
                offenders[f"{path.relative_to(root)}:{lineno}"] = line.strip()[:200]
    assert not offenders, offenders


def test_the_session_document_carries_no_cross_arm_arithmetic(valid_session_document):
    doc = valid_session_document

    def all_keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from all_keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from all_keys(v)

    keys = set(all_keys(doc))
    for ratio_key in ("ratio", "R_c", "R_agg", "speedup", "candidate_over_baseline"):
        assert ratio_key not in keys
    # both arms' raw numbers coexist; only a P6 analysis may divide them
    assert doc["kv_capacities_tokens"]["baseline_b1"] is not None
    assert doc["kv_capacities_tokens"]["candidate_v2"] is not None
    assert doc["no_verdict_note"]


def test_summaries_are_per_arm_only(valid_session_document):
    root = Path(valid_session_document["run_directory"])
    for arm_dir in root.iterdir():
        if not arm_dir.is_dir():
            continue
        summary = json.loads((arm_dir / "summary.json").read_text())
        assert set(summary["statistics"].keys()) <= {"W1", "W2", "W3", "W4"}
        for stats in summary["statistics"].values():
            assert set(stats.keys()) == {
                "label", "measured_generations", "decode_tok_s", "ttft_ms",
                "prefill_tok_s", "inter_token_ms", "prompt_output_counts", "note",
            }
    assert not list(root.glob("comparison*")), "no cross-arm comparison artifact may exist"


def test_no_loaded_campaign_module_defines_a_cross_arm_function():
    """No function in the package suggests or implements cross-arm arithmetic."""
    import inspect

    for module_name, module in sys.modules.items():
        if not module_name.startswith("inferswarm_phase1.campaign"):
            continue
        for name, member in vars(module).items():
            if not inspect.isfunction(member):
                continue
            segments = set(name.lower().split("_"))
            assert not segments & {
                "ratio", "speedup", "r", "c", "agg", "verdict", "bootstrap",
                "significance",
            }, (module_name, name)
