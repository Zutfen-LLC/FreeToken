"""The authoritative campaign-validity gate, and the summary that reports it.

The rule this file pins down: a Phase-0 artifact must never be able to say

    canonical: true / status: COMPLETE / label: MEASURED

while a campaign-invalidating condition is recorded inside it.
"""

from __future__ import annotations

import json

import pytest

from inferswarm_phase0 import validity as V
from inferswarm_phase0.artifacts import RunWriter
from inferswarm_phase0.protocol import BlockTally


# --- the verdict --------------------------------------------------------------------------

def test_a_canonical_campaign_with_no_invalidations_is_valid():
    assert V.CampaignValidity(canonical_intent=True).verdict() == V.VALIDITY_VALID


def test_one_invalidation_makes_a_canonical_campaign_invalid():
    state = V.CampaignValidity(canonical_intent=True)
    state.add(V.PREFILL_MISSING, "no fresh prefill record", arm_id="B1", class_id="W2")
    assert state.verdict() == V.VALIDITY_INVALID


def test_a_smoke_run_is_non_canonical_rather_than_invalid():
    """It was never claiming to be a baseline, so its invalidations are observations."""
    state = V.CampaignValidity(canonical_intent=False)
    state.add(V.PREFILL_DISABLED, "instrumentation off")
    assert state.verdict() == V.VALIDITY_NON_CANONICAL


def test_a_canonical_blocker_downgrades_to_non_canonical_not_valid():
    """--allow-missing-provenance leaves the repetition protocol untouched; the campaign is
    still not a canonical one."""
    state = V.CampaignValidity(canonical_intent=True)
    state.canonical_blockers.append("--allow-missing-provenance")
    assert state.verdict() == V.VALIDITY_NON_CANONICAL


def test_invalidations_are_structured_and_located():
    state = V.CampaignValidity(canonical_intent=True)
    state.add(V.PROMPT_SHAPE_VIOLATION, "too long", arm_id="B2", class_id="W3", execution_index=17)
    record = state.record()["campaign_invalidations"][0]
    assert record == {
        "code": V.PROMPT_SHAPE_VIOLATION,
        "message": "too long",
        "arm_id": "B2",
        "class_id": "W3",
        "execution_index": 17,
    }


def test_codes_are_deduplicated_but_every_occurrence_is_kept():
    state = V.CampaignValidity(canonical_intent=True)
    for index in range(3):
        state.add(V.PREFILL_MISSING, "gone", arm_id="B1", execution_index=index)
    assert state.codes == [V.PREFILL_MISSING]
    assert len(state.record()["campaign_invalidations"]) == 3


@pytest.mark.parametrize(
    "execution,verdict,expected",
    [
        (V.EXECUTION_COMPLETE, V.VALIDITY_VALID, "VALID CANONICAL CAMPAIGN"),
        (V.EXECUTION_COMPLETE, V.VALIDITY_INVALID, "INVALID CANONICAL ATTEMPT"),
        (V.EXECUTION_COMPLETE, V.VALIDITY_NON_CANONICAL, "NON-CANONICAL DEVELOPER RUN"),
        (V.EXECUTION_INCOMPLETE, V.VALIDITY_VALID, "INCOMPLETE RUN"),
        (V.EXECUTION_INCOMPLETE, V.VALIDITY_INVALID, "INCOMPLETE RUN"),
    ],
)
def test_the_headline_is_one_of_four_unambiguous_states(execution, verdict, expected):
    assert V.headline(execution, verdict) == expected


# --- what reaches the artifact ---------------------------------------------------------------

def _write(tmp_path, state, *, complete=True):
    tally = BlockTally("B1", "W2", expected_measured=1)
    if complete:
        tally.observed_measured = 1
    writer = RunWriter(tmp_path / "run", {"protocol": {"session_id": "s", "canonical": True}})
    return writer.finalize([tally], state)


def test_an_invalid_campaign_never_presents_itself_as_a_complete_success(tmp_path):
    state = V.CampaignValidity(canonical_intent=True)
    state.add(V.BENCH_BW_FAILED, "`ft bench bw` returned 1")
    doc = _write(tmp_path, state)
    # execution and validity are separate answers, and the second one is not flattering
    assert doc["execution_status"] == V.EXECUTION_COMPLETE
    assert doc["validity"] == V.VALIDITY_INVALID
    assert doc["headline"] == "INVALID CANONICAL ATTEMPT"
    assert doc["campaign_invalidation_codes"] == [V.BENCH_BW_FAILED]
    # there is no bare `status` key a reader could mistake for the verdict
    assert "status" not in doc


def test_measured_labels_an_observation_never_the_campaign(tmp_path):
    state = V.CampaignValidity(canonical_intent=True)
    state.add(V.GPU_MISMATCH, "ran on another card")
    doc = _write(tmp_path, state)
    assert doc["label"] == "MEASURED"
    assert "MEASURED describes an observation, never the campaign" in doc["label_note"]
    assert doc["validity"] == V.VALIDITY_INVALID


def test_the_summary_leads_with_the_verdict_and_lists_the_reasons(tmp_path):
    state = V.CampaignValidity(canonical_intent=True)
    state.add(V.PREFILL_AMBIGUOUS, "two fresh records", arm_id="B2", class_id="W1",
              execution_index=9)
    _write(tmp_path, state)
    summary = (tmp_path / "run" / "SUMMARY.md").read_text()
    lines = summary.splitlines()
    assert lines[0] == "# INVALID CANONICAL ATTEMPT"
    assert "Campaign-invalidating conditions (1)" in summary
    assert V.PREFILL_AMBIGUOUS in summary
    assert "B2/W1 #9" in summary


def test_the_summary_says_valid_only_when_it_is(tmp_path):
    doc = _write(tmp_path, V.CampaignValidity(canonical_intent=True))
    summary = (tmp_path / "run" / "SUMMARY.md").read_text()
    assert summary.splitlines()[0] == "# VALID CANONICAL CAMPAIGN"
    assert "every precommitted prerequisite" in summary
    assert doc["campaign_invalidations"] == []


def test_an_incomplete_run_leads_with_incompleteness(tmp_path):
    state = V.CampaignValidity(canonical_intent=True)
    doc = _write(tmp_path, state, complete=False)
    summary = (tmp_path / "run" / "SUMMARY.md").read_text()
    assert doc["headline"] == "INCOMPLETE RUN"
    assert summary.splitlines()[0] == "# INCOMPLETE RUN"
    assert "no verdict about the configurations can be read from it" in summary


def test_the_run_json_leads_with_the_verdict(tmp_path):
    state = V.CampaignValidity(canonical_intent=True)
    state.add(V.DIRTY_WORKING_TREE, "dirty")
    _write(tmp_path, state)
    doc = json.loads((tmp_path / "run" / "run.json").read_text())
    # a `head` of the file shows the verdict before any number
    assert list(doc)[:3] == ["schema", "headline", "execution_status"]


def test_a_pipe_in_a_message_cannot_break_the_summary_table(tmp_path):
    state = V.CampaignValidity(canonical_intent=True)
    state.add(V.RUNTIME_CONFIG_MISSING_FIELD, "field 'a|b' is null")
    _write(tmp_path, state)
    summary = (tmp_path / "run" / "SUMMARY.md").read_text()
    assert r"a\|b" in summary
