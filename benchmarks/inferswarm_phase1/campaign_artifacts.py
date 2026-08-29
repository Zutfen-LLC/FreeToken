"""Session artifact layout, raw per-generation records, and derived summaries.

Layout (local only; nothing is written into the InferSwarm repository):

    <out-root>/
      campaign-plan.json          the whole two-session plan (the ``plan`` subcommand)
      session-1/
        plan.json                 this session's plan: every expected generation
        provenance.json           preflight provenance + prerequisites + thermal record
        baseline_b1/
          startup.json            launch/ready timestamps, M-start duration, command
          runtime.json            runtime report + validation record
          W1.jsonl .. W4.jsonl    ONE LINE PER GENERATION, warmups tagged, failures kept
          block-mechanism-W*.json candidate only: per-block instrumentation window
          summary.json            per-block descriptive statistics
          server.log
        candidate_v2/ ...
        session-summary.json      execution order, block identities, completion, checks
      session-2/ ...

Rules inherited from the Phase-0 artifact discipline:

* raw generations are the artifact: every inter-token gap is preserved so a later
  analysis can compute variance, medians and bootstrap intervals;
* a successful-looking summary is impossible while generations are missing:
  ``execution_status`` is computed from expected-vs-observed counts, never asserted;
* failed generations are preserved in place (tagged ``failed``) — never deleted, never
  replaced by an invisible retry;
* summaries contain per-arm descriptive values only; no cross-arm ratio and no
  campaign verdict exists anywhere in this package's output.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferswarm_phase0.artifacts import block_stats

PLAN_SCHEMA = "inferswarm.phase1.session-plan/1"
PROVENANCE_SCHEMA = "inferswarm.phase1.session-provenance/1"
REPETITION_SCHEMA = "inferswarm.phase1.repetition/1"
STARTUP_SCHEMA = "inferswarm.phase1.arm-startup/1"
RUNTIME_SCHEMA = "inferswarm.phase1.arm-runtime/1"
MECHANISM_SCHEMA = "inferswarm.phase1.block-mechanism/1"
ARM_SUMMARY_SCHEMA = "inferswarm.phase1.arm-summary/1"
SESSION_SUMMARY_SCHEMA = "inferswarm.phase1.session-summary/1"
CAMPAIGN_PLAN_SCHEMA = "inferswarm.phase1.campaign-plan/1"


def _write_json(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class BlockTally:
    """Expected vs observed generations for one (arm, class, block).

    ``complete`` is computed from the counts and failures, never asserted by the
    writer — an incomplete block cannot look healthy in any summary.
    """

    arm_id: str
    class_id: str
    block_id: str
    expected_warmups: int
    expected_measured: int
    observed_warmups: int = 0
    observed_measured: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    rerun_reason: str | None = None
    discarded: bool = False

    @property
    def complete(self) -> bool:
        return (
            self.observed_measured == self.expected_measured
            and self.observed_warmups == self.expected_warmups
            and not self.failures
            and not self.discarded
        )

    def record(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "arm_id": self.arm_id,
            "class_id": self.class_id,
            "expected_warmups": self.expected_warmups,
            "observed_warmups": self.observed_warmups,
            "expected_measured": self.expected_measured,
            "observed_measured": self.observed_measured,
            "failures": list(self.failures),
            "rerun_reason": self.rerun_reason,
            "discarded": self.discarded,
            "complete": self.complete,
        }


class SessionWriter:
    """Writes one session's self-contained result directory.

    Directories are created lazily on first write: a session refused at preflight
    (boundary unproven, refusals outstanding) leaves no empty result directory that
    could be mistaken for an attempt.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._generations: list[dict[str, Any]] = []

    def _ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    # ---- per-session files ----------------------------------------------------------

    def write_plan(self, plan_doc: Mapping[str, Any]) -> Path:
        path = self._ensure_root() / "plan.json"
        _write_json(path, {"schema": PLAN_SCHEMA, **plan_doc})
        return path

    def write_provenance(self, provenance_doc: Mapping[str, Any]) -> Path:
        path = self._ensure_root() / "provenance.json"
        _write_json(path, {"schema": PROVENANCE_SCHEMA, **provenance_doc})
        return path

    def write_session_summary(self, doc: Mapping[str, Any]) -> Path:
        path = self._ensure_root() / "session-summary.json"
        _write_json(path, {"schema": SESSION_SUMMARY_SCHEMA, **doc})
        return path

    # ---- per-arm files ---------------------------------------------------------------

    def arm_dir(self, arm_id: str) -> Path:
        path = self._ensure_root() / arm_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def server_log_path(self, arm_id: str) -> Path:
        return self.arm_dir(arm_id) / "server.log"

    def write_startup(self, arm_id: str, doc: Mapping[str, Any]) -> Path:
        path = self.arm_dir(arm_id) / "startup.json"
        _write_json(path, {"schema": STARTUP_SCHEMA, **doc})
        return path

    def write_runtime(self, arm_id: str, doc: Mapping[str, Any]) -> Path:
        path = self.arm_dir(arm_id) / "runtime.json"
        _write_json(path, {"schema": RUNTIME_SCHEMA, **doc})
        return path

    def class_jsonl_path(self, arm_id: str, class_id: str) -> Path:
        return self.arm_dir(arm_id) / f"{class_id}.jsonl"

    def write_generation(self, arm_id: str, record: Mapping[str, Any]) -> None:
        """Append one generation — warmups and failures included, tagged, never dropped."""
        line = {"schema": REPETITION_SCHEMA, **record}
        self._generations.append(dict(line))
        with self.class_jsonl_path(arm_id, str(record["class_id"])).open(
            "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(line) + "\n")

    def write_block_mechanism(
        self, arm_id: str, class_id: str, doc: Mapping[str, Any]
    ) -> Path:
        path = self.arm_dir(arm_id) / f"block-mechanism-{class_id}.json"
        _write_json(path, {"schema": MECHANISM_SCHEMA, **doc})
        return path

    def write_arm_summary(self, arm_id: str, doc: Mapping[str, Any]) -> Path:
        path = self.arm_dir(arm_id) / "summary.json"
        _write_json(path, {"schema": ARM_SUMMARY_SCHEMA, **doc})
        return path

    # ---- reads -----------------------------------------------------------------------

    def generations(self) -> list[dict[str, Any]]:
        return list(self._generations)

    def measured(self, arm_id: str | None = None) -> list[dict[str, Any]]:
        return [
            g
            for g in self._generations
            if g.get("measured") and (arm_id is None or g.get("arm_id") == arm_id)
        ]

    def artifact_sha256_index(self) -> dict[str, str]:
        """SHA-256 of every file in the session directory, keyed by relative path."""
        index: dict[str, str] = {}
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                index[str(path.relative_to(self.root))] = _sha256_file(path)
        return index


def summarize_block(reps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-(arm, class) descriptive statistics — never a cross-arm comparison.

    Permitted per-arm values only (min/median/max/IQR/CV and derived percentiles of
    the per-repetition token latencies), computed after a block is complete; nothing
    here can affect whether later blocks execute.
    """
    measured = [r for r in reps if r.get("measured") and not r.get("failed")]
    decode = block_stats([r.get("decode_tok_s") for r in measured])
    ttft = block_stats([r.get("ttft_ms") for r in measured])
    prefill = block_stats(
        [(r.get("prefill") or {}).get("prefill_tok_s") for r in measured]
    )
    p50 = block_stats([r.get("inter_token_ms_p50") for r in measured])
    p95 = block_stats([r.get("inter_token_ms_p95") for r in measured])
    pmax = block_stats([r.get("inter_token_ms_max") for r in measured])
    return {
        "label": "CALCULATED",
        "measured_generations": len(measured),
        "decode_tok_s": decode,
        "ttft_ms": ttft,
        "prefill_tok_s": prefill,
        "inter_token_ms": {
            "p50_of_p50": p50.get("median"),
            "p95_of_p95": p95.get("median"),
            "max_of_max": pmax.get("max"),
        },
        "prompt_output_counts": {
            "prompt_tokens": sorted(
                {r.get("prompt_tokens") for r in measured}, key=str
            ),
            "completion_tokens": sorted(
                {r.get("completion_tokens") for r in measured}, key=str
            ),
        },
        "note": (
            "per-arm descriptive values computed after block completion; no cross-arm "
            "ratio is computed by this runner"
        ),
    }


def baseline_noise_floor(tallies: Sequence[BlockTally], reps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Baseline CV vs the frozen 5% noise-floor ceiling, per class — reported, not gating.

    This is a status record for the session summary (the P6 analysis applies the
    re-run rule); the runner itself never re-runs anything automatically.
    """
    per_class: dict[str, Any] = {}
    for tally in tallies:
        block_reps = [
            r
            for r in reps
            if r.get("arm_id") == tally.arm_id
            and r.get("class_id") == tally.class_id
            and r.get("measured")
            and not r.get("failed")
        ]
        stats = block_stats([r.get("decode_tok_s") for r in block_reps])
        cv = stats.get("cv_percent")
        per_class[tally.class_id] = {
            "cv_percent": cv,
            "n": stats.get("n", 0),
            "within_5_percent_ceiling": (cv is None or cv <= 5.0),
        }
    return {
        "rule": "baseline CV > 5% in any class means the environment is not quiet "
        "enough; the campaign is re-run after identifying the source (criteria "
        "section 10). Recorded here as status; the runner never re-runs on its own.",
        "per_class": per_class,
        "all_within_ceiling": all(
            v["within_5_percent_ceiling"] for v in per_class.values()
        )
        if per_class
        else None,
    }


def execution_status(tallies: Sequence[BlockTally], failure_count: int) -> str:
    from .campaign_validity import EXECUTION_COMPLETE, EXECUTION_INCOMPLETE

    incomplete = [t for t in tallies if not t.complete]
    return (
        EXECUTION_COMPLETE
        if not incomplete and not failure_count
        else EXECUTION_INCOMPLETE
    )
