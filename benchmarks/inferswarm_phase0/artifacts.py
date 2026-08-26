"""Run-directory layout, raw artifacts, and a summary that cannot hide a shortfall.

Layout (created locally; **nothing here is written into the InferSwarm repository**):

    <out-root>/<YYYY-MM-DD>-<session-id>-<short-name>/
        run.json            provenance, configuration, protocol, per-arm resolved config
        repetitions.jsonl   ONE LINE PER GENERATION, warmups included and tagged
        failures.jsonl      every failed generation, with its reason
        SUMMARY.md          human-readable; states expected vs observed reps per block
        server-logs/        one ft serve log per arm

Two rules shape this module:

* **Raw repetitions are the artifact.** ``repetitions.jsonl`` keeps every measured
  generation with its full inter-token gap list, so variance, CV, medians and a bootstrap
  can be computed later. Averages are never emitted in place of the data.
* **A successful-looking summary must be impossible while repetitions are missing.** The
  run status is computed from expected-vs-observed counts and the failure list, so an
  incomplete campaign says ``INCOMPLETE`` at the top of both ``run.json`` and ``SUMMARY.md``.

Copying a completed run into ``inferswarm/docs/benchmarks/results/YYYY-MM-DD-short-name/``
is a deliberate human step, done once the numbers have been read.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from . import REPETITION_SCHEMA, RUN_SCHEMA

STATUS_COMPLETE = "COMPLETE"
STATUS_INCOMPLETE = "INCOMPLETE"


def run_dir_name(date: str, session_id: str, short_name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in short_name).strip("-")
    session = "".join(c if c.isalnum() or c in "-_" else "-" for c in session_id).strip("-")
    return f"{date}-{session}-{safe}" if safe else f"{date}-{session}"


@dataclass
class RunWriter:
    root: Path
    header: Dict[str, Any]
    _reps: List[Dict[str, Any]] = field(default_factory=list)
    _failures: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "server-logs").mkdir(exist_ok=True)
        self._reps_path = self.root / "repetitions.jsonl"
        self._failures_path = self.root / "failures.jsonl"
        # Truncate on open so a re-run into the same directory cannot blend two campaigns.
        self._reps_path.write_text("")
        self._failures_path.write_text("")

    def server_log_path(self, arm_id: str) -> Path:
        return self.root / "server-logs" / f"{arm_id}.log"

    def write_repetition(self, record: Dict[str, Any]) -> None:
        """Append one generation. Called for warmups too -- tagged, never dropped."""
        record = {"schema": REPETITION_SCHEMA, **record}
        self._reps.append(record)
        with self._reps_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def write_failure(self, record: Dict[str, Any]) -> None:
        self._failures.append(record)
        with self._failures_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    @property
    def failures(self) -> List[Dict[str, Any]]:
        return list(self._failures)

    def measured(self) -> List[Dict[str, Any]]:
        return [r for r in self._reps if r.get("measured")]

    def finalize(self, tallies: Sequence[Any], extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        tally_records = [t.record() for t in tallies]
        incomplete = [t for t in tally_records if not t["complete"]]
        status = STATUS_COMPLETE if (not incomplete and not self._failures) else STATUS_INCOMPLETE
        doc = {
            "schema": RUN_SCHEMA,
            "status": status,
            "label": "MEASURED" if status == STATUS_COMPLETE else "INCOMPLETE",
            "label_note": (
                "Per InferSwarm BENCHMARKING.md every number carries a label. Raw "
                "repetitions in repetitions.jsonl are MEASURED observations of this "
                "configuration; the per-block statistics in SUMMARY.md are CALCULATED from "
                "them. This harness computes no cross-configuration ratio and selects no "
                "baseline."
            ),
            **self.header,
            **(extra or {}),
            "blocks": tally_records,
            "incomplete_blocks": [
                f"{t['arm_id']}/{t['class_id']}: {t['observed_measured']}/{t['expected_measured']} "
                f"measured, {len(t['failures'])} failure(s)"
                for t in incomplete
            ],
            "failure_count": len(self._failures),
            "measured_repetition_count": len(self.measured()),
            "artifacts": {
                "repetitions": self._reps_path.name,
                "failures": self._failures_path.name,
                "summary": "SUMMARY.md",
                "server_logs": "server-logs/",
            },
        }
        (self.root / "run.json").write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
        (self.root / "SUMMARY.md").write_text(render_summary(doc, self.measured()))
        return doc


def block_stats(values: Sequence[float]) -> Dict[str, Any]:
    """Descriptive statistics for ONE (arm, class) block -- never a comparison.

    Criteria section 10 requires min / median / max / IQR / CV per block. Reported here as
    CALCULATED values over the preserved raw repetitions; the bootstrap CIs and the
    per-class ratios the criteria also require belong to the analysis step, after the
    campaign is complete, and are deliberately not computed by the runner.
    """
    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        return {"n": 0}
    ordered = sorted(clean)
    n = len(ordered)
    median = statistics.median(ordered)
    mean = statistics.fmean(ordered)
    stdev = statistics.stdev(ordered) if n > 1 else 0.0
    q1 = statistics.median(ordered[: n // 2]) if n > 1 else ordered[0]
    q3 = statistics.median(ordered[(n + 1) // 2:]) if n > 1 else ordered[0]
    return {
        "n": n,
        "min": ordered[0],
        "median": median,
        "max": ordered[-1],
        "iqr": q3 - q1,
        "mean": mean,
        "stdev": stdev,
        "cv_percent": (stdev / mean * 100.0) if mean else None,
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_summary(doc: Dict[str, Any], measured: Iterable[Dict[str, Any]]) -> str:
    """SUMMARY.md. The status line is derived, so it cannot flatter an incomplete run."""
    measured = list(measured)
    lines: List[str] = []
    status = doc.get("status")
    lines.append(f"# Phase-0 baseline run - {status}")
    lines.append("")
    if status != STATUS_COMPLETE:
        lines.append(
            "> **This run is INCOMPLETE.** Blocks below did not produce their full "
            "repetition count, or a generation failed. Nothing here is a baseline."
        )
        lines.append("")
    protocol = doc.get("protocol", {})
    if not protocol.get("canonical", False):
        lines.append(
            "> **NON-CANONICAL developer smoke test.** The precommitted protocol "
            "(criteria section 10) was overridden: "
            + "; ".join(protocol.get("deviations", []) or ["unspecified"])
            + ". These numbers are not a Phase-0 baseline and must not be published as one."
        )
        lines.append("")
    lines.append(f"- session: `{protocol.get('session_id')}`")
    lines.append(f"- started: {doc.get('started_at')}")
    lines.append(f"- finished: {doc.get('finished_at')}")
    lines.append(
        f"- protocol: {protocol.get('warmups_per_block')} warmup + "
        f"{protocol.get('measured_repetitions_per_block')} measured per (arm, class)"
    )
    lines.append(f"- harness: {doc.get('software', {}).get('harness_version')}")
    model = doc.get("model", {})
    revision = model.get("revision")
    revision_text = revision.get("value") if isinstance(revision, dict) else revision
    lines.append(f"- model: `{model.get('repository')}` @ `{revision_text}`")
    lines.append("")
    lines.append(
        "This harness does **not** select `CANONICAL_PERFORMANCE_BASELINE` and computes no "
        "cross-configuration ratio: that selection is made from the completed campaign "
        "(criteria section 2.2), and computing ratios mid-campaign is prohibited "
        "(section 10)."
    )
    lines.append("")

    lines.append("## Completeness")
    lines.append("")
    lines.append("| arm | class | measured | expected | warmups | failures | complete |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for block in doc.get("blocks", []):
        lines.append(
            f"| {block['arm_id']} | {block['class_id']} | {block['observed_measured']} | "
            f"{block['expected_measured']} | {block['observed_warmups']} | "
            f"{len(block['failures'])} | {'yes' if block['complete'] else '**NO**'} |"
        )
    lines.append("")

    lines.append("## Per-block statistics (CALCULATED from the raw repetitions)")
    lines.append("")
    lines.append(
        "Every repetition is preserved in `repetitions.jsonl`; none was discarded "
        "(criteria section 10 prohibits selective outlier removal). "
        "`decode_tok_s` is the primary metric (section 6); `prefill_tok_s` is the "
        "instrumented prefill interval, **not** prompt_tokens/TTFT."
    )
    lines.append("")
    lines.append(
        "| arm | class | n | decode tok/s min | median | max | IQR | CV% | TTFT ms median | prefill tok/s median |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    keyed: Dict[tuple, List[Dict[str, Any]]] = {}
    for rep in measured:
        keyed.setdefault((rep.get("arm_id"), rep.get("class_id")), []).append(rep)
    for (arm_id, class_id), reps in keyed.items():
        decode = block_stats([r.get("decode_tok_s") for r in reps])
        ttft = block_stats([r.get("ttft_ms") for r in reps])
        prefill = block_stats(
            [(r.get("prefill") or {}).get("prefill_tok_s") for r in reps]
        )
        lines.append(
            f"| {arm_id} | {class_id} | {decode.get('n', 0)} | {_fmt(decode.get('min'))} | "
            f"{_fmt(decode.get('median'))} | {_fmt(decode.get('max'))} | "
            f"{_fmt(decode.get('iqr'))} | {_fmt(decode.get('cv_percent'))} | "
            f"{_fmt(ttft.get('median'), 1)} | {_fmt(prefill.get('median'))} |"
        )
    lines.append("")

    lines.append("## Resolved configuration per arm")
    lines.append("")
    lines.append(
        "Read back from the running engine (`/v1/instrumentation`), not from the flags. "
        "`auto` is not a configuration record (criteria section 2.3)."
    )
    lines.append("")
    for arm_id, resolved in (doc.get("resolved_configuration") or {}).items():
        lines.append(f"### {arm_id}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(resolved, indent=2)[:8000])
        lines.append("```")
        lines.append("")

    if doc.get("failure_count"):
        lines.append("## Failures")
        lines.append("")
        lines.append(f"{doc['failure_count']} generation(s) failed; see `failures.jsonl`.")
        lines.append("")

    lines.append("## Provenance")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(
            {k: doc.get(k) for k in ("software", "model", "host", "gpu", "workload_manifest")},
            indent=2,
        )[:20000]
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "Authoritative InferSwarm results live in `Zutfen-LLC/inferswarm` under "
        "`docs/benchmarks/results/`; this directory is the raw local artifact they are "
        "derived from."
    )
    lines.append("")
    return "\n".join(lines)
