"""Generate the immutable R4 test-summary artifact (+ .sha256 sidecar).

Parses the retained pytest logs (focused R4 run and predecessor regression
run), captures environment facts, and writes docs/inferswarm_r4/
test-summary.json so test provenance never lives only in PR prose.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha

RESULT_RE = re.compile(r"(=?\s*)?(\d+) passed([^\n]*)")
FAILED_RE = re.compile(r"(\d+) failed")
ERROR_RE = re.compile(r"(\d+) error")


def parse_pytest_log(text: str) -> dict:
    tail = text.strip().splitlines()[-40:]
    joined = "\n".join(tail)
    passed = failed = errors = 0
    warnings = []
    for match in RESULT_RE.finditer(joined):
        passed = max(passed, int(match.group(2)))
    for match in FAILED_RE.finditer(joined):
        failed = max(failed, int(match.group(1)))
    for match in ERROR_RE.finditer(joined):
        errors = max(errors, int(match.group(1)))
    for match in re.finditer(r"(\d+) warnings", joined):
        warnings.append(int(match.group(1)))
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "warnings": sum(warnings),
        "ok": failed == 0 and errors == 0 and passed > 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-sha", required=True)
    parser.add_argument("--focused-log", type=Path, required=True)
    parser.add_argument("--predecessor-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    focused_text = args.focused_log.read_text(errors="replace")
    predecessor_text = args.predecessor_log.read_text(errors="replace")
    focused_cmd = _grep_command(focused_text) or (
        "python -m pytest tests/research/test_r4_wire.py "
        "tests/research/test_r4_preflight_gate.py -v"
    )
    predecessor_cmd = _grep_command(predecessor_text) or (
        "python -m pytest tests/research/ -v"
    )
    summary = {
        "schema": "inferswarm.r4.test-summary/1",
        "generated_at_unix": time.time(),
        "generated_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "producer_sha": args.producer_sha,
        "environment": {
            "hostname": subprocess.check_output(["hostname"], text=True).strip(),
            "kernel": subprocess.check_output(
                ["uname", "-r"], text=True
            ).strip(),
            "python": subprocess.check_output(
                [sys.executable, "--version"], text=True, stderr=subprocess.STDOUT
            ).strip(),
        },
        "test_commands": {
            "focused_r4": focused_cmd,
            "predecessor_regressions": predecessor_cmd,
        },
        "focused_r4_result": parse_pytest_log(focused_text),
        "predecessor_regression_result": parse_pytest_log(predecessor_text),
    }
    ok = summary["focused_r4_result"]["ok"] and summary[
        "predecessor_regression_result"
    ]["ok"]
    summary["result"] = "ALL_TESTS_PASSED" if ok else "TEST_FAILURES"
    write_json_with_sha(args.out, summary)
    print(json.dumps({k: summary[k] for k in
                      ("focused_r4_result", "predecessor_regression_result",
                       "result")}))
    return 0 if ok else 1


def _grep_command(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("$ ") or line.startswith("CMD: "):
            return line.split(" ", 1)[1].strip()
    return None


if __name__ == "__main__":
    raise SystemExit(main())
