"""Validate independent R2 v2 reference sessions and select session A mechanically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha

from .v2_support import select_reference_pair, validate_v2_output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-a", type=Path, required=True)
    parser.add_argument("--session-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    validate_v2_output_path(args.out)
    if args.out.name != "reference-v2-selection.json":
        raise ValueError("pair selection must write reference-v2-selection.json")
    selection = select_reference_pair(
        json.loads(args.session_a.read_text()),
        json.loads(args.session_b.read_text()),
        session_a_path=args.session_a,
        session_b_path=args.session_b,
    )
    write_json_with_sha(args.out, selection)
    print(json.dumps(selection, indent=2))
    return 0 if selection["self_consistency_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
