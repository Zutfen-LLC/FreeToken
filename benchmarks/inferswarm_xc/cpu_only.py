"""CPU-only helpers for the external-Coordinator proof (inferswarm #67).

Torch-free equivalents of the small utilities the accepted research modules
pull in via ``freetoken.research.n0_model_block`` (which imports torch for its
weight-side work).  The Coordinator host must remain free of Torch; these
helpers preserve the exact on-disk artifact format so retained evidence
remains interchangeable with the accepted R4/R5A/R5B records.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


def write_json_with_sha(path: str | os.PathLike[str], payload: dict) -> None:
    target = Path(path)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    target.with_suffix(target.suffix + ".sha256").write_text(
        f"{hashlib.sha256(data).hexdigest()}  {target.name}\n"
    )


def load_json_with_sha(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a JSON file and verify its sidecar sha256 (fail-closed)."""
    target = Path(path)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    if sidecar.exists():
        expected = sidecar.read_text().split()[0]
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if expected != actual:
            raise RuntimeError(f"sha256 sidecar mismatch for {target}")
    return json.loads(target.read_text())


def printable_increment(tokenizer: Any, token_ids: list[int], *, finished: bool) -> str:
    """Incremental printable detokenization for the bounded proof workload.

    Applies the accepted streaming policy (hold back a trailing partial word
    until it resolves; flush everything on finish) without importing the
    torch-carrying server detokenizer.
    """

    def find_printable(text: str) -> str:
        if text.endswith("\n"):
            return text
        if text and len(text) > 1 and _is_cjk(ord(text[-2])):
            return text[:-1]
        if text and _is_cjk(ord(text[-1])):
            return text
        return text[: text.rfind(" ") + 1]

    decoded = tokenizer.decode(token_ids)
    if finished:
        return decoded
    return find_printable(decoded)


def _is_cjk(cp: int) -> bool:
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF
        or 0x2F800 <= cp <= 0x2FAFF
    )


__all__ = ["load_json_with_sha", "printable_increment", "write_json_with_sha"]
