"""Generic extend-prefill execution-unit partition policy.

Issue #117 Arm-C remediation (InferSwarm #153, corrected classification:
``BACKEND_REQUIRES_MULTI_CHUNK``).  The admitted capacity is the FROZEN
boundary contract's per-call row limit (e.g. the R6 dense strategy's
``PREFILL_CHUNK``).  Semantics:

- a logical unit at or below the admitted capacity must execute as ONE
  semantic backend call — subdividing a legal unit is policy, not
  necessity, and the #137 corrected intervention (probe C2) demonstrated
  that splitting a legal 53-row unit into ``32 + 21`` is sufficient to
  destabilize the committed token while the single-chunk arm is
  deterministic;
- a logical unit ABOVE the admitted capacity partitions deterministically
  at the capacity — this is REQUIRED, not optional: the frozen boundary
  geometry and the frozen wire contract both reject a call carrying more
  than the admitted rows, so e.g. the accepted 65-67-row Arm-C failing
  units must remain ``64 + remainder`` under any admissible unchanged
  contract.

This module is deliberately model-opaque and CPU-pure (stdlib only): it
knows row counts and an admitted capacity, nothing else.  The capacity
is supplied by the frozen execution contract (e.g. the R6 dense
strategy's ``PREFILL_CHUNK`` boundary-geometry constant and the frozen
participant plan's ``semantic_contract.prefill_chunk_rows``), never by a
hand-written literal at a call site.

Semantics (frozen by tests/test_issue117_arm_c_remediation.py):

- ``plan_prefill_partitions(n, c)`` with ``1 <= n <= c`` returns exactly
  ``[(0, n)]`` — one logical unit that fits the admitted capacity is ONE
  backend call.
- ``n > c`` partitions deterministically at the admitted capacity:
  full ``c``-row chunks followed by the remainder — the pre-existing
  over-limit semantics, unchanged and contract-required.
- Zero/negative/non-integer row counts or capacities fail closed.
- Every returned partition list satisfies the coverage invariants
  (ordered, non-overlapping, complete, each logical row exactly once);
  ``assert_partition_invariants`` re-derives them from the returned
  value so callers and tests can police any partition, not just ours.

NOTE: this policy does NOT remediate the numerical instability of the
required multi-chunk extend path for over-limit units — that is the
retained #137 necessary-path finding, and remediation of it requires a
deeper authority than InferSwarm #153 (see the #153 terminal
``ISSUE117_ARM_C_REMEDIATION_BLOCKED``).
"""

from __future__ import annotations

from typing import List, Tuple

Partition = Tuple[int, int]  # (offset, row_count)


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def plan_prefill_partitions(
    total_rows: int, admitted_max_rows: int
) -> List[Partition]:
    """Partition one logical extend-prefill unit into backend calls.

    A unit that fits the admitted capacity stays ONE call; a larger unit
    partitions deterministically at the capacity.  The returned list is
    complete, ordered, non-overlapping, and covers each logical row
    exactly once (verified by :func:`assert_partition_invariants`).
    """
    total = _require_positive_int(total_rows, "total_rows")
    capacity = _require_positive_int(admitted_max_rows, "admitted_max_rows")
    partitions: List[Partition] = []
    offset = 0
    while offset < total:
        count = min(capacity, total - offset)
        partitions.append((offset, count))
        offset += count
    assert_partition_invariants(partitions, total_rows=total)
    return partitions


def assert_partition_invariants(
    partitions: List[Partition], *, total_rows: int
) -> None:
    """Fail closed unless ``partitions`` covers ``total_rows`` exactly once.

    Ordered: offsets strictly increasing from 0.  Non-overlapping and
    gap-free: each partition starts where the previous ended.  Complete:
    the final partition ends exactly at ``total_rows``.  Well-formed:
    every count is a positive integer (no zero/negative row sizes).
    """
    total = _require_positive_int(total_rows, "total_rows")
    if not isinstance(partitions, list):
        raise ValueError("partitions must be a list of (offset, count) pairs")
    expected_offset = 0
    for index, item in enumerate(partitions):
        if not (isinstance(item, tuple) and len(item) == 2):
            raise ValueError(f"partition {index} is not an (offset, count) pair")
        offset, count = item
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise ValueError(f"partition {index} offset is not an integer")
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError(f"partition {index} count is not an integer")
        if count < 1:
            raise ValueError(f"partition {index} has non-positive row count {count}")
        if offset != expected_offset:
            raise ValueError(
                f"partition {index} offset {offset} breaks ordered/non-overlapping/"
                f"gap-free coverage (expected {expected_offset})"
            )
        expected_offset = offset + count
    if expected_offset != total:
        raise ValueError(
            f"partitions cover {expected_offset} rows, not the complete "
            f"logical unit of {total}"
        )


def admitted_capacity_from_boundary_contract(contract: object) -> int:
    """Read the admitted per-call row capacity from a frozen boundary contract.

    Accepts a mapping carrying ``prefill_chunk_rows`` (the frozen
    boundary-geometry field, e.g. the R6 strategy constant or a frozen
    participant plan's ``semantic_contract``).  Fails closed on a
    missing/non-positive field so a drifted contract can never silently
    legalize oversized single calls.
    """
    if not isinstance(contract, dict):
        raise ValueError("boundary contract must be a mapping")
    if "prefill_chunk_rows" not in contract:
        raise ValueError("boundary contract lacks prefill_chunk_rows")
    return _require_positive_int(
        contract["prefill_chunk_rows"], "prefill_chunk_rows"
    )


__all__ = [
    "Partition",
    "admitted_capacity_from_boundary_contract",
    "assert_partition_invariants",
    "plan_prefill_partitions",
]
