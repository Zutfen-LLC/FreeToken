"""Opt-in exact generation-state capture for the frozen InferSwarm C3 check.

The recorder is allocated only when explicitly configured. It is intentionally bounded and
correctness-only: capturing selected logits performs a device-to-host copy and therefore must
never be enabled for performance measurements.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch

CORRECTNESS_DIAGNOSTICS_SCHEMA = "inferswarm.phase1.c3-generation-state/1"
N1_SELECTED_LOGIT_STEPS = frozenset((0, 1, 15, 31))


@dataclass
class _GenerationRecord:
    uid: int
    generated_token_ids: list[int] = field(default_factory=list)
    step0_logits: torch.Tensor | None = None
    step0_source_dtype: str | None = None
    step0_argmax: int | None = None
    step0_top5: list[int] | None = None
    selected_logits: dict[int, torch.Tensor] = field(default_factory=dict)
    sampler_logit_rows_seen: int = 0


class CorrectnessDiagnostics:
    """Capture accepted tokens and the first sampler-input logit row per request."""

    def __init__(self, *, max_requests: int = 8) -> None:
        if max_requests < 1:
            raise ValueError("correctness diagnostics max_requests must be positive")
        self.max_requests = int(max_requests)
        self._records: OrderedDict[int, _GenerationRecord] = OrderedDict()
        self._overflow_requests = 0

    def _record(self, uid: int) -> _GenerationRecord | None:
        uid = int(uid)
        record = self._records.get(uid)
        if record is not None:
            return record
        if len(self._records) >= self.max_requests:
            self._overflow_requests += 1
            return None
        record = _GenerationRecord(uid=uid)
        self._records[uid] = record
        return record

    def capture_step0_logits(self, uid: int, logits: torch.Tensor) -> None:
        """Copy the actual first sampler-input row before text decoding or sampling changes."""

        record = self._record(uid)
        if record is None:
            return
        step = record.sampler_logit_rows_seen
        record.sampler_logit_rows_seen += 1
        if step not in N1_SELECTED_LOGIT_STEPS or step in record.selected_logits:
            return
        if logits.ndim != 1 or logits.numel() < 5:
            raise ValueError(
                f"C3 step-0 logits must be a vocabulary vector, got {tuple(logits.shape)}"
            )
        source = logits.detach()
        # Float32 preserves every value exactly when the model already emits float32 and is an
        # exact widening for fp16/bf16. The clone protects graph-output buffers from replay.
        cpu = source.to(device="cpu", dtype=torch.float32).clone()
        record.selected_logits[step] = cpu
        if step != 0:
            return
        top5 = torch.topk(cpu, 5, dim=-1).indices.tolist()
        record.step0_logits = cpu
        record.step0_source_dtype = str(source.dtype)
        record.step0_argmax = int(torch.argmax(cpu).item())
        record.step0_top5 = [int(value) for value in top5]

    def record_accepted_token(self, uid: int, token_id: int) -> None:
        record = self._record(uid)
        if record is not None:
            record.generated_token_ids.append(int(token_id))

    def snapshot(self) -> dict[str, Any]:
        records = []
        for record in self._records.values():
            logits = record.step0_logits
            records.append(
                {
                    "uid": record.uid,
                    "generated_token_ids": list(record.generated_token_ids),
                    "generated_token_count": len(record.generated_token_ids),
                    "step0": {
                        "available": logits is not None,
                        "source": "actual model logits at the first generated token before sampling",
                        "source_dtype": record.step0_source_dtype,
                        "serialized_dtype": "float32",
                        "vocab_size": int(logits.numel())
                        if logits is not None
                        else None,
                        "argmax": record.step0_argmax,
                        "top5_order": record.step0_top5,
                        "full_logits": logits.tolist() if logits is not None else None,
                    },
                    "selected_logit_steps": {
                        str(step): {
                            "source": "actual model logits before greedy selection",
                            "serialized_dtype": "float32",
                            "vocab_size": int(values.numel()),
                            "argmax": int(values.argmax().item()),
                            "full_logits": values.tolist(),
                        }
                        for step, values in sorted(record.selected_logits.items())
                    },
                }
            )
        return {
            "schema": CORRECTNESS_DIAGNOSTICS_SCHEMA,
            "enabled": True,
            "correctness_only": True,
            "performance_compatible": False,
            "ordinary_sampling_unchanged": True,
            "ordinary_sse_unchanged": True,
            "token_source": "scheduler-accepted token IDs before detokenization",
            "max_requests": self.max_requests,
            "records_retained": len(records),
            "overflow_requests": self._overflow_requests,
            "truncated": self._overflow_requests > 0,
            "records": records,
        }

    def reset(self) -> None:
        self._records.clear()
        self._overflow_requests = 0


def absent_correctness_diagnostics_report() -> dict[str, Any]:
    return {
        "schema": CORRECTNESS_DIAGNOSTICS_SCHEMA,
        "enabled": False,
        "correctness_only": True,
        "performance_compatible": False,
        "ordinary_sampling_unchanged": True,
        "ordinary_sse_unchanged": True,
        "records_retained": 0,
        "overflow_requests": 0,
        "truncated": False,
        "records": [],
    }
