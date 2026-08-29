"""Opt-in exact generation-state capture for the frozen InferSwarm C3 check.

The recorder is allocated only when explicitly configured. It is intentionally bounded and
correctness-only: capturing step-0 logits performs a device-to-host copy and therefore must
never be enabled for performance measurements.
"""

from __future__ import annotations

import base64
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch

CORRECTNESS_DIAGNOSTICS_SCHEMA = "inferswarm.phase1.c3-generation-state/1"
C3_ROOT_CAUSE_SCHEMA = "inferswarm.phase1.c3-root-cause-tensors/1"
C3_ROOT_CAUSE_MODES = ("off", "trace", "DIAGNOSTIC_SPLIT_GPU0")
C3_ROOT_CAUSE_EXPECTED_LAYERS = 40
C3_ROOT_CAUSE_DEFAULT_MAX_BYTES = 8 * 1024 * 1024


@dataclass
class _GenerationRecord:
    uid: int
    generated_token_ids: list[int] = field(default_factory=list)
    step0_logits: torch.Tensor | None = None
    step0_source_dtype: str | None = None
    step0_argmax: int | None = None
    step0_top5: list[int] | None = None


class CorrectnessDiagnostics:
    """Capture accepted tokens and the first sampler-input logit row per request."""

    def __init__(
        self,
        *,
        max_requests: int = 8,
        root_cause_mode: str = "off",
        root_cause_decode_step: int = 0,
        root_cause_expected_layers: int = C3_ROOT_CAUSE_EXPECTED_LAYERS,
        root_cause_max_tensor_bytes: int = C3_ROOT_CAUSE_DEFAULT_MAX_BYTES,
    ) -> None:
        if max_requests < 1:
            raise ValueError("correctness diagnostics max_requests must be positive")
        if root_cause_mode not in C3_ROOT_CAUSE_MODES:
            raise ValueError(f"invalid C3 root-cause mode {root_cause_mode!r}")
        if root_cause_decode_step < 0:
            raise ValueError("C3 root-cause decode step cannot be negative")
        if root_cause_expected_layers < 1:
            raise ValueError("C3 root-cause expected layer count must be positive")
        if root_cause_max_tensor_bytes < 1:
            raise ValueError("C3 root-cause tensor bound must be positive")
        self.max_requests = int(max_requests)
        self._records: OrderedDict[int, _GenerationRecord] = OrderedDict()
        self._overflow_requests = 0
        self.root_cause_mode = root_cause_mode
        self.root_cause_decode_step = int(root_cause_decode_step)
        self.root_cause_expected_layers = int(root_cause_expected_layers)
        self.root_cause_max_tensor_bytes = int(root_cause_max_tensor_bytes)
        self._current_decode_step: int | None = None
        self._moe_layers: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._moe_tensor_bytes = 0
        self._moe_overflow_records = 0
        self._transport_capture_layer: int | None = None

    @property
    def moe_capture_enabled(self) -> bool:
        return self.root_cause_mode != "off"

    @property
    def split_gpu0_enabled(self) -> bool:
        return self.root_cause_mode == "DIAGNOSTIC_SPLIT_GPU0"

    def begin_decode_step(self, step: int) -> None:
        """Set the engine-owned decode-step identity for subsequent MoE boundaries."""

        self._current_decode_step = int(step)

    def _capturing_layer(self, layer_id: int) -> bool:
        return (
            self.moe_capture_enabled
            and self._current_decode_step == self.root_cause_decode_step
            and 0 <= int(layer_id) < self.root_cause_expected_layers
        )

    def _layer_record(self, layer_id: int) -> dict[str, Any] | None:
        layer_id = int(layer_id)
        if not self._capturing_layer(layer_id):
            return None
        record = self._moe_layers.get(layer_id)
        if record is None:
            record = {
                "layer_id": layer_id,
                "decode_step": self.root_cause_decode_step,
                "tensors": OrderedDict(),
                "ownership": None,
                "selected_expert_weights": None,
            }
            self._moe_layers[layer_id] = record
        return record

    def _store_tensor(self, layer_id: int, name: str, tensor: torch.Tensor) -> None:
        record = self._layer_record(layer_id)
        if record is None:
            return
        tensors = record["tensors"]
        if name in tensors:
            raise RuntimeError(
                f"C3 root-cause tensor {name!r} for layer {layer_id} was captured twice"
            )
        source = tensor.detach().contiguous()
        size = int(source.numel() * source.element_size())
        if self._moe_tensor_bytes + size > self.root_cause_max_tensor_bytes:
            self._moe_overflow_records += 1
            raise RuntimeError(
                "C3 root-cause tensor capture exceeded its explicit byte bound: "
                f"{self._moe_tensor_bytes + size} > {self.root_cause_max_tensor_bytes}"
            )
        # A same-device clone is enqueued on the current stream.  It protects replay/staging
        # buffers without introducing a per-layer host synchronization that could mask an
        # overlap-ordering defect.  The idle instrumentation endpoint synchronizes before
        # snapshot(), where the bounded tensors are finally copied to CPU.
        tensors[name] = source.clone()
        self._moe_tensor_bytes += size

    def capture_moe_input(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        raw_topk_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> None:
        if not self._capturing_layer(layer_id):
            return
        self._store_tensor(layer_id, "hidden_input", hidden_states)
        self._store_tensor(layer_id, "raw_topk_ids", raw_topk_ids)
        self._store_tensor(layer_id, "routing_weights", routing_weights)

    def capture_ownership(
        self,
        layer_id: int,
        *,
        local_mask: torch.Tensor,
        remote_mask: torch.Tensor,
        remote_slot_ids: torch.Tensor,
    ) -> None:
        record = self._layer_record(layer_id)
        if record is None:
            return
        # Counts are derived at the already-synchronized snapshot boundary.  Avoid a hot-path
        # ``.item()`` here because the overlap trace must not gain a per-layer host wait.
        record["ownership"] = "derive_from_captured_masks"
        self._store_tensor(layer_id, "local_mask", local_mask)
        self._store_tensor(layer_id, "remote_mask", remote_mask)
        self._store_tensor(layer_id, "remote_slot_ids", remote_slot_ids)

    def capture_gpu0_slot_ids(self, layer_id: int, gpu0_slot_ids: torch.Tensor) -> None:
        self._store_tensor(layer_id, "gpu0_slot_ids", gpu0_slot_ids)

    def capture_selected_expert_weights(
        self, layer_id: int, raw_topk_ids: torch.Tensor, cache
    ) -> None:
        """Hash the exact selected source-bank rows for the R/G isolation proof.

        This deliberately runs only in the ordinary GPU0 and split-GPU0 diagnostics.  The
        selected source rows are authoritative model bytes already retained by the offload
        cache, so hashes prove R and G used the same weights without duplicating those large
        rows in the bounded tensor sidecar.  It is not called in O/S, where a per-layer host
        synchronization could mask an overlap defect.
        """

        record = self._layer_record(layer_id)
        if record is None:
            return
        if record["selected_expert_weights"] is not None:
            raise RuntimeError(
                f"C3 selected expert weights for layer {layer_id} were captured twice"
            )
        selected = sorted(
            {int(value) for value in raw_topk_ids.detach().cpu().reshape(-1).tolist()}
        )
        if not selected:
            raise RuntimeError("C3 selected expert-weight proof has an empty route set")
        banks = []
        for name in cache.bank_schema:
            source = cache.bank_sources[name][int(layer_id)]
            rows = []
            combined = hashlib.sha256()
            byte_count = 0
            for expert_id in selected:
                if not 0 <= expert_id < int(source.shape[0]):
                    raise RuntimeError(
                        f"C3 expert ID {expert_id} is outside source bank {name!r}"
                    )
                row = source[expert_id].detach().contiguous().cpu()
                raw = row.view(torch.uint8).reshape(-1).numpy().tobytes()
                digest = hashlib.sha256(raw).hexdigest()
                combined.update(raw)
                byte_count += len(raw)
                rows.append(
                    {
                        "expert_id": expert_id,
                        "raw_byte_count": len(raw),
                        "raw_byte_sha256": digest,
                    }
                )
            banks.append(
                {
                    "name": name,
                    "dtype": str(source.dtype).removeprefix("torch."),
                    "row_shape": list(source.shape[1:]),
                    "selected_rows_raw_byte_count": byte_count,
                    "selected_rows_raw_byte_sha256": combined.hexdigest(),
                    "rows": rows,
                }
            )
        record["selected_expert_weights"] = {
            "source": "authoritative offload-cache host bank rows",
            "selected_raw_expert_ids": selected,
            "bank_order": list(cache.bank_schema),
            "banks": banks,
        }

    def capture_candidate_partials(
        self,
        layer_id: int,
        *,
        local_partial: torch.Tensor,
        remote_partial: torch.Tensor,
        combined_partial: torch.Tensor,
    ) -> None:
        self._store_tensor(layer_id, "local_partial", local_partial)
        self._store_tensor(layer_id, "remote_partial", remote_partial)
        self._store_tensor(layer_id, "combined_partial", combined_partial)

    def capture_moe_output(self, layer_id: int, output: torch.Tensor) -> None:
        self._store_tensor(layer_id, "moe_output", output)

    def enable_transport_capture(self, layer_id: int) -> None:
        """Enable the conditional §11 staging trace for one targeted replay layer."""

        if not self.moe_capture_enabled:
            raise RuntimeError("transport capture requires C3 root-cause diagnostics")
        self._transport_capture_layer = int(layer_id)

    def capture_transport_tensor(
        self, layer_id: int, stage: str, tensor: torch.Tensor
    ) -> None:
        if self._transport_capture_layer != int(layer_id):
            return
        self._store_tensor(layer_id, f"transport_{stage}", tensor)

    @staticmethod
    def _tensor_evidence(tensor: torch.Tensor) -> dict[str, Any]:
        cpu = tensor.detach().contiguous().cpu()
        raw = cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
        return {
            "dtype": str(cpu.dtype).removeprefix("torch."),
            "shape": list(cpu.shape),
            "raw_byte_count": len(raw),
            "raw_byte_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_bytes_base64": base64.b64encode(raw).decode("ascii"),
        }

    def _root_cause_snapshot(self) -> dict[str, Any]:
        layers = []
        for layer_id in sorted(self._moe_layers):
            record = self._moe_layers[layer_id]
            tensors = record["tensors"]
            ownership = None
            if record["ownership"] is not None:
                local_mask = tensors["local_mask"].detach().cpu()
                remote_mask = tensors["remote_mask"].detach().cpu()
                local_count = int(local_mask.sum().item())
                remote_count = int(remote_mask.sum().item())
                complete = bool(torch.all(local_mask | remote_mask).item())
                disjoint = not bool(torch.any(local_mask & remote_mask).item())
                ownership = {
                    "local_selection_count": local_count,
                    "remote_selection_count": remote_count,
                    "total_routed_selections": int(local_mask.numel()),
                    "masks_complete": complete,
                    "masks_disjoint": disjoint,
                    "every_route_exactly_once": complete
                    and disjoint
                    and local_count + remote_count == local_mask.numel(),
                }
            layers.append(
                {
                    "layer_id": layer_id,
                    "decode_step": record["decode_step"],
                    "ownership": ownership,
                    "selected_expert_weights": record["selected_expert_weights"],
                    "tensors": {
                        name: self._tensor_evidence(tensor)
                        for name, tensor in tensors.items()
                    },
                }
            )
        layer_ids = [record["layer_id"] for record in layers]
        expected_ids = list(range(self.root_cause_expected_layers))
        return {
            "schema": C3_ROOT_CAUSE_SCHEMA,
            "enabled": self.moe_capture_enabled,
            "mode": self.root_cause_mode,
            "diagnostic_label": (
                "DIAGNOSTIC_SPLIT_GPU0" if self.split_gpu0_enabled else None
            ),
            "correctness_only": True,
            "performance_compatible": False,
            "performance_fields_collected": False,
            "target_decode_step": self.root_cause_decode_step,
            "expected_moe_layers": self.root_cause_expected_layers,
            "layer_ids": layer_ids,
            "exactly_expected_layers": layer_ids == expected_ids,
            "layers_retained": len(layers),
            "tensor_bytes_retained": self._moe_tensor_bytes,
            "max_tensor_bytes": self.root_cause_max_tensor_bytes,
            "overflow_records": self._moe_overflow_records,
            "truncated": self._moe_overflow_records > 0,
            "transport_capture_layer": self._transport_capture_layer,
            "layers": layers,
        }

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
        if record is None or record.step0_logits is not None:
            return
        if logits.ndim != 1 or logits.numel() < 5:
            raise ValueError(
                f"C3 step-0 logits must be a vocabulary vector, got {tuple(logits.shape)}"
            )
        source = logits.detach()
        # Float32 preserves every value exactly when the model already emits float32 and is an
        # exact widening for fp16/bf16. The clone protects graph-output buffers from replay.
        cpu = source.to(device="cpu", dtype=torch.float32).clone()
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
            "moe_root_cause": self._root_cause_snapshot(),
        }

    def reset(self) -> None:
        self._records.clear()
        self._overflow_requests = 0
        self._current_decode_step = None
        self._moe_layers.clear()
        self._moe_tensor_bytes = 0
        self._moe_overflow_records = 0
        self._transport_capture_layer = None


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
        "moe_root_cause": {
            "schema": C3_ROOT_CAUSE_SCHEMA,
            "enabled": False,
            "mode": "off",
            "diagnostic_label": None,
            "correctness_only": True,
            "performance_compatible": False,
            "performance_fields_collected": False,
            "target_decode_step": None,
            "expected_moe_layers": C3_ROOT_CAUSE_EXPECTED_LAYERS,
            "layer_ids": [],
            "exactly_expected_layers": False,
            "layers_retained": 0,
            "tensor_bytes_retained": 0,
            "max_tensor_bytes": 0,
            "overflow_records": 0,
            "truncated": False,
            "transport_capture_layer": None,
            "layers": [],
        },
    }
