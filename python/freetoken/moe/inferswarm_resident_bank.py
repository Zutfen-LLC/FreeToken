"""InferSwarm Phase-1 frozen placement and secondary expert storage.

This module deliberately stops at startup residency.  It has no route, fetch, execute,
or cache-miss API.  P3 exposes deterministic read-only-by-contract tensor views to its
narrow executor, while all transport, partitioning, and execution remain elsewhere.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch

from .expert_banks import ExpertBanks
from .offload_cache import _BANK_BYTES_PER_EXPERT, _BANK_SCHEMAS

PLACEMENT_SCHEMA = "inferswarm.phase1.placement/1"
PLACEMENT_STATUS = "FROZEN_BEFORE_PHASE1_PERFORMANCE"
CANONICAL_MODEL_REPOSITORY = "nvidia/Qwen3.6-35B-A3B-NVFP4"
CANONICAL_MODEL_REVISION = "491c2f1ea524c639598bf8fa787a93fed5a6fbce"
CANONICAL_WORKLOAD_MANIFEST_SHA256 = (
    "10f81e5418a71a68f387632de422c3337cc7ba0518111a8746ad856d0210b24a"
)
CANONICAL_RUN_JSON_SHA256 = (
    "1ecd14c8c157eb6cde62f8514b8f2af82a36dfe69e3a7241f5be6c9908e539dc"
)
CANONICAL_EXACT_ROUTING_SHA256 = (
    "4071e2bfd3c18f39e5c5a0b5ff8913ca0fb99b843cf7abca0ecc1f4ebd0a252f"
)
CANONICAL_CACHE_PRESSURE_SHA256 = (
    "f02d96a079a6af94b3fec9d2c571322a1274cd408a3a9ff2ef162f538babab3a"
)
V1_PLACEMENT_POLICY = "phase1-qwen36-placement-v1"
V1_CANONICAL_PLACEMENT = "complement_5442"
V1_ARTIFACT_SHA256 = (
    "255dce5d335c5017de06eff54cfd1c8a0599d2dbd6c84c7fb0fb856701596a2c"
)
V2_PLACEMENT_POLICY = "phase1-qwen36-placement-v2"
V2_CANONICAL_PLACEMENT = "coverage_constrained_complement_5442"
V2_ARTIFACT_SHA256 = (
    "2f62bb84df40d4cc5649e940a39cb53d2975eadecbc320fb97d2b037d4e005f4"
)
# Compatibility names remain pinned to historical v1. Runtime auto-resolution below accepts
# only the exact SHA-addressed v1/v2 descriptors; it never treats this alias as a wildcard.
PLACEMENT_POLICY = V1_PLACEMENT_POLICY
CANONICAL_PLACEMENT = V1_CANONICAL_PLACEMENT
CANONICAL_ARTIFACT_SHA256 = V1_ARTIFACT_SHA256
MAPPING_RULE = "remote_slot = index within canonical flat_ids_in_rank_order"


@dataclass(frozen=True)
class PlacementContract:
    schema: str = PLACEMENT_SCHEMA
    policy: str = PLACEMENT_POLICY
    status: str = PLACEMENT_STATUS
    canonical_placement: str = CANONICAL_PLACEMENT
    model_repository: str = CANONICAL_MODEL_REPOSITORY
    model_revision: str = CANONICAL_MODEL_REVISION
    num_layers: int = 40
    num_experts: int = 256
    remote_slots: int = 5_442
    bytes_per_slot: int = 1_775_616
    remote_resident_bytes: int = 9_662_902_272
    remote_budget_bytes: int = 9_663_676_416
    hidden_size: int = 2_048
    intermediate_size: int = 512
    architecture: str = "Qwen3_5MoeForConditionalGeneration"
    workload_manifest_sha256: str = CANONICAL_WORKLOAD_MANIFEST_SHA256
    run_json_sha256: str = CANONICAL_RUN_JSON_SHA256
    exact_routing_sha256: str = CANONICAL_EXACT_ROUTING_SHA256
    cache_pressure_sha256: str = CANONICAL_CACHE_PRESSURE_SHA256
    artifact_sha256: str = V1_ARTIFACT_SHA256


V1_CONTRACT = PlacementContract()
V2_CONTRACT = PlacementContract(
    policy=V2_PLACEMENT_POLICY,
    canonical_placement=V2_CANONICAL_PLACEMENT,
    artifact_sha256=V2_ARTIFACT_SHA256,
)
KNOWN_PLACEMENT_CONTRACTS: Mapping[str, PlacementContract] = MappingProxyType(
    {
        V1_CONTRACT.artifact_sha256: V1_CONTRACT,
        V2_CONTRACT.artifact_sha256: V2_CONTRACT,
    }
)
CANONICAL_CONTRACT = V1_CONTRACT


@dataclass(frozen=True)
class PlacementIdentity:
    flat_id: int
    layer_id: int
    expert_id: int
    remote_slot: int


@dataclass(frozen=True)
class LayerPlacement:
    layer_id: int
    expert_ids: tuple[int, ...]
    remote_slots: tuple[int, ...]


@dataclass(frozen=True)
class FrozenPlacement:
    artifact_sha256: str
    schema: str
    policy: str
    status: str
    canonical_placement: str
    model_repository: str
    model_revision: str
    num_layers: int
    num_experts: int
    total_expert_slots: int
    bytes_per_slot: int
    remote_slots: int
    remote_resident_bytes: int
    remote_budget_bytes: int
    flat_ids_in_rank_order: tuple[int, ...]
    identities_in_rank_order: tuple[PlacementIdentity, ...]
    per_layer: tuple[LayerPlacement, ...]
    _slot_by_identity: Mapping[tuple[int, int], int]

    def remote_slot(self, layer_id: int, expert_id: int) -> int:
        return self._slot_by_identity[(layer_id, expert_id)]


def _expect_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"placement field {field!r} must be an object")
    return value


def _expect_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"placement field {field!r} must be an array")
    return value


def _expect_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"placement field {field!r} must be an integer")
    return value


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ValueError(
            f"placement {field} disagreement: expected {expected!r}, got {actual!r}"
        )


def parse_frozen_placement_bytes(
    raw: bytes,
    *,
    expected_sha256: str | None = None,
    contract: PlacementContract | None = None,
) -> FrozenPlacement:
    """Parse and mechanically cross-check every canonical placement representation."""
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is None and contract is None:
        contract = KNOWN_PLACEMENT_CONTRACTS.get(digest)
        if contract is None:
            raise ValueError(
                "placement artifact SHA-256 is not a known frozen Phase-1 policy: "
                f"{digest}"
            )
        expected_sha256 = contract.artifact_sha256
    elif expected_sha256 is None or contract is None:
        raise ValueError(
            "custom placement parsing requires both expected_sha256 and contract"
        )
    if digest != expected_sha256:
        raise ValueError(
            "placement artifact SHA-256 disagreement: "
            f"expected {expected_sha256}, got {digest}"
        )
    try:
        doc = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"placement artifact is not valid UTF-8 JSON: {exc}") from exc
    doc = _expect_dict(doc, "root")

    _require_equal(doc.get("schema"), contract.schema, "schema")
    _require_equal(doc.get("policy_id"), contract.policy, "policy_id")
    _require_equal(doc.get("status"), contract.status, "status")
    _require_equal(
        doc.get("canonical_remote_placement"),
        contract.canonical_placement,
        "canonical_remote_placement",
    )

    source = _expect_dict(doc.get("source"), "source")
    _require_equal(
        source.get("model_repository"), contract.model_repository, "source.model_repository"
    )
    _require_equal(
        source.get("model_revision"), contract.model_revision, "source.model_revision"
    )
    _require_equal(
        source.get("workload_manifest_sha256"),
        contract.workload_manifest_sha256,
        "source.workload_manifest_sha256",
    )
    _require_equal(
        source.get("run_json_sha256"),
        contract.run_json_sha256,
        "source.run_json_sha256",
    )
    _require_equal(
        source.get("exact_routing_sha256"),
        contract.exact_routing_sha256,
        "source.exact_routing_sha256",
    )
    _require_equal(
        source.get("cache_pressure_sha256"),
        contract.cache_pressure_sha256,
        "source.cache_pressure_sha256",
    )

    geometry = _expect_dict(doc.get("geometry"), "geometry")
    num_layers = _expect_int(geometry.get("num_moe_layers"), "geometry.num_moe_layers")
    num_experts = _expect_int(
        geometry.get("num_experts_per_layer"), "geometry.num_experts_per_layer"
    )
    total = _expect_int(geometry.get("total_expert_slots"), "geometry.total_expert_slots")
    _require_equal(num_layers, contract.num_layers, "geometry.num_moe_layers")
    _require_equal(num_experts, contract.num_experts, "geometry.num_experts_per_layer")
    _require_equal(total, num_layers * num_experts, "geometry.total_expert_slots arithmetic")

    budget = _expect_dict(doc.get("budget"), "budget")
    bytes_per_slot = _expect_int(budget.get("bytes_per_slot"), "budget.bytes_per_slot")
    remote_budget = _expect_int(
        budget.get("remote_budget_bytes"), "budget.remote_budget_bytes"
    )
    remote_slots = _expect_int(budget.get("remote_slots"), "budget.remote_slots")
    remote_bytes = _expect_int(
        budget.get("remote_resident_bytes"), "budget.remote_resident_bytes"
    )
    _require_equal(bytes_per_slot, contract.bytes_per_slot, "budget.bytes_per_slot")
    _require_equal(remote_slots, contract.remote_slots, "budget.remote_slots")
    _require_equal(remote_budget, contract.remote_budget_bytes, "budget.remote_budget_bytes")
    _require_equal(
        remote_bytes, remote_slots * bytes_per_slot, "budget.remote_resident_bytes arithmetic"
    )
    _require_equal(remote_bytes, contract.remote_resident_bytes, "budget.remote_resident_bytes")
    if remote_bytes > remote_budget:
        raise ValueError(
            f"placement is over budget: {remote_bytes} resident bytes > {remote_budget}"
        )

    placements = _expect_dict(doc.get("placements"), "placements")
    if contract.canonical_placement not in placements:
        raise ValueError(
            f"placement is missing canonical placement {contract.canonical_placement!r}"
        )
    selected = _expect_dict(
        placements[contract.canonical_placement],
        f"placements.{contract.canonical_placement}",
    )
    slot_count = _expect_int(selected.get("slot_count"), "canonical.slot_count")
    _require_equal(slot_count, remote_slots, "canonical.slot_count")

    flat_raw = _expect_list(
        selected.get("flat_ids_in_rank_order"), "canonical.flat_ids_in_rank_order"
    )
    if len(flat_raw) != slot_count:
        raise ValueError(
            "canonical flat-id slot count disagreement: "
            f"expected {slot_count}, got {len(flat_raw)}"
        )
    flat_ids = tuple(
        _expect_int(value, f"canonical.flat_ids_in_rank_order[{slot}]")
        for slot, value in enumerate(flat_raw)
    )
    if len(set(flat_ids)) != len(flat_ids):
        raise ValueError("canonical placement contains duplicate flat IDs")

    records = _expect_list(
        selected.get("identities_in_rank_order"), "canonical.identities_in_rank_order"
    )
    if len(records) != slot_count:
        raise ValueError(
            "canonical identity-record slot count disagreement: "
            f"expected {slot_count}, got {len(records)}"
        )
    identities: list[PlacementIdentity] = []
    seen_identities: set[tuple[int, int]] = set()
    for remote_slot, (flat_id, raw_record) in enumerate(zip(flat_ids, records)):
        record = _expect_dict(raw_record, f"canonical.identities_in_rank_order[{remote_slot}]")
        record_flat = _expect_int(record.get("flat_id"), f"identity[{remote_slot}].flat_id")
        layer_id = _expect_int(record.get("layer"), f"identity[{remote_slot}].layer")
        expert_id = _expect_int(record.get("expert_id"), f"identity[{remote_slot}].expert_id")
        if not 0 <= layer_id < num_layers:
            raise ValueError(f"identity at slot {remote_slot} has out-of-range layer {layer_id}")
        if not 0 <= expert_id < num_experts:
            raise ValueError(
                f"identity at slot {remote_slot} has out-of-range expert_id {expert_id}"
            )
        calculated = layer_id * num_experts + expert_id
        if record_flat != calculated:
            raise ValueError(
                f"identity at slot {remote_slot} violates flat_id arithmetic: "
                f"{record_flat} != {layer_id} * {num_experts} + {expert_id}"
            )
        if flat_id != record_flat:
            raise ValueError(
                f"rank-order/identity disagreement at slot {remote_slot}: "
                f"flat_ids has {flat_id}, identity has {record_flat}"
            )
        identity = (layer_id, expert_id)
        if identity in seen_identities:
            raise ValueError(f"canonical placement contains duplicate identity {identity}")
        seen_identities.add(identity)
        identities.append(PlacementIdentity(flat_id, layer_id, expert_id, remote_slot))

    per_layer_raw = _expect_list(selected.get("per_layer"), "canonical.per_layer")
    if len(per_layer_raw) != num_layers:
        raise ValueError(
            "canonical per_layer count disagreement: "
            f"expected {num_layers}, got {len(per_layer_raw)}"
        )
    expected_by_layer: list[list[int]] = [[] for _ in range(num_layers)]
    rank_by_layer: list[list[PlacementIdentity]] = [[] for _ in range(num_layers)]
    for identity in identities:
        expected_by_layer[identity.layer_id].append(identity.expert_id)
        rank_by_layer[identity.layer_id].append(identity)
    observed_layers: set[int] = set()
    for index, raw_layer in enumerate(per_layer_raw):
        layer = _expect_dict(raw_layer, f"canonical.per_layer[{index}]")
        layer_id = _expect_int(layer.get("layer"), f"canonical.per_layer[{index}].layer")
        if not 0 <= layer_id < num_layers:
            raise ValueError(f"per_layer record {index} has out-of-range layer {layer_id}")
        if layer_id in observed_layers:
            raise ValueError(f"canonical per_layer contains duplicate layer {layer_id}")
        observed_layers.add(layer_id)
        expert_values = _expect_list(
            layer.get("expert_ids"), f"canonical.per_layer[{index}].expert_ids"
        )
        expert_ids = [
            _expect_int(value, f"canonical.per_layer[{index}].expert_ids[{j}]")
            for j, value in enumerate(expert_values)
        ]
        if any(not 0 <= expert_id < num_experts for expert_id in expert_ids):
            raise ValueError(f"canonical per_layer layer {layer_id} has out-of-range expert ID")
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError(f"canonical per_layer layer {layer_id} has duplicate expert IDs")
        expected = sorted(expected_by_layer[layer_id])
        if expert_ids != expected:
            raise ValueError(
                f"canonical per_layer disagrees with rank-order identities for layer {layer_id}"
            )
    if observed_layers != set(range(num_layers)):
        raise ValueError("canonical per_layer does not represent every MoE layer exactly once")

    layer_placements = tuple(
        LayerPlacement(
            layer_id=layer_id,
            expert_ids=tuple(identity.expert_id for identity in rank_by_layer[layer_id]),
            remote_slots=tuple(identity.remote_slot for identity in rank_by_layer[layer_id]),
        )
        for layer_id in range(num_layers)
    )
    slot_by_identity = MappingProxyType(
        {(identity.layer_id, identity.expert_id): identity.remote_slot for identity in identities}
    )
    return FrozenPlacement(
        artifact_sha256=digest,
        schema=contract.schema,
        policy=contract.policy,
        status=contract.status,
        canonical_placement=contract.canonical_placement,
        model_repository=contract.model_repository,
        model_revision=contract.model_revision,
        num_layers=num_layers,
        num_experts=num_experts,
        total_expert_slots=total,
        bytes_per_slot=bytes_per_slot,
        remote_slots=remote_slots,
        remote_resident_bytes=remote_bytes,
        remote_budget_bytes=remote_budget,
        flat_ids_in_rank_order=flat_ids,
        identities_in_rank_order=tuple(identities),
        per_layer=layer_placements,
        _slot_by_identity=slot_by_identity,
    )


def load_frozen_placement(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    contract: PlacementContract | None = None,
) -> FrozenPlacement:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read --inferswarm-placement: {exc}") from exc
    return parse_frozen_placement_bytes(
        raw, expected_sha256=expected_sha256, contract=contract
    )


@dataclass(frozen=True)
class ResolvedBankLayout:
    quant_format: str
    nvfp4_backend: str
    bank_layout: str
    bank_schema: tuple[str, ...]
    bank_row_bytes: int
    auxiliary_row_bytes: int
    actual_row_bytes: int
    artifact_row_bytes_match: bool
    artifact_contract_reconciled: bool
    reconciliation: str


def _row_bytes(tensor: torch.Tensor) -> int:
    return math.prod(tensor.shape[1:]) * tensor.element_size()


def validate_runtime_bank_layout(
    placement: FrozenPlacement,
    banks: ExpertBanks,
    model_config: Any,
    *,
    contract: PlacementContract = CANONICAL_CONTRACT,
) -> ResolvedBankLayout:
    """Refuse a runtime geometry/layout that cannot satisfy the frozen experiment."""
    fields = {
        "num_moe_layers": (getattr(model_config, "num_moe_layers", None), placement.num_layers),
        "num_experts": (getattr(model_config, "num_experts", None), placement.num_experts),
        "hidden_size": (getattr(model_config, "hidden_size", None), contract.hidden_size),
        "moe_intermediate_size": (
            getattr(model_config, "moe_intermediate_size", None),
            contract.intermediate_size,
        ),
        "expert_quant": (getattr(model_config, "expert_quant", None), "nvfp4"),
    }
    for name, (actual, expected) in fields.items():
        if actual != expected:
            raise ValueError(
                f"--inferswarm-placement runtime {name} disagreement: "
                f"expected {expected!r}, got {actual!r}"
            )
    architectures = tuple(getattr(model_config, "architectures", ()) or ())
    if contract.architecture not in architectures:
        raise ValueError(
            "--inferswarm-placement requires the canonical Qwen3.6 MoE architecture "
            f"{contract.architecture!r}; runtime architectures are {architectures!r}"
        )

    quant_format = banks.quant_format
    allowed = {"nvfp4", "nvfp4_marlin", "nvfp4_b12x"}
    if quant_format not in allowed:
        raise ValueError(
            "frozen Qwen3.6 placement requires an NVFP4 production bank layout; "
            f"got {quant_format!r}"
        )
    schema = _BANK_SCHEMAS[quant_format]
    if set(banks.sources) != set(schema):
        raise ValueError(
            f"expert banks {sorted(banks.sources)} do not match {quant_format!r} schema {schema}"
        )

    row_bytes = 0
    for name in schema:
        per_layer = banks.sources[name]
        if len(per_layer) != placement.num_layers:
            raise ValueError(
                f"bank {name!r} has {len(per_layer)} layers; expected {placement.num_layers}"
            )
        head = per_layer[0]
        if head.device.type != "cpu":
            raise ValueError(f"bank {name!r} is not an existing host expert bank")
        if not head.is_contiguous() or head.size(0) != placement.num_experts:
            raise ValueError(f"bank {name!r} has incompatible source shape {tuple(head.shape)}")
        for layer_id, source in enumerate(per_layer):
            if source.device.type != "cpu":
                raise ValueError(f"bank {name!r} layer {layer_id} is not on the host")
            if (
                not source.is_contiguous()
                or source.shape != head.shape
                or source.dtype != head.dtype
            ):
                raise ValueError(
                    f"bank {name!r} layer {layer_id} has inconsistent dtype/shape/layout"
                )
        row_bytes += _row_bytes(head)

    expected_native = _BANK_BYTES_PER_EXPERT["nvfp4"](
        contract.hidden_size, contract.intermediate_size
    )
    if expected_native != placement.bytes_per_slot:
        raise ValueError(
            "runtime NVFP4 geometry does not reproduce the artifact bytes per identity: "
            f"{expected_native} != {placement.bytes_per_slot}"
        )

    gate_alpha, down_alpha = banks.gate_up_alpha, banks.down_alpha
    auxiliary_row_bytes = 0
    if quant_format == "nvfp4":
        if gate_alpha is not None or down_alpha is not None:
            raise ValueError("native NVFP4 layout must not carry folded alpha tensors")
        if row_bytes != placement.bytes_per_slot:
            raise ValueError(
                f"native NVFP4 source rows total {row_bytes} bytes; "
                f"artifact requires {placement.bytes_per_slot}"
            )
        backend, layout = "triton", "native_modelopt_nvfp4"
        reconciliation = "actual native bank rows equal the frozen artifact bytes per identity"
    else:
        if gate_alpha is None or down_alpha is None:
            raise ValueError(f"{quant_format} requires both gate_up_alpha and down_alpha")
        expected_shape = (placement.total_expert_slots,)
        if gate_alpha.shape != expected_shape or down_alpha.shape != expected_shape:
            raise ValueError(
                f"{quant_format} alpha shape disagreement: expected {expected_shape}, "
                f"got {tuple(gate_alpha.shape)} and {tuple(down_alpha.shape)}"
            )
        expected_dtype = torch.bfloat16 if quant_format == "nvfp4_marlin" else torch.float32
        if gate_alpha.dtype != expected_dtype or down_alpha.dtype != expected_dtype:
            raise ValueError(
                f"{quant_format} alpha dtype disagreement: expected {expected_dtype}, "
                f"got {gate_alpha.dtype} and {down_alpha.dtype}"
            )
        auxiliary_row_bytes = gate_alpha.element_size() + down_alpha.element_size()
        packed_scale_bytes = (
            2 * contract.intermediate_size
            * (contract.hidden_size // 2 + contract.hidden_size // 16)
            + contract.hidden_size
            * (contract.intermediate_size // 2 + contract.intermediate_size // 16)
        )
        if row_bytes != packed_scale_bytes:
            raise ValueError(
                f"{quant_format} packed/scale source rows total {row_bytes} bytes; "
                f"expected {packed_scale_bytes}"
            )
        backend = "marlin" if quant_format == "nvfp4_marlin" else "b12x"
        layout = quant_format
        reconciliation = (
            "the frozen byte count describes native packed/scale/global rows; the resolved "
            f"{backend} production layout retains the packed/scale bytes and replaces the "
            "native per-output global rows with two explicitly accounted per-expert alphas"
        )

    actual_row_bytes = row_bytes + auxiliary_row_bytes
    actual_total = actual_row_bytes * placement.remote_slots
    if actual_total > placement.remote_budget_bytes:
        raise ValueError(
            f"resolved resident representation needs {actual_total} bytes, over artifact "
            f"budget {placement.remote_budget_bytes}"
        )
    return ResolvedBankLayout(
        quant_format=quant_format,
        nvfp4_backend=backend,
        bank_layout=layout,
        bank_schema=schema,
        bank_row_bytes=row_bytes,
        auxiliary_row_bytes=auxiliary_row_bytes,
        actual_row_bytes=actual_row_bytes,
        artifact_row_bytes_match=actual_row_bytes == placement.bytes_per_slot,
        artifact_contract_reconciled=True,
        reconciliation=reconciliation,
    )


@dataclass(frozen=True)
class BankAccounting:
    name: str
    dtype: str
    row_shape: tuple[int, ...]
    bytes_per_row: int
    resident_rows: int
    total_resident_bytes: int
    device: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "per_row_shape": list(self.row_shape),
            "bytes_per_resident_row": self.bytes_per_row,
            "resident_rows": self.resident_rows,
            "total_resident_bytes": self.total_resident_bytes,
            "cuda_device": self.device,
        }


@dataclass(frozen=True)
class VerificationAccounting:
    name: str
    kind: str
    verified_rows: int
    verified_bytes: int
    source_sha256: str
    resident_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class CudaMemoryAccounting:
    allocated_before: int
    allocated_after: int
    allocated_delta: int
    reserved_before: int
    reserved_after: int
    reserved_delta: int
    free_before: int
    free_after: int
    free_delta: int
    primary_current_after_initialization: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_allocated_bytes": {
                "before": self.allocated_before,
                "after": self.allocated_after,
                "delta": self.allocated_delta,
            },
            "memory_reserved_bytes": {
                "before": self.reserved_before,
                "after": self.reserved_after,
                "delta": self.reserved_delta,
                "note": "allocator reservation granularity is not model payload size",
            },
            "mem_get_info_free_bytes": {
                "before": self.free_before,
                "after": self.free_after,
                "delta": self.free_delta,
            },
            "primary_current_after_initialization": self.primary_current_after_initialization,
        }


@dataclass(frozen=True)
class ResidentBankReport:
    placement: FrozenPlacement
    layout: ResolvedBankLayout
    secondary_uuid: str | None
    secondary_visible_ordinal: int
    banks: tuple[BankAccounting, ...]
    auxiliary: tuple[BankAccounting, ...]
    verification: tuple[VerificationAccounting, ...]
    memory: CudaMemoryAccounting

    @property
    def expert_bank_tensor_bytes(self) -> int:
        return sum(bank.total_resident_bytes for bank in self.banks)

    @property
    def auxiliary_resident_bytes(self) -> int:
        return sum(bank.total_resident_bytes for bank in self.auxiliary)

    @property
    def total_live_resident_bytes(self) -> int:
        return self.expert_bank_tensor_bytes + self.auxiliary_resident_bytes

    def as_dict(self) -> dict[str, Any]:
        p = self.placement
        layer_bytes = self.layout.bank_row_bytes
        aux_bytes = self.layout.auxiliary_row_bytes
        return {
            "placement_configured": True,
            "resident_bank_loaded": True,
            "artifact": {
                "path": None,
                "path_disclosure": "withheld: host-local paths are not runtime provenance",
                "sha256": p.artifact_sha256,
                "schema": p.schema,
                "policy": p.policy,
                "status": p.status,
                "canonical_placement": p.canonical_placement,
                "model_repository": p.model_repository,
                "model_revision": p.model_revision,
                "bytes_per_identity": p.bytes_per_slot,
                "remote_resident_expert_bytes": p.remote_resident_bytes,
                "remote_byte_budget": p.remote_budget_bytes,
            },
            "secondary_device": {
                "uuid": self.secondary_uuid,
                "visible_cuda_ordinal": self.secondary_visible_ordinal,
            },
            "resolved_quant_format": self.layout.quant_format,
            "resolved_nvfp4_backend": self.layout.nvfp4_backend,
            "resolved_bank_layout": self.layout.bank_layout,
            "mapping_rule": MAPPING_RULE,
            "resident_slots": p.remote_slots,
            "selected_identities": p.remote_slots,
            "layers": [
                {
                    "layer_id": layer.layer_id,
                    "resident_expert_count": len(layer.expert_ids),
                    "expert_ids_in_remote_rank_order": list(layer.expert_ids),
                    "remote_slots": list(layer.remote_slots),
                    "expert_bank_payload_bytes": len(layer.expert_ids) * layer_bytes,
                    "auxiliary_resident_bytes": len(layer.expert_ids) * aux_bytes,
                    "artifact_planned_bytes": len(layer.expert_ids) * p.bytes_per_slot,
                }
                for layer in p.per_layer
            ],
            "banks": [bank.as_dict() for bank in self.banks],
            "auxiliary_tensors": [bank.as_dict() for bank in self.auxiliary],
            "accounting": {
                "artifact_planned_expert_bytes": p.remote_resident_bytes,
                "expert_bank_tensor_bytes": self.expert_bank_tensor_bytes,
                "auxiliary_resident_bytes": self.auxiliary_resident_bytes,
                "total_live_resident_bank_bytes": self.total_live_resident_bytes,
                "actual_bytes_per_identity": self.layout.actual_row_bytes,
                "artifact_bytes_per_identity_match": self.layout.artifact_row_bytes_match,
                "artifact_contract_reconciled": self.layout.artifact_contract_reconciled,
                "layout_reconciliation": self.layout.reconciliation,
            },
            "cuda_memory": self.memory.as_dict(),
            "source_byte_verification": {
                "status": "passed",
                "algorithm": "sha256 over exact raw row bytes in deterministic slot order",
                "verified_rows": sum(item.verified_rows for item in self.verification),
                "verified_bytes": sum(item.verified_bytes for item in self.verification),
                "tensors": [item.as_dict() for item in self.verification],
                "mismatch": None,
            },
            "startup_expert_weight_bytes_host_to_gpu1": self.total_live_resident_bytes,
            "steady_state_expert_weight_bytes_host_to_gpu1": 0,
            "storage_boundary": (
                "startup-only resident storage; execution views are read-only by contract; "
                "no fetch, ensure, eviction, planning, transport, or execute method"
            ),
        }


def absent_resident_bank_report() -> dict[str, Any]:
    return {
        "placement_configured": False,
        "resident_bank_loaded": False,
        "artifact": None,
        "secondary_device": None,
        "resolved_quant_format": None,
        "resolved_nvfp4_backend": None,
        "resolved_bank_layout": None,
        "mapping_rule": None,
        "resident_slots": 0,
        "selected_identities": 0,
        "layers": [],
        "banks": [],
        "auxiliary_tensors": [],
        "accounting": None,
        "cuda_memory": None,
        "source_byte_verification": {"status": "not_configured"},
        "startup_expert_weight_bytes_host_to_gpu1": 0,
        "steady_state_expert_weight_bytes_host_to_gpu1": 0,
        "storage_boundary": "--inferswarm-placement was not supplied",
    }


@dataclass(frozen=True, slots=True)
class SecondaryResidentExpertBank:
    """Persistent P2 storage with a minimal P3 execution-view surface.

    The returned tensors are read-only by contract.  This object intentionally owns no
    cache-miss, fetch, load-on-demand, eviction, planning, transport, or execution API.
    """

    placement: FrozenPlacement
    report: ResidentBankReport
    _bank_tensors: dict[str, torch.Tensor]
    _auxiliary_tensors: dict[str, torch.Tensor]

    def bank_views(self) -> tuple[torch.Tensor, ...]:
        """Resident expert tensors in the validated production schema order."""
        return tuple(self._bank_tensors[name] for name in self.report.layout.bank_schema)

    def alpha_views(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Optional resident per-slot alphas for layouts that carry them."""
        gate = self._auxiliary_tensors.get("gate_up_alpha")
        down = self._auxiliary_tensors.get("down_alpha")
        if gate is None and down is None:
            return None
        if gate is None or down is None:
            raise RuntimeError("resident alpha storage is incomplete")
        return gate, down


def _copy_and_verify_rows(
    *,
    name: str,
    kind: str,
    destination: torch.Tensor,
    source_rows_for_layer,
    placement: FrozenPlacement,
    device: torch.device,
    chunk_rows: int,
    profile: dict[str, Any] | None = None,
) -> VerificationAccounting:
    source_hash = hashlib.sha256()
    resident_hash = hashlib.sha256()
    verified_rows = 0
    verified_bytes = 0
    for layer in placement.per_layer:
        for start in range(0, len(layer.expert_ids), chunk_rows):
            expert_ids = layer.expert_ids[start : start + chunk_rows]
            remote_slots = layer.remote_slots[start : start + chunk_rows]
            if not expert_ids:
                continue
            tick = time.perf_counter()
            source = source_rows_for_layer(layer.layer_id, expert_ids)
            if profile is not None:
                profile["cpu_source_gather_s"] += time.perf_counter() - tick
            if source.device.type != "cpu":
                source = source.to("cpu")
            tick = time.perf_counter()
            source = source.contiguous()
            if profile is not None:
                profile["contiguous_materialization_s"] += time.perf_counter() - tick
            if source.dtype != destination.dtype or tuple(source.shape[1:]) != tuple(
                destination.shape[1:]
            ):
                raise RuntimeError(
                    f"resident source layout mismatch for {kind} {name!r}, "
                    f"layer {layer.layer_id}: source {source.dtype}/{tuple(source.shape[1:])}, "
                    f"destination {destination.dtype}/{tuple(destination.shape[1:])}"
                )
            slots_device = torch.tensor(remote_slots, dtype=torch.long, device=device)
            # CUDA index_copy does not implement every storage dtype (notably FP8).
            # Copy the exact contiguous row bytes instead: no conversion, dequantization,
            # or layout adaptation occurs, and one primitive covers every bank dtype.
            destination_bytes = destination.view(placement.remote_slots, -1).view(torch.uint8)
            source_bytes = source.view(len(expert_ids), -1).view(torch.uint8)
            tick = time.perf_counter()
            staged = source_bytes.to(device)
            if profile is not None:
                torch.cuda.synchronize(device)
                profile["h2d_s"] += time.perf_counter() - tick
                profile["h2d_bytes"] += source_bytes.numel()
            tick = time.perf_counter()
            destination_bytes.index_copy_(0, slots_device, staged)
            if profile is not None:
                torch.cuda.synchronize(device)
                profile["gpu_scatter_s"] += time.perf_counter() - tick
            tick = time.perf_counter()
            observed_bytes = (
                destination_bytes.index_select(0, slots_device).to("cpu").contiguous()
            )
            if profile is not None:
                torch.cuda.synchronize(device)
                profile["d2h_verification_s"] += time.perf_counter() - tick
                profile["d2h_bytes"] += observed_bytes.numel()
            tick = time.perf_counter()
            if not torch.equal(source_bytes, observed_bytes):
                unequal = source_bytes.ne(observed_bytes).any(dim=1)
                row = int(unequal.nonzero()[0].item())
                raise RuntimeError(
                    f"resident byte mismatch for {kind} {name!r}: layer {layer.layer_id}, "
                    f"expert {expert_ids[row]}, remote_slot {remote_slots[row]}"
                )
            source_hash.update(source_bytes.numpy())
            resident_hash.update(observed_bytes.numpy())
            if profile is not None:
                profile["cpu_equality_hash_s"] += time.perf_counter() - tick
                profile["copy_chunks"] += 1
            rows = len(expert_ids)
            byte_count = source_bytes.numel()
            verified_rows += rows
            verified_bytes += byte_count
            del source, staged, observed_bytes, slots_device
    if source_hash.digest() != resident_hash.digest():
        raise RuntimeError(f"resident aggregate hash mismatch for {kind} {name!r}")
    return VerificationAccounting(
        name=name,
        kind=kind,
        verified_rows=verified_rows,
        verified_bytes=verified_bytes,
        source_sha256=source_hash.hexdigest(),
        resident_sha256=resident_hash.hexdigest(),
    )


def _memory_snapshot(cuda, ordinal: int) -> tuple[int, int, int]:
    allocated = int(cuda.memory_allocated(ordinal))
    reserved = int(cuda.memory_reserved(ordinal))
    try:
        free, _total = cuda.mem_get_info(ordinal)
    except TypeError:  # small fake CUDA surfaces and older torch builds
        with cuda.device(ordinal):
            free, _total = cuda.mem_get_info()
    return allocated, reserved, int(free)


def _source_alpha_rows(
    alpha: torch.Tensor,
    layer_id: int,
    expert_ids: tuple[int, ...],
    num_experts: int,
    cuda,
) -> torch.Tensor:
    flat = [layer_id * num_experts + expert_id for expert_id in expert_ids]
    if alpha.device.type == "cuda":
        with cuda.device(alpha.device):
            indices = torch.tensor(flat, dtype=torch.long, device=alpha.device)
            return alpha.index_select(0, indices).to("cpu")
    indices = torch.tensor(flat, dtype=torch.long, device="cpu")
    return alpha.index_select(0, indices)


def _construct_storage(
    placement: FrozenPlacement,
    banks: ExpertBanks,
    layout: ResolvedBankLayout,
    device: torch.device,
    cuda,
    *,
    chunk_rows: int,
    profile: dict[str, Any] | None = None,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    tuple[BankAccounting, ...],
    tuple[BankAccounting, ...],
    tuple[VerificationAccounting, ...],
]:
    if chunk_rows <= 0:
        raise ValueError("resident verification chunk_rows must be positive")
    resident_banks: dict[str, torch.Tensor] = {}
    bank_reports: list[BankAccounting] = []
    verification: list[VerificationAccounting] = []
    for name in layout.bank_schema:
        head = banks.sources[name][0]
        tick = time.perf_counter()
        destination = torch.empty(
            (placement.remote_slots, *head.shape[1:]), dtype=head.dtype, device=device
        )
        resident_banks[name] = destination
        if profile is not None:
            profile["allocation_s"] += time.perf_counter() - tick
            tensor_profile = {key: 0 for key in profile if key not in ("tensors", "mode")}
            tensor_profile["name"] = name
            tensor_profile["kind"] = "expert_bank"
        verification.append(
            _copy_and_verify_rows(
                name=name,
                kind="expert_bank",
                destination=destination,
                source_rows_for_layer=lambda layer_id, expert_ids, n=name: banks.sources[n][
                    layer_id
                ].index_select(0, torch.tensor(expert_ids, dtype=torch.long)),
                placement=placement,
                device=device,
                chunk_rows=chunk_rows,
                profile=tensor_profile if profile is not None else None,
            )
        )
        if profile is not None:
            profile["tensors"].append(tensor_profile)
            for key, value in tensor_profile.items():
                if key in profile and isinstance(value, (int, float)):
                    profile[key] += value
        row_bytes = _row_bytes(destination)
        bank_reports.append(
            BankAccounting(
                name=name,
                dtype=str(destination.dtype).removeprefix("torch."),
                row_shape=tuple(destination.shape[1:]),
                bytes_per_row=row_bytes,
                resident_rows=placement.remote_slots,
                total_resident_bytes=destination.numel() * destination.element_size(),
                device=str(destination.device),
            )
        )

    resident_aux: dict[str, torch.Tensor] = {}
    aux_reports: list[BankAccounting] = []
    for name, source in (
        ("gate_up_alpha", banks.gate_up_alpha),
        ("down_alpha", banks.down_alpha),
    ):
        if source is None:
            continue
        tick = time.perf_counter()
        destination = torch.empty((placement.remote_slots,), dtype=source.dtype, device=device)
        resident_aux[name] = destination
        if profile is not None:
            profile["allocation_s"] += time.perf_counter() - tick
            tensor_profile = {key: 0 for key in profile if key not in ("tensors", "mode")}
            tensor_profile["name"] = name
            tensor_profile["kind"] = "auxiliary"
        verification.append(
            _copy_and_verify_rows(
                name=name,
                kind="auxiliary",
                destination=destination,
                source_rows_for_layer=lambda layer_id, expert_ids, s=source: _source_alpha_rows(
                    s, layer_id, expert_ids, placement.num_experts, cuda
                ),
                placement=placement,
                device=device,
                chunk_rows=chunk_rows,
                profile=tensor_profile if profile is not None else None,
            )
        )
        if profile is not None:
            profile["tensors"].append(tensor_profile)
            for key, value in tensor_profile.items():
                if key in profile and isinstance(value, (int, float)):
                    profile[key] += value
        aux_reports.append(
            BankAccounting(
                name=name,
                dtype=str(destination.dtype).removeprefix("torch."),
                row_shape=(),
                bytes_per_row=destination.element_size(),
                resident_rows=placement.remote_slots,
                total_resident_bytes=destination.numel() * destination.element_size(),
                device=str(destination.device),
            )
        )
    return (
        resident_banks,
        resident_aux,
        tuple(bank_reports),
        tuple(aux_reports),
        tuple(verification),
    )


def _sha256_tensor_bytes(tensor: torch.Tensor, *, chunk_bytes: int = 256 << 20) -> str:
    """Hash exact CPU storage bytes without constructing a second bank-sized object."""
    raw = tensor.contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, raw.numel(), chunk_bytes):
        digest.update(raw[start : start + chunk_bytes].numpy())
    return digest.hexdigest()


def _bulk_stage_bank(
    *, source_rows_for_layer, placement: FrozenPlacement, row_shape: tuple[int, ...],
    dtype: torch.dtype, cpu_workers: int, profile: dict[str, Any],
) -> torch.Tensor:
    tick = time.perf_counter()
    staging = torch.empty(
        (placement.remote_slots, *row_shape), dtype=dtype, device="cpu", pin_memory=True
    )
    profile["allocation_s"] += time.perf_counter() - tick

    def gather(layer: LayerPlacement) -> None:
        if not layer.expert_ids:
            return
        indices = torch.tensor(layer.expert_ids, dtype=torch.long)
        slots = torch.tensor(layer.remote_slots, dtype=torch.long)
        selected = source_rows_for_layer(layer.layer_id, layer.expert_ids)
        if selected.device.type != "cpu":
            selected = selected.to("cpu")
        selected = selected.contiguous()
        if selected.dtype != dtype or tuple(selected.shape[1:]) != row_shape:
            raise RuntimeError("bulk resident source layout mismatch")
        staging.index_copy_(0, slots, selected)

    tick = time.perf_counter()
    if cpu_workers == 1:
        for layer in placement.per_layer:
            gather(layer)
    else:
        with ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix="d5-stage") as pool:
            list(pool.map(gather, placement.per_layer))
    profile["cpu_source_gather_s"] += time.perf_counter() - tick
    return staging


def _construct_storage_bulk(
    placement: FrozenPlacement, banks: ExpertBanks, layout: ResolvedBankLayout,
    device: torch.device, cuda, *, cpu_workers: int, profile: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], tuple[BankAccounting, ...],
           tuple[BankAccounting, ...], tuple[VerificationAccounting, ...]]:
    if cpu_workers not in (1, 2, 4, 8):
        raise ValueError("D5 bulk loader cpu_workers must be one of 1, 2, 4, 8")
    resident_banks: dict[str, torch.Tensor] = {}
    resident_aux: dict[str, torch.Tensor] = {}
    bank_reports: list[BankAccounting] = []
    aux_reports: list[BankAccounting] = []
    verification: list[VerificationAccounting] = []

    entries: list[tuple[str, str, torch.Tensor, Any]] = []
    for name in layout.bank_schema:
        head = banks.sources[name][0]
        entries.append((name, "expert_bank", head,
                        lambda layer_id, expert_ids, n=name: banks.sources[n][layer_id].index_select(
                            0, torch.tensor(expert_ids, dtype=torch.long))))
    for name, source in (("gate_up_alpha", banks.gate_up_alpha), ("down_alpha", banks.down_alpha)):
        if source is not None:
            entries.append((name, "auxiliary", source,
                            lambda layer_id, expert_ids, s=source: _source_alpha_rows(
                                s, layer_id, expert_ids, placement.num_experts, cuda)))

    for name, kind, head, getter in entries:
        row_shape = tuple(head.shape[1:]) if kind == "expert_bank" else ()
        item = {"name": name, "kind": kind, "allocation_s": 0.0,
                "cpu_source_gather_s": 0.0, "contiguous_materialization_s": 0.0,
                "h2d_s": 0.0, "gpu_scatter_s": 0.0, "d2h_verification_s": 0.0,
                "cpu_equality_hash_s": 0.0, "synchronization_s": 0.0,
                "h2d_bytes": 0, "d2h_bytes": 0, "copy_chunks": 1}
        staging = _bulk_stage_bank(source_rows_for_layer=getter, placement=placement,
                                   row_shape=row_shape, dtype=head.dtype,
                                   cpu_workers=cpu_workers, profile=item)
        tick = time.perf_counter()
        destination = torch.empty_like(staging, device=device)
        item["allocation_s"] += time.perf_counter() - tick
        tick = time.perf_counter()
        destination.copy_(staging, non_blocking=True)
        cuda.synchronize(device)
        item["h2d_s"] += time.perf_counter() - tick
        item["h2d_bytes"] = staging.numel() * staging.element_size()

        tick = time.perf_counter()
        observed = torch.empty_like(staging, pin_memory=True)
        item["allocation_s"] += time.perf_counter() - tick
        tick = time.perf_counter()
        observed.copy_(destination, non_blocking=True)
        cuda.synchronize(device)
        item["d2h_verification_s"] += time.perf_counter() - tick
        item["d2h_bytes"] = observed.numel() * observed.element_size()
        tick = time.perf_counter()
        if not torch.equal(staging.view(torch.uint8), observed.view(torch.uint8)):
            unequal = staging.view(placement.remote_slots, -1).view(torch.uint8).ne(
                observed.view(placement.remote_slots, -1).view(torch.uint8)).any(dim=1)
            slot = int(unequal.nonzero()[0].item())
            raise RuntimeError(f"bulk resident byte mismatch for {kind} {name!r}, remote_slot {slot}")
        source_sha = _sha256_tensor_bytes(staging)
        resident_sha = _sha256_tensor_bytes(observed)
        item["cpu_equality_hash_s"] += time.perf_counter() - tick
        if source_sha != resident_sha:
            raise RuntimeError(f"bulk resident aggregate hash mismatch for {kind} {name!r}")
        byte_count = staging.numel() * staging.element_size()
        verification.append(VerificationAccounting(name, kind, placement.remote_slots,
                                                     byte_count, source_sha, resident_sha))
        accounting = BankAccounting(name, str(destination.dtype).removeprefix("torch."),
                                    row_shape, _row_bytes(destination), placement.remote_slots,
                                    byte_count, str(destination.device))
        if kind == "expert_bank":
            resident_banks[name] = destination
            bank_reports.append(accounting)
        else:
            resident_aux[name] = destination
            aux_reports.append(accounting)
        profile["tensors"].append(item)
        for key, value in item.items():
            if key in profile and isinstance(value, (int, float)):
                profile[key] += value
        del staging, observed
    return resident_banks, resident_aux, tuple(bank_reports), tuple(aux_reports), tuple(verification)


def load_secondary_resident_bank_bulk(
    placement: FrozenPlacement, banks: ExpertBanks, model_config: Any, secondary_device,
    *, primary_visible_ordinal: int, cpu_workers: int = 4, torch_module=torch,
    cuda_module=None, resident_device: torch.device | None = None,
    contract: PlacementContract = CANONICAL_CONTRACT,
    profile: dict[str, Any] | None = None,
) -> SecondaryResidentExpertBank:
    """D5-only bank-at-a-time pinned staging with exact final-slot verification."""
    cuda = cuda_module or torch_module.cuda
    secondary_ordinal = int(secondary_device.secondary.visible_ordinal)
    device = resident_device or torch_module.device("cuda", secondary_ordinal)
    result_profile = profile if profile is not None else {}
    result_profile.clear()
    result_profile.update(mode="bulk", cpu_workers=cpu_workers, tensors=[], allocation_s=0.0,
                          cpu_source_gather_s=0.0, contiguous_materialization_s=0.0,
                          h2d_s=0.0, gpu_scatter_s=0.0, d2h_verification_s=0.0,
                          cpu_equality_hash_s=0.0, synchronization_s=0.0,
                          h2d_bytes=0, d2h_bytes=0, copy_chunks=0)
    started = time.perf_counter()
    layout = before = built = after = None
    try:
        layout = validate_runtime_bank_layout(placement, banks, model_config, contract=contract)
        before = _memory_snapshot(cuda, secondary_ordinal)
        cuda.set_device(secondary_ordinal)
        built = _construct_storage_bulk(placement, banks, layout, device, cuda,
                                        cpu_workers=cpu_workers, profile=result_profile)
        tick = time.perf_counter(); cuda.synchronize(secondary_ordinal)
        result_profile["synchronization_s"] += time.perf_counter() - tick
        after = _memory_snapshot(cuda, secondary_ordinal)
    finally:
        cuda.set_device(primary_visible_ordinal)
    restored = int(cuda.current_device()) == primary_visible_ordinal
    result_profile["total_s"] = time.perf_counter() - started
    result_profile["average_chunk_bytes"] = (result_profile["h2d_bytes"] /
                                               result_profile["copy_chunks"])
    if not restored:
        raise RuntimeError("D5 bulk resident initialization failed to restore primary CUDA device")
    assert layout is not None and before is not None and built is not None and after is not None
    resident_banks, resident_aux, bank_reports, aux_reports, verification = built
    memory = CudaMemoryAccounting(before[0], after[0], after[0] - before[0], before[1],
                                  after[1], after[1] - before[1], before[2], after[2],
                                  after[2] - before[2], restored)
    report = ResidentBankReport(placement, layout, secondary_device.secondary.uuid,
                                secondary_ordinal, bank_reports, aux_reports, verification, memory)
    if report.expert_bank_tensor_bytes != layout.bank_row_bytes * placement.remote_slots:
        raise RuntimeError("D5 bulk resident expert-bank accounting does not reconcile")
    if report.auxiliary_resident_bytes != layout.auxiliary_row_bytes * placement.remote_slots:
        raise RuntimeError("D5 bulk resident auxiliary accounting does not reconcile")
    return SecondaryResidentExpertBank(placement, report, resident_banks, resident_aux)


def load_secondary_resident_bank(
    placement: FrozenPlacement,
    banks: ExpertBanks,
    model_config: Any,
    secondary_device,
    *,
    primary_visible_ordinal: int,
    torch_module=torch,
    cuda_module=None,
    resident_device: torch.device | None = None,
    chunk_rows: int = 32,
    contract: PlacementContract = CANONICAL_CONTRACT,
    profile: dict[str, Any] | None = None,
) -> SecondaryResidentExpertBank:
    """Allocate, copy, and exactly verify the frozen rows, restoring primary always."""
    cuda = cuda_module or torch_module.cuda
    secondary_ordinal = int(secondary_device.secondary.visible_ordinal)
    device = resident_device or torch_module.device("cuda", secondary_ordinal)
    layout = None
    before = None
    built = None
    after = None
    started = time.perf_counter()
    if profile is not None:
        profile.clear()
        profile.update(mode="legacy", tensors=[], allocation_s=0.0,
                       cpu_source_gather_s=0.0, contiguous_materialization_s=0.0,
                       h2d_s=0.0, gpu_scatter_s=0.0, d2h_verification_s=0.0,
                       cpu_equality_hash_s=0.0, synchronization_s=0.0,
                       h2d_bytes=0, d2h_bytes=0, copy_chunks=0)
    try:
        layout = validate_runtime_bank_layout(
            placement, banks, model_config, contract=contract
        )
        before = _memory_snapshot(cuda, secondary_ordinal)
        cuda.set_device(secondary_ordinal)
        built = _construct_storage(
            placement, banks, layout, device, cuda, chunk_rows=chunk_rows,
            profile=profile,
        )
        tick = time.perf_counter()
        cuda.synchronize(secondary_ordinal)
        if profile is not None:
            profile["synchronization_s"] += time.perf_counter() - tick
        after = _memory_snapshot(cuda, secondary_ordinal)
    finally:
        cuda.set_device(primary_visible_ordinal)
    restored = int(cuda.current_device()) == primary_visible_ordinal
    if profile is not None:
        profile["total_s"] = time.perf_counter() - started
        profile["average_chunk_bytes"] = (profile["h2d_bytes"] / profile["copy_chunks"]
                                            if profile["copy_chunks"] else 0)
    if not restored:
        raise RuntimeError(
            "resident bank initialization failed to restore primary CUDA device "
            f"{primary_visible_ordinal}"
        )
    assert (
        layout is not None
        and before is not None
        and built is not None
        and after is not None
    )
    resident_banks, resident_aux, bank_reports, aux_reports, verification = built
    for name, tensor in (*resident_banks.items(), *resident_aux.items()):
        if tensor.device != device:
            raise RuntimeError(
                f"resident tensor {name!r} landed on {tensor.device}, expected {device}"
            )
    memory = CudaMemoryAccounting(
        allocated_before=before[0],
        allocated_after=after[0],
        allocated_delta=after[0] - before[0],
        reserved_before=before[1],
        reserved_after=after[1],
        reserved_delta=after[1] - before[1],
        free_before=before[2],
        free_after=after[2],
        free_delta=after[2] - before[2],
        primary_current_after_initialization=restored,
    )
    report = ResidentBankReport(
        placement=placement,
        layout=layout,
        secondary_uuid=secondary_device.secondary.uuid,
        secondary_visible_ordinal=secondary_ordinal,
        banks=bank_reports,
        auxiliary=aux_reports,
        verification=verification,
        memory=memory,
    )
    expected_bank_bytes = layout.bank_row_bytes * placement.remote_slots
    expected_aux_bytes = layout.auxiliary_row_bytes * placement.remote_slots
    if report.expert_bank_tensor_bytes != expected_bank_bytes:
        raise RuntimeError("resident expert-bank tensor accounting does not reconcile")
    if report.auxiliary_resident_bytes != expected_aux_bytes:
        raise RuntimeError("resident auxiliary tensor accounting does not reconcile")
    return SecondaryResidentExpertBank(
        placement, report, resident_banks, resident_aux
    )
