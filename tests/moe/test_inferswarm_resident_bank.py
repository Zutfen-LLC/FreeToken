from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import freetoken.moe.inferswarm_resident_bank as resident_mod
from freetoken.engine.engine import Engine
from freetoken.layers.moe import OffloadMoELayer
from freetoken.moe.expert_banks import ExpertBanks
from freetoken.moe.inferswarm_resident_bank import (
    LayerPlacement,
    PlacementContract,
    ResolvedBankLayout,
    _construct_storage,
    load_frozen_placement,
    load_secondary_resident_bank,
    parse_frozen_placement_bytes,
    validate_runtime_bank_layout,
)
from freetoken.moe.offload_cache import _BANK_SCHEMAS


CONTRACT = PlacementContract(
    schema="test.placement/1",
    policy="test-policy",
    status="FROZEN_BEFORE_PHASE1_PERFORMANCE",
    canonical_placement="selected",
    model_repository="test/repo",
    model_revision="a" * 40,
    num_layers=2,
    num_experts=3,
    remote_slots=3,
    bytes_per_slot=6,
    remote_resident_bytes=18,
    remote_budget_bytes=24,
    hidden_size=1,
    intermediate_size=1,
    architecture="TestArch",
)
FLAT_IDS = [4, 0, 5]


def _placement_doc(contract: PlacementContract = CONTRACT) -> dict:
    identities = [
        {
            "flat_id": flat_id,
            "layer": flat_id // contract.num_experts,
            "expert_id": flat_id % contract.num_experts,
        }
        for flat_id in FLAT_IDS
    ]
    per_layer = []
    for layer_id in range(contract.num_layers):
        per_layer.append(
            {
                "layer": layer_id,
                "expert_ids": sorted(
                    item["expert_id"] for item in identities if item["layer"] == layer_id
                ),
            }
        )
    return {
        "schema": contract.schema,
        "policy_id": contract.policy,
        "status": contract.status,
        "canonical_remote_placement": contract.canonical_placement,
        "source": {
            "model_repository": contract.model_repository,
            "model_revision": contract.model_revision,
        },
        "geometry": {
            "num_moe_layers": contract.num_layers,
            "num_experts_per_layer": contract.num_experts,
            "total_expert_slots": contract.num_layers * contract.num_experts,
        },
        "budget": {
            "bytes_per_slot": contract.bytes_per_slot,
            "remote_budget_bytes": contract.remote_budget_bytes,
            "remote_slots": contract.remote_slots,
            "remote_resident_bytes": contract.remote_resident_bytes,
        },
        "placements": {
            contract.canonical_placement: {
                "slot_count": contract.remote_slots,
                "flat_ids_in_rank_order": list(FLAT_IDS),
                "identities_in_rank_order": identities,
                "per_layer": per_layer,
            }
        },
    }


def _raw(doc: dict) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


def _parse(doc: dict | None = None, contract: PlacementContract = CONTRACT):
    raw = _raw(doc or _placement_doc(contract))
    return parse_frozen_placement_bytes(
        raw, expected_sha256=hashlib.sha256(raw).hexdigest(), contract=contract
    )


def test_valid_synthetic_placement_and_deterministic_remote_slots():
    placement = _parse()
    assert placement.flat_ids_in_rank_order == tuple(FLAT_IDS)
    assert placement.remote_slot(1, 1) == 0
    assert placement.remote_slot(0, 0) == 1
    assert placement.remote_slot(1, 2) == 2
    assert placement.per_layer == (
        LayerPlacement(layer_id=0, expert_ids=(0,), remote_slots=(1,)),
        LayerPlacement(layer_id=1, expert_ids=(1, 2), remote_slots=(0, 2)),
    )


def test_load_computes_the_hash_of_exact_file_bytes(tmp_path):
    raw = _raw(_placement_doc()) + b"\n"
    path = tmp_path / "placement.json"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    placement = load_frozen_placement(
        path, expected_sha256=digest, contract=CONTRACT
    )
    assert placement.artifact_sha256 == digest
    with pytest.raises(ValueError, match="SHA-256 disagreement"):
        load_frozen_placement(path, expected_sha256="0" * 64, contract=CONTRACT)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda d: d.__setitem__("schema", "unsupported"), "schema disagreement"),
        (lambda d: d.__setitem__("policy_id", "wrong"), "policy_id disagreement"),
        (lambda d: d.__setitem__("status", "DRAFT"), "status disagreement"),
        (
            lambda d: d["source"].__setitem__("model_repository", "wrong/repo"),
            "model_repository disagreement",
        ),
        (
            lambda d: d["source"].__setitem__("model_revision", "b" * 40),
            "model_revision disagreement",
        ),
        (
            lambda d: d["geometry"].__setitem__("num_moe_layers", 3),
            "num_moe_layers disagreement",
        ),
        (
            lambda d: d["budget"].__setitem__("remote_slots", 2),
            "remote_slots disagreement",
        ),
        (
            lambda d: d["budget"].__setitem__("bytes_per_slot", 7),
            "bytes_per_slot disagreement",
        ),
        (
            lambda d: d["budget"].__setitem__("remote_resident_bytes", 17),
            "remote_resident_bytes arithmetic",
        ),
        (
            lambda d: d.__setitem__("canonical_remote_placement", "missing"),
            "canonical_remote_placement disagreement",
        ),
    ],
)
def test_unsupported_or_disagreeing_frozen_fields_are_rejected(mutation, match):
    doc = _placement_doc()
    mutation(doc)
    with pytest.raises(ValueError, match=match):
        _parse(doc)


def test_duplicate_flat_id_is_rejected():
    doc = _placement_doc()
    doc["placements"]["selected"]["flat_ids_in_rank_order"][1] = FLAT_IDS[0]
    with pytest.raises(ValueError, match="duplicate flat IDs"):
        _parse(doc)


def test_duplicate_identity_is_rejected_even_when_other_redundancy_is_malformed():
    doc = _placement_doc()
    records = doc["placements"]["selected"]["identities_in_rank_order"]
    records[2] = copy.deepcopy(records[0])
    with pytest.raises(ValueError):
        _parse(doc)


@pytest.mark.parametrize(
    "field,value,match",
    [("layer", 2, "out-of-range layer"), ("expert_id", 3, "out-of-range expert_id")],
)
def test_out_of_range_identity_is_rejected(field, value, match):
    doc = _placement_doc()
    doc["placements"]["selected"]["identities_in_rank_order"][0][field] = value
    with pytest.raises(ValueError, match=match):
        _parse(doc)


def test_bad_flat_id_arithmetic_is_rejected():
    doc = _placement_doc()
    doc["placements"]["selected"]["identities_in_rank_order"][0]["flat_id"] = 3
    with pytest.raises(ValueError, match="flat_id arithmetic"):
        _parse(doc)


def test_rank_order_and_identity_records_must_agree():
    doc = _placement_doc()
    doc["placements"]["selected"]["flat_ids_in_rank_order"][0] = 3
    with pytest.raises(ValueError, match="rank-order/identity disagreement"):
        _parse(doc)


def test_per_layer_must_agree_with_identities():
    doc = _placement_doc()
    doc["placements"]["selected"]["per_layer"][1]["expert_ids"] = [0, 1]
    with pytest.raises(ValueError, match="per_layer disagrees"):
        _parse(doc)


def test_slot_count_disagreement_is_rejected():
    doc = _placement_doc()
    doc["placements"]["selected"]["slot_count"] = 2
    with pytest.raises(ValueError, match="slot_count disagreement"):
        _parse(doc)


def test_over_budget_placement_is_rejected():
    contract = replace(CONTRACT, remote_budget_bytes=17)
    doc = _placement_doc(contract)
    with pytest.raises(ValueError, match="over budget"):
        _parse(doc, contract)


def _native_banks() -> ExpertBanks:
    sources = {}
    for bank_index, name in enumerate(_BANK_SCHEMAS["nvfp4"]):
        sources[name] = [
            (torch.arange(3, dtype=torch.uint8) + bank_index * 20 + layer * 3).view(3, 1)
            for layer in range(2)
        ]
    return ExpertBanks("nvfp4", sources)


def _model_config():
    return SimpleNamespace(
        num_moe_layers=2,
        num_experts=3,
        hidden_size=1,
        moe_intermediate_size=1,
        expert_quant="nvfp4",
        architectures=["TestArch"],
    )


class _FakeCuda:
    def __init__(self, current=0):
        self.current = current
        self.set_calls = []

    def set_device(self, ordinal):
        self.current = int(ordinal)
        self.set_calls.append(int(ordinal))

    def current_device(self):
        return self.current

    def memory_allocated(self, _ordinal):
        return 100

    def memory_reserved(self, _ordinal):
        return 200

    def mem_get_info(self, ordinal=None):
        ordinal = self.current if ordinal is None else int(ordinal)
        return 1000 - ordinal, 2000

    def synchronize(self, _ordinal=None):
        pass


def _secondary(ordinal=1):
    return SimpleNamespace(
        secondary=SimpleNamespace(visible_ordinal=ordinal, uuid=f"GPU-secondary-{ordinal}")
    )


def test_resident_bank_selects_exact_rows_in_canonical_slot_order_and_accounts_bytes():
    cuda = _FakeCuda(current=0)
    bank = load_secondary_resident_bank(
        _parse(),
        _native_banks(),
        _model_config(),
        _secondary(),
        primary_visible_ordinal=0,
        cuda_module=cuda,
        resident_device=torch.device("cpu"),
        chunk_rows=1,
        contract=CONTRACT,
    )
    assert set(bank._bank_tensors) == set(_BANK_SCHEMAS["nvfp4"])
    for bank_index, name in enumerate(_BANK_SCHEMAS["nvfp4"]):
        assert bank._bank_tensors[name].flatten().tolist() == [
            bank_index * 20 + 4,
            bank_index * 20,
            bank_index * 20 + 5,
        ]
        assert bank._bank_tensors[name].dtype == torch.uint8
        assert bank._bank_tensors[name].shape == (3, 1)
    assert bank.report.expert_bank_tensor_bytes == 18
    assert bank.report.auxiliary_resident_bytes == 0
    assert bank.report.total_live_resident_bytes == 18
    report = bank.report.as_dict()
    assert report["source_byte_verification"]["status"] == "passed"
    assert report["source_byte_verification"]["verified_bytes"] == 18
    assert report["startup_expert_weight_bytes_host_to_gpu1"] == 18
    assert report["steady_state_expert_weight_bytes_host_to_gpu1"] == 0
    assert report["remote_execution_enabled"] is False
    assert report["decode_dispatches_to_secondary"] == 0
    assert cuda.current_device() == 0


def test_wrong_or_missing_bank_schema_is_rejected():
    banks = _native_banks()
    banks.sources.pop("down_global")
    with pytest.raises(ValueError, match="do not match"):
        validate_runtime_bank_layout(_parse(), banks, _model_config(), contract=CONTRACT)


def test_auxiliary_alpha_rows_are_selected_and_accounted_for_the_resolved_layout():
    placement = _parse()
    sources = {}
    for bank_index, name in enumerate(_BANK_SCHEMAS["nvfp4_marlin"]):
        sources[name] = [
            (torch.arange(3, dtype=torch.uint8) + bank_index * 20 + layer * 3).view(3, 1)
            for layer in range(2)
        ]
    gate = torch.arange(6, dtype=torch.bfloat16) + 100
    down = torch.arange(6, dtype=torch.bfloat16) + 200
    banks = ExpertBanks(
        "nvfp4_marlin", sources, gate_up_alpha=gate, down_alpha=down
    )
    layout = ResolvedBankLayout(
        quant_format="nvfp4_marlin",
        nvfp4_backend="marlin",
        bank_layout="nvfp4_marlin",
        bank_schema=_BANK_SCHEMAS["nvfp4_marlin"],
        bank_row_bytes=4,
        auxiliary_row_bytes=4,
        actual_row_bytes=8,
        artifact_row_bytes_match=False,
        artifact_contract_reconciled=True,
        reconciliation="synthetic",
    )
    _, aux, _, aux_reports, verification = _construct_storage(
        placement, banks, layout, torch.device("cpu"), _FakeCuda(), chunk_rows=2
    )
    assert aux["gate_up_alpha"].tolist() == [104, 100, 105]
    assert aux["down_alpha"].tolist() == [204, 200, 205]
    assert sum(item.total_resident_bytes for item in aux_reports) == 12
    assert {item.name for item in verification if item.kind == "auxiliary"} == {
        "gate_up_alpha",
        "down_alpha",
    }


def test_copy_uses_exact_byte_views_for_fp8_rows():
    placement = _parse()
    sources = {
        "scale": [
            torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float8_e4m3fn),
            torch.tensor([[4.0], [5.0], [6.0]], dtype=torch.float8_e4m3fn),
        ]
    }
    banks = ExpertBanks("synthetic", sources)
    layout = ResolvedBankLayout(
        quant_format="synthetic",
        nvfp4_backend="synthetic",
        bank_layout="synthetic",
        bank_schema=("scale",),
        bank_row_bytes=1,
        auxiliary_row_bytes=0,
        actual_row_bytes=1,
        artifact_row_bytes_match=False,
        artifact_contract_reconciled=True,
        reconciliation="synthetic",
    )
    resident, _, _, _, verification = _construct_storage(
        placement, banks, layout, torch.device("cpu"), _FakeCuda(), chunk_rows=2
    )
    assert resident["scale"].tolist() == [[5.0], [1.0], [6.0]]
    assert verification[0].source_sha256 == verification[0].resident_sha256


def test_primary_device_is_restored_after_success():
    cuda = _FakeCuda(current=0)
    load_secondary_resident_bank(
        _parse(), _native_banks(), _model_config(), _secondary(),
        primary_visible_ordinal=0, cuda_module=cuda, resident_device=torch.device("cpu"),
        contract=CONTRACT,
    )
    assert cuda.set_calls[-1] == 0
    assert cuda.current_device() == 0


def test_resident_allocation_uses_the_resolved_secondary_ordinal_not_cuda_one():
    cuda = _FakeCuda(current=2)
    bank = load_secondary_resident_bank(
        _parse(), _native_banks(), _model_config(), _secondary(0),
        primary_visible_ordinal=2, cuda_module=cuda, resident_device=torch.device("cpu"),
        contract=CONTRACT,
    )
    assert cuda.set_calls[0] == 0
    assert cuda.set_calls[-1] == 2
    assert bank.report.secondary_visible_ordinal == 0


def test_primary_device_is_restored_after_construction_failure(monkeypatch):
    cuda = _FakeCuda(current=0)

    def fail(*_args, **_kwargs):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(resident_mod, "_construct_storage", fail)
    with pytest.raises(RuntimeError, match="copy failed"):
        load_secondary_resident_bank(
            _parse(), _native_banks(), _model_config(), _secondary(),
            primary_visible_ordinal=0, cuda_module=cuda, resident_device=torch.device("cpu"),
            contract=CONTRACT,
        )
    assert cuda.current_device() == 0
    assert cuda.set_calls[-1] == 0


def test_engine_p2_helper_consumes_the_already_loaded_banks(monkeypatch):
    engine = Engine.__new__(Engine)
    engine.inferswarm_placement = object()
    engine.inferswarm_secondary_device = object()
    engine.inferswarm_resident_bank = None
    engine.device = SimpleNamespace(index=0)
    banks = object()
    captured = {}

    def fake_load(placement, supplied_banks, model_config, secondary, **kwargs):
        captured.update(
            placement=placement,
            banks=supplied_banks,
            model_config=model_config,
            secondary=secondary,
            kwargs=kwargs,
        )
        return SimpleNamespace(
            report=SimpleNamespace(
                placement=SimpleNamespace(remote_slots=3),
                total_live_resident_bytes=18,
                secondary_visible_ordinal=1,
            )
        )

    monkeypatch.setattr(resident_mod, "load_secondary_resident_bank", fake_load)
    config = SimpleNamespace(model_config=object())
    engine._init_inferswarm_resident_bank(config, banks)
    assert captured["banks"] is banks
    assert captured["kwargs"]["primary_visible_ordinal"] == 0
    source = inspect.getsource(Engine._init_offload_moe_cache)
    assert source.count("load_expert_banks(") == 1
    assert "_init_inferswarm_resident_bank(config, banks)" in source


def test_resident_bank_is_not_an_execution_backend_or_moe_layer_attachment():
    assert not hasattr(OffloadMoELayer, "inferswarm_resident_bank")
    assert "inferswarm" not in inspect.getsource(OffloadMoELayer._decode_routed)
    assert "inferswarm" not in inspect.getsource(OffloadMoELayer._prefill_routed)
    public = {
        name
        for name in dir(resident_mod.SecondaryResidentExpertBank)
        if not name.startswith("_")
    }
    assert public <= {"placement", "report"}


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two visible CUDA devices",
)
def test_actual_secondary_cuda_allocation_uses_explicit_ordinal_and_restores_primary():
    primary, secondary = 0, 1
    torch.cuda.set_device(primary)
    bank = load_secondary_resident_bank(
        _parse(),
        _native_banks(),
        _model_config(),
        _secondary(secondary),
        primary_visible_ordinal=primary,
        resident_device=torch.device("cuda", secondary),
        contract=CONTRACT,
    )
    assert all(t.device.index == secondary for t in bank._bank_tensors.values())
    assert torch.cuda.current_device() == primary
