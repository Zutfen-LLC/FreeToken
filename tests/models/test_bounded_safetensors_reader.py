from __future__ import annotations

import json

import safetensors
import torch
from freetoken.models.loader import BoundedSafetensorsReader
from freetoken.research.r6_dense_census import load_selective_dense_block
from safetensors.torch import save_file


def _checkpoint(tmp_path, tensors):
    root = tmp_path / "checkpoint"
    root.mkdir()
    save_file(tensors, root / "model.safetensors")
    return root


def test_single_file_uses_one_short_lived_mapping_per_tensor(tmp_path, monkeypatch):
    root = _checkpoint(
        tmp_path,
        {
            "a": torch.arange(8, dtype=torch.float32),
            "b": torch.arange(3, dtype=torch.float32),
        },
    )
    events = []
    real_safe_open = safetensors.safe_open

    class TrackingOpen:
        def __init__(self, *args, **kwargs):
            self._inner = real_safe_open(*args, **kwargs)

        def __enter__(self):
            events.append("open")
            return self._inner.__enter__()

        def __exit__(self, *args):
            result = self._inner.__exit__(*args)
            events.append("close")
            return result

    monkeypatch.setattr(safetensors, "safe_open", TrackingOpen)

    reader = BoundedSafetensorsReader(
        root,
        {"a", "b"},
        page_cache_advisor=lambda *_: events.append("advise"),
    )
    retained_sources = []
    for key in ("a", "b"):
        with reader.open_tensor(key) as source:
            retained_sources.append(source)
            torch.empty_like(source).copy_(source)
            assert reader.active_mapping_count == 1
        assert source.numel() == 0
        assert events[-2:] == ["close", "advise"]
        assert reader.active_mapping_count == 0

    assert events == ["open", "close", "advise"] * 2
    assert reader.mapping_open_count == reader.mapping_close_count == 2
    assert reader.page_cache_advisory_calls == 2
    assert all(source.numel() == 0 for source in retained_sources)


def test_missing_and_unplanned_keys_fail_closed(tmp_path):
    root = _checkpoint(tmp_path, {"planned": torch.ones(2)})
    reader = BoundedSafetensorsReader(root, {"planned"})

    try:
        reader.record("unplanned")
    except RuntimeError as exc:
        assert "unplanned key" in str(exc)
    else:
        raise AssertionError("unplanned lookup did not fail closed")

    try:
        BoundedSafetensorsReader(root, {"missing"})
    except ValueError as exc:
        assert "absent from checkpoint" in str(exc)
    else:
        raise AssertionError("missing key did not fail closed")


def test_index_resolves_each_planned_key_to_its_exact_shard(tmp_path):
    root = tmp_path / "sharded"
    root.mkdir()
    save_file({"left": torch.tensor([1.0])}, root / "left.safetensors")
    save_file(
        {"right": torch.tensor([2.0]), "unplanned": torch.tensor([3.0])},
        root / "right.safetensors",
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "left": "left.safetensors",
                    "right": "right.safetensors",
                    "unplanned": "right.safetensors",
                }
            }
        )
    )
    reader = BoundedSafetensorsReader(root, {"left", "right"})

    assert reader.record("left").path.endswith("left.safetensors")
    assert reader.record("right").path.endswith("right.safetensors")
    with reader.open_tensor("right") as source:
        assert source.item() == 2.0
    assert reader.fetched_keys == ["right"]
    assert "unplanned" not in reader.fetched_keys


def test_staging_peak_is_maximum_live_bytes_not_cumulative(tmp_path):
    root = _checkpoint(
        tmp_path,
        {
            "large": torch.ones(11, dtype=torch.float32),
            "small": torch.ones(3, dtype=torch.float32),
        },
    )
    reader = BoundedSafetensorsReader(root, {"large", "small"})

    for key in ("large", "small"):
        with reader.open_tensor(key) as source:
            torch.empty_like(source).copy_(source)

    assert reader.fetched_bytes == (11 + 3) * 4
    assert reader.host_staging_peak_live_tensor_bytes == 11 * 4
    assert reader.largest_raw_tensor_bytes == 11 * 4
    assert reader.host_staging_current_bytes == 0


def test_explicit_group_rejects_a_bound_smaller_than_header_bytes(tmp_path):
    root = _checkpoint(
        tmp_path,
        {"a": torch.ones(4), "b": torch.ones(5)},
    )
    reader = BoundedSafetensorsReader(root, {"a", "b"})

    try:
        with reader.open_group(("a", "b"), max_live_bytes=35):
            pass
    except ValueError as exc:
        assert "exceeds bound" in str(exc)
    else:
        raise AssertionError("undersized explicit group bound was accepted")
    assert reader.fetched_keys == []


def test_selective_standalone_transfer_does_not_retain_source_storage(tmp_path):
    values = torch.arange(6, dtype=torch.float32)
    root = _checkpoint(tmp_path, {"weight": values})
    reader = BoundedSafetensorsReader(root, {"weight"})

    class Module:
        def __init__(self):
            self.weight = torch.empty(6)

        def state_dict(self):
            return {"weight": self.weight}

        def load_state_dict(self, state):
            self.weight = state["weight"]

    result = load_selective_dense_block(
        str(root),
        {
            "spec": {
                "start_layer": 0,
                "end_layer": 1,
                "owns_embeddings": True,
                "owns_final_norm_head": True,
            },
            "allowed_tensor_keys": ["weight"],
        },
        lambda _spec: Module(),
        device="cpu",
        reader=reader,
    )

    torch.testing.assert_close(result["module"].weight, values)
    assert reader.host_staging_current_bytes == 0
    assert reader.mapping_open_count == reader.mapping_close_count == 1
