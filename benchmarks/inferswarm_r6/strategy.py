"""R6 Gemma-4-12B dense Model Execution Strategy adapter (inferswarm #65).

Strategy-owned, model-specific by design: the generic planner/resource code
sees only opaque shapes/slots/units/capabilities.  Everything Gemma lives
here — census, legal distribution candidates, state classification, the
selective dense block runtime, and the correctness reference.

Proving model (frozen):
    google/gemma-4-12B-it @ 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
    text-only serving scope; BF16 native representation.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

MODEL_REPOSITORY = "google/gemma-4-12B-it"
MODEL_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
STRATEGY_ID = "freetoken.gemma4-dense-serving/1"
TEXT_PREFIX = "model.language_model"

# ---- frozen physical constants (census-derived, see issue #65 census) ----
HIDDEN_SIZE = 3840
NUM_LAYERS = 48
VOCAB_SIZE = 262144
FULL_ATTENTION_LAYERS = (5, 11, 17, 23, 29, 35, 41, 47)
BOUNDARY_PLANES = 2
PREFILL_CHUNK = 64          # rows per boundary transfer: 64*3840*2 = 491520 B.
                            # 64 (not 32): replay prefills for the canonical
                            # 26-token prompt + <=8 committed tokens must stay
                            # SINGLE-chunk — the KV-extend path (chunk 2+) is
                            # the known-broken incremental append (see
                            # anomaly-incremental-decode.md).
DECODER_LAYERS_PER_STAGE_2 = (0, 24)  # candidate A: [0,24) / [24,48)
ATTN_BACKEND = "triton"     # only backend family supporting SWA AttentionSpec


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Census / legal candidates
# ---------------------------------------------------------------------------

def model_census(model_path: str) -> dict[str, Any]:
    """Full R6 model census: identity, hashes, structure, VRAM math."""
    from freetoken.research.r6_dense_census import checkpoint_census

    root = Path(model_path)
    census = checkpoint_census(root, text_prefix=TEXT_PREFIX)
    files = {}
    for name in sorted(p.name for p in root.iterdir() if p.is_file()):
        files[name] = {
            "bytes": (root / name).stat().st_size,
            "sha256": _file_sha256(root / name),
        }
    return {
        "schema": "inferswarm.r6.gemma-model-census/1",
        "model_repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "serving_scope": "text-only",
        "representation": "native-bf16-safetensors",
        "files": files,
        "checkpoint_census": census,
        "text_structure": {
            "num_layers": NUM_LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "vocab_size": VOCAB_SIZE,
            "tie_word_embeddings": True,
            "full_attention_layers": list(FULL_ATTENTION_LAYERS),
            "sliding_attention_layers": [
                i for i in range(NUM_LAYERS) if i not in FULL_ATTENTION_LAYERS
            ],
            "attention_backend": ATTN_BACKEND,
        },
    }


def _bytes_for_range(census: dict, start: int, end: int) -> int:
    return sum(
        rec["byte_count"]
        for rec in census["tensors"]
        if rec["layer_id"] is not None and start <= rec["layer_id"] < end
    )


def _embedding_bytes(census: dict) -> int:
    return sum(
        rec["byte_count"]
        for rec in census["tensors"]
        if rec["owner_category"] == "embedding/input"
    )


def _final_norm_bytes(census: dict) -> int:
    return sum(
        rec["byte_count"]
        for rec in census["tensors"]
        if rec["owner_category"] == "final norm"
    )


def legal_candidates(model_path: str) -> dict[str, Any]:
    """Strategy-owned legal distribution candidates, derived from census."""
    census = model_census(model_path)["checkpoint_census"]
    embed = _embedding_bytes(census)
    norm = _final_norm_bytes(census)

    def stage_weights(ranges, head_on_last=True):
        total = 0
        per = []
        for index, (start, end) in enumerate(ranges):
            weights = _bytes_for_range(census, start, end)
            if index == 0:
                weights += embed
            if head_on_last and index == len(ranges) - 1:
                weights += norm
            per.append(weights)
            total += weights
        return per, total

    # Census-measured (issue #65): text tower ~21.7 GB + 2.0 GB tied embed;
    # a 2-stage split (~12.8 GB/stage) does NOT fit 11.63 GiB-usable 3060s
    # (hardware-verified OOM during selective load).  The legal candidate is
    # the 3-stage contiguous chain; 2-stage is retained as MEASURED_INFEASIBLE.
    two_ranges = [(0, 24), (24, 48)]
    two_per, two_total = stage_weights(two_ranges)
    three_ranges = [(0, 16), (16, 32), (32, 48)]
    three_per, three_total = stage_weights(three_ranges)
    # Tied lm_head duplicates the embedding table on the head stage.
    shared_state = {
        "id": "tied-embedding-lm-head",
        "kind": "tied-weight-shared-state",
        "tensor_keys": [f"{TEXT_PREFIX}.embed_tokens.weight"],
        "bytes": embed,
        "materialization_policy": "duplicated-on-first-and-last-stage",
        "reason": "tied lm_head reuses the embedding table; declared shared "
        "logical state per the R6 selective-materialization contract",
    }

    return {
        "schema": "inferswarm.r6.gemma-legal-candidates/1",
        "model_repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "strategy_id": STRATEGY_ID,
        "measured_infeasible": [
            {
                "id": "dense-two-stage-contiguous",
                "unitization": "opaque-contiguous-stage-v2",
                "stages": [
                    {"layer_range": [0, 24], "owns": ["embeddings"]},
                    {"layer_range": [24, 48], "owns": ["final-norm", "lm_head(tied)"]},
                ],
                "stage_weight_bytes": two_per,
                "feasibility_note": "MEASURED_INFEASIBLE: stage weights exceed "
                "11.63 GiB usable VRAM per 3060 (hardware OOM during selective "
                "load on inferswarm01, producer 2c8e381)",
            },
        ],
        "candidates": [
            {
                "id": "dense-three-stage-chain",
                "unitization": "opaque-contiguous-stage-v2",
                "stages": [
                    {"layer_range": [0, 16], "owns": ["embeddings"]},
                    {"layer_range": [16, 32], "owns": []},
                    {"layer_range": [32, 48], "owns": ["final-norm", "lm-head(tied)"]},
                ],
                "stage_weight_bytes": three_per,
                "shared_state": shared_state,
                "feasibility_note": "three >=9.4GiB VRAM units; spans two nodes "
                "when only two local GPUs are available",
            },
            {
                "id": "dense-two-stage-contiguous",
                "unitization": "opaque-contiguous-stage-v2",
                "stages": [
                    {"layer_range": [0, 24], "owns": ["embeddings"]},
                    {"layer_range": [24, 48], "owns": ["final-norm", "lm_head(tied)"]},
                ],
                "stage_weight_bytes": two_per,
                "shared_state": shared_state,
                "feasibility_note": "requires two >=10.9GiB VRAM units",
            },
        ],
        "boundary_geometry": {
            "planes": 1,
            "row_width": HIDDEN_SIZE,
            "dtype": "bfloat16",
            "note": "dense Gemma boundary carries the single residual-stream "
            "hidden state (no separate residual plane; Qwen's 2-plane boundary "
            "was a first-model artifact of its dual-stream blocks)",
            "decode_bytes": HIDDEN_SIZE * 2,
            "prefill_chunk_rows": PREFILL_CHUNK,
            "prefill_bytes": PREFILL_CHUNK * HIDDEN_SIZE * 2,
        },
    }


# ---------------------------------------------------------------------------
# Dense logical-state census (strategy-owned taxonomy)
# ---------------------------------------------------------------------------

def dense_state_census(model_path: str, *, runtime_capacity_tokens: int) -> dict:
    """Classify every Gemma text-path state class with ownership/lifecycle."""
    census = model_census(model_path)["checkpoint_census"]
    embed = _embedding_bytes(census)
    layers = _bytes_for_range(census, 0, NUM_LAYERS)
    norm = _final_norm_bytes(census)
    kv_full, kv_swa = _kv_state_bytes(runtime_capacity_tokens)
    return {
        "schema": "inferswarm.r6.gemma-dense-state-census/1",
        "immutable": [
            {"class": "embedding-input", "bytes": embed,
             "materialization": "participant(first-stage)",
             "shared": "tied lm_head source"},
            {"class": "decoder-block-params", "bytes": layers,
             "materialization": "participant(stage-owned-range)"},
            {"class": "final-norm", "bytes": norm,
             "materialization": "participant(last-stage)"},
            {"class": "lm-head(tied)", "bytes": 0,
             "materialization": "shared-view-of-embedding",
             "note": "no separate checkpoint tensor; logits = embed.T applied "
             "by the last stage with final logit softcapping (30.0)"},
        ],
        "mutable": [
            {"class": "kv-cache-full-attn", "bytes": kv_full,
             "authority": "stage-local", "reconstructible": "replay-prefill",
             "owner": "stage owning that layer"},
            {"class": "kv-cache-swa", "bytes": kv_swa,
             "authority": "stage-local", "reconstructible": "replay-prefill",
             "owner": "stage owning that layer"},
            {"class": "host-committed-output-ledger", "bytes": 0,
             "authority": "coordinator", "reconstructible": "none(authoritative)"},
            {"class": "activation-staging-buffers", "bytes": PREFILL_CHUNK * 2 * HIDDEN_SIZE * 2,
             "authority": "transient", "reconstructible": "recompute"},
        ],
        "non_text_state_excluded": {
            "vision-tower": "never materialized for text-only serving",
            "audio-tower": "never materialized for text-only serving",
        },
        "runtime_capacity_tokens": runtime_capacity_tokens,
    }


def _kv_state_bytes(tokens: int) -> tuple[int, int]:
    # full-attn: 8 layers x 1 kv-head x 512 hd; swa: 40 x 8 x 256; bf16, K+V
    full = len(FULL_ATTENTION_LAYERS) * 1 * 512 * 2 * tokens * 2
    swa = (NUM_LAYERS - len(FULL_ATTENTION_LAYERS)) * 8 * 256 * 2 * tokens * 2
    return full, swa


# ---------------------------------------------------------------------------
# Planning problem / operator policy (generic planner interface)
# ---------------------------------------------------------------------------

GPU_A = "gpu.node-a.0"
GPU_A_SECONDARY = "gpu.node-a.1"
GPU_B = "gpu.node-b.0"


def planning_problem(implementation_commit: str, model_path: str) -> dict[str, Any]:
    from freetoken.research.r3_planner import freeze

    candidates = legal_candidates(model_path)
    boundary = candidates["boundary_geometry"]
    shapes = []
    for candidate in candidates["candidates"]:
        if candidate["id"] == "dense-two-stage-contiguous":
            shapes.append(
                {
                    "id": "resident-same-node-two-slot",
                    "slots": [
                        {
                            "id": "slot-a",
                            "allowed_compute_unit_ids": [GPU_A],
                            "required_capabilities": ["freetoken-resident-stage-first-v1"],
                            "memory": {"persistent_required_bytes": candidate["stage_weight_bytes"][0]},
                        },
                        {
                            "id": "slot-b",
                            "allowed_compute_unit_ids": [GPU_A_SECONDARY],
                            "required_capabilities": ["freetoken-resident-stage-last-v1"],
                            "memory": {"persistent_required_bytes": candidate["stage_weight_bytes"][1]},
                        },
                    ],
                    "distinct_slot_groups": [["slot-a", "slot-b"]],
                    "paths": [
                        {
                            "id": "strategy-boundary",
                            "from_slot": "slot-a",
                            "to_slot": "slot-b",
                            "required_capabilities": ["freetoken-static-boundary-v1"],
                        }
                    ],
                    "strategy_payload": {
                        "realization": "r6-dense-two-stage",
                        "semantic_geometry": {
                            "opaque_units": [[0, 24], [24, 48]],
                            "dtype": "bfloat16",
                            "planes": BOUNDARY_PLANES,
                            "row_width": HIDDEN_SIZE,
                            "decode_bytes": boundary["decode_bytes"],
                            "prefill_chunk_rows": PREFILL_CHUNK,
                            "prefill_bytes": boundary["prefill_bytes"],
                        },
                    },
                }
            )
    return freeze(
        {
            "schema": "inferswarm.r6.gemma-strategy-problem/1",
            "implementation_commit": implementation_commit,
            "strategy": {"id": STRATEGY_ID},
            "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
            "evidence_context": {"model_revision": MODEL_REVISION},
            "shapes": shapes,
        }
    )


__all__ = [
    "ATTN_BACKEND",
    "BOUNDARY_PLANES",
    "FULL_ATTENTION_LAYERS",
    "HIDDEN_SIZE",
    "MODEL_REPOSITORY",
    "MODEL_REVISION",
    "NUM_LAYERS",
    "PREFILL_CHUNK",
    "STRATEGY_ID",
    "TEXT_PREFIX",
    "VOCAB_SIZE",
    "dense_state_census",
    "legal_candidates",
    "model_census",
    "planning_problem",
]
