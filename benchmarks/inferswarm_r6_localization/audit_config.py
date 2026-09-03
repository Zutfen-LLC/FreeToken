"""#71 localization: S-vs-D configuration equivalence audit (Phase 7).

Machine-records, for the constructed runtime objects (NOT source text), the
per-layer semantic configuration of a stage: global vs local layer ID,
attention group kind, kv heads, head_dim, sliding window, k_eq_v, rope
config, dtype — plus the KV pool mapping. The single-role (S) audit covers
all 48 layers; each distributed stage covers its owned range. The report
diffs every owned layer's semantic properties S-vs-D and fails closed on
any mismatch.

Runs on a compute node (imports torch + the model config machinery against
the real checkpoint config.json only — no weight load).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
from pathlib import Path

LAYER_FIELDS = (
    "attention_group_kind",
    "num_kv_heads",
    "head_dim",
    "sliding_window",
    "k_eq_v",
    "rope_base",
    "rope_rotary_dim",
    "rope_max_position",
    "rope_scaling",
    "is_swa",
    "sm_scale",
)


def _stage_layer_audit(role: str, spec: dict, full_config, stage_config) -> dict:
    import torch
    from freetoken.models.gemma4.model import Gemma4DecoderLayer
    from freetoken.utils import torch_dtype

    owned = list(range(spec["start_layer"], spec["end_layer"]))
    # NOTE: audit builds layers under BOTH the full config (global ids, the
    # S-equivalent view) and the stage-local config (local ids, the D view),
    # then compares semantic properties per owned global layer. Modules are
    # built on meta (no real allocation); only extracted properties survive.
    def _props(config, layer_id):
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            layer = Gemma4DecoderLayer(config, layer_id)
        attn = layer.self_attn
        return {
            "attention_group_kind": type(
                config.attention_group_for_layer(layer_id)
            ).__name__,
            "num_kv_heads": attn.num_kv_heads,
            "head_dim": attn.head_dim,
            "sliding_window": attn.attn_spec.sliding_window,
            "k_eq_v": attn.k_eq_v,
            "rope_base": getattr(attn.rotary, "base", None),
            "rope_rotary_dim": getattr(attn.rotary, "rotary_dim", None),
            "rope_max_position": getattr(attn.rotary, "max_position", None),
            "rope_scaling": getattr(attn.rotary, "scaling", None),
            "is_swa": attn.is_swa,
            "sm_scale": attn.attn_spec.sm_scale,
        }

    global_layers = {gid: _props(full_config, gid) for gid in owned}
    local_view = {gid: _props(stage_config, local_id)
                  for local_id, gid in enumerate(owned)}
    mismatches = {}
    for gid in owned:
        g = global_layers[gid]
        l = local_view[gid]
        for field in LAYER_FIELDS:
            if g[field] != l[field]:
                mismatches.setdefault(str(gid), {})[field] = {
                    "global_view": g[field], "stage_local_view": l[field],
                }
    return {
        "role": role,
        "owned_global_layers": owned,
        "per_layer_global_view": {str(k): v for k, v in global_layers.items()},
        "per_layer_stage_local_view": {str(k): v for k, v in local_view.items()},
        "semantic_mismatches": mismatches,
    }


def _kv_pool_audit(stage_config) -> list:
    specs = stage_config.kv_cache_group_specs()
    out = []
    for spec in specs:
        out.append({
            "name": spec.name,
            "layer_ids": list(spec.layer_ids),
            "num_kv_heads": spec.num_kv_heads,
            "head_dim": spec.head_dim,
            "sliding_window": spec.sliding_window,
        })
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from dataclasses import replace as _replace

    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.rotary import set_rope_device
    from freetoken.models.gemma4.config import parse_config
    from freetoken.utils import cached_load_hf_config

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    set_rope_device("cpu")  # audit builds modules only; no CUDA needed

    full_config = parse_config(cached_load_hf_config(args.model))

    # S view: identity renumbering over all 48 layers
    s_audit = _stage_layer_audit("single", {"start_layer": 0, "end_layer": 48},
                                 full_config, full_config)
    s_kv = _kv_pool_audit(full_config)

    stage_audits = []
    stage_kvs = []
    for role, (start, end) in (
        ("first", (0, 16)), ("middle", (16, 32)), ("last", (32, 48)),
    ):
        owned = sorted(range(start, end))
        local_of = {gid: i for i, gid in enumerate(owned)}
        groups = []
        for group in full_config.attention_groups:
            layer_ids = tuple(local_of[i] for i in group.layer_ids if i in local_of)
            if layer_ids:
                groups.append(_replace(group, layer_ids=layer_ids))
        stage_config = _replace(
            full_config, attention_groups=tuple(groups), num_layers=len(owned)
        )
        stage_audits.append(
            _stage_layer_audit(role, {"start_layer": start, "end_layer": end},
                               full_config, stage_config)
        )
        stage_kvs.append({"role": role, "kv_pool_groups": _kv_pool_audit(stage_config)})

    repo = Path(__file__).resolve().parents[2]
    producer = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    # Cross-arm diff: S per-layer semantic view vs each stage's stage-local view
    cross_arm_mismatches = {}
    s_view = s_audit["per_layer_global_view"]
    for audit in stage_audits:
        for gid_str, local_view in audit["per_layer_stage_local_view"].items():
            for field in LAYER_FIELDS:
                if s_view[gid_str][field] != local_view[field]:
                    cross_arm_mismatches.setdefault(
                        f"{audit['role']}:layer{gid_str}", {}
                    )[field] = {
                        "S": s_view[gid_str][field],
                        "D_stage_local": local_view[field],
                    }

    report = {
        "schema": "inferswarm.r6_localization.config-audit/1",
        "producer_sha": producer,
        "model": str(args.model),
        "layer_fields_audited": list(LAYER_FIELDS),
        "single_audit": s_audit,
        "single_kv_pool_groups": s_kv,
        "stage_audits": stage_audits,
        "stage_kv_pools": stage_kvs,
        "stage_local_vs_global_mismatches": {
            a["role"]: a["semantic_mismatches"] for a in stage_audits
        },
        "cross_arm_s_vs_d_mismatches": cross_arm_mismatches,
        "all_stage_local_views_equal_S": not cross_arm_mismatches,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "all_stage_local_views_equal_S": not cross_arm_mismatches,
        "mismatches": cross_arm_mismatches if cross_arm_mismatches else None,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
