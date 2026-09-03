"""#71 localization analyzer: S-vs-D comparisons from capture bundles.

Pure-stdlib+torch offline analyzer (runs on any host with torch; the
coarse summary is JSON-only so the coordinator can ingest it too).

Inputs: capture bundles (torch.save'd dicts from CaptureSink.save) for S
(single role) and D stages (first/middle/last), boundary wire logs.
Output: localization-summary.json with per-(step, checkpoint) rows:
S hash, D hash, exact_equal, max/mean/RMS absdiff, max coordinate,
S/D values at max — plus boundary sender/receiver identity records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ANALYZER_VERSION = "inferswarm.r6_localization.analyzer/1"

# Coarse checkpoint -> (S bundle checkpoint, D producer checkpoint)
# after_layer_15: S = single's after_layer_15 (hook) == D stage-1 output
# after_layer_31: S hook == D stage-2 output
COMPARE_POINTS = [
    "embedding_output",
    "after_layer_15",
    "after_layer_31",
    "after_layer_47",
    "final_norm",
    "bf16_logits",
    "final_row_fp32",
]
BOUNDARY_POINTS = {
    "boundary1": ("boundary_send_hidden", "boundary_recv_hidden",
                  "stage1-out", "stage2-in"),
    "boundary2": ("boundary_send_hidden", "boundary_recv_hidden",
                  "stage2-out", "stage3-in"),
}


def _load_bundle(path: Path) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload


def _index_records(bundle: dict) -> dict[tuple[int, str], dict]:
    index = {}
    for meta, tensor in zip(bundle["records"], bundle["tensors"]):
        index[(meta["step"], meta["checkpoint"])] = (meta, tensor)
    return index


def _tensor_bytes(t) -> bytes:
    import torch

    return t.detach().contiguous().view(torch.uint8).numpy().tobytes()


def _compare(s_tensor, d_tensor) -> dict:
    import torch

    if s_tensor.shape != d_tensor.shape or s_tensor.dtype != d_tensor.dtype:
        return {
            "shape_or_dtype_mismatch": True,
            "s_shape": list(s_tensor.shape),
            "d_shape": list(d_tensor.shape),
            "s_dtype": str(s_tensor.dtype),
            "d_dtype": str(d_tensor.dtype),
            "exact_equal": False,
        }
    exact = bool(torch.equal(s_tensor, d_tensor))
    s64 = s_tensor.detach().to(torch.float64)
    d64 = d_tensor.detach().to(torch.float64)
    diff = (s64 - d64).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    rms = float(math.sqrt((diff * diff).mean().item()))
    flat_index = int(diff.argmax().item())
    coord = tuple(divmod(flat_index, s_tensor.shape[-1])) if s_tensor.dim() >= 2 else (flat_index,)
    return {
        "exact_equal": exact,
        "shape_or_dtype_mismatch": False,
        "max_absdiff": max_abs,
        "mean_absdiff": mean_abs,
        "rms_diff": rms,
        "max_coordinate_rows_then_dim": list(coord),
        "s_at_max": float(s_tensor.flatten()[flat_index].item())
        if s_tensor.dtype.is_floating_point else None,
        "d_at_max": float(d_tensor.flatten()[flat_index].item())
        if d_tensor.dtype.is_floating_point else None,
    }


def analyze(s_bundle_path: Path, d_bundle_paths: dict[str, Path],
            boundary_tx: Path | None = None,
            boundary_rx: Path | None = None) -> dict:
    s_index = _index_records(_load_bundle(s_bundle_path))
    d_indexes = {role: _index_records(_load_bundle(p))
                 for role, p in d_bundle_paths.items()}

    def find_d(step: int, checkpoint: str):
        # D producer of a coarse point: embedding/after15 -> stage1(first),
        # after31 -> stage2(middle), after47/final_norm/logits -> stage3(last)
        order = {
            "embedding_output": ["first", "middle", "last"],
            "after_layer_15": ["first", "middle", "last"],
            "after_layer_31": ["middle", "last"],
            "after_layer_47": ["last"],
            "final_norm": ["last"],
            "bf16_logits": ["last"],
            "final_row_fp32": ["last"],
        }.get(checkpoint, ["first", "middle", "last"])
        for role in order:
            if role in d_indexes and (step, checkpoint) in d_indexes[role]:
                return role, d_indexes[role][(step, checkpoint)]
        return None, None

    rows = []
    steps = sorted({step for step, _ in s_index})
    for step in steps:
        for checkpoint in COMPARE_POINTS:
            if (step, checkpoint) not in s_index:
                continue
            s_meta, s_tensor = s_index[(step, checkpoint)]
            d_role, found = find_d(step, checkpoint)
            if found is None:
                rows.append({
                    "step": step, "checkpoint": checkpoint,
                    "s_sha256": s_meta["sha256"],
                    "d_role": None, "d_sha256": None,
                    "error": "D capture missing",
                })
                continue
            d_meta, d_tensor = found
            row = {
                "step": step,
                "checkpoint": checkpoint,
                "s_sha256": s_meta["sha256"],
                "d_role": d_role,
                "d_sha256": d_meta["sha256"],
            }
            row.update(_compare(s_tensor, d_tensor))
            rows.append(row)

    boundaries = {}
    # boundary 1: stage1 boundary_send_hidden vs stage2 boundary_recv_hidden
    first = d_indexes.get("first", {})
    middle = d_indexes.get("middle", {})
    b1 = []
    for (step, cp), (meta, tensor) in sorted(first.items()):
        if cp != "boundary_send_hidden":
            continue
        key = (step, "boundary_recv_hidden")
        if key not in middle:
            continue
        rmeta, rtensor = middle[key]
        record = {
            "step": step,
            "sender_sha256": meta["sha256"],
            "receiver_sha256": rmeta["sha256"],
            "byte_count": meta["byte_count"],
            "bytes_identical": meta["sha256"] == rmeta["sha256"]
            and meta["byte_count"] == rmeta["byte_count"],
            "shape_dtype_identical": (
                meta["shape"] == rmeta["shape"] and meta["dtype"] == rmeta["dtype"]
            ),
        }
        record.update({
            "exact_equality": _compare(tensor, rtensor)["exact_equal"],
        })
        b1.append(record)
    boundaries["boundary1"] = b1

    # boundary 2: stage2 boundary_send_hidden vs stage3 boundary_recv_hidden
    last = d_indexes.get("last", {})
    b2 = []
    for (step, cp), (meta, tensor) in sorted(middle.items()):
        if cp != "boundary_send_hidden":
            continue
        key = (step, "boundary_recv_hidden")
        if key not in last:
            continue
        rmeta, rtensor = last[key]
        record = {
            "step": step,
            "sender_sha256": meta["sha256"],
            "receiver_sha256": rmeta["sha256"],
            "byte_count": meta["byte_count"],
            "bytes_identical": meta["sha256"] == rmeta["sha256"]
            and meta["byte_count"] == rmeta["byte_count"],
            "shape_dtype_identical": (
                meta["shape"] == rmeta["shape"] and meta["dtype"] == rmeta["dtype"]
            ),
            "exact_equality": _compare(tensor, rtensor)["exact_equal"],
        }
        b2.append(record)
    boundaries["boundary2"] = b2

    # wire-level byte proof (payload bytes sha256 sender vs receiver)
    wire_rows = []
    if boundary_tx is not None and boundary_rx is not None:
        tx = json.loads(boundary_tx.read_text()) if boundary_tx.exists() else None
        rx = json.loads(boundary_rx.read_text()) if boundary_rx.exists() else None
        if tx and rx:
            tx_by = {}
            for r in tx:
                tx_by[(r["operation"], r["position"], r["token_count"])] = r
            for r in rx:
                key = (r["operation"], r["position"], r["token_count"])
                t = tx_by.get(key)
                if t is None:
                    continue
                tx_hash = t.get("payload_sha256") or (
                    "sha256:" + hashlib.sha256(t["payload_bytes"]).hexdigest()
                )
                wire_rows.append({
                    "operation": r["operation"],
                    "position": r["position"],
                    "token_count": r["token_count"],
                    "tx_payload_sha256": tx_hash,
                    "rx_payload_sha256": r["payload_sha256"],
                    "bytes_identical": tx_hash == r["payload_sha256"],
                })

    # verdict skeleton (fail-closed: descriptive only)
    summary = {
        "schema": "inferswarm.r6_localization.summary/1",
        "analyzer_version": ANALYZER_VERSION,
        "rows": rows,
        "boundaries": boundaries,
        "boundary2_wire_bytes": wire_rows,
        "notes": (
            "Localization evidence only. Exact equality means no drift has "
            "appeared yet at that checkpoint; first nonzero residual marks "
            "the first observed divergent checkpoint. No threshold defined; "
            "historical R6 result unchanged."
        ),
    }
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s-bundle", required=True)
    parser.add_argument("--d-first", required=True)
    parser.add_argument("--d-middle", required=True)
    parser.add_argument("--d-last", required=True)
    parser.add_argument("--boundary-tx")
    parser.add_argument("--boundary-rx")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    summary = analyze(
        Path(args.s_bundle),
        {"first": Path(args.d_first), "middle": Path(args.d_middle),
         "last": Path(args.d_last)},
        Path(args.boundary_tx) if args.boundary_tx else None,
        Path(args.boundary_rx) if args.boundary_rx else None,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "rows": len(summary["rows"]),
        "first_divergent": next(
            (r for r in summary["rows"]
             if not r.get("exact_equal", True) and "exact_equal" in r),
            None),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
