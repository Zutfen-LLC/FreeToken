"""#76 single-arm case runner: matched FreeToken reference on the RTX 3090.

Case-driven replay-prefill greedy generation with the full 15-envelope
checkpoint capture at positions 0/1/3/7 and per-step top-1 margin
diagnostics. One process may run MANY cases sequentially (model load is the
dominant cost); Phase-A fresh-process realizations are created by launching
this runner again.

Per case it writes:
  <out>/<case_id>/capture-<tag>.pt     raw host tensors (#71 bundle format)
  <out>/<case_id>/case-<tag>.json      case summary (tokens, margins,
                                        checkpoint hashes, NaN/Inf counts,
                                        producer, gpu identity)

The case summary is the atomic unit of committed evidence; the .pt bundle is
retained immutably next to it. Nothing is overwritten: a repeat run must use
a different --tag.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.inferswarm_76 import (
    CAPTURE_POSITIONS,
    GENERATED_TOKENS,
    RUNTIME_CAPACITY_TOKENS,
    load_corpus,
    verify_case_identity,
)


def _producer(repo: Path) -> dict:
    sha = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo),
         "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo),
         "status", "--porcelain"], text=True)
    return {"commit": sha, "dirty": bool(status)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True,
                        help="JSON file: corpus manifest OR a list of case rows")
    parser.add_argument("--case-ids", default=None,
                        help="comma-separated subset of case ids to run")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args(argv)

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    repo = Path(__file__).resolve().parents[2]

    producer = _producer(repo)
    if producer["dirty"]:
        print(json.dumps({"status": "BLOCKED_DIRTY_SOURCE"}))
        return 2

    raw = json.loads(Path(args.corpus).read_text())
    if isinstance(raw, list):
        cases = [verify_case_identity(row) for row in raw]
    else:
        cases = load_corpus(args.corpus)
    if args.case_ids:
        wanted = set(args.case_ids.split(","))
        cases = [c for c in cases if c["case_id"] in wanted]
        missing = wanted - {c["case_id"] for c in cases}
        if missing:
            raise SystemExit(f"unknown case ids: {sorted(missing)}")

    import torch

    from benchmarks.inferswarm_76.capture import RowPruningSink, arm_full_capture
    from benchmarks.inferswarm_r6.stage_runtime import GemmaDenseStage
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.rotary import set_rope_device
    from freetoken.research.r6_dense_census import (
        DenseBlockSpec,
        checkpoint_census,
        freeze_dense_block_plan,
    )

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    set_rope_device(torch.device("cuda:0"))

    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True).strip()

    model = Path(args.model).resolve()
    census = checkpoint_census(model, text_prefix="model.language_model")
    shared = {
        "id": "tied-embedding-lm-head",
        "kind": "tied-weight-shared-state",
        "tensor_keys": ["model.language_model.embed_tokens.weight"],
        "bytes": census["bytes_by_owner_category"]["embedding/input"],
        "materialization_policy": "single-cuda-tensor-used-for-input-and-output",
    }
    plan = freeze_dense_block_plan(
        census, [DenseBlockSpec(0, 48, True, True)], declared_shared_state=shared
    )
    runtime = GemmaDenseStage(
        role="single",
        model_path=str(model),
        adapter_data={
            **plan["blocks"][0],
            "declared_shared_state": shared,
            "runtime_capacity_tokens": RUNTIME_CAPACITY_TOKENS,
        },
    )

    runtime._capture_sink = RowPruningSink(role="single", gpu_uuid=gpu_uuid)
    runtime._capture_after_layers = frozenset({15, 31})
    # wrappers bind runtime._emit dynamically (which reads _capture_sink),
    # so they are installed EXACTLY ONCE; per-case isolation comes from
    # swapping the sink below.
    arm_full_capture(runtime, runtime._capture_sink)

    results = []
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for case in cases:
        case_dir = out_root / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        runtime._capture_sink = RowPruningSink(role="single", gpu_uuid=gpu_uuid)
        sink = runtime._capture_sink

        prompt = list(case["token_ids"])
        generated: list[int] = []
        margins: list[dict] = []
        nan_inf_total = 0
        t0 = time.perf_counter()
        for step in range(GENERATED_TOKENS):
            runtime.reset_session_state()
            replay = prompt + generated
            capture_now = step in CAPTURE_POSITIONS
            runtime._capture_step = step if capture_now else None
            token, logits = runtime.prefill(replay, None, 0)
            row = logits  # final-row FP32 consumer logits [vocab]
            nan_inf_total += int(
                torch.isnan(row).sum().item() + torch.isinf(row).sum().item())
            top2 = torch.topk(row, 2)
            margins.append({
                "step": step,
                "top1_index": int(top2.indices[0].item()),
                "top1_value_hex": float(top2.values[0].item()).hex(),
                "top2_index": int(top2.indices[1].item()),
                "top2_value_hex": float(top2.values[1].item()).hex(),
                "margin_hex": float(
                    top2.values[0].item() - top2.values[1].item()).hex(),
            })
            generated.append(int(token))
            del logits, row

        manifest = sink.save(str(case_dir), args.tag)
        summary = {
            "schema": "inferswarm.issue76.single-case-run/1",
            "attempt_id": args.attempt_id,
            "case_id": case["case_id"],
            "case_sha256": case["case_sha256"],
            "prompt_sha256": case["prompt_sha256"],
            "token_ids_sha256": case["token_ids_sha256"],
            "generated_token_ids": generated,
            "step_margins": margins,
            "min_top1_margin_hex": min(
                float.fromhex(m["margin_hex"]) for m in margins).hex(),
            "nan_inf_count": nan_inf_total,
            "producer": producer,
            "gpu_uuid": gpu_uuid,
            "role": "single",
            "capture_positions": list(CAPTURE_POSITIONS),
            "capture_manifest": manifest,
            "wall_seconds": time.perf_counter() - t0,
        }
        path = case_dir / f"case-{args.tag}.json"
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        results.append(summary)
        print(json.dumps({
            "case_id": case["case_id"], "status": "CASE_COMMITTED",
            "tokens": generated,
            "nan_inf": nan_inf_total,
            "records": manifest["record_count"]}), flush=True)
        # sink replaced per case at loop head

    index = {
        "schema": "inferswarm.issue76.single-run-index/1",
        "attempt_id": args.attempt_id,
        "producer": producer,
        "gpu_uuid": gpu_uuid,
        "tag": args.tag,
        "case_count": len(results),
        "cases": [
            {"case_id": r["case_id"], "case_sha256": r["case_sha256"],
             "generated_token_ids": r["generated_token_ids"],
             "nan_inf_count": r["nan_inf_count"],
             "min_top1_margin_hex": r["min_top1_margin_hex"]}
            for r in results
        ],
    }
    index_path = out_root / f"index-{args.tag}.json"
    if index_path.exists():
        raise SystemExit(f"refusing to overwrite {index_path}")
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "RUN_COMPLETE", "cases": len(results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
