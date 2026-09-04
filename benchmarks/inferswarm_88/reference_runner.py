"""#88 v3 reference runner: RTX 3090 canonical reference execution.

Runs the frozen c86-*/p86-*/h86-* case manifests through the #76 harness
core (verbatim execution path) and ADDS the v3 semantic layer per case:

- canonical 8-decision trajectory under the frozen argmax/tie-break rule
  with per-decision executor rule proofs;
- FULL decision evidence at ALL 8 decisions: the complete FP32
  consumer-logit row is appended to the case bundle (decision-<i>.f32
  sidecars, sha256-recorded) — sufficient for D(r) construction,
  E_full-class full-vocabulary evidence, and later
  decision_local_error derivation;
- the frozen decision domain D(r) per decision, computed on this
  reference row only, with canonical membership hashes;
- unchanged 15-envelope checkpoint capture at positions 0/1/3/7 and
  per-decision margin diagnostics (frozen min-over-8 margin definition).

One process runs many cases sequentially (model load dominates). Each
case appends its own capture bundle + decision rows; nothing is ever
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from benchmarks.inferswarm_76 import verify_case_identity
from benchmarks.inferswarm_76.reference_runner import resolve_cases
from benchmarks.inferswarm_88 import (
    ARGMAX_TIE_BREAK_IDENTITY,
    DECISION_DOMAIN_CONSTRUCTION,
    GENERATED_TOKENS,
    V3_CONTRACT_ID,
    decision_domain_row,
    executor_rule_proof,
    prefix_sha256,
    producer_identity,
)


def _row_bytes(row) -> bytes:
    import torch

    return (
        row.detach().to(torch.float32).contiguous()
        .view(torch.uint8).numpy().tobytes()
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--resolve-corpus", default=None)
    parser.add_argument("--case-ids", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args(argv)

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    repo = Path(__file__).resolve().parents[2]

    producer = producer_identity(repo)
    if producer["dirty"]:
        print(json.dumps({"status": "BLOCKED_DIRTY_SOURCE"}))
        return 2

    cases = resolve_cases(args.corpus, args.resolve_corpus)
    if args.case_ids:
        wanted = set(args.case_ids.split(","))
        cases = [c for c in cases if c["case_id"] in wanted]
        missing = wanted - {c["case_id"] for c in cases}
        if missing:
            raise SystemExit(f"unknown case ids: {sorted(missing)}")

    import torch

    from benchmarks.inferswarm_76 import (
        CAPTURE_POSITIONS,
        RUNTIME_CAPACITY_TOKENS,
    )
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

    # wrappers bind runtime._emit dynamically (which reads _capture_sink),
    # so they are installed EXACTLY ONCE (the accepted #76 pattern); per-case
    # isolation comes from swapping the sink below. Arming per case would
    # CHAIN wrapper layers and duplicate every capture record.
    runtime._capture_sink = RowPruningSink(role="single", gpu_uuid=gpu_uuid)
    runtime._capture_after_layers = frozenset({15, 31})
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
        decision_rows: list[dict] = []
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
            # --- v3 semantic layer on the exact row --------------------
            host_row = row.detach().to("cpu", torch.float32).contiguous()
            values = host_row.tolist()
            proof = executor_rule_proof(values, int(token))
            domain = decision_domain_row(values)
            row_bytes = host_row.view(torch.uint8).numpy().tobytes()
            row_path = case_dir / f"decision-{step}.f32"
            if row_path.exists():
                raise SystemExit(f"refusing to overwrite {row_path}")
            row_path.write_bytes(row_bytes)
            decision_rows.append({
                "decision_index": step,
                "prefix_len": len(replay),
                "prefix_sha256": prefix_sha256(replay),
                "domain_membership_sha256": domain["domain_membership_sha256"],
                "domain_size": domain["domain_size"],
                "domain_cutoff_hex": domain["cutoff_hex"],
                "emitted_token": int(token),
                "emitted_rule": ARGMAX_TIE_BREAK_IDENTITY,
                "row_f32_sha256": _sha256_bytes(row_bytes),
                "row_element_count": len(values),
                "rule_proof": proof,
            })
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
            del logits, row, host_row, values

        manifest = sink.save(str(case_dir), args.tag)
        summary = {
            "schema": "inferswarm.issue88.v3-reference-case/1",
            "contract_id": V3_CONTRACT_ID,
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
            "decision_domain_construction": DECISION_DOMAIN_CONSTRUCTION,
            "decisions": decision_rows,
            "producer": producer,
            "gpu_uuid": gpu_uuid,
            "role": "reference-single",
            "capture_positions": list(CAPTURE_POSITIONS),
            "capture_manifest": manifest,
            "wall_seconds": time.perf_counter() - t0,
        }
        path = case_dir / f"reference-case-{args.tag}.json"
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        results.append(summary)
        print(json.dumps({
            "case_id": case["case_id"], "status": "CASE_COMMITTED",
            "tokens": generated, "nan_inf": nan_inf_total,
            "records": manifest["record_count"]}), flush=True)

    index = {
        "schema": "inferswarm.issue88.v3-reference-run-index/1",
        "contract_id": V3_CONTRACT_ID,
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
