"""#76 chain-arm case runner: three-stage RTX 3060 distributed candidate.

Drives the accepted R6 chain topology (stage 1 = node-01 GPU-0 layers [0,16),
stage 2 = node-01 GPU-1 layers [16,32), stage 3 = node-03 last stage via the
#76 R4 wire service) through the frozen replay-prefill greedy semantics,
per case:

  for step in 0..7:
      RESET all stages; replay = prompt + generated[0:step]
      stage1 PREFILL(token_ids) -> hidden -> stage2 PREFILL(hidden)
          -> stage3 PREFILL(hidden) -> token (capture_step at 0/1/3/7)

Capture handling mirrors the single arm: stages 1-2 use local
``RowPruningSink`` instances; the remote stage saves via CASE_SAVE.
Margins/NaN/Inf for the final row come from the last-stage response.

Run from a #76 worktree on inferswarm01 with the last-stage service already
listening on inferswarm03.
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
    parser.add_argument("--plan", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--resolve-corpus", default=None,
                        help="full calibration corpus for identity-only subsets")
    parser.add_argument("--case-ids", default=None)
    parser.add_argument("--last-stage-host", default="10.0.0.219")
    parser.add_argument("--last-stage-port", type=int, default=18485)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args(argv)

    import multiprocessing

    repo = Path(__file__).resolve().parents[2]
    producer = _producer(repo)
    if producer["dirty"]:
        print(json.dumps({"status": "BLOCKED_DIRTY_SOURCE"}))
        return 2

    from benchmarks.inferswarm_76.reference_runner import resolve_cases
    cases = resolve_cases(args.corpus, args.resolve_corpus)
    if args.case_ids:
        wanted = set(args.case_ids.split(","))
        cases = [c for c in cases if c["case_id"] in wanted]
        missing = wanted - {c["case_id"] for c in cases}
        if missing:
            raise SystemExit(f"unknown case ids: {sorted(missing)}")

    from benchmarks.inferswarm_76.stage_entry import I76StageClient
    from benchmarks.inferswarm_76.wire_client import I76LastStageClient

    plan = json.loads(Path(args.plan).read_text())
    shared = plan.get("declared_shared_state")

    context = multiprocessing.get_context("spawn")
    stages = []
    try:
        for index, block in enumerate(plan["blocks"][:-1]):
            stages.append(
                I76StageClient(
                    context,
                    role="first" if index == 0 else "middle",
                    adapter_data={
                        **block,
                        "declared_shared_state": shared if index == 0 else None,
                        "runtime_capacity_tokens": 64,
                    },
                    model_path=args.model,
                    gpu_index=index,
                )
            )
        stages.append(
            I76LastStageClient(
                host=args.last_stage_host,
                port=args.last_stage_port,
                experiment_id=plan["digest"],
                connect_timeout=600.0,
            )
        )
        for stage in stages[:-1]:
            ready = stage.recv()
            if ready.get("op") == "ERROR":
                raise RuntimeError(f"stage failed: {ready}")

        results = []
        out_root = Path(args.out_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        for case in cases:
            case_dir = out_root / case["case_id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()

            # arm per-case capture on stages 1-2 (fresh sink per case)
            gpu_uuids = [
                subprocess.check_output(
                    ["nvidia-smi", "-i", str(i), "--query-gpu=uuid",
                     "--format=csv,noheader"], text=True).strip()
                for i in (0, 1)
            ]
            for stage_index, (stage, uuid) in enumerate(
                zip(stages[:-1], gpu_uuids)
            ):
                stage.request({
                    "op": "CASE_ARM",
                    "out_dir": str(case_dir),
                    "tag": args.tag,
                    "gpu_uuid": uuid,
                    "after_layers": [15] if stage_index == 0 else [31],
                })
            stages[-1].request({"op": "CASE_BEGIN", "case_id": case["case_id"]})

            prompt = list(case["token_ids"])
            generated: list[int] = []
            margins: list[dict] = []
            nan_inf_total = 0
            for step in range(GENERATED_TOKENS):
                for stage in stages:
                    stage.request({"op": "RESET"})
                replay = prompt + generated
                capture_now = step in CAPTURE_POSITIONS
                hidden = None
                response: dict = {}
                for index, stage in enumerate(stages):
                    if index == 0:
                        response = stage.request({
                            "op": "PREFILL",
                            "token_ids": replay,
                            "position": 0,
                            **({"capture_step": step} if capture_now else {}),
                        })
                        hidden = response.get("hidden")
                    else:
                        response = stage.request({
                            "op": "PREFILL",
                            "hidden": hidden,
                            "position": 0,
                            **({"capture_step": step} if capture_now else {}),
                        })
                        hidden = response.get("hidden")
                token = response["token_id"]
                if response.get("top1_index") is None:
                    raise RuntimeError(
                        "last-stage response missing margin diagnostics")
                margin = {
                    "step": step,
                    "top1_index": response["top1_index"],
                    "top1_value_hex": response["top1_value_hex"],
                    "top2_index": response["top2_index"],
                    "top2_value_hex": response["top2_value_hex"],
                    "margin_hex": response["margin_hex"],
                    "nan_inf_count": response["nan_inf_count"],
                }
                margins.append(margin)
                nan_inf_total += int(response["nan_inf_count"])
                generated.append(int(token))

            # persist per-case captures from stages 1-2 and the remote stage
            manifests = {}
            for stage_index, stage in enumerate(stages[:-1]):
                ack = stage.request({"op": "SAVE_CAPTURE",
                                     "suffix": args.tag})
                manifests[f"stage{stage_index + 1}"] = ack["manifest"]
            save_ack = stages[-1].request(
                {"op": "CASE_SAVE", "tag": args.tag})
            manifests["stage3"] = save_ack["manifest"]

            summary = {
                "schema": "inferswarm.issue76.chain-case-run/1",
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
                "role": "chain",
                "capture_positions": list(CAPTURE_POSITIONS),
                "capture_manifests": manifests,
                "wall_seconds": time.perf_counter() - t0,
            }
            path = case_dir / f"chain-case-{args.tag}.json"
            if path.exists():
                raise SystemExit(f"refusing to overwrite {path}")
            path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            results.append(summary)
            print(json.dumps({
                "case_id": case["case_id"], "status": "CASE_COMMITTED",
                "tokens": generated, "nan_inf": nan_inf_total}), flush=True)

        index = {
            "schema": "inferswarm.issue76.chain-run-index/1",
            "attempt_id": args.attempt_id,
            "producer": producer,
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
    finally:
        for stage in stages:
            try:
                stage.shutdown()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    raise SystemExit(main())
