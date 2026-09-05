"""#97 v4 chain runner: teacher-forced three-stage RTX 3060 candidate.

Drives the accepted chain topology (stage 1 = node-01 GPU-0 [0,16),
stage 2 = node-01 GPU-1 [16,32), stage 3 = node-03 last stage via the
#76 R4 wire service) with the v3 canonical-prefix contract:

At each of all 8 decisions the candidate consumes EXACTLY the frozen
reference prefix for that decision (proved byte-identical via
``assert_teacher_forcing`` BEFORE execution), and the ACTUAL candidate
full-vocabulary FP32 winner for that same canonical-prefix row is
retained (row persisted as decision-<i>.f32 on the last-stage node,
sha256-bound in the case summary, with an executor rule proof under the
frozen argmax/tie-break rule).

Free-running candidate continuation is never executed: the candidate is
teacher-forced at every decision, so every retained row is a
canonical-prefix row. The 15-envelope checkpoint capture at positions
0/1/3/7 is unchanged from the #76 harness.

Run from a #97 worktree on inferswarm01 with the #97 last-stage service
already listening on inferswarm03.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmarks.inferswarm_76 import verify_case_identity
from benchmarks.inferswarm_76.reference_runner import resolve_cases
from benchmarks.inferswarm_97 import (
    ARGMAX_TIE_BREAK_IDENTITY,
    GENERATED_TOKENS,
    V4_CONTRACT_ID,
    assert_teacher_forcing,
    build_chain_case_summary,
    executor_rule_proof,
    prefix_sha256,
    producer_identity,
)


def _load_reference_case(path: Path, case: dict) -> dict:
    reference = json.loads(path.read_text())
    if reference.get("schema") != "inferswarm.issue97.v4-reference-case/1":
        raise SystemExit(f"{path}: not a v4 reference case summary")
    for field in ("case_id", "case_sha256", "prompt_sha256",
                  "token_ids_sha256"):
        if reference[field] != case[field]:
            raise SystemExit(
                f"{path}: reference {field} mismatch for {case['case_id']}")
    return reference


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--resolve-corpus", default=None)
    parser.add_argument("--case-ids", default=None)
    parser.add_argument("--reference-dir", required=True,
                        help="v3 reference run root (per-case dirs with "
                             "reference-case-<tag>.json)")
    parser.add_argument("--reference-tag", required=True)
    parser.add_argument("--last-stage-host", default="10.0.0.219")
    parser.add_argument("--last-stage-port", type=int, default=18485)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args(argv)

    import multiprocessing

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
            ref_path = (Path(args.reference_dir) / case["case_id"] /
                        f"reference-case-{args.reference_tag}.json")
            reference = _load_reference_case(ref_path, case)
            ref_decisions = sorted(
                reference["decisions"], key=lambda d: d["decision_index"])
            forced = list(reference["generated_token_ids"])
            t0 = time.perf_counter()

            # arm per-case capture on stages 1-2 (fresh sink per case)
            import subprocess

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
            margins: list[dict] = []
            decision_rows: list[dict] = []
            nan_inf_total = 0
            for step in range(GENERATED_TOKENS):
                # --- v3 teacher forcing: exact reference prefix ----------
                replay = prompt + [int(t) for t in forced[:step]]
                assert_teacher_forcing(
                    prefix=replay, reference_decision=ref_decisions[step])

                for stage in stages:
                    stage.request({"op": "RESET"})
                # every decision is evidence-bearing in v3: capture_step is
                # sent on ALL 8 decisions (the last-stage service keys its
                # retained decision-<i>.f32 rows off this value), while the
                # 15-envelope capture positions remain the frozen 0/1/3/7.
                hidden = None
                response: dict = {}
                for index, stage in enumerate(stages):
                    if index == 0:
                        response = stage.request({
                            "op": "PREFILL",
                            "token_ids": replay,
                            "position": 0,
                            "capture_step": step,
                        })
                        hidden = response.get("hidden")
                    else:
                        response = stage.request({
                            "op": "PREFILL",
                            "hidden": hidden,
                            "position": 0,
                            "capture_step": step,
                        })
                        hidden = response.get("hidden")
                token = response["token_id"]
                row_sha = response["row_f32_sha256"]
                row_count = response["row_element_count"]
                rule = response["rule_proof"]
                if response.get("top1_index") is None:
                    raise RuntimeError(
                        "last-stage response missing margin diagnostics")
                decision_rows.append({
                    "decision_index": step,
                    "prefix_len": len(replay),
                    "prefix_sha256": prefix_sha256(replay),
                    "emitted_token": int(token),
                    "emitted_rule": ARGMAX_TIE_BREAK_IDENTITY,
                    "row_f32_sha256": row_sha,
                    "row_element_count": row_count,
                    "rule_proof": rule,
                    "row_retained_at": "last-stage-node",
                })
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

            # persist per-case captures from stages 1-2 and the remote stage
            manifests = {}
            for stage_index, stage in enumerate(stages[:-1]):
                ack = stage.request(
                    {"op": "SAVE_CAPTURE",
                     "suffix": f"{args.tag}-stage{stage_index + 1}"})
                manifests[f"stage{stage_index + 1}"] = ack["manifest"]
            save_ack = stages[-1].request(
                {"op": "CASE_SAVE", "tag": args.tag})
            manifests["stage3"] = save_ack["manifest"]

            summary = build_chain_case_summary(
                case=case,
                reference_case=reference,
                decision_rows=decision_rows,
                margins=margins,
                nan_inf_total=nan_inf_total,
                capture_manifests=manifests,
                producer=producer,
                tag=args.tag,
                attempt_id=args.attempt_id,
                wall_seconds=time.perf_counter() - t0,
            )
            path = case_dir / f"chain-case-{args.tag}.json"
            if path.exists():
                raise SystemExit(f"refusing to overwrite {path}")
            path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            results.append(summary)
            print(json.dumps({
                "case_id": case["case_id"], "status": "CASE_COMMITTED",
                "nan_inf": nan_inf_total}), flush=True)

        index = {
            "schema": "inferswarm.issue97.v4-chain-run-index/1",
            "contract_id": V4_CONTRACT_ID,
            "attempt_id": args.attempt_id,
            "producer": producer,
            "tag": args.tag,
            "case_count": len(results),
            "cases": [
                {"case_id": r["case_id"], "case_sha256": r["case_sha256"],
                 "nan_inf_count": r["nan_inf_count"]}
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
