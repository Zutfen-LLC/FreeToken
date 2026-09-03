"""#71 D-arm runner: three-stage distributed chain with semantic checkpoints.

Drives the accepted R6 distributed topology (stages 1-2 spawned locally on
node A, last stage via the R4 wire service on node B) through the SAME
replay-prefill semantics as the accepted R6 D logit capture arm: per
comparator step k, RESET all stages, full replay prefill of
prompt + generated[0:k] in a single 64-row chunk at position 0, greedy
argmax at the last stage. No speculative decode calls.

Captures per stage (capture steps 0/1/7 + bisection after-layers):
  stage 1: embedding_output, after_layer_15 (=boundary1 send)
  stage 2: boundary1 recv, after_layer_31 (=boundary2 send)
  stage 3: boundary2 recv, after_layer_47, final_norm, bf16_logits (final
           row via final_row_fp32; full-matrix BF16 retention is NOT taken
           here — 3060 VRAM — the final-row and hash suffice for coarse
           localization; the head interval, if divergent, is localized by
           the dedicated head diagnostic).

Requires: last_stage_service running with --localization-capture and
--allow-producer <running-sha> (evidence arm), same plan file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

CAPTURE_STEPS = (0, 1, 7)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--last-stage-host", default="10.0.0.219")
    parser.add_argument("--last-stage-port", type=int, default=18485)
    parser.add_argument("--reference", default="docs/inferswarm_r6/reference-generation.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", default="coarse")
    parser.add_argument("--capture-steps", type=int, nargs="+", default=list(CAPTURE_STEPS))
    parser.add_argument("--capture-layers", type=int, nargs="*", default=[])
    args = parser.parse_args(argv)

    import multiprocessing

    import torch

    from benchmarks.inferswarm_r6.stage_chain import GemmaStageChainRuntime, StageClient
    from benchmarks.inferswarm_r6.wire_client import (
        RemoteLastStageClient,
        arm_boundary_byte_log,
    )

    repo = Path(__file__).resolve().parents[2]
    producer_sha = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    reference = json.loads((repo / args.reference).read_text())
    prompt = list(reference["prompt_token_ids"])
    expected_generated = list(reference["generated_token_ids"])
    maximum = len(expected_generated)
    capture_steps = set(args.capture_steps)
    after_layers = sorted(set(int(x) for x in args.capture_layers))

    plan = json.loads(Path(args.plan).read_text())
    shared = plan.get("declared_shared_state")
    context = multiprocessing.get_context("spawn")

    stages = []
    byte_log = arm_boundary_byte_log()

    class _Chain(GemmaStageChainRuntime):
        def __init__(self):
            self.stages = []
            try:
                for index, block in enumerate(plan["blocks"][:-1]):
                    self.stages.append(
                        StageClient(
                            context,
                            role="first" if index == 0 else "middle",
                            adapter_data={
                                **block,
                                "declared_shared_state": shared if index == 0 else None,
                                "runtime_capacity_tokens": plan["runtime_capacity_tokens"],
                            },
                            model_path=args.model,
                            gpu_index=index,
                        )
                    )
                self.stages.append(
                    RemoteLastStageClient(
                        host=args.last_stage_host,
                        port=args.last_stage_port,
                        experiment_id=plan["digest"],
                    )
                )
                self.ready = []
                for stage in self.stages[:-1]:
                    ready = stage.recv()
                    if ready.get("op") == "ERROR":
                        raise RuntimeError(f"stage failed: {ready}")
                    self.ready.append(ready)
                self.ready.append({"op": "READY", "role": "last", "remote": True})
            except BaseException:
                for stage in self.stages:
                    stage.shutdown()
                raise
            self._sessions = []
            self._closed = False
            self.reclamation_report = {}

    chain = _Chain()
    try:
        # arm capture on stages 1-2 (stage 3 armed at service launch)
        gpu_uuids = [
            subprocess.check_output(
                ["nvidia-smi", "-i", str(i), "--query-gpu=uuid",
                 "--format=csv,noheader"], text=True
            ).strip()
            for i in (0, 1)
        ]
        for stage_index, (stage, uuid) in enumerate(zip(chain.stages[:-1], gpu_uuids)):
            # stage 1 owns globals [0,16): coarse checkpoint after layer 15;
            # stage 2 owns [16,32): after layer 31. Bisection extras appended.
            coarse = 15 if stage_index == 0 else 31
            stage.request({
                "op": "ARM_CAPTURE",
                "out_dir": args.out_dir,
                "tag": args.tag,
                "gpu_uuid": uuid,
                "after_layers": sorted({coarse} | set(after_layers)),
            })

        generated: list[int] = []
        step_records = []
        for step in range(maximum):
            for stage in chain.stages:
                stage.request({"op": "RESET"})
            replay = prompt + generated
            capture_now = step in capture_steps
            hidden = None
            token_id = None
            for index, stage in enumerate(chain.stages):
                if index == 0:
                    response = stage.request({
                        "op": "PREFILL",
                        "token_ids": replay,
                        "position": 0,
                        **({"capture_step": step} if capture_now else {}),
                    })
                else:
                    response = stage.request({
                        "op": "PREFILL",
                        "hidden": hidden,
                        "position": 0,
                        **({"capture_step": step} if capture_now else {}),
                    })
                if response.get("op") == "TOKEN_RESULT":
                    token_id = response["token_id"]
                    break
                hidden = response["hidden"]
            if token_id is None:
                raise RuntimeError("chain ended without token result")
            generated.append(int(token_id))
            step_records.append({"step": step, "rows": len(replay),
                                 "captured": capture_now, "token": int(token_id)})

        # persist stage 1-2 captures
        manifests = {}
        for index, stage in enumerate(chain.stages[:-1]):
            ack = stage.request({
                "op": "SAVE_CAPTURE", "suffix": f"step{maximum}",
            })
            manifests[f"stage{index + 1}"] = ack["manifest"]

        # persist sender-side wire byte log (boundary 2)
        boundary_sends = []
        for record in byte_log:
            entry = {k: v for k, v in record.items() if k != "payload_bytes"}
            import hashlib

            entry["payload_bytes_sha256"] = hashlib.sha256(
                record["payload_bytes"]
            ).hexdigest()
            payload_path = (
                Path(args.out_dir)
                / f"boundary2-send-{entry['operation']}-p{entry['position']}.bin"
            )
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_bytes(record["payload_bytes"])
            boundary_sends.append(entry)

        exact = generated == expected_generated
        result = {
            "schema": "inferswarm.r6_localization.d-arm-run/1",
            "status": "D_ARM_COMPLETE" if exact else "D_ARM_TOKEN_MISMATCH",
            "producer_sha": producer_sha,
            "plan_digest": plan.get("digest"),
            "generated_token_ids": generated,
            "expected_generated_token_ids": expected_generated,
            "exact_generated_token_match": exact,
            "capture_steps": sorted(capture_steps),
            "after_layer_checkpoints": after_layers,
            "steps": step_records,
            "stage_manifests": manifests,
            "boundary2_sends": boundary_sends,
        }
        out = Path(args.out_dir) / f"d-arm-result-{args.tag}.json"
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": result["status"], "exact": exact}))
        return 0 if exact else 2
    finally:
        for stage in chain.stages:
            try:
                stage.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
