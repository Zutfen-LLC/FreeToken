"""#71 same-device diagnostic: stage-1 (first role) executed alone on THIS GPU.

Phase 6/§9 discriminator. Runs the distributed stage-1 code path (role
"first", layers [0,16) + embeddings, the exact #71 plan block) through the
SAME replay-prefill loop, capturing embedding/after_layer_0/after_layer_15.

- On the 3090 (inferswarm04): if after_layer_0 == S's after_layer_0
  byte-exact, the stage-1 distributed code path is bit-faithful on the same
  device -> drift is device-class execution, not stage code/config.
- On a 3060 (inferswarm01 GPU1): if after_layer_0 == D stage-1's
  (GPU0) byte-exact, execution is deterministic within a device class.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--reference", default="docs/inferswarm_r6/reference-generation.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args(argv)

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    repo = Path(__file__).resolve().parents[2]
    reference = json.loads((repo / args.reference).read_text())
    prompt = list(reference["prompt_token_ids"])
    expected = list(reference["generated_token_ids"])

    import torch

    from benchmarks.inferswarm_r6_localization.capture import CaptureSink
    from benchmarks.inferswarm_r6.stage_runtime import GemmaDenseStage
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.rotary import set_rope_device

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    set_rope_device(torch.device("cuda:0"))

    plan = json.loads(Path(args.plan).read_text())
    block = plan["blocks"][0]
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()

    runtime = GemmaDenseStage(
        role="first",
        model_path=args.model,
        adapter_data={
            **block,
            "declared_shared_state": plan.get("declared_shared_state"),
            "runtime_capacity_tokens": plan["runtime_capacity_tokens"],
        },
    )
    sink = CaptureSink(role="first-diag", gpu_uuid=gpu_uuid)
    runtime._capture_sink = sink
    runtime._capture_after_layers = frozenset({0, 15})

    # replay loop: capture steps 0/1/7 (token advance uses the CANONICAL
    # committed sequence so inputs match both arms exactly)
    generated = []
    for step in range(len(expected)):
        runtime.reset_session_state()
        replay = prompt + generated
        runtime._capture_step = step if step in (0, 1, 7) else None
        hidden, _ = runtime.prefill(replay, None, 0)
        del hidden
        generated.append(expected[step])  # canonical advance (diagnostic arm)

    manifest = sink.save(args.out_dir, args.tag)
    result = {
        "schema": "inferswarm.r6_localization.stage1-diag/1",
        "status": "STAGE1_DIAG_COMPLETE",
        "gpu_uuid": gpu_uuid,
        "role": "first-diag",
        "plan_digest": plan["digest"],
        "canonical_generated_token_ids": expected,
        "capture_manifest": manifest,
    }
    out = Path(args.out_dir) / f"stage1-diag-{args.tag}.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "gpu": gpu_uuid,
                      "records": manifest["record_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
