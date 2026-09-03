"""#71 S-arm runner: single-GPU control with semantic tensor checkpoints.

Runs the accepted R6 single-GPU control replay semantics (reset KV per step,
replay = prompt + generated[0:k], single prefill at position 0, greedy) with
the #71 coarse checkpoints captured: embedding output, after global layers
15/31/47, final norm, full final-row BF16 logits (pre-FP32), final-row FP32
logits. Steps 0/1/7 by default (methodology §6).

Layer-checkpoint capture is opt-in per step via --capture-layers so bisection
rounds (methodology §8) reuse the identical arm.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

EXPECTED_CHECKPOINT_SHA256 = (
    "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d"
)
CAPTURE_STEPS = (0, 1, 7)
COARSE_AFTER_LAYERS = (15, 31, 47)


def _git_identity(repo: Path) -> dict:
    sha = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), "status", "--porcelain"],
        text=True,
    )
    return {"sha": sha, "dirty": bool(status), "status": status.splitlines()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--reference", default="docs/inferswarm_r6/reference-generation.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", default="coarse")
    parser.add_argument("--capture-steps", type=int, nargs="+", default=list(CAPTURE_STEPS))
    parser.add_argument(
        "--capture-layers", type=int, nargs="*", default=[],
        help="additional AFTER-layer global checkpoints (bisection rounds)",
    )
    args = parser.parse_args(argv)

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    repo = Path(__file__).resolve().parents[2]
    reference = json.loads((repo / args.reference).read_text())
    prompt = list(reference["prompt_token_ids"])
    expected_generated = list(reference["generated_token_ids"])
    maximum = len(expected_generated)

    import torch

    from benchmarks.inferswarm_r6_localization.capture import CaptureSink
    from benchmarks.inferswarm_r6.stage_runtime import GemmaDenseStage
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.rotary import set_rope_device
    from freetoken.research.r6_dense_census import (
        DenseBlockSpec,
        checkpoint_census,
        freeze_dense_block_plan,
    )

    producer = _git_identity(repo)
    if producer["dirty"]:
        print(json.dumps({"status": "S_ARM_BLOCKED_DIRTY_SOURCE"}))
        return 2
    import subprocess as sp

    gpu_uuid = sp.check_output(
        ["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    set_rope_device(torch.device("cuda:0"))

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
    block = plan["blocks"][0]

    runtime = GemmaDenseStage(
        role="single",
        model_path=str(model),
        adapter_data={
            **block,
            "declared_shared_state": shared,
            "runtime_capacity_tokens": 64,
        },
    )
    capture_steps = set(args.capture_steps)
    after_layers = tuple(
        sorted(set(COARSE_AFTER_LAYERS) | set(int(x) for x in args.capture_layers))
    )
    sink = CaptureSink(role="single", gpu_uuid=gpu_uuid)
    runtime._capture_sink = sink
    # after-layer hook checkpoints (15/31 via hook; 47 is emitted explicitly
    # in the single prefill path, so keep it out of the hook set to avoid a
    # duplicate record).
    runtime._capture_after_layers = frozenset(
        ({15, 31} | set(int(x) for x in args.capture_layers)) - {47}
    )

    # --- instrumented single-role replay -------------------------------
    # Mirrors GemmaDenseStage.prefill(single) exactly, with checkpoints at
    # the frozen semantic points. Same _prepare/batch/forward semantics;
    # identical math; capture copies are host-side only.
    generated: list[int] = []
    nan_inf_total = 0
    for step in range(maximum):
        runtime.reset_session_state()
        replay = prompt + generated
        capture_now = step in capture_steps
        runtime._capture_step = step if capture_now else None
        token, logits = runtime.prefill(replay, None, 0)
        nan_inf_total += int(
            torch.isnan(logits).sum().item() + torch.isinf(logits).sum().item()
        )
        generated.append(int(token))
        del logits

    exact = generated == expected_generated
    result = {
        "schema": "inferswarm.r6_localization.s-arm-run/1",
        "status": "S_ARM_COMPLETE",
        "producer": producer,
        "gpu_uuid": gpu_uuid,
        "model_path": str(model),
        "generated_token_ids": generated,
        "expected_generated_token_ids": expected_generated,
        "exact_generated_token_match": exact,
        "nan_inf_count": nan_inf_total,
        "capture_steps": sorted(capture_steps),
        "after_layer_checkpoints": list(after_layers),
    }
    if not exact:
        result["status"] = "S_ARM_TOKEN_MISMATCH"
    manifest = sink.save(args.out_dir, args.tag)
    result["capture_manifest"] = manifest
    out = Path(args.out_dir) / f"s-arm-result-{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "exact": exact,
                      "records": manifest["record_count"]}))
    return 0 if result["status"] == "S_ARM_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
