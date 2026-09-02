"""R6 correctness reference: unpartitioned Gemma-4-12B text generation.

Independent-framework comparator run on a temporary reference device
(a single GPU that fits the BF16 text tower, or CPU).  Deterministic
greedy decoding; retains exact prompt token IDs, generated token IDs,
and selected top-k logits at declared checkpoints.

The comparator contract (token equality primary; top-32 logits absdiff
< 0.25 per token) is frozen HERE before any distributed result exists.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

COMPARATOR_CONTRACT = {
    "schema": "inferswarm.r6.comparator-contract/1",
    "frozen_before_distributed_results": True,
    "primary": "exact generated-token-id equality (greedy, 8 tokens)",
    "secondary": {
        "metric": "max |logit_ref - logit_dist| over top-32 ref logits per step",
        "threshold": 0.25,
        "note": "top-32 union of reference and distributed top-32, float32",
    },
    "nan_inf_policy": "any NaN/Inf in distributed logits fails the run",
}


def generate_reference(
    *,
    model_path: str,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    device: str = "cuda:0",
    logits_steps: tuple[int, ...] = (0, 1, 7),
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    load_seconds = time.perf_counter() - started

    tokens: list[int] = []
    captured: dict[str, dict] = {}
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    generated = []
    step_logits = {}
    t0 = time.perf_counter()
    with torch.inference_mode():
        for step in range(max_new_tokens):
            out = model(input_ids=input_ids)
            logits = out.logits[0, -1].float()
            if step in logits_steps:
                top = torch.topk(logits, 32)
                step_logits[str(step)] = {
                    "top_indices": top.indices.tolist(),
                    "top_values": top.values.tolist(),
                }
            next_id = int(torch.argmax(logits).item())
            generated.append(next_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=device)], dim=1
            )
    wall = time.perf_counter() - t0
    return {
        "schema": "inferswarm.r6.reference-generation/1",
        "model_path": str(model_path),
        "prompt_token_ids": list(prompt_token_ids),
        "generated_token_ids": generated,
        "step_top32_logits": step_logits,
        "comparator_contract": COMPARATOR_CONTRACT,
        "load_seconds": load_seconds,
        "decode_wall_seconds": wall,
        "device": device,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", required=True,
                        help="JSON file or comma-separated ids")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    raw = args.prompt_tokens
    p = Path(raw)
    if p.exists():
        prompt = json.loads(p.read_text())["prompt_token_ids"]
    else:
        prompt = [int(x) for x in raw.split(",")]

    result = generate_reference(
        model_path=args.model,
        prompt_token_ids=prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"generated_token_ids": result["generated_token_ids"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
