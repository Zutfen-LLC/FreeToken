"""#71 intra-layer-0 operator probe: which op first differs across devices?

Runs the layer-0 sub-ops on THIS device from the captured S embedding input
(the byte-exact shared input): input_layernorm -> qkv GEMM -> split/norms ->
rope -> attention -> o_proj -> post-norm/add -> MLP. Captures after each op.

Compare the two probe bundles (3090 vs 3060) to name the earliest divergent
operator inside the first divergent layer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--embedding-capture", required=True,
                        help="captures bundle containing step0 embedding_output")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args(argv)

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    repo = Path(__file__).resolve().parents[2]

    from benchmarks.inferswarm_r6_localization.capture import CaptureSink
    from benchmarks.inferswarm_r6.stage_runtime import GemmaDenseStage
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.rotary import set_rope_device
    from freetoken.research.r6_dense_census import checkpoint_census
    from freetoken.research.r6_dense_census import DenseBlockSpec
    from freetoken.research.r6_dense_census import freeze_dense_block_plan

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    set_rope_device(torch.device("cuda:0"))

    census = checkpoint_census(args.model, text_prefix="model.language_model")
    shared = {
        "id": "tied-embedding-lm-head",
        "kind": "tied-weight-shared-state",
        "tensor_keys": ["model.language_model.embed_tokens.weight"],
        "bytes": census["bytes_by_owner_category"]["embedding/input"],
        "materialization_policy": "single-cuda-tensor-used-for-input-and-output",
    }
    # probe builds a [0,1) block via the plan machinery over a synthetic
    # single-layer census view: freeze_dense_block_plan requires the last
    # block to end at the final layer, so probe the census of a layer-trimmed
    # view instead — simplest correct route: reuse the full plan's block 0
    # keys filtered to layer 0.
    full = freeze_dense_block_plan(
        census,
        [DenseBlockSpec(0, 48, True, True)],
        declared_shared_state=shared,
    )
    layer0_prefix = "model.language_model.layers.0."
    block = {
        "spec": {"start_layer": 0, "end_layer": 1, "owns_embeddings": True,
                 "owns_final_norm_head": False},
        "allowed_tensor_keys": [
            k for k in full["blocks"][0]["allowed_tensor_keys"]
            if k.startswith(layer0_prefix)
            or k == "model.language_model.embed_tokens.weight"
        ],
        "owned_checkpoint_bytes": 0,
    }

    runtime = GemmaDenseStage(
        role="first",
        model_path=args.model,
        adapter_data={
            **block,
            "declared_shared_state": shared,
            "runtime_capacity_tokens": 64,
        },
    )
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()

    # shared input: step-0 embedding_output from the S capture bundle
    bundle = torch.load(args.embedding_capture, map_location="cpu", weights_only=False)
    emb = None
    for meta, tensor in zip(bundle["records"], bundle["tensors"]):
        if meta["step"] == 0 and meta["checkpoint"] == "embedding_output":
            emb = tensor
            break
    if emb is None:
        raise RuntimeError("embedding_output step0 not found in capture bundle")

    layer = runtime.block.layers[0]
    dev = runtime.device

    from freetoken.core import get_global_ctx

    sink = CaptureSink(role="op-probe", gpu_uuid=gpu_uuid)
    runtime._capture_sink = sink
    runtime._capture_step = 0

    def emit(name, t):
        sink.emit(checkpoint=name, step=0, global_layer=0,
                  position_range=[0, int(t.shape[0])],
                  source_device=str(dev), tensor=t)

    x = emb.to(device=dev, dtype=torch.bfloat16)
    batch = runtime._prepare(start=0, token_count=x.shape[0], phase="prefill")
    with runtime.ctx.forward_batch(batch):
        emit("probe_input", x)
        h = layer.input_layernorm.forward(x)
        emit("op1_input_layernorm", h)
        attn = layer.self_attn
        qkv = attn.qkv_proj.forward(h)
        emit("op2_qkv_gemm", qkv)
        q_lin, k_lin, v_lin = qkv.split((attn.q_dim, attn.kv_dim, attn.kv_dim), dim=-1)
        T = x.shape[0]
        q = q_lin.view(T, attn.num_qo_heads, attn.head_dim)
        k = k_lin.view(T, attn.num_kv_heads, attn.head_dim)
        v = v_lin.view(T, attn.num_kv_heads, attn.head_dim)
        q = attn.q_norm.forward(q)
        k = attn.k_norm.forward(k)
        v = attn.v_norm.forward(v)
        emit("op3_qkv_norms", torch.cat([q.flatten(1), k.flatten(1), v.flatten(1)], dim=1))
        positions = batch.positions
        q, k = attn._apply_rope(positions, q, k)
        emit("op4_rope", torch.cat([q.flatten(1), k.flatten(1)], dim=1))
        o = get_global_ctx().attn_backend.forward(
            q.contiguous(), k.contiguous().reshape(T, -1).view(T, attn.num_kv_heads, attn.head_dim).contiguous(),
            v.contiguous().reshape(T, -1).view(T, attn.num_kv_heads, attn.head_dim).contiguous(),
            attn.layer_id, batch, attn_spec=attn.attn_spec,
        )
        emit("op5_attention_out", o)
        o_proj = attn.o_proj.forward(o.reshape(T, -1))
        emit("op6_o_proj", o_proj)
        post = layer.post_attention_layernorm.forward(o_proj)
        emit("op7_post_attn_norm", post)
        pre_ff, res = layer.pre_feedforward_layernorm.forward_add_residual(post, x)
        emit("op8_pre_ff_norm_residual", pre_ff)
        ff = layer.feed_forward.forward(pre_ff, res)
        emit("op9_mlp_out", ff)

    manifest = sink.save(args.out_dir, args.tag)
    result = {
        "schema": "inferswarm.r6_localization.op-probe/1",
        "status": "OP_PROBE_COMPLETE",
        "gpu_uuid": gpu_uuid,
        "layer": 0,
        "ops": ["input_layernorm", "qkv_gemm", "qkv_norms", "rope",
                "attention", "o_proj", "post_attn_norm", "pre_ff_norm_residual",
                "mlp"],
        "capture_manifest": manifest,
    }
    out = Path(args.out_dir) / f"op-probe-{args.tag}.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "gpu": gpu_uuid}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
