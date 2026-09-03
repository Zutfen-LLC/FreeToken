# R6 Localization (#71) — Code-Path Map (Phase 1)

Producer base: `inferswarm-research@51e772ab88643df61888c8860c8e67e307190565`.
This document maps the ACTUAL landed execution paths for the single-GPU (S) and
distributed (D) FreeToken arms before any instrumentation is written. It is
descriptive code reading, not measurement; every claim here is re-verified
mechanically in the configuration-equivalence audit (Phase 7).

## Shared implementation (both arms)

`benchmarks/inferswarm_r6/stage_runtime.py::GemmaDenseStage` is the ONLY
execution implementation. The two arms differ in role and placement, not in
layer code:

- construction (`__init__` → `_selective_load` → `_setup_context`):
  - `parse_config(cached_load_hf_config(model_path))` → `full_config`
    (all 48 layers' attention groups; global layer IDs).
  - `self.config = self._stage_config(full_config)`:
    `attention_groups` renumbered order-preserving to stage-local IDs
    (`local_of[gid]`), `num_layers = owned count`. Everything else in the
    dataclass is inherited unchanged from `full_config`.
  - modules built on meta: `VocabParallelEmbedding` (from `full_config`,
    embed_scale=√hidden), `Gemma4DecoderLayer(self.config, local_id)` per
    owned layer, `GemmaRMSNorm` final norm (from `full_config`).
  - weight streaming: checkpoint `q/k/v` merged into `qkv_proj`
    (`k_eq_v` full-attn layers have no stored `v`; `v := k`),
    `gate/up` merged into `gate_up_proj`, via `materialize_dense_state`
    (same code both arms; destination slices on cuda:0).
  - `_setup_context`: `Context(1)`; KV pool from the STAGE-LOCAL config
    (`create_kvcache_pool(self.config, num_pages=max_seq_len, page_size=1,
    bf16)`); `ctx.page_table = arange(max_seq_len).unsqueeze(0)`;
    `ctx.attn_backend = create_attention_backend("triton", self.config)`.
- execution (`prefill` / `decode`, both `@torch.inference_mode()`):
  - `_prepare` → `_make_batch(start, token_count, phase)`:
    `positions = arange(start, end)` int32; `out_loc = arange(start, end)`;
    then `attn_backend.prepare_metadata(batch)`
    (`TritonMetadata`: indptr/indices from page_table rows, cu_seqlens,
    q_to_req, q_positions, swa_indices when the pool is swa-paged).
  - role first/single: `hidden = embed(input_ids)` →
    `forward_layers` → (single only) `finalize` (final norm) →
    `final_row_logits` = `full_bf16_logits(final)[-1].float()`.
  - role middle/last: receive `hidden_or_ids` tensor → `forward_layers` →
    (last only) finalize + head.
  - `full_bf16_logits(final)`: `logits = final @ embed_tokens.weight.t()`
    (full [seq, vocab] BF16 GEMM against the shared/tied embedding table),
    then softcap (`softcap_inplace` by default: div_/tanh_/mul_, cap from
    `full_config.final_logit_softcapping`).
- layer math (`python/freetoken/models/gemma4/model.py::Gemma4DecoderLayer.forward`):
  attention sandwich (input_layernorm → Gemma4Attention →
  post_attention_layernorm → pre_feedforward_layernorm.forward_add_residual)
  → dense MLP. `Gemma4Attention.forward`: fused qkv GEMM → split →
  q/k/v RMSNorms → `_apply_rope(positions, q, k)` (in-place on contiguous
  views via the shared `get_rope` module; rope device set once per process
  via `set_rope_device(cuda:0)`) → `ctx.attn_backend.forward(
  q, k, v, self.layer_id /* STAGE-LOCAL ID */, batch, attn_spec)` → o_proj.
- Triton backend (`python/freetoken/attention/triton.py`):
  `store_kv(k, v, batch.out_loc, layer_id)` writes the stage-local pool
  slab, then prefill routes to `extend_paged_attention` when
  `q.shape[-1] <= 256 or max_q_len >= 128` (SWA layers, head_dim 256),
  else the naive `paged_attention` kernel (full-attn layers, head_dim 512,
  short replays); decode routes to `decode_paged_attention`
  (split-K, `max_kv_splits = 8`).

## Single (S) role — accepted RTX 3090 control path

`benchmarks/inferswarm_r6/single_gpu_control.py` on inferswarm04
(GPU-ecda1aaa-0c66-857b-8218-3d511dc75c03), model `/srv/models/gemma-r6`
(sha256 5a84cb31…ff18d verified per-run):

```
per step k in 0..7:
    runtime.reset_session_state()          # KV pools zeroed + sync
    replay = prompt + generated[0:k]        # 26..33 rows
    token, logits = runtime.prefill(replay, None, start=0)
        embedding → layers 0..47 → final norm → full BF16 GEMM →
        in-place softcap → final-row FP32
    argmax(logits) → generated.append(token)   # greedy
```

One process, one GPU, no inter-stage boundary
(`semantic_boundaries_for_role("single") == []`). Stage-local config for the
full range is the identity renumbering (local i == global i), so S itself
also runs "stage-local" layer IDs — the renumbering map is just the identity.

## Distributed (D) role — canonical three-stage chain

Topology (accepted R6): stage 1 = inferswarm01 GPU0 (embed + layers [0,16)),
stage 2 = inferswarm01 GPU1 (layers [16,32)), stage 3 = inferswarm03 GPU0
(layers [32,48) + final norm + tied head). All three GPUs are RTX 3060
(sm_86, 12 GiB). Coordinator inferswarm00 drives the node agent on 01
(`benchmarks/inferswarm_r6/node_agent.py` → `chain_runtime.py`).

- stages 1–2: `stage_chain.py::StageClient` spawns one process per GPU
  (`CUDA_VISIBLE_DEVICES` pinning; spawn context). Control protocol over a
  multiprocessing pipe: PREFILL / DECODE / REPORT / RESET / SHUTDOWN.
- stage 3: `last_stage_service.py` on inferswarm03 over the accepted R4 wire
  (`r4_wire` framing, sha256 payload checksum in diagnostic mode), driven by
  `wire_client.py::RemoteLastStageClient` from 01.

Per committed step k (R5B controller semantics, `serve_tokens`):

```
RESET all stages (KV zeroed)
replay-prefill(prompt + committed[0:k]):
    stage 1: embed → layers 0..15 → hidden.cpu() → pipe
             (BOUNDARY_PAYLOAD; pickled bf16 tensor)
    stage 2: hidden.to(cuda:0, bf16) → layers 16..31 → hidden.cpu() →
             wire_client serializes contiguous bf16 bytes (frame + checksum)
             → 03: torch.frombuffer → view(bf16) → reshape → .to(cuda:0)
    stage 3: layers 32..47 → final norm → full BF16 GEMM → softcap →
             final-row FP32 → argmax → TOKEN_RESULT (token id only on wire)
decode(speculative, discarded): 1 row at position len(replay)
```

Both arms therefore compute the same replay-prefill per step from a zeroed
KV state with identical token inputs, positions (arange(0, T)), and
checkpoint weights (same merged layout code). The accepted D logit capture
(`LogitCapture`) recorded every prefill+decode call; retained comparator
steps 0/1/7 correspond to the replay-prefill rows, matching the S control's
step capture semantics.

## Where things live (localization hooks)

| concern | location |
| --- | --- |
| hidden tensors | `GemmaDenseStage.prefill/decode` locals; `execute_dense_layer_sequence` loop |
| boundary 1 send | `stage_chain._stage_entry` (role=first): `hidden.cpu()` in BOUNDARY_PAYLOAD |
| boundary 1 recv | `stage_chain._stage_entry` (role!=first): `message["hidden"].to(cuda:0, bf16)` |
| boundary 2 send | `wire_client._boundary`: `hidden.contiguous().view(uint8).cpu().tobytes()` |
| boundary 2 recv | `last_stage_service.serve`: `torch.frombuffer(...).view(bf16).reshape(...).to(cuda:0)` |
| stage-local vs global layer identity | `_StageModules.global_layer_ids` (global) vs `Gemma4DecoderLayer(self.config, local)` (local into attention/KV) |
| stage-local config construction | `GemmaDenseStage._stage_config` |
| attention-group remapping | `_stage_config` (order-preserving renumber) |
| KV/cache ownership | `_setup_context` (pool from stage-local config), `reset_session_state`, `logical_state_records` |
| position/batch metadata | `_make_batch` + `TritonAttentionBackend.prepare_metadata` |
| final norm/head | `finalize` / `full_bf16_logits` / `final_row_logits` |

## A priori suspect classes (to be tested, not assumed)

1. Stage-local config / layer-identity misuse (`MODEL_STRATEGY_LOCAL`) —
   the renumbering is order-preserving so per-layer group kind, rope,
   k_eq_v, head_dim SHOULD be preserved; Phase 7 verifies mechanically.
2. Backend/kernel execution differences (`BACKEND_EXECUTION_LOCAL`) — S
   computes on an RTX 3090 (82 SMs) while all D stages compute on RTX 3060s
   (28 SMs): same sm_86 arch, but cuBLAS/Triton launch heuristics can differ
   by SM count, changing accumulation order in GEMMs. The same-input stage
   replay (Phase 6) plus a same-GPU-model diagnostic replay separates this
   from config causes.
3. Boundary transport (`BOUNDARY_TRANSPORT`) — only if sender/receiver byte
   hashes differ (both boundaries carry value-preserving byte copies; the
   wire adds a checksum).
4. Generic InferSwarm seam (`GENERIC_INFERSWARM_SEAM`) — control/plan/
   execution contract altering numerics; no candidate identified in code
   reading (the coordinator only drives token commits and never touches
   tensor values), retained as the residual class.

Non-claims: this map asserts nothing about which class actually produces
`D−S`; that is what the physical campaign decides.
