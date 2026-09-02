# Semantic conflict resolutions

The explicit merge reported two conflicts. Each was resolved against the
accepted InferSwarm behavior and the changed upstream runtime contract.

## `python/freetoken/engine/engine.py`

Accepted behavior retained:

- InferSwarm decode-step observation/instrumentation;
- accepted prefill timing and serving integration semantics.

Upstream behavior incorporated:

- `self.model.forward_host_ctx(batch, use_cuda_graph)` around forward, required
  by the new Qwen4 PLE host-context lifecycle;
- `swiglu_clamp` in the CPU MoE supported-activation contract for GLM-5.3.

This resolution preserves the accepted serving waist while honoring the new
model/runtime dependency contract. It does not restore pre-InferSwarm engine
semantics.

## `python/freetoken/models/nvfp4_banks.py`

Accepted behavior retained:

- selective layer bank mapping;
- bounded per-layer shard remapping and page-cache dropping;
- InferSwarm fetch/accounting callbacks.

Upstream behavior incorporated:

- `_canon_kind` normalization;
- `_ingest_global` handling for mixed-precision/global reciprocal values.

This keeps R5A/R5B residency, staging, and accounting behavior while adapting
the bank loader to the upstream representation contract.

No other textual conflicts occurred. Conflict-sensitive changes in model
registration/configuration, MoE offload, CPU native code, attention/KV-cache,
server arguments, runtime reporting, and R5A/R5B serving/epoch paths were
reviewed and qualified from the merged tree.

