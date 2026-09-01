# R2 correctness diagnostics

All files in this directory are `NONCANONICAL_DIAGNOSTIC_EVIDENCE`. They do not
replace the retained `correctness.json`, `benchmark.json`, `result.json`, or frozen
plan. The historical verdict remains `R2_LOCAL_SPLIT_EXECUTION_BLOCKED_CORRECTNESS`.

## Diagnosis

Classification: `REFERENCE_GEOMETRY_MISMATCH`.

The frozen R2 split is byte-identical to a one-GPU full-model matched control under
the same W4 chunk64/state protocol. The legacy external reference does not retain
enough runtime provenance and was generated under a different prefill geometry.
Chunk geometry first changes the retained global layer-0 GatedDeltaNet recurrent
state, before the layer-19 boundary. No R2 implementation bug was found or fixed.

## Physical matrix

The unchanged comparison threshold is `rtol=0.002, atol=0.002`.

| Case | Prompt | Chunk | Chunks | Tokens vs legacy | Selected-logit result |
| --- | ---: | ---: | ---: | --- | --- |
| W2 current | 54 | 64 | 1 | exact | byte-exact at steps 0/1/15/31 |
| W2 diagnostic | 54 | 32 | 2 | exact | fail; max abs 1.15234375; NaN/Inf 0/0 |
| W4 current | 121 | 64 | 2 | exact | fail; max abs 0.75390625; NaN/Inf 0/0 |
| W4 diagnostic | 121 | 128 | 1 | diverges later | steps 0/1 byte-exact; steps 15/31 diverge |
| W4 matched local | 121 | 64 | 2 | exact | byte-identical to R2 split at every selected step |

W2 changing from one to two chunks creates the same class of selected-logit drift.
W4 changing from two chunks to one restores byte-exact legacy-reference logits at
steps 0 and 1. Later W4 decode divergence is separate evidence that the legacy
artifact's unreported graph/state protocol is also not sufficiently matched.

## Matched local and boundary result

The one-GPU control uses GPU
`GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176` (RTX 3060), complete full-model
execution, FI attention, NVFP4 Triton, ordinary expert offload, cache size 3,774,
page size 1, capacity 17,152, concurrency 1, chunk64 manual prefill, fresh zeroed
KV/recurrent state, and one bs=1 full-model decode graph. It removes the second
Compute Unit, process split, activation transport, and R2 execution edge.

At both W4 prefill chunks, the matched local control and R2 split have byte-identical:

- layer-18 hidden and residual;
- every logical Block A and Block B KV layer;
- every logical Block A and Block B conv/recurrent state;
- Block B output hidden/residual;
- final norm and logits.

The local layer-18 pair equals the R2 Block A producer hash, and the Block B consumer
hash remains equal to the producer hash. All 32 generated tokens and selected logits
are also byte-identical between matched local and split execution. There is no
split-specific first divergent checkpoint, global layer, or mutable state.

## Geometry first divergence

After the complete W4 prompt, matched-local chunk64 and chunk128 first differ in
global layer 0 recurrent state:

```text
state:       GatedDeltaNet recurrent
shape:       [32, 128, 128]
dtype:       float32
chunk64 SHA: 406bb2e95ba1e2f3ae1dac181affb99ff8ca48a38ce871a828897547130a2adc
chunk128 SHA:313c280746aff6d1f889248198d4dd81fc5475525b50bcbf81de6a06e1131cbb
max abs:     0.043923377990722656
max rel:     18153.4453125
NaN/Inf:     0/0
```

Layer 0 convolution state remains byte-exact. The first KV difference appears at
global layer 3. The first observed difference from the legacy reference is W4's
step-0 logits after the second chunk; that artifact retained no layer/state captures,
so it cannot support a more specific legacy-side operation comparison.

## Reference provenance

Legacy reference SHA-256:
`cc5d7b64323fa2864f3add1193f72f35387780ea9abc8c9f85acc42695864952`.

It records schema, model, revision, producer commit, manifest, prompts, workloads,
and greedy generation, but omits resolved runtime configuration, prefill chunk,
capacity, reset protocol, and graph policy. The enhanced harness fails closed on
those omissions for canonical comparisons. An explicit legacy diagnostic allowance
is available only with `NONCANONICAL_DIAGNOSTIC_OVERRIDE`, which refuses canonical
`correctness.json` and `result.json` output paths.

## Artifact index

| Artifact | SHA-256 |
| --- | --- |
| `chunk-controls.json` | `c5254e57ffb79f1627a629b994474404ac75a011c1b89b9d960224147141a9e5` |
| `matched-local-control.json` | `2434b6bd4279969a1bb49dcfc84fcfdcaf4e8554e328fe5a53b2bfc9245911a4` |
| `first-divergence.json` | `09b81c3e8c8f9cab338a84016e2f813d5dfff2fffb5d52fb64e14bc81c7e6cab` |
| `w2-chunk32.json` | `558e2e4388c9ab92458f4f46dd14297e0aaaff85ae457b85a12ba8f6f2676ba0` |
| `w4-chunk128.json` | `88f60d4128281852b57d7dc513fbe7289688e08cfe250539061aedcc174f5e21` |
| `w4-split-chunk64-state.json` | `4febe65b846de70f04c160e586b838b56386e23645cbaa344a63ded20b6cf8a1` |
| `w4-matched-local-chunk64.json` | `7a28beb99de00694be97b7030452ae97a00748de1115ae4be875fd055a2c6198` |
| `w4-matched-local-chunk128.json` | `304b3b02b374c5f4ab92db4998404758ff802964aaf3ddf1db4eedf41d71baea` |

Every JSON has a sibling `.sha256`. The proposed corrected methodology is frozen
for review in InferSwarm PR #52. Per the experiment firewall, no corrected retained
R2 candidate evaluation has been run.
