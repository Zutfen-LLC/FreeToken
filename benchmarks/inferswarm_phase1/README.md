# InferSwarm Phase-1 diagnostics

This namespace contains narrow Phase-1 engineering diagnostics. It is not the P5 campaign
runner and does not collect candidate throughput or produce a Phase-1 performance verdict.

## P1 device probe

The P1 probe does not allocate the frozen expert bank, execute remote experts, benchmark
throughput, or change model outputs.

Run from the FreeToken repository root with both physical GPU UUIDs explicit:

```bash
PYTHONPATH=python:benchmarks python -m inferswarm_phase1.device_probe \
  --primary-gpu GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55 \
  --secondary-gpu GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176
```

The command emits JSON on stdout. Add `--output phase1-device-probe.json` to write a file.
It records CUDA-visible device identity, full UUIDs when available, memory at probe time,
compute capability, peer-access capability in both directions, the resulting descriptive
transport classification, and whether the primary remained current. It also captures
`nvidia-smi topo -m`, `nvidia-smi topo -p2p r`, PCI bus IDs, and PCIe link generation/width
when `nvidia-smi` is available. Missing NVIDIA CLI data is recorded explicitly and does not
invalidate the CUDA-level probe.

On `inferswarm01`, the measured acceptance assertions are: both UUIDs identify RTX 3060s;
the UUIDs are distinct; peer access is false in both directions; transport is
`host_staged_required`; secondary total memory is approximately 12 GB nominal; and
`primary_current_after_probe` is true. These are physical-host checks—ordinary CI does not
claim to reproduce them.

## P3 correctness fixture

The P3 fixture loads the canonical model and frozen placement, then compares identical
native-NVFP4 routed-layer inputs through (1) ordinary complete GPU0 execution and (2) the
serialized host-staged local/remote partition. It exercises mixed, local-only, remote-only,
and 40-layer ownership cases; applies the C1 `rtol=2e-3, atol=2e-3` gate; checks the GPU0
residency/copy plan; and reports C2 ownership plus C4 NaN/Inf health. It records no timing or
throughput fields.

```bash
PYTHONPATH=python:benchmarks python -m inferswarm_phase1.p3_correctness \
  --model /path/to/nvidia/Qwen3.6-35B-A3B-NVFP4/491c2f1... \
  --primary-gpu GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55 \
  --secondary-gpu GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176 \
  --placement /path/to/phase1-qwen36-placement-v1.json
```

The JSON label deliberately says `MEASURED ENGINEERING FIXTURE (not canonical C3)`. This
command does not reproduce the full warmed W1-W4 correctness-reference protocol and must
not be cited as canonical C3 evidence.

## P4 overlap fixture

The P4 fixture runs one bounded multi-token deterministic mixed routed layer in explicit
`serialized` and `overlap` modes. It checks numerical output, ownership, transfer-byte
accounting, dispatch count, fallback state, direct GPU1-event pending state at GPU0 local
service entry, and the persistent staging-buffer lifecycle. Complete-layer wall,
GPU0-local branch, and GPU1-branch intervals are measured independently; the tool never
sums concurrent branch durations or reports end-to-end token throughput.

```bash
PYTHONPATH=python:benchmarks python -m inferswarm_phase1.p4_overlap \
  --model /path/to/nvidia/Qwen3.6-35B-A3B-NVFP4/491c2f1... \
  --primary-gpu GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55 \
  --secondary-gpu GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176 \
  --placement /path/to/phase1-qwen36-placement-v1.json \
  --output p4-overlap-engineering.json
```

## P4 W1-W4 mechanism/timing smoke

`p4_workload_smoke` drives an already-running candidate or B1 server without collecting
request timestamps, throughput, TTFT, or a Phase-1 verdict. It performs two warmups,
resets the idle instrumentation boundary, and snapshots every requested workload class
independently. Candidate snapshots mechanically evaluate F1/F2/F3/F5/F6. B1 snapshots
require active CUDA graph replay and use the same MoE timing schema with remote operations
explicitly marked `not_applicable`.

The ordinary OpenAI-compatible stream remains unchanged and does not expose exact generated
token IDs or step-0 logits. This mechanism tool therefore retains its decoded-text diagnostic
without calling it C3. Exact C3 evidence is collected separately through the opt-in internal
generation-state recorder described below.

```bash
PYTHONPATH=python:benchmarks python -m inferswarm_phase1.p4_workload_smoke \
  --origin http://127.0.0.1:48145 \
  --manifest /path/to/phase0-canonical-manifest.json \
  --reference-jsonl /path/to/p0h-warmed-reference.jsonl \
  --tokenizer-model /path/to/model-revision \
  --model-id nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --role baseline \
  --class W4 \
  --output p4-b1-graph-timing.json
```

## Exact C3 correctness diagnostic

`c3_correctness` uses the disabled-by-default
`--inferswarm-correctness-diagnostics` recorder. The recorder copies the actual step-0
sampler-input logit vector inside the engine and appends scheduler-accepted token IDs before
detokenization. It does not change ordinary SSE payloads, sampling parameters, or generation
logic. Its first-logit host copy is intentionally incompatible with performance evidence, so
never enable it for a performance run.

Start the frozen `CORRECTNESS_REFERENCE` configuration with
`--inferswarm-correctness-diagnostics`, then capture two independent exact reference
sequences per W1-W4 fixture after the frozen two warmups:

```bash
PYTHONPATH=python:benchmarks python -m inferswarm_phase1.c3_correctness \
  --origin http://127.0.0.1:48145 \
  --manifest /path/to/phase0-canonical-manifest.json \
  --model-id nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --role reference \
  --output c3-reference-exact.json
```

Restart with the frozen distributed candidate, also with the recorder enabled, and compare
against those exact reference bytes:

```bash
PYTHONPATH=python:benchmarks python -m inferswarm_phase1.c3_correctness \
  --origin http://127.0.0.1:48145 \
  --manifest /path/to/phase0-canonical-manifest.json \
  --model-id nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --role candidate \
  --reference-evidence c3-reference-exact.json \
  --output c3-candidate-exact.json
```

The tool applies the frozen criterion without a new tolerance: first 64 generated token IDs
exact, step-0 argmax and top-5 ordering exact, and the full step-0 logit vector within
`rtol=2e-3, atol=2e-3`. Reference self-consistency must pass first. The JSON explicitly
contains no throughput, TTFT, prefill ratio, aggregate speedup, or Phase-1 verdict field.

## C3 numerical root-cause diagnostic

`c3_root_cause` captures decode step 0 at every MoE `_decode_routed` boundary. Every tensor
has dtype, shape, exact raw-byte SHA-256, and a raw binary sidecar. The four shapes must be
run from separately restarted servers. The client uses the unchanged frozen request bodies,
two warmups, greedy sampling, fixed output length, `ignore_eos`, and the existing
reset-delimited cache semantics. It records no request or kernel performance field and
rejects such a field before writing JSON.

Use the exact pinned model snapshot and v2 artifact. This common server fragment applies to
all four shapes:

```bash
ft serve \
  --model /path/to/nvidia/Qwen3.6-35B-A3B-NVFP4/491c2f1ea524c639598bf8fa787a93fed5a6fbce \
  --served-model-name nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --gpu GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55 \
  --moe-backend offload \
  --moe-cpu-layers 0 \
  --nvfp4-backend triton \
  --moe-cache-size 3774 \
  --kv-reserve-tokens 17075 \
  --num-tokens 17075 \
  --memory-ratio 0.85 \
  --cuda-graph-max-bs 0 \
  --max-running-requests 1 \
  --sampling-defaults none \
  --inferswarm-correctness-diagnostics \
  --inferswarm-c3-root-cause-mode trace
```

For R, use that command unchanged. For O, append the v2 placement, secondary UUID, remote
decode, and overlap selector:

```bash
--inferswarm-placement /path/to/phase1-qwen36-placement-v2.json \
--inferswarm-secondary-gpu GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176 \
--inferswarm-remote-decode \
--inferswarm-remote-mode overlap
```

For S, change only the last value to `serialized`. For G, append only the placement and
replace the root-cause mode value with `DIAGNOSTIC_SPLIT_GPU0`; do not supply a secondary
GPU, remote-decode flag, or remote mode. The parser rejects those combinations. G executes
two complementary calls to the production native-NVFP4/Triton expert GEMM on GPU0 and one
combine, and is explicitly ineligible for F-gate evidence.

After each restarted server is ready, capture its shape. Omitting `--class` captures all
W1-W4 controls; pass `--class W3 --class W4` only when the requested investigation does not
need W1/W2 route-geometry controls.

```bash
PYTHONPATH=python:benchmarks python -m inferswarm_phase1.c3_root_cause capture \
  --origin http://127.0.0.1:48145 \
  --manifest /home/zutfen/inferswarm/docs/benchmarks/workloads/phase0-v1/manifest.json \
  --model-id nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --shape R \
  --output /path/to/phase1-v2-evidence/c3-root-cause-reference.json
```

Repeat with `--shape O`, `S`, and `G` and the corresponding output names, then compare all
six required pairs mechanically:

```bash
PYTHONPATH=python:benchmarks python -m inferswarm_phase1.c3_root_cause compare \
  --reference /path/to/phase1-v2-evidence/c3-root-cause-reference.json \
  --overlap /path/to/phase1-v2-evidence/c3-root-cause-overlap.json \
  --serialized /path/to/phase1-v2-evidence/c3-root-cause-serialized.json \
  --split-gpu0 /path/to/phase1-v2-evidence/c3-root-cause-split-gpu0.json \
  --output /path/to/phase1-v2-evidence/c3-root-cause-comparison.json
```

The comparison reports the first exact and first C1-tolerance divergence independently for
hidden input and MoE output, plus the first router-ID and routing-weight divergence. It also
checks every R/G layer for identical raw routes, routing weights, selected expert-row hashes,
production kernel, GPU0 device, and complementary ownership masks.

Use the first relevant class/layer from that report for the serialized fixed-input replay:

```bash
PYTHONPATH=python:benchmarks python -m inferswarm_phase1.c3_layer_replay \
  --model /path/to/nvidia/Qwen3.6-35B-A3B-NVFP4/491c2f1ea524c639598bf8fa787a93fed5a6fbce \
  --primary-gpu GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55 \
  --secondary-gpu GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176 \
  --placement /path/to/phase1-qwen36-placement-v2.json \
  --reference-trace /path/to/phase1-v2-evidence/c3-root-cause-reference.json \
  --class W3 \
  --layer FIRST_DIVERGENT_LAYER \
  --output /path/to/phase1-v2-evidence/c3-root-cause-layer-replay.json
```

The replay evaluates U/GL/GR/GS and actual RL/RR/RC from the same losslessly restored
input/routing tensors. It activates transport-stage capture and selected GPU1 resident-row
revalidation only if GR and RR differ. It characterizes a proven split-reduction result but
does not change the production combine or implement a numerical fix.
