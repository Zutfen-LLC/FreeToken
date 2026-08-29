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

## P5 canonical campaign runner

`campaign*.py` + the `benchmarks/phase1_campaign.py` entry point are the P5 canonical
Phase-1 A/B campaign runner, built on the Phase-0 benchmark contract implementation in
`benchmarks/inferswarm_phase0/` (manifest, provenance, client, statistics, validity) —
not a second benchmarking framework.

### P5 / P6 boundary

P5 delivers the runner and proves by dry-run/provenance validation that baseline and
candidate are comparable. P5 collects **no** candidate/baseline ratio and emits no
campaign verdict. P6 (only after the InferSwarm campaign-order amendment merges) runs
the physical two-session campaign and the analysis that applies the frozen decision
criteria.

### Canonical commands

```bash
# the complete two-session plan, no model started
python benchmarks/phase1_campaign.py plan \
  --model /path/to/nvidia/Qwen3.6-35B-A3B-NVFP4/491c2f1... \
  --manifest /path/to/phase0-v1-2026-08-27/manifest.json \
  --model-revision 491c2f1ea524c639598bf8fa787a93fed5a6fbce \
  --placement /path/to/phase1-qwen36-placement-v2.json \
  --inferswarm-commit <40-hex> \
  --correctness-prerequisites /path/to/prerequisites.json \
  --output campaign-plan.json

# dry-run validation: preflight refusals, held-constant comparison, counts, ordering
python benchmarks/phase1_campaign.py validate ...same flags... --output campaign-validation.json

# P6 execution surface (one session at a time; session 2 needs the thermal attestation)
python benchmarks/phase1_campaign.py run-session --session 1 ...same flags...
python benchmarks/phase1_campaign.py run-session --session 2 \
  --thermal-reset-attested "independently cooled reset observed at ..." ...same flags...
```

`plan` and `validate` never start a model. A canonical session refuses to start on a
dirty FreeToken tree, a non-40-hex model revision, the wrong workload-manifest
identity (`phase0-v1-2026-08-27` at its frozen SHA-256), a placement artifact whose
SHA-256 is not the frozen `2f62bb84...` value, unresolvable GPU UUIDs, or a missing
correctness prerequisite manifest. The prerequisite manifest is bound to the exact
campaign checkout: `freetoken_runtime_commit` must be a valid 40-hex SHA **and equal
the current clean FreeToken HEAD** (current-commit equality is mandatory —
correctness qualified on another build is not correctness for this campaign), every
declared evidence digest must be a lowercase normalized 64-hex SHA-256, and when the
manifest supplies artifact paths the bytes are rehashed and must agree (a missing
path is recorded as "identity syntactically verified; bytes not independently
rehashed"). Session 2 additionally refuses to start unless a COMPLETE, VALID
session-1 record whose campaign-build baseline identity gate passed exists under
`--out-root`.

### Two counterbalanced arm-major sessions

Repetition-level `A/B/A/B` interleaving is physically impossible on this rig: both
arms require exclusive use of the same 12-GB GPU 0, and reloading a server between
individual measured repetitions would destroy the warmed serving state the protocol
measures. The executable ordering (frozen by the InferSwarm campaign-order amendment
before any candidate performance existed) is:

- **Session 1** (fresh thermal reset): `baseline_b1` → `candidate_v2`;
- **Session 2** (independent operator-attested thermal reset): `candidate_v2` →
  `baseline_b1`;
- within every arm/server process: `W1 → W2 → W3 → W4` in **both** sessions (workload
  order is never reversed), 2 discarded warmups + 10 measured generations per class,
  one fresh server process per arm per session, no restart between classes, no radix
  cache clearing between classes.

Per arm: 48 generations. Per session: 96 primary generations. Campaign: 192 primary
generations. The predeclared conditional supplementary arm adds up to 48 *possible*
generations per session, counted separately and executed only when its fixed
condition resolves true. The whole plan is built before the first server starts and
cannot be shortened dynamically.

### The arms

`baseline_b1` is the frozen Phase-0 `CANONICAL_PERFORMANCE_BASELINE` identity (B1)
remeasured on the exact campaign build — single GPU 0, offload backend, auto NVFP4,
auto expert cache, graphs enabled, no InferSwarm treatment. **Session 1's B1 runtime
resolution is the campaign-build baseline identity gate**: it must pass before the
first candidate measurement anywhere in the campaign, and session 2 (candidate
first) refuses to start without a passing session-1 record under `--out-root`. If
Session-1 B1 no longer resolves to the recorded configuration (Triton, GPU decode,
zero CPU MoE layers, ~3,774 slots), the session stops with **no candidate
generation** and the Phase-0 baseline must be refreshed; no other arm is
substituted. **Session 2 revalidates B1 when its counterbalanced B1 arm runs** —
by design it cannot stop before candidate performance (no preliminary B1 server is
started before the Session-2 candidate, because that would perturb the
candidate-first thermal/cache condition). If the revalidation drifts, session 2 is
`INVALID`, its already-collected candidate measurements are retained as invalid
evidence and are not eligible for the Phase-1 analysis, the baseline must be
refreshed, and the complete affected campaign is rerun with no candidate data
reused or spliced.

`candidate_v2` is the exact landed candidate: GPU0 cache fixed at 3,774 slots,
`--num-tokens 17075`, graphs disabled for the cross-device path, the frozen
`phase1-qwen36-placement-v2` placement (SHA verified before startup) on the secondary
GPU, host-staged remote decode in overlap mode. Its runtime contract requires the
resolved KV capacity to equal 17,075 tokens — the pin that makes the conditional
supplementary arm fully specified before performance.

`baseline_b1_kv_matched` is the **predeclared conditional supplementary arm** (never
primary). Everything about it is fixed in every canonical plan before execution:
B1 plus exactly `--num-tokens 17075`; the trigger
`candidate_resolved_kv_capacity != baseline_resolved_kv_capacity`, evaluated from
the two primary arms' resolved runtime reports after both exist; up to 48 possible
generations per session positioned after both primary arms; and its non-gating
supplementary status. After both primary runtime reports exist: equal capacities →
the arm is recorded `NOT_REQUIRED_BY_KV_RULE` and its generations are not executed;
different capacities → the already-predeclared arm executes. No performance number
controls this branch. Completion accounting counts required primary generations and
conditional supplementary generations separately and records whether the condition
resolved and whether the required supplementary block completed; a canonical session
cannot be `COMPLETE`/`VALID` when the condition is true and the block is missing.
Manual `--kv-matched-tokens` exists only as a dev-smoke/testing override that forces
the arm unconditionally; canonical runs never pass it.

### Held constants and intended differences

`validate` emits the machine-readable comparison. Held equal: GPU0, model revision,
workload manifest, output lengths, sampling, memory ratio, KV reserve, batch size
(`--max-running-requests 1`), prefill instrumentation. Intended differences
(exhaustively enumerated; an undeclared difference fails validation): InferSwarm
remote execution flags, the fixed candidate expert placement/cache flags, CUDA graph
state, the candidate's KV-capacity pin, and the timing-role label.

### Dry-run evidence

A passing `validate` document has `canonical: true`, empty `preflight_refusals`,
`held_equal_all: true`, no `undeclared_differences`, counts exactly
96 primary generations per session / 192 per campaign, session orders
`[baseline_b1, candidate_v2]` then `[candidate_v2, baseline_b1]`, identical
`W1 → W2 → W3 → W4` class order in both sessions, and
`supplementary_predeclaration.predeclared: true` with the conditional arm's exact
flags, fixed trigger, and pinned 17,075-token capacity stated in advance.

### Artifact layout

```
phase1-campaign/
  campaign-plan.json              (from `plan`)
  session-1/
    plan.json                     every expected generation of the session
    provenance.json               repo/model/workload/GPU/software/host/placement/prerequisites
    baseline_b1/                  startup.json runtime.json W1..W4.jsonl summary.json server.log
    candidate_v2/                 ... + block-mechanism-W*.json
    session-summary.json          execution order, block identities, completion, checks
  session-2/ ...
```

Every generation (warmups and failures included) is one JSON line with schema, block
identity, execution index, full per-token timings, TTFT, prefill record, and output
hash. No repetition is ever deleted or invisibly retried; a failed generation is
preserved in place and its block is incomplete. Block reruns create a NEW block
identity (`block-2`, `block-3`, ...) with the discard reason recorded; the original
block is never overwritten.

### Mechanism and issue-#5 timing evidence

Per (arm, class) measured block the runner resets the engine instrumentation at an
idle boundary after the discarded warmups and snapshots it after the last measured
repetition. The candidate block artifacts retain `selected_for_gpu1`,
`executed_on_gpu1`, `explicit_failure`, `fallback_elsewhere`, dispatch/
reconstruction/reduction counts, remote prefill dispatches, startup vs steady-state
expert-weight H2D bytes, activation/routing and route-contribution traffic, and the
complete MoE-layer timing (`complete_layer`, GPU0 branch, remote dispatch/control,
GPU1 branch, join/reconstruct/reduce) from the shared timing schema. Baseline blocks
retain cache service, weight fetch, local expert execution, and complete-layer wall.
Complete-layer wall is measured independently, never summed from overlapping
components; absolute timestamps from independent GPU clocks are never combined.

### Instrumentation performance-compatibility audit

Surfaces used by canonical performance runs and their classification:

| Surface | Classification |
|---|---|
| SSE streaming client with arrival timestamps (`measure_generation`) | compatible (it IS the measurement) |
| `FREETOKEN_INSTRUMENT_PREFILL=1` CUDA-event prefill timing (both arms) | compatible (Phase-0 contract) |
| `OffloadMoeCache` miss/counters via idle-boundary snapshot | compatible (graph-safe) |
| InferSwarm ownership counters + transport byte counters | compatible (execution-boundary counters) |
| `--moe-layer-timing-max-steps` marker-kernel timing ring (both arms) | compatible (device markers, bounded ring, no per-step host sync) |
| `--inferswarm-correctness-diagnostics` C3 full-logit recorder | **incompatible** — forbidden in every campaign arm; correctness is a separate preflight run |
| serialized `--inferswarm-remote-mode` control | **diagnostic only** — not part of the canonical candidate arm |

### Dev-smoke distinction and the no-early-stopping rule

`--dev-smoke` is the only way to alter repetitions, warmups, or class selection.
Every alteration is recorded as a deviation, forces `canonical=false`, and the
artifacts are labelled `NON_CANONICAL_DEV_SMOKE` — unusable by the canonical
analysis. Without `--dev-smoke`, protocol changes are rejected.

While a campaign is executing, the runner computes no candidate/baseline ratio, no
aggregate statistic, and no significance test. Per-arm descriptive values (min/
median/max/IQR/CV, TTFT/prefill/token-latency percentiles) are computed only after a
block completes and can never affect whether later blocks execute. The runner's
complete output vocabulary is `COMPLETE` / `INCOMPLETE` / `VALID` / `INVALID` /
`NON_CANONICAL_DEV_SMOKE`; the performance-decision vocabulary belongs to P6.
