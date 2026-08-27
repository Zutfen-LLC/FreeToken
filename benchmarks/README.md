# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.

---

## InferSwarm Phase-0 baseline harness

**`phase0_baseline.py`** (package: `inferswarm_phase0/`) — the reproducible Phase-0
baseline campaign for [InferSwarm issue #2](https://github.com/Zutfen-LLC/inferswarm/issues/2).
It drives the same real serving path as `bench_decode_moe.py` and adds what a *baseline*
needs on top: the explicit B1–B5 configuration matrix, a frozen workload manifest, the
precommitted warmup/repetition protocol, full provenance capture, and raw per-repetition
artifacts.

The rules it implements are fixed in advance by InferSwarm's
[Phase-1 success criteria](https://github.com/Zutfen-LLC/inferswarm/blob/main/docs/phase1-poc-success-criteria.md)
(sections 1.1, 2, 3, 9, 10, 13) and its
[benchmark contract](https://github.com/Zutfen-LLC/inferswarm/blob/main/BENCHMARKING.md).
Those documents are canonical and are **not** duplicated here — this section is usage only.

### What it measures

Per generation, through `/v1/chat/completions` with streaming: TTFT, warm decode tokens/sec,
the full raw inter-token gap list (so p50/p95/max, variance, CV and a later bootstrap all
remain computable), prompt and completion token counts, a VRAM observation, and the output
hash (with full text for the correctness reference).

**Prefill throughput is measured, not derived from TTFT.** TTFT contains transport,
tokenization, chat-template rendering, queueing, sampling, detokenization and the SSE hop;
`prompt_tokens / TTFT` would attribute all of that to prefill. Instead the server is started
with `FREETOKEN_INSTRUMENT_PREFILL=1`, which brackets *the prefill model forward* with CUDA
events on the engine stream (`engine.forward_batch`) and sums them per request across
chunked prefill (`Scheduler._accumulate_prefill`). The harness reads the result from
`GET /v1/instrumentation` and refuses any record that could belong to an earlier
generation. When instrumentation is off, the field is an explicit null with a reason — never
a TTFT-derived substitute.

The prefill record is attributed to *this* request by uid where possible: the response id is
`chatcmpl-<uid>` and every instrumentation record is stamped with the same uid, so the match
is request identity rather than "newest record above a sequence floor". Where the id shape is
unrecognized the sequence rule still applies, but more than one candidate is reported as
**ambiguous** rather than resolved by taking the newest — which would attribute another
request's interval to this one.

`GET /v1/instrumentation` also serves the engine's **resolved** configuration: which MoE
backend `auto` actually selected, whether `--nvfp4-backend` was inert for the executing
expert path, whether `_auto_cpu_layers` locked any layers, the resolved cache slots and
bytes, the resolved hybrid fetch fraction, the resolved `--max-prefill-length` and
`--cache-type`, and whether the Marlin 992-slot cap applies and whether it bound. These are
read back off the running engine rather than re-derived, so the record cannot disagree with
what executed. A canonical run requires the fields criteria section 2.3 lists to be present
and non-null; a hole where a resolved value should be invalidates the campaign.

### Running the sweep

```bash
python benchmarks/phase0_baseline.py sweep \
    --model /path/to/nvidia--Qwen3.6-35B-A3B-NVFP4 \
    --model-repository nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --model-revision <exact 40-hex upstream commit SHA> \
    --gpu GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
    --manifest /path/to/frozen-workloads.json \
    --inferswarm-commit <inferswarm commit> \
    --session-id session-1 \
    --out-root phase0-runs
```

- **Pinned model revision.** `--model-revision` must be the exact upstream commit SHA.
  Branch names, tags and `main` are refused for a canonical run: no Phase-0 measurement may
  begin until the revision is recorded, and inventing one is a fabricated provenance record.
- **Fixed model and loaded format.** Canonical sweep and correctness-reference runs require
  the exact repository `nvidia/Qwen3.6-35B-A3B-NVFP4`; another repository requires
  `--dev-smoke`. Every live arm must additionally report `model.expert_quant = nvfp4` from
  the engine. Agreement across arms is not sufficient if they all loaded another format.
- **Physical GPU by stable UUID.** `--gpu` is **required** for a canonical run and takes
  what `ft serve --gpu` takes. Prefer the UUID from `nvidia-smi -L`: indices move between
  boots, and the criteria require the sweep and the later candidate to use the *same
  physical card*. An index is accepted but is immediately resolved to its UUID through
  `freetoken.gpu_select` (the same selector `ft serve` uses) and **both** are recorded; the
  resolved UUID is then what every child process is given — the servers, `ft bench bw` and
  the microbenchmarks — so they cannot silently disagree. After each server is ready the
  UUID the *engine* reports for itself (`/v1/stats` `gpus`, filled from its own bound
  device) is compared with the resolved one; a mismatch invalidates the campaign.
  The selected `nvidia-smi` row is also checked before measurement, and the matching live
  engine identity is checked again, for the exact model `NVIDIA GeForce RTX 3060` and an
  inclusive total-memory range of 11--13 GiB. The bounded range tolerates normal reporting
  variance around a 12 GB card while deliberately excluding 8, 16, and 24 GB devices;
  adjacent names such as RTX 3060 Ti are rejected.
- **A fresh `ft bench bw` profile is a session-level prerequisite**, not a B2 side effect.
  It runs **once, before the sweep traversal**, in either ordering direction, because two
  arms read it: B2 resolves its per-decode-step fetch split from it, and B3's
  `--moe-backend auto` reads the same profile to decide whether to upgrade offload to
  hybrid. In a reversed session B3 runs first, so a B2-local refresh would let it consume a
  stale profile. Canonical Phase 0 is locked to `--bench-bw-dtype nvfp4`; alternate formats
  are developer-smoke only. `--no-bench-bw` is **refused** for a canonical sweep. A failed
  command, an unreadable profile, a profile without a positively matching GPU UUID, or an
  incomplete NVFP4 CPU-MoE/PCIe calibration aborts the campaign *before any server starts*.
  The captured file is checked through the same backend-recommendation and hybrid-fraction
  readers the engine uses. After B2 starts, a zero resolved hybrid fraction is invalidating:
  it is the fixed-cap fallback, not evidence that the calibration was consumed. The artifact
  records the command, requested dtype, both timestamps, return code, selected GPU UUID,
  resolved profile path, profile contents, reader results and sha256.
- **A clean FreeToken checkout.** A canonical run refuses to start from a dirty working
  tree and names the modified paths: a dirty tree cannot be reproduced from its commit SHA,
  and recording only the filenames does not make it reproducible.
- **`--inferswarm-commit` must be a full 40-hex SHA** for a canonical run, for the same
  reason `--model-revision` must be. If the checkpoint path is a Hugging Face
  `.../snapshots/<sha>` layout, that SHA and the `models--<org>--<name>` cache entry are
  cross-checked against `--model-revision` / `--model-repository` and a disagreement is a
  refusal; a non-snapshot path records "cannot cross-check" instead of guessing.
- **Frozen workload manifest.** `--manifest` points at a version-controlled JSON file
  pinning, per class, the fixture (or inline content), its sha256, the output-token count,
  the sampling parameters, `ignore_eos`, and the chat-template settings. Schema:
  `inferswarm_phase0/workload-manifest.schema.json`; the authoritative validator is
  `inferswarm_phase0/manifest.py`, and it runs before a server is started.
  `python benchmarks/phase0_baseline.py hash <fixture>` prints the digest to freeze.
  The W1/W3/W4 fixtures come from
  [InferSwarm issue #3](https://github.com/Zutfen-LLC/inferswarm/issues/3); the example
  manifest in `inferswarm_phase0/examples/` is a smoke-test fixture and declares
  `canonical: false`, so a canonical run refuses it.
- **Sessions, warmups, repetitions.** Per (arm, workload class): 2 discarded warmups then
  10 measured generations. `--session-id` distinguishes campaign sessions so the whole
  campaign can be repeated on a different day and thermal state; `--reverse-order` gives
  that second session the reversed traversal. `--warmups` / `--repetitions` exist for
  developer smoke tests only, require `--dev-smoke`, and stamp the run NON-CANONICAL
  everywhere it is recorded.
- **`--dry-run`** prints the whole plan — every arm's exact `ft serve` command line, the
  bandwidth-profile prerequisite and which arms consume it, the execution order, and an
  unambiguous CANONICAL / NON-CANONICAL banner — without touching a GPU.

### Correctness reference

`CORRECTNESS_REFERENCE` is a **separate subcommand**, because it answers a different
question and must never be conflated with the performance baseline: it is fixed in advance,
never chosen by speed, and no ratio is ever computed against it.

```bash
python benchmarks/phase0_baseline.py reference \
    --model ... --model-revision ... --gpu ... --manifest ... \
    --nvfp4-backend triton \      # the RESOLVED backend the candidate's expert GEMM uses
    --moe-cache-size 992          # fixed; >= num_experts, and <= 992 under marlin
```

It runs `--moe-backend offload --moe-cpu-layers 0 --sampling-defaults none` and always
stores full output text. Its request sampling is **forced greedy**
(`temperature 0.0 / top_p 1.0 / top_k -1`) on the same frozen prompt fixture the sweep uses,
and the override is recorded per repetition alongside the manifest's own values.
`--sampling-defaults none` is not sufficient on its own: the manifest states sampling
explicitly in every request body and a request-level value beats a server default, so a
manifest whose frozen *performance* sampling is deliberately realistic would otherwise make
the correctness reference sampled — and FreeToken exposes no seed to make that reproducible.
The performance sweep keeps the manifest's frozen sampling untouched. Its self-consistency check is two
independent runs with different `--session-id`s, compared on the recorded `output_sha256`
per class. Note that identical HTTP text hashes are **not** the deeper C1/C2/C3 Phase-1
correctness instrumentation (per-layer outputs, router selections, step-0 logits); they are
the reproducible fixture those gates will later be applied to.

### Hardware profile

```bash
nvidia-smi -L

PYTHONPATH=python:. python benchmarks/phase0_baseline.py profile \
    --gpu GPU-<UUID> \
    --dtype nvfp4 \
    --device-bandwidth \
    --expert-microbench \
    --out phase0-runs/hardware-profile.json
```

`phase0-runs/` is repository-ignored, so this hardware profile and the sweep's default
artifacts do not make a clean checkout dirty between canonical sessions. Replace
`GPU-<UUID>` with the value reported by `nvidia-smi -L`; do not guess it.

Captures GPU/VRAM identity, driver, compute capability, PCIe link generation and width
(current *and* max), `nvidia-smi topo -m`, CPU/RAM/OS, and the bandwidth measurements from
`ft bench bw` (STREAM-style CPU DRAM, pinned↔device copy, the real CPU MoE GEMV and the real
`OffloadMoeCache.copy_missing` PCIe gather).

- **`--device-bandwidth`** measures the card's own **VRAM** bandwidth, which is what issue
  #2's "memory bandwidth" asks for and which `ft bench bw` does *not* measure — its ceilings
  are host DRAM and the PCIe link. A device-resident buffer is copied device-to-device with
  a working set far beyond L2 (512 MiB per buffer by default), timed with CUDA events after
  warmup; every repetition is reported individually, and the byte accounting is stated both
  ways (read+write and read-only) so neither has to be reverse-engineered.
- **`--expert-microbench`** adds **true single-expert** NVFP4 decode-GEMV latency at
  `top_k = 1` — one routed expert per call, timed directly. It optionally also reports the
  grouped `top_k` routed-expert step, clearly named as a grouped/batched diagnostic and
  **never divided by `top_k`**: expert work inside a grouped call executes concurrently, so
  `step_ms / top_k` is an amortized throughput-like quantity, not a latency.

Both bind the process to the requested GPU through FreeToken's own `gpu_select` binding path
and record the UUID of the device they actually bound; a mismatch is refused rather than
mislabelled. Both are **diagnostic only**: per the benchmark contract a microbenchmark never
constitutes evidence about end-to-end inference and is never combined into one.

### Artifacts

```
phase0-runs/<YYYY-MM-DD>-<session-id>-<short-name>/
    run.json            verdict, provenance, configuration, protocol, resolved per-arm config
    repetitions.jsonl   ONE LINE PER GENERATION, warmups included and tagged
    failures.jsonl      every failed generation, with its reason
    SUMMARY.md          human summary; leads with the campaign verdict and its reasons
    server-logs/        one ft serve log per arm
```

Every measured repetition is preserved with its raw timings; no repetition is ever
discarded, and averages are never emitted in place of the data.

**Execution completeness and campaign validity are two different answers, and the artifact
keeps them apart.**

- `execution_status` — `COMPLETE` / `INCOMPLETE` — is computed from expected-vs-observed
  repetition counts and the failure list, so a successful-looking summary cannot hide a
  missing repetition.
- `validity` — `VALID` / `INVALID` / `NON_CANONICAL` — answers whether this is a valid
  canonical Phase-0 baseline campaign. A campaign that produced every repetition it planned
  is still `INVALID` when a precommitted requirement failed, and `campaign_invalidations`
  lists each one as a structured record with a stable reason code (missing or failed
  bandwidth refresh, unavailable instrumentation, a missing resolved-configuration field,
  an expert-quant or held-constant disagreement across arms, a B3 resolution outside B1/B2,
  a prompt outside its frozen class shape, a completion length that is not the requested
  one, a stale / missing / ambiguous / shared-batch / unusable prefill record, an unproven
  or mismatched physical GPU, …).
- `label` is the InferSwarm evidence label and describes an **observation**: repetitions
  that really happened are `MEASURED` whatever the campaign verdict is. That never promotes
  the campaign to a valid baseline.

The first line of both `run.json` and `SUMMARY.md` is one of `VALID CANONICAL CAMPAIGN`,
`INVALID CANONICAL ATTEMPT`, `NON-CANONICAL DEVELOPER RUN`, `INCOMPLETE RUN`, with the
reasons immediately below it — derived from the *overall* campaign state, not from the
repetition protocol alone (`--allow-missing-provenance`, for instance, leaves the protocol
untouched and still makes the campaign non-canonical).

The harness computes **no** cross-configuration ratio and selects no baseline:
`CANONICAL_PERFORMANCE_BASELINE` is chosen by a human from the completed campaign, and
computing ratios mid-campaign is prohibited.

### Where results live

**CI never produces hardware numbers.** The tests under `tests/benchmarks/` mock the
server, GPU and HTTP layer and run on CPU; nothing in this repository's CI measures
hardware, and nothing may.

The authoritative InferSwarm results live in the **InferSwarm repository**, under
`docs/benchmarks/results/YYYY-MM-DD-short-name/` (`result.json` + `SUMMARY.md`). A run
directory here is the raw local artifact those entries are derived from; copying one over
is a deliberate human step taken once the numbers have been read.
