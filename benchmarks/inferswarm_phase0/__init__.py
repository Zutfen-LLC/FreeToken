"""InferSwarm Phase-0 baseline harness (FreeToken side).

Canonical issue: https://github.com/Zutfen-LLC/inferswarm/issues/2
Criteria:        inferswarm/docs/phase1-poc-success-criteria.md (sections 1.1, 2, 3, 9, 10, 13)

This package is *instrumentation*, not a performance path. It drives the real FreeToken
serving path (``ft serve`` + streamed ``/v1/chat/completions``) exactly as
``benchmarks/bench_decode_moe.py`` does, and adds what a reproducible baseline campaign
needs on top: the explicit B1-B5 configuration matrix, a frozen workload manifest, the
precommitted warmup/repetition protocol, full provenance capture, and raw per-repetition
artifacts.

It deliberately does NOT:

* decide a winner -- ``CANONICAL_PERFORMANCE_BASELINE`` is selected by a human reading the
  completed campaign (criteria section 2.2), and computing ratios mid-campaign is
  prohibited (section 10, "no early stopping");
* invent workload content -- InferSwarm issue #3 supplies the W1/W3/W4 fixtures;
* write anything into ``inferswarm/docs/benchmarks/results/``.

Nothing here is torch-dependent: the harness talks to a server over HTTP, so its tests run
in CPU-only CI. The one GPU-touching module (``expert_microbench``) imports torch lazily.
"""

from __future__ import annotations

# The checkpoint family fixed by the Phase-0/1 experiment.  The revision remains an
# independently supplied, exact upstream commit SHA; only the repository is fixed here.
CANONICAL_MODEL_REPOSITORY = "nvidia/Qwen3.6-35B-A3B-NVFP4"

# Bumped whenever the harness changes in a way that could move a number. Recorded in every
# run artifact so two result directories can be told apart by more than their date.
HARNESS_VERSION = "0.3.0"

# The artifact schema the runner writes. A consumer must refuse an unknown major version
# rather than guess at field meanings.
# /2: `status` split into `execution_status` (did every generation return?) and
# `validity` (is this a valid canonical campaign?), plus `campaign_invalidations`.
RUN_SCHEMA = "inferswarm.phase0.run/2"
REPETITION_SCHEMA = "inferswarm.phase0.repetition/1"
