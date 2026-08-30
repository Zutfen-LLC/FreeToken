"""InferSwarm Phase-1 diagnostics and the P5 canonical campaign runner.

The ``campaign_*`` modules build the canonical two-session A/B campaign on top of the
Phase-0 benchmark contract implementation in ``benchmarks/inferswarm_phase0``. The
runner collects raw observations and per-arm descriptive statistics only; it computes
no cross-arm ratio and emits no campaign verdict — that analysis belongs to P6.
"""

from __future__ import annotations

# Bumped whenever the campaign runner changes in a way that could move a number or an
# artifact shape. Recorded in every plan/provenance artifact.
# 0.2.0: session-aware baseline-identity gate semantics; prerequisites bound to the
# exact current clean FreeToken HEAD with SHA validation/rehashing; the conditional
# KV-matched supplementary arm predeclared with its trigger and pinned capacity.
# 0.3.0: deterministic campaign identity (SHA-256 over canonical-JSON components)
# recorded by session 1 and required exactly by the session-2 gate, with the expected
# session-1 artifact set verified; every pre-warmup runtime-contract failure now stops
# measurement (candidate contract findings abort the arm before the first warmup,
# baseline InferSwarm leakage is a B1 identity failure, engine-GPU mismatch stops the
# arm); the resolved baseline expert-cache slot count is exact provenance with no
# numeric validity band — KV-capacity differences belong to the supplementary-KV rule.
# 0.4.0: instrumentation-control fix — the frozen reset/snapshot operation budget is
# 300 s (MEASURED canonical B1/W1 snapshot = 136.93 s for ~204,800 retained timing
# records; the frozen 60 s budget returned HTTP 504 after a completed block), recorded
# in every plan/provenance artifact; the expected HTTP control statuses (409/422/503/
# 504) are decoded and converted to a fail-closed ServerError instead of a raw urllib
# exception crashing the session; a failed reset/snapshot stops the session with every
# already-collected generation preserved and every remaining planned generation
# preserved as not-executed evidence. Control-plane only: no timing truncation, no
# runtime/mechanism change, no performance threshold moved.
CAMPAIGN_RUNNER_VERSION = "0.4.0"
