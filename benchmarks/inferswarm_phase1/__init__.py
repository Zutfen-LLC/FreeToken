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
CAMPAIGN_RUNNER_VERSION = "0.2.0"
