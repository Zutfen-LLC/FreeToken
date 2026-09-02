# InferSwarm R5A harness

This package supports the frozen methodology in
`docs/inferswarm_r5a/METHODOLOGY.md`.

- `preflight.py` freezes both Nodes, checkpoints, runtime, network, memory, and
  derives the current participant plan before realization.
- `strategy.py` owns Qwen-specific legal candidates and compilation.
- `runtime.py` realizes the selected two-Node candidate through the accepted R4
  backend-native primitive.
- `http_campaign.py` only drives ordinary FreeToken HTTP requests.
- `compose.py` applies the retained R2/R4 comparator and composes network-arm
  evidence.
- `compose_local.py` composes the ordinary local HTTP control and normalized
  ranking evidence.

These are research interfaces, not final public InferSwarm APIs.
