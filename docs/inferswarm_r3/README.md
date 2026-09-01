# InferSwarm R3 minimum automatic planning

This directory is the canonical retained campaign for InferSwarm issue #55. It
was regenerated only after the implementation tree was clean and committed.

- Accepted R3 base: `2fc64ae7c79bdc494a52468da329ddafd0adb8ba`
- Exact implementation commit: `f2ea03738a0162f1f26c57a90e548e2d22119a3b`
- Model: `nvidia/Qwen3.6-35B-A3B-NVFP4`
- Revision: `491c2f1ea524c639598bf8fa787a93fed5a6fbce`
- Disposition: `R3_MINIMUM_AUTOMATIC_PLANNING_PASS`

## Architectural disposition

R3 proves the minimum research-internal strategy/planner seam. The generic
planner knows only generic resource, capability, lifecycle, policy, evidence,
objective, ranking, and tie-breaking concepts. The Qwen adapter owns model and
backend semantics, exposes S0 and S1 as legal shapes, and compiles the planner's
frozen selection into the existing supported offload or R2/#53 realization path.
The decision and all input digests exist before heavyweight materialization.
Resource, evidence, policy, objective, implementation, or R2-plan identity drift
fails closed rather than repairing a selection.

Technical feasibility, integrity eligibility, operator policy eligibility,
evidence applicability, and ranking are separate planner axes. A quarantined
resource remains technically evaluated but is classified `INTEGRITY_EXCLUDED`;
it is never relabeled as operator-excluded when policy still allows it. Evidence
unit and statistic must match the objective, and each candidate's evidence audit
is self-describing with class, freshness, confidence, and metric metadata.

The standalone `synthetic-planner-proof.json` uses the same generic planner with
arbitrary `violet-shape`, `amber-slot`, and `copper-unit` names. It is CPU-testable
and passes; no second simplified planner is involved.

## Candidate and objective behavior

Scenario A maximizes applicable measured median warm decode throughput. It selects
S0 on GPU-A (`73.61368231952689 tok/s`). The accepted GPU-A to GPU-B S1 mapping is
technically feasible and ranked second (`67.01495086043339 tok/s`), not unsupported.
GPU-B is available but unused because the highest-ranked candidate does not need it.
S1 has exactly one legal R3 mapping: slot A on GPU-A and slot B on GPU-B, matching
the physically proven R2 implementation. S0 still enumerates both GPU-A and GPU-B;
S0 on GPU-B remains the principal technically feasible unranked real alternative.

Scenario B minimizes applicable measured median W4 warm-request TTFT. It selects S1
on GPU-A to GPU-B (`466.800324 ms`). S0 remains technically feasible and ranked
second (`3504.4774749985663 ms`).

Scenario C leaves the physical graph unchanged but excludes GPU-B by hard operator
policy. GPU-B-dependent S1 remains technically feasible and is classified
`POLICY_EXCLUDED`; allowed S0 on GPU-A is selected. Unmeasured otherwise feasible
resource permutations remain `FEASIBLE_UNRANKED`. If no applicable evidence exists,
the planner abstains explicitly rather than inventing a score.

## Physical execution

Both planner-selected shapes were freshly executed from the implementation commit
above through the decision/compile seam for W2 and W4. S0 used GPU-A
`GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099`; S1 used that GPU-A for slot A and
GPU-B `GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55` for slot B. Live identities,
capacity, driver, and the S1 R2-plan identities were validated before materialization.
For S1, the loaded plan was also validated with the existing R2 machinery; its
recomputed canonical digest matched both its own frozen digest and the compiled
digest, while `opaque-slot-a`/`opaque-slot-b` matched
`exec.block-a`/`exec.block-b` on GPU-A/GPU-B. Compiled input digests also matched
the frozen decision inputs exactly.

For both shapes, W2 and W4 generated the exact frozen token sequences and all
selected logits were byte-exact against the frozen R2-v2 corrected reference:
maximum absolute deviation `0.0`, maximum relative deviation `0.0`, NaN `0`, and
Inf `0`. S0 correctly retained its ordinary required host source backing and
recorded source fetch activity rather than being held to resident-only semantics.

S1 passed startup reconciliation and retained exact state-ownership records and the
accepted boundary. Both participants were backend-native and resident-only. Each
reported unexplained persistent host model mirror bytes `0`, steady-state model-state
movement bytes `0`, post-finalization host source fetches `0`, resident source
accesses `0`, fallbacks `0`, and graph recaptures `0`. The #53 RELEASE lifecycle was
physically applied: routed source staging counted at realization peak, then
`8,636,596,224` and `9,545,711,616` bytes were released, leaving current source
staging `0`. RETAIN is never credited as live-evictable capacity, while ordinary S0
source backing remains persistent.

## Frozen digest identities

- Resource snapshot: `sha256:188a619d7cc2363ac7217d037bdfdc3a2ee8b3868dcfc32b3f8d536db7119946`
- Strategy problem: `sha256:193ba70caffa4ec42e58ab9e62274e460a8e62668c305fb94a776392cd8f9c02`
- Evidence catalog: `sha256:ab504a3fdca223b79645fbf80be17f826811933d12a3d82460d78fda5a3d0c92`
- Policies A/B/C: `sha256:f79a770d1124ae5d69fc4aa32bb305f775ef77c8fdb5b85d74a700c487dc53f2`, `sha256:4b17c1541df0160ab0232d69ed83f040be0c8cc7d9effb8e2ec0d68cc3b94759`, `sha256:49144f9050aa017b1f7fc2feab72b153acd998b2cf7b9fc4f5485bd29217be4e`
- Objectives A/B/C: `sha256:4c049d14be818479c2660cc01d0d7f1907ac868e48b5e6c03c70aafb68641a27`, `sha256:b349b411f9f93c4a6d2fbe21c108eb766ed751a86448965b22cf66ec279ce5b5`, `sha256:4c049d14be818479c2660cc01d0d7f1907ac868e48b5e6c03c70aafb68641a27`
- Decisions A/B/C: `sha256:9698ac95fb70cd54f843302ef8b9da6d4e57bae58d2005221ae506182d790218`, `sha256:3b1da297de13bd55f2252e18c231eb6eac4cc3ee61cfd9e712260db95d8c433d`, `sha256:eded345c2244da16648665d51b7dd0229fe835613373858182f3ca177b47cad7`
- Compiled plans A/B: `sha256:717ec95c76162c200e6927a2cc607e13de628c43ab0c5a62e2cf79ebac25374d`, `sha256:c6648b5118bc8ebc8a0e5ce42f2b3b3bc32d2c837ff1a98b21a5d686b161226a`
- Physical file SHA-256 A/B: `9a089277a4b4f87cc710b080cc9c10509504ba603a48509fa5a12ba9789c089f`, `0ccc527a84f2847b20f32dd01e320297bf45878eebe601b78e355b8a0d5bf9c1`

## Tests and environmental limitation

The retained focused/predecessor command passed `186` tests. The pinned-memory file
passed `7` tests with exactly one deselection, for `193 passed, 0 failed, 1
deselected` overall. The deselected node is
`tests/kernels/test_pinned_tensor.py::test_host_device_ptr_is_identity_under_uva`.
Run separately on Linux 6.12.105, Python 3.13.5, PyTorch 2.11.0+cu130/CUDA 13.0,
and NVIDIA driver 610.57.04, it fails because `cudaHostGetDevicePointer` rejects an
unregistered pageable pointer with `invalid argument` despite reported UVA identity.
This is outside R3 semantics: both R3 paths use registered mapped host memory, no R3
or required predecessor test is excluded, and both physical paths pass. The test was
not weakened, hidden globally, modified, or deleted.

Ruff is not installed in the accepted test environment. Changed paths pass Python
byte-compilation and `git diff --check`; no repository-wide clean-suite or lint claim
is made beyond the retained commands.

## Proof boundary

R3 does not prove production planner quality, dynamic replanning, adaptive demand
learning, multi-node execution, R4 network viability, plan epochs or elasticity,
stable public planner/strategy APIs, generalized performance prediction,
multi-vendor execution, or transparent cache eviction/rematerialization. It does not
begin R4. Human evidence and mergeworthiness review is the next gate.

Regenerate frozen inputs/decisions with:

```bash
PYTHONPATH=python:benchmarks /home/zutfen/FreeToken-r2/.venv/bin/python -m benchmarks.inferswarm_r3.build_artifacts
```

Physical execution uses `benchmarks.inferswarm_r3.run_selected`; final composition
uses `benchmarks.inferswarm_r3.compose_result`. Every retained JSON and this README
has a sibling SHA-256 sidecar.
