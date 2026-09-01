# InferSwarm R3 minimum automatic planning

This directory is the canonical retained campaign for InferSwarm issue #55. It
was regenerated only after the implementation tree was clean and committed.

- Accepted R3 base: `2fc64ae7c79bdc494a52468da329ddafd0adb8ba`
- Exact implementation commit: `012a37d3bb7dde2e53a42aedadab06364dba8a9a`
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

The standalone `synthetic-planner-proof.json` uses the same generic planner with
arbitrary `violet-shape`, `amber-slot`, and `copper-unit` names. It is CPU-testable
and passes; no second simplified planner is involved.

## Candidate and objective behavior

Scenario A maximizes applicable measured median warm decode throughput. It selects
S0 on GPU-A (`73.61368231952689 tok/s`). The accepted GPU-A to GPU-B S1 mapping is
technically feasible and ranked second (`67.01495086043339 tok/s`), not unsupported.
GPU-B is available but unused because the highest-ranked candidate does not need it.

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

- Resource snapshot: `sha256:648e1098a72a4b714af6dc347cdcc97bffb467a331e5937f53cf02a5cbc56f58`
- Strategy problem: `sha256:00061ee5d11115b31df178f0ef013c36508b078688aea0be68fa4b4d99910231`
- Evidence catalog: `sha256:0401eef4f967cfa2f5314be8fa64e80dd1c211e7b61878f9200249fe39bd2ea7`
- Policies A/B/C: `sha256:3558ba98a260124c681549e0cb093bb5ed4d81e30ace179643c7e30a34a6153c`, `sha256:64e85fccab2b26aa7f989bd85870bb6cd4f49d7fe1d7afe57210583942aae2ca`, `sha256:c59e6acb44b65fcafc61f613339665544268af5eebf7268fd0aa3af319ce564b`
- Objectives A/B/C: `sha256:79162ce197bd379d20183c038693423e8da4a6d524f721bf1e114a448bd1c52d`, `sha256:d09950f44bafb3698b305b0c966f4028b538584aa1bee829cb6f4e3737bae0ce`, `sha256:79162ce197bd379d20183c038693423e8da4a6d524f721bf1e114a448bd1c52d`
- Decisions A/B/C: `sha256:bbc82f52ab26ccba14d943ed393cdb4c87f92e4b2442b69a8001f9c4108abd85`, `sha256:5726a3ce4b8993b9dbca7906ca2b8985eca7fdc4774a6f7805b24222335799ed`, `sha256:b3c6a6e4bf6536f0b351e42a027fa4a5947a8fe6c405bd31c623ffa528111296`
- Compiled plans A/B: `sha256:4718ceca0a59b0908d55a99cdf4007874ff36360fa37a82e76c564b0b8142ac4`, `sha256:b9a3fc1226f436b54eb5f3af5bcc23d8f7c5d522f3de9b2a083143145cf7a8bd`
- Physical file SHA-256 A/B: `a4d9c9c63f8cf1b3d9b9c2265c0d66c3a35058076b843f684adcd282f71d1036`, `d7b064d5af950602813bd8c33b0adc7b8a96f80c9500d8d9c57e0d8ef6dc243d`

## Tests and environmental limitation

The retained focused/predecessor command passed `178` tests. The pinned-memory file
passed `7` tests with exactly one deselection, for `185 passed, 0 failed, 1
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
