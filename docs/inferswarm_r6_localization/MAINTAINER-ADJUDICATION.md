# InferSwarm #71 — Maintainer Adjudication

Status: **ACCEPTED LOCALIZATION EVIDENCE**

Issue #71 successfully localizes the additional single-vs-distributed (`D−S`) numerical residual to a **backend-execution-local, device-dependent BF16 GEMM difference**.

Accepted evidence:

- the single-GPU and distributed arms are byte-identical at the embedding output;
- the first divergent semantic checkpoint is after global layer 0;
- bisection localizes the first divergent operation inside layer 0 to the attention output projection (`o_proj`) BF16 GEMM;
- the input to that GEMM is byte-identical across devices, while the output is not;
- the same distributed stage code fed the same input on the RTX 3090 reproduces the single-GPU arm byte-exactly, and on the RTX 3060 reproduces the distributed stage-1 arm byte-exactly;
- both distributed boundaries preserve exact bytes sender→receiver, excluding transport corruption;
- the audited stage-local Gemma configuration is equivalent to the single-GPU configuration across all 48 layers;
- the generic planner and external-Coordinator seams are not implicated.

Accepted classification:

`BACKEND_EXECUTION_LOCAL`

## Causal precision

The retained experiment **proves device-dependent BF16 GEMM numerics** at `o_proj` under otherwise matched inputs, weights, semantic configuration, model representation, and execution code.

The evidence is consistent with a different backend GEMM algorithm / tiling / accumulation order being selected on RTX 3060 versus RTX 3090 hardware, and such a mechanism is the leading explanation for the observed 1–2-ULP differences. However, the campaign did **not** retain a cuBLAS algorithm identifier or equivalent kernel-selection record that directly proves the specific causal chain “different SM count → different cuBLAS kernel/tile selection.” That mechanism therefore remains an explanatory hypothesis, not a separately proven sub-finding.

This distinction does not weaken the localization result: the first differing operator and its device dependence are physically established.

## Architectural disposition

This result does **not** support imposing same-GPU-model placement as InferSwarm's general correctness rule. InferSwarm is explicitly a heterogeneous fabric; ordinary correctness must be able to tolerate legitimate backend floating-point variation across qualified Compute Units.

A homogeneous-device / bit-exact execution policy may remain a valid optional constraint where an operator specifically requires bit identity.

The next work should instead define a **prospective heterogeneous numerical-equivalence contract** that keeps exact byte identity for transport/state-transfer invariants while defining evidence-backed numerical correctness for backend execution across qualified heterogeneous devices. No such future tolerance may be retrofitted onto historical R6, whose frozen `<0.25` comparator and FAIL verdict remain immutable.

## Historical status

- Historical R6 remains `R6_DENSE_ARCHITECTURE_FALSIFICATION_FAIL`.
- Issue #65 remains open.
- No historical threshold/reference is changed.
- No R6 PASS is implied by accepting this localization evidence.
