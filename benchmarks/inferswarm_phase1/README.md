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
