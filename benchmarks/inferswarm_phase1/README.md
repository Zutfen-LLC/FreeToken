# InferSwarm Phase-1 device probe

This namespace contains the Phase-1 **P1 device-discovery diagnostic only**. It does not
allocate the frozen expert bank, execute remote experts, benchmark throughput, or change
model outputs.

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
