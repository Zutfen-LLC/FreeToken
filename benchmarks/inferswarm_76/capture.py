"""#76 capture arming: full 15-envelope checkpoint set from the R6 runtime.

Arms the accepted #71 ``CaptureSink`` seam (``runtime._capture_sink``) and
adds the three checkpoints the frozen checkpoint-family map requires that the
R6/#71 seams do not emit:

- ``layer0_o_proj_input``  (input rows of layer-0 attention o_proj GEMM)
- ``layer0_o_proj_output``
- ``layer15_attn_o_proj_output`` (representative global-attention projection)
- ``full_final_row_bf16_logits`` (complete final-row BF16 logits; the #71
  seam emits the full [seq, vocab] matrix which the harness reduces to the
  final row OFF-device, on host, without touching device math)

All wrappers are instance-level (bound-method rebind), installed by the
harness process only, and add zero device-side math: each wrapper calls the
original method, then hands the EXACT native tensor to the sink for a host
copy. Un-instrumented execution is unchanged (wrappers absent).

Checkpoint identity vs. the stage's own layer numbering: the wrappers key off
GLOBAL layer ids (``block.global_layer_ids``), so the same wrapper works for
the single arm (stage owns [0,48)) and chain stages owning [0,16)/[16,32).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def arm_full_capture(runtime, sink) -> None:
    """Install the #76 checkpoint wrappers on one runtime instance."""
    import torch  # noqa: F401  (presence check; wrappers run under it)

    block = runtime.block
    global_ids = list(block.global_layer_ids)

    # --- layer-0 attention o_proj in/out (layer 0 must be owned) ----------
    if 0 in global_ids:
        local0 = global_ids.index(0)
        attn0 = block.layers[local0].self_attn
        o_proj = attn0.o_proj
        orig_forward = o_proj.forward

        def o_proj_forward(x, *args, **kwargs):
            runtime._emit(
                "layer0_o_proj_input",
                x,
                global_layer=0,
                extra={"op": "o_proj_gemm_input"},
            )
            out = orig_forward(x, *args, **kwargs)
            runtime._emit(
                "layer0_o_proj_output",
                out,
                global_layer=0,
                extra={"op": "o_proj_gemm_output"},
            )
            return out

        o_proj.forward = o_proj_forward

    # --- representative global-attention o_proj output (layer 15) ---------
    if 15 in global_ids:
        local15 = global_ids.index(15)
        attn15 = block.layers[local15].self_attn
        o_proj15 = attn15.o_proj
        orig15 = o_proj15.forward

        def o_proj15_forward(x, *args, **kwargs):
            out = orig15(x, *args, **kwargs)
            runtime._emit(
                "layer15_attn_o_proj_output",
                out,
                global_layer=15,
                extra={"op": "global_attention_o_proj_output"},
            )
            return out

        o_proj15.forward = o_proj15_forward

    # --- full final-row BF16 logits (host-side row slice) ------------------
    # The runtime emits the full [seq, vocab] BF16 matrix at "bf16_logits".
    # The harness post-processes that record to the final row on host; no
    # device math is added here.
    runtime._capture_sink = sink


def final_row_from_bf16_record(record_tensor):
    """Return the exact final row of a captured full BF16 logits matrix."""
    if record_tensor.dim() != 2:
        raise ValueError("bf16_logits capture must be a 2-D [seq, vocab] tensor")
    return record_tensor[record_tensor.shape[0] - 1]


def reduce_capture_records(
    records: list[dict[str, Any]],
    *,
    mapping: dict[str, str],
) -> dict[tuple[int, str], Any]:
    """Group sink records by (capture_position, frozen_checkpoint_id).

    Returns raw host tensors keyed for the reducer. Records whose seam name
    is not in the mapping are ignored (diagnostic-only seams).
    """
    out: dict[tuple[int, str], Any] = {}
    for record in records:
        meta = record["meta"]
        name = meta.get("checkpoint")
        if name not in mapping:
            continue
        checkpoint_id = mapping[name]
        position = int(meta["step"])
        out[(position, checkpoint_id)] = record["tensor"]
    return out


class RowPruningSink:
    """#76 per-case capture sink (#71 CaptureSink-compatible surface).

    Differences from the R6/#71 sink, all host-side only:

    - ``bf16_logits`` (full [seq, vocab] BF16 matrix) is reduced to its
      FINAL ROW immediately after the host copy; the retained artifact is
      the exact frozen ``full-final-row-bf16-logits`` checkpoint domain.
    - tensors are NOT kept after ``save``; only hashes/metadata plus the
      pruning-carried final row flow into the persisted bundle.
    """

    def __init__(self, *, role: str, gpu_uuid: str | None = None):
        import socket
        import time as _time

        self.role = role
        self.host = socket.gethostname()
        self.gpu_uuid = gpu_uuid
        self.records: list[dict[str, Any]] = []
        self._now = _time.time

    def emit(
        self,
        *,
        checkpoint: str,
        step: int | None,
        global_layer: int | None,
        position_range: list[int] | None,
        source_device: str,
        tensor,
        extra: dict[str, Any] | None = None,
    ) -> None:
        import hashlib

        import torch

        host_copy = tensor.detach().cpu()
        if checkpoint == "bf16_logits":
            host_copy = final_row_from_bf16_record(host_copy)
            checkpoint = "full_final_row_bf16_logits"
        raw = host_copy.detach().contiguous()
        meta = {
            "schema": "inferswarm.issue76.row-pruned-capture/1",
            "checkpoint": checkpoint,
            "step": step,
            "global_layer": global_layer,
            "position_range": position_range,
            "source_device": source_device,
            "role": self.role,
            "host": self.host,
            "gpu_uuid": self.gpu_uuid,
            "captured_at": self._now(),
            "shape": list(raw.shape),
            "dtype": str(raw.dtype).replace("torch.", ""),
            "byte_count": raw.numel() * raw.element_size(),
            "sha256": hashlib.sha256(
                raw.view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
            "nan_count": (
                int(torch.isnan(raw).sum().item())
                if raw.dtype.is_floating_point
                else 0
            ),
            "inf_count": (
                int(torch.isinf(raw).sum().item())
                if raw.dtype.is_floating_point
                else 0
            ),
        }
        if extra:
            meta["extra"] = extra
        self.records.append({"meta": meta, "tensor": raw})

    def save(self, out_dir: str | Path, tag: str) -> dict[str, Any]:
        import torch

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        metas = [r["meta"] for r in self.records]
        tensors = [r["tensor"] for r in self.records]
        torch.save(
            {"records": metas, "tensors": tensors},
            out / f"capture-{tag}.pt",
        )
        self.records = []
        return {
            "schema": "inferswarm.issue76.capture-manifest/1",
            "out_dir": str(out),
            "tag": tag,
            "record_count": len(metas),
            "record_sha256": [m["sha256"] for m in metas],
        }
