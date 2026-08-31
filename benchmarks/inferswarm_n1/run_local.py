from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import subprocess
import time
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha
from freetoken.research.n1_local_boundary import HEADER, MessageType

from .client import N1BlockClient


def _service_entry(argv: list[str]) -> None:
    from .service import main
    raise SystemExit(main(argv))


def _wait_ready(path: Path, process: multiprocessing.Process, timeout: float = 240.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text())
        if not process.is_alive():
            raise RuntimeError(f"Block B exited during startup with code {process.exitcode}")
        time.sleep(0.25)
    raise TimeoutError("Block B did not become ready")


def _device_record(uuid: str) -> dict:
    fields = "index,uuid,pci.bus_id,name,pcie.link.gen.current,pcie.link.width.current"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if values[1] == uuid:
            return dict(zip(fields.split(","), values, strict=True))
    raise RuntimeError(f"GPU UUID {uuid} disappeared")


def _prompt_ids(tokenizer, workload) -> list[int]:
    body = workload.greedy_reference_body("nvidia/Qwen3.6-35B-A3B-NVFP4")
    encoded = tokenizer.apply_chat_template(
        body["messages"], tokenize=True, add_generation_prompt=True,
        **body["chat_template_kwargs"],
    )
    return list(encoded["input_ids"])


def _run_session(
    *, runtime, client: N1BlockClient, session_id: int, prompt_ids: list[int],
    max_new_tokens: int, chunk_size: int,
) -> dict:
    client.open(session_id)
    generated: list[int] = []
    boundaries = []
    step = 0
    for start in range(0, len(prompt_ids), chunk_size):
        chunk = prompt_ids[start:start + chunk_size]
        hidden, residual = runtime.prefill_a(chunk, start=start)
        payload = runtime.boundary_payload(hidden, residual)
        final = start + len(chunk) == len(prompt_ids)
        token = client.hidden(
            session_id=session_id,
            operation=MessageType.PREFILL,
            position=start,
            token_count=len(chunk),
            payload=payload,
            final_prefill=final,
        )
        boundaries.append({
            "step_id": step,
            "operation": "PREFILL",
            "position": start,
            "token_count": len(chunk),
            "payload_bytes": len(payload),
            "frame_bytes": HEADER.size + len(payload),
            "sha256_before_send": hashlib.sha256(payload).hexdigest(),
        })
        step += 1
        if token is not None:
            generated.append(token)
    runtime.populate_all_experts()
    while len(generated) < max_new_tokens:
        position = len(prompt_ids) + len(generated) - 1
        hidden, residual = runtime.decode_a(generated[-1], position=position)
        payload = runtime.boundary_payload(hidden, residual)
        token = client.hidden(
            session_id=session_id,
            operation=MessageType.DECODE,
            position=position,
            token_count=1,
            payload=payload,
        )
        boundaries.append({
            "step_id": step,
            "operation": "DECODE",
            "position": position,
            "token_count": 1,
            "payload_bytes": len(payload),
            "frame_bytes": HEADER.size + len(payload),
            "sha256_before_send": hashlib.sha256(payload).hexdigest(),
        })
        step += 1
        generated.append(token)
    state_hashes = runtime.state_hashes()
    client.close(session_id)
    return {
        "session_id": session_id,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated,
        "complete_token_sequence": [*prompt_ids, *generated],
        "boundaries": boundaries,
        "a_state_hashes_at_close": state_hashes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--classes", default="W1,W2,W3,W4")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prefill-chunk-size", type=int, default=8192)
    parser.add_argument("--label", default="retained")
    args = parser.parse_args(argv)

    root = Path(args.evidence_root)
    root.mkdir(parents=True, exist_ok=True)
    socket_path = root / f"n1-{os.getpid()}.sock"
    ready_path = root / f"n1-{os.getpid()}.ready"
    b_report = root / f"n1-{args.label}-block-b.json"
    for path in (socket_path, ready_path):
        if path.exists():
            path.unlink()

    context = multiprocessing.get_context("spawn")
    service_args = [
        "--socket", str(socket_path), "--model", args.model, "--plan", args.plan,
        "--device-index", "0", "--report-out", str(b_report),
        "--ready-file", str(ready_path),
    ]
    process_b = context.Process(target=_service_entry, args=(service_args,), daemon=False)
    process_b.start()
    ready = _wait_ready(ready_path, process_b)

    # Importing/initializing CUDA happens only after Process B was spawned fresh.
    import torch
    from transformers import AutoTokenizer
    from inferswarm_phase0.manifest import load_manifest
    from .runtime import N1BlockRuntime

    runtime_a = N1BlockRuntime(
        role="a", model_path=args.model, plan_path=args.plan, device_index=1
    )
    client = N1BlockClient.connect(str(socket_path))
    manifest = load_manifest(args.manifest, canonical=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    reference = json.loads(Path(args.reference).read_text())
    reference_by_class = {row["class_id"]: row for row in reference["workloads"]}
    selected = [item.strip() for item in args.classes.split(",") if item.strip()]
    rows = []
    session_id = 1000
    try:
        for class_id in selected:
            workload = manifest.by_class()[class_id]
            prompt_ids = _prompt_ids(tokenizer, workload)
            row = _run_session(
                runtime=runtime_a, client=client, session_id=session_id,
                prompt_ids=prompt_ids, max_new_tokens=args.max_new_tokens,
                chunk_size=args.prefill_chunk_size,
            )
            expected = reference_by_class[class_id]["generated_token_ids"][:args.max_new_tokens]
            row.update({
                "class_id": class_id,
                "expected_generated_token_ids": expected,
                "exact_generated_sequence": row["generated_token_ids"] == expected,
            })
            rows.append(row)
            session_id += 1
        # Required same-prompt post-close leakage check.
        leak_class = selected[-1]
        workload = manifest.by_class()[leak_class]
        repeat = _run_session(
            runtime=runtime_a, client=client, session_id=session_id,
            prompt_ids=_prompt_ids(tokenizer, workload),
            max_new_tokens=args.max_new_tokens, chunk_size=args.prefill_chunk_size,
        )
        repeat_expected = reference_by_class[leak_class]["generated_token_ids"][:args.max_new_tokens]
        leakage = {
            "class_id": leak_class,
            "session_id": session_id,
            "repeat_generated_token_ids": repeat["generated_token_ids"],
            "matches_fresh_reference": repeat["generated_token_ids"] == repeat_expected,
            "matches_first_n1_session": repeat["generated_token_ids"] == rows[-1]["generated_token_ids"],
            "different_prompt_sessions_also_exact": all(row["exact_generated_sequence"] for row in rows),
        }
    finally:
        client.close_transport()
    process_b.join(timeout=60)
    if process_b.is_alive():
        process_b.terminate()
        process_b.join(timeout=20)
    if process_b.exitcode not in (0,):
        raise RuntimeError(f"Block B service exit code {process_b.exitcode}")

    block_b_document = json.loads(b_report.read_text())
    for row in rows:
        actual_steps = {
            item["generated_step"]: item
            for item in block_b_document["local_logit_checkpoints"]
            if item.get("session_id", row["session_id"]) == row["session_id"]
        }
        # Older exploratory service records predate the explicit session field. Retained
        # runs always carry it; fail closed if any frozen checkpoint is absent.
        reference_steps = reference_by_class[row["class_id"]].get("selected_logit_steps", {})
        comparisons = []
        for step in (0, 1, 15, 31):
            expected_entry = reference_steps.get(str(step))
            actual_entry = actual_steps.get(step)
            if expected_entry is None or actual_entry is None:
                comparisons.append({"step": step, "available": False, "exact": False})
                continue
            actual = torch.tensor(actual_entry["full_logits"], dtype=torch.float32)
            expected = torch.tensor(expected_entry["full_logits"], dtype=torch.float32)
            difference = (actual - expected).abs()
            nonzero = expected.abs() > 0
            relative = torch.where(
                nonzero, difference / expected.abs(),
                torch.where(difference == 0, 0.0, float("inf")),
            )
            comparisons.append({
                "step": step,
                "available": True,
                "exact": bool(torch.equal(actual, expected)),
                "max_absolute_deviation": float(difference.max().item()),
                "max_relative_deviation": float(relative.max().item()),
                "nan_count": int(torch.isnan(actual).sum().item()),
                "inf_count": int(torch.isinf(actual).sum().item()),
                "actual_float32_sha256": actual_entry["float32_sha256"],
                "reference_float32_sha256": expected_entry["float32_sha256"],
            })
        row["logit_checkpoints"] = comparisons
        row["all_selected_logits_exact"] = all(item["exact"] for item in comparisons)

    payload = {
        "schema": "inferswarm.n1.local-split-run/1",
        "label": args.label,
        "process_a": {
            "pid": os.getpid(),
            "device": _device_record("GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"),
            "runtime": runtime_a.report(),
        },
        "process_b": {
            "pid": ready["pid"],
            "device": _device_record("GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"),
            "report_path": str(b_report),
        },
        "protocol": {
            "boundary_dtype": "torch.bfloat16",
            "boundary_layout": "residual_pair[2,token_count,2048] row-major",
            "decode_hidden_payload_bytes": 8192,
            "token_result_payload_bytes": 4,
            "state_bytes_transferred": 0,
            "full_logits_transferred": 0,
        },
        "workloads": rows,
        "session_leakage": leakage,
        "client_timings": client.timings,
        "all_sequences_exact": all(row["exact_generated_sequence"] for row in rows),
        "all_selected_logits_exact": all(row["all_selected_logits_exact"] for row in rows),
        "all_session_leakage_checks_passed": all(leakage[key] for key in (
            "matches_fresh_reference", "matches_first_n1_session",
            "different_prompt_sessions_also_exact",
        )),
    }
    out = root / f"n1-{args.label}-run.json"
    write_json_with_sha(out, payload)
    print(json.dumps({
        "output": str(out),
        "process_a_pid": os.getpid(),
        "process_b_pid": ready["pid"],
        "all_sequences_exact": payload["all_sequences_exact"],
        "all_selected_logits_exact": payload["all_selected_logits_exact"],
        "session_leakage": leakage,
        "classes": [{"class_id": row["class_id"], "exact": row["exact_generated_sequence"]}
                    for row in rows],
    }, indent=2))
    return 0 if (
        payload["all_sequences_exact"]
        and payload["all_selected_logits_exact"]
        and payload["all_session_leakage_checks_passed"]
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
