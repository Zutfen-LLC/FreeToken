"""Run retained diagnostic correctness and session-isolation evidence for R2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha
from freetoken.research.r2_local_split import RESULT_SCHEMA

from .coordinator import LocalSplitCoordinator
from .correctness_support import (
    DIAGNOSTIC_OVERRIDE,
    NONCANONICAL_LABEL,
    reference_provenance,
    validate_diagnostic_output,
)


def _prompt_ids(tokenizer, workload) -> list[int]:
    body = workload.greedy_reference_body("nvidia/Qwen3.6-35B-A3B-NVFP4")
    encoded = tokenizer.apply_chat_template(
        body["messages"],
        tokenize=True,
        add_generation_prompt=True,
        **body["chat_template_kwargs"],
    )
    return list(encoded["input_ids"])


def _compare_logits(actual: dict, expected: dict) -> dict:
    import torch

    actual_tensor = torch.tensor(actual["full_logits"], dtype=torch.float32)
    expected_tensor = torch.tensor(expected["full_logits"], dtype=torch.float32)
    actual_shape = list(actual_tensor.shape)
    expected_shape = list(expected_tensor.shape)
    actual_tensor = actual_tensor.reshape(-1)
    expected_tensor = expected_tensor.reshape(-1)
    if actual_tensor.numel() != expected_tensor.numel():
        raise RuntimeError(
            "selected logit checkpoint element count differs: "
            f"actual={actual_shape}, expected={expected_shape}"
        )
    difference = (actual_tensor - expected_tensor).abs()
    relative = torch.where(
        expected_tensor.abs() > 0,
        difference / expected_tensor.abs(),
        torch.where(difference == 0, 0.0, float("inf")),
    )
    return {
        "exact": bool(torch.equal(actual_tensor, expected_tensor)),
        "actual_shape": actual_shape,
        "reference_shape": expected_shape,
        "logical_row_element_count": actual_tensor.numel(),
        "within_canonical_threshold": bool(
            torch.allclose(actual_tensor, expected_tensor, rtol=2e-3, atol=2e-3)
        ),
        "rtol": 2e-3,
        "atol": 2e-3,
        "max_absolute_deviation": float(difference.max().item()),
        "max_relative_deviation": float(relative.max().item()),
        "nan_count": int(torch.isnan(actual_tensor).sum().item()),
        "inf_count": int(torch.isinf(actual_tensor).sum().item()),
        "actual_float32_sha256": actual["float32_sha256"],
        "reference_float32_sha256": expected["float32_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--classes", default="W1,W2,W3,W4")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--diagnostic-prefill-chunk", type=int)
    parser.add_argument("--allow-legacy-reference-diagnostic", action="store_true")
    parser.add_argument("--capture-prefill-state", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from inferswarm_phase0.manifest import load_manifest
    from transformers import AutoTokenizer

    plan = json.loads(args.plan.read_text())
    reference = json.loads(args.reference.read_text())
    reference_by_class = {item["class_id"]: item for item in reference["workloads"]}
    manifest = load_manifest(args.manifest, canonical=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    selected = [value.strip() for value in args.classes.split(",") if value.strip()]
    diagnostic_override = args.diagnostic_prefill_chunk is not None
    if args.allow_legacy_reference_diagnostic and not diagnostic_override:
        raise ValueError("legacy reference allowance requires a diagnostic override")
    validate_diagnostic_output(args.out, diagnostic_override=diagnostic_override)
    prefill_chunk = (
        args.diagnostic_prefill_chunk
        if diagnostic_override
        else plan["runtime_capacity"]["prefill_chunk_tokens"]
    )
    if prefill_chunk <= 0:
        raise ValueError("prefill chunk must be positive")
    prompt_ids_by_class = {
        class_id: _prompt_ids(tokenizer, manifest.by_class()[class_id])
        for class_id in selected
    }
    reference_record = reference_provenance(
        reference,
        args.reference,
        required_model=plan["model"]["repository"],
        required_revision=plan["model"]["revision"],
        required_classes=selected,
        required_prompt_ids=prompt_ids_by_class,
        allow_legacy_diagnostic=args.allow_legacy_reference_diagnostic,
    )
    rows = []
    with LocalSplitCoordinator(
        plan_path=str(args.plan),
        model_path=args.model,
        diagnostic=True,
        diagnostic_prefill_chunk=args.diagnostic_prefill_chunk,
    ) as coordinator:
        startup = coordinator.ready
        for index, class_id in enumerate(selected):
            prompt_ids = prompt_ids_by_class[class_id]
            expected = reference_by_class[class_id]
            if prompt_ids != expected["prompt_token_ids"]:
                raise RuntimeError(
                    f"{class_id} prompt IDs differ from frozen reference"
                )
            row = coordinator.run_session(
                session_id=1000 + index,
                prompt_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                prefill_chunk=prefill_chunk,
                capture_steps={0, 1, 15, 31} & set(range(args.max_new_tokens)),
                capture_prefill_state=args.capture_prefill_state,
            )
            row["class_id"] = class_id
            row["prefill_chunk_tokens"] = prefill_chunk
            row["prefill_chunk_count"] = (
                len(prompt_ids) + prefill_chunk - 1
            ) // prefill_chunk
            row["expected_generated_token_ids"] = expected["generated_token_ids"][
                : args.max_new_tokens
            ]
            row["exact_generated_sequence"] = (
                row["generated_token_ids"] == row["expected_generated_token_ids"]
            )
            actual_logits = {
                item["generated_step"]: item["logits"]
                for item in row["boundaries"]
                if "logits" in item
            }
            comparisons = []
            for step in sorted(actual_logits):
                comparison = _compare_logits(
                    actual_logits[step], expected["selected_logit_steps"][str(step)]
                )
                comparisons.append({"generated_step": step, **comparison})
            row["logit_checkpoints"] = comparisons
            row["all_selected_logits_exact"] = len(comparisons) == min(
                4, args.max_new_tokens
            ) and all(item["exact"] for item in comparisons)
            for boundary in row["boundaries"]:
                boundary.pop("logits", None)
            rows.append(row)
        repeat_class = selected[-1]
        repeat_ids = _prompt_ids(tokenizer, manifest.by_class()[repeat_class])
        repeat = coordinator.run_session(
            session_id=2000,
            prompt_ids=repeat_ids,
            max_new_tokens=args.max_new_tokens,
            prefill_chunk=prefill_chunk,
            capture_steps=set(),
        )
        session_isolation = {
            "class_id": repeat_class,
            "matches_fresh_reference": repeat["generated_token_ids"]
            == reference_by_class[repeat_class]["generated_token_ids"][
                : args.max_new_tokens
            ],
            "matches_original_split_session": repeat["generated_token_ids"]
            == rows[-1]["generated_token_ids"],
            "different_prompt_sessions_exact": all(
                row["exact_generated_sequence"] for row in rows
            ),
        }
        reports = coordinator.reports()

    graph_pass = all(
        reports[role]["runtime"]["decode_graph"]["captures"] == 1
        and reports[role]["runtime"]["decode_graph"]["recaptures"] == 0
        and reports[role]["runtime"]["decode_graph"]["replays"]
        >= (args.max_new_tokens - 1) * (len(rows) + 1)
        and reports[role]["runtime"]["host_expert_fetches"] == 0
        and reports[role]["runtime"]["resident_source_accesses"] == 0
        and reports[role]["runtime"]["fallbacks"] == 0
        for role in ("a", "b")
    )
    host_mirror_pass = all(
        reports[role]["runtime"]["unexplained_persistent_host_mirror_bytes"] == 0
        and reports[role]["runtime"]["host_staging_current_bytes"] == 0
        for role in ("a", "b")
    )
    payload = {
        "schema": RESULT_SCHEMA,
        "result_kind": "correctness",
        "evidence_label": NONCANONICAL_LABEL
        if diagnostic_override
        else "CANONICAL_CANDIDATE_EVIDENCE",
        "diagnostic_override": DIAGNOSTIC_OVERRIDE if diagnostic_override else None,
        "reference": reference_record,
        "architectural_result": "AWAITING_PHYSICAL_REVIEW",
        "plan": {
            "digest": plan["digest"],
            "file_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
            "split": {"block_a": [0, 19], "block_b": [19, 40]},
        },
        "correctness": {
            "workloads": rows,
            "all_generated_sequences_exact": all(
                row["exact_generated_sequence"] for row in rows
            ),
            "all_selected_logits_exact": all(
                row["all_selected_logits_exact"] for row in rows
            ),
            "all_selected_logits_within_canonical_threshold": all(
                item["within_canonical_threshold"]
                for row in rows
                for item in row["logit_checkpoints"]
            ),
            "max_absolute_deviation": max(
                item["max_absolute_deviation"]
                for row in rows
                for item in row["logit_checkpoints"]
            ),
            "max_relative_deviation": max(
                item["max_relative_deviation"]
                for row in rows
                for item in row["logit_checkpoints"]
            ),
            "nan_count": sum(
                item["nan_count"] for row in rows for item in row["logit_checkpoints"]
            ),
            "inf_count": sum(
                item["inf_count"] for row in rows for item in row["logit_checkpoints"]
            ),
        },
        "session_isolation": {
            **session_isolation,
            "passed": all(session_isolation.values()),
        },
        "backend_native": {
            "participants": {
                role: reports[role]["runtime"]["decode_graph"] for role in ("a", "b")
            },
            "passed": graph_pass,
        },
        "host_mirror": {
            "participants": {
                role: reports[role]["runtime"][
                    "unexplained_persistent_host_mirror_bytes"
                ]
                for role in ("a", "b")
            },
            "passed": host_mirror_pass,
        },
        "state_ownership": {
            role: reports[role]["runtime"]["state_ownership"] for role in ("a", "b")
        },
        "boundary": {
            "decode_payload_bytes": plan["boundary"]["contract"][
                "decode_payload_bytes"
            ],
            "prefill_chunk_payload_bytes": plan["boundary"]["contract"][
                "prefill_chunk_payload_bytes"
            ],
            "all_checksums_matched": all(
                item.get("producer_sha256") == item.get("consumer_sha256")
                for row in rows
                for item in row["boundaries"]
            ),
        },
        "participants": reports,
        "startup": startup,
        "passed": all(row["exact_generated_sequence"] for row in rows)
        and all(
            item["within_canonical_threshold"]
            for row in rows
            for item in row["logit_checkpoints"]
        )
        and all(session_isolation.values())
        and graph_pass
        and host_mirror_pass,
    }
    write_json_with_sha(args.out, payload)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "plan_digest": plan["digest"],
                "passed": payload["passed"],
                "classes": selected,
                "graphs": payload["backend_native"],
                "host_mirror": payload["host_mirror"],
            },
            indent=2,
        )
    )
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
