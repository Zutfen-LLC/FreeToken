"""R6 result composer: re-derive every passing condition from retained
artifacts (no trust in this-session claims, no constant-passing checks).
Emits result.json for the docs/inferswarm_r6 evidence directory.

Fail-closed: any missing artifact, any stage lacking required evidence,
any accounting mismatch, or any comparator violation fails the gate.
Exit code is nonzero when any check fails.

This composer deliberately separates provenance identities:
- the canonical physical producer (run-bound, from chain-plan.json);
- the evidence-arm requalification producer (secondary comparator
  capture, recorded as EXPLICIT_OVERRIDE_EVIDENCE_ARM);
- the repo/evidence-assembly HEAD at composition time.

Run from the FreeToken repo root with the evidence dir path as argv[1].
"""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

EXPECTED_TOKENS = [818, 6073, 529, 74413, 46515, 600, 2557, 532]
EXPECTED_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
EXPECTED_CHECKPOINT_SHA = (
    "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d"
)
CANONICAL_PRODUCER = "44d6c94e4fd2ee967451cc959f930883ca3f4a25"
LOGIT_THRESHOLD = 0.25
COMPARATOR_STEPS = ("0", "1", "7")
VOCAB = 262144


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"required evidence artifact missing: {path}")
    return json.loads(path.read_text())


def evaluate_secondary_comparator(evidence: Path) -> dict:
    """Independently derive the frozen secondary comparator from retained
    artifacts: reference top-32 (reference-generation.json) vs the
    distributed full-vocab rows (binary float32 retention).  Returns a
    detail dict; never a constant."""
    reference = load(evidence / "reference-generation.json")
    blob = (evidence / "lifecycle/distributed-logits-0-1-7.f32.bin").read_bytes()
    if len(blob) != len(COMPARATOR_STEPS) * VOCAB * 4:
        raise ValueError(
            f"retained distributed logits blob is {len(blob)} bytes; "
            f"expected {len(COMPARATOR_STEPS) * VOCAB * 4}"
        )
    rows = {}
    offset = 0
    for step in COMPARATOR_STEPS:
        rows[step] = list(
            struct.unpack_from(f"<{VOCAB}f", blob, offset)
        )
        offset += VOCAB * 4

    nan_inf = 0
    for step in COMPARATOR_STEPS:
        nan_inf += sum(1 for v in rows[step] if v != v or v in (float("inf"), float("-inf")))

    per_step = {}
    aggregate = 0.0
    for step in COMPARATOR_STEPS:
        top = reference["step_top32_logits"][step]
        diffs = [
            abs(rows[step][index] - value)
            for index, value in zip(top["top_indices"], top["top_values"])
        ]
        per_step[step] = max(diffs)
        aggregate = max(aggregate, per_step[step])
    return {
        "threshold": LOGIT_THRESHOLD,
        "per_step_max_absdiff": per_step,
        "aggregate_max_absdiff": aggregate,
        "nan_inf_count": nan_inf,
        "domain": "reference top-32 per declared step (retained reference "
                  "domain; union-domain max is necessarily >= this max)",
        "passes": aggregate < LOGIT_THRESHOLD and nan_inf == 0,
    }


def stage_evidence(evidence: Path, chain_plan: dict, serving_report: dict) -> dict:
    """Assemble per-stage observed evidence, fail-closed on absence.

    Stage 1/2: the serving report's final_runtime_report.stages[0..1].
    Stage 3 (remote): the retained last-stage final report
    (lifecycle/last-stage-final-report.json), validated against the frozen
    plan (digest + producer) before use.
    """
    stages = serving_report["epochs"][0]["final_runtime_report"]["stages"]
    if len(stages) < 2:
        raise ValueError("serving report lacks local stage reports")
    last_report = load(evidence / "lifecycle/last-stage-final-report.json")
    if last_report["plan_digest"] != chain_plan["digest"]:
        raise ValueError(
            "retained last-stage final report is not bound to the frozen "
            "participant plan digest"
        )
    if last_report["producer_freetoken_sha"] != CANONICAL_PRODUCER:
        raise ValueError(
            "retained last-stage final report producer does not match the "
            "canonical run producer"
        )
    runtime = last_report["runtime"]
    roles = [stages[0].get("role"), stages[1].get("role"), runtime.get("role")]
    if roles != ["first", "middle", "last"]:
        raise ValueError(f"unexpected stage roles: {roles}")
    return {
        "first": stages[0],
        "middle": stages[1],
        "last": runtime,
        "_last_report": last_report,
    }


def main() -> int:
    evidence = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/inferswarm_r6")
    checks: dict[str, bool] = {}
    detail: dict = {}

    repo_root = evidence.resolve().parents[1]
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        head = None  # synthetic/offline evidence fixture; recorded as null

    chain_plan = load(evidence / "chain-plan.json")
    producer = chain_plan["provenance"]["r6"]["producer_sha"]

    # 1. Frozen identity chain
    environment = load(evidence / "environment.json")
    checks["environment_producer_matches_run_producer"] = (
        environment["implementation_commit"] == producer
    )
    checks["run_producer_is_canonical"] = producer == CANONICAL_PRODUCER
    detail["run_producer"] = producer
    detail["repo_head_at_composition"] = head
    checks["model_revision_frozen"] = (
        environment["model"]["revision"] == EXPECTED_REVISION
    )
    checks["checkpoint_sha_frozen"] = (
        environment["model"]["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA
    )
    checks["chain_plan_producer_frozen"] = (
        chain_plan["provenance"]["r6"]["producer_sha"] == producer
    )

    # Methodology chronology: original freeze + amendment retained.
    methodology = (evidence / "METHODOLOGY.md").read_text()
    amendment_path = evidence / "METHODOLOGY-AMENDMENT-001.md"
    checks["methodology_original_retained"] = (
        "245,760" in methodology and "32-row prefill chunks" in methodology
    )
    amendment_present = amendment_path.exists()
    checks["methodology_amendment_retained"] = amendment_present
    if amendment_present:
        amendment = amendment_path.read_text()
        checks["methodology_amendment_chronology_explicit"] = (
            "ff561e5" in amendment
            and "44d6c94" in amendment
            and "BEFORE" in amendment
        )
        frozen_geometry = chain_plan.get("boundary_geometry", {})
        checks["chain_plan_uses_amended_geometry"] = (
            frozen_geometry.get("prefill_chunk_rows") == 64
            and frozen_geometry.get("prefill_bytes") == 491520
        )

    # 2. Canonical serving correctness
    report = load(evidence / "lifecycle/serving-report.json")
    requests = report["coordinator_scope"]["requests"]
    canonical = next(
        (r for r in requests if not r.get("fencing_arm_injections")), None
    )
    fencing = next(
        (r for r in requests if r.get("fencing_arm_injections")), None
    )
    checks["canonical_request_retained"] = canonical is not None
    checks["token_equality_exact"] = (
        canonical is not None
        and canonical["generated_token_ids"] == EXPECTED_TOKENS
    )
    reference = load(evidence / "reference-generation.json")
    checks["reference_token_equality"] = (
        reference["generated_token_ids"] == EXPECTED_TOKENS
    )
    checks["prompt_ids_match_canonical_prompt"] = (
        canonical is not None
        and canonical["prompt_token_ids"]
        == load(evidence / "canonical-prompt.json")["prompt_token_ids"]
    )

    # 3. Attribution / fencing / epoch authority
    if canonical is not None:
        epochs = set(canonical["committed_epoch_ids"])
        detail["distinct_committed_epochs"] = sorted(epochs)
        checks["all_tokens_one_epoch_lineage"] = len(epochs) == 1
    checks["fencing_arm_rejections"] = (
        fencing is not None
        and len(fencing["fencing_arm_injections"]) == 2
        and all(
            inj["accepted"] is False
            for inj in fencing["fencing_arm_injections"]
        )
    )
    checks["fencing_arm_output_unharmed"] = (
        fencing is not None
        and fencing["generated_token_ids"] == EXPECTED_TOKENS
    )

    # 4. Epoch lifecycle
    epoch_states = [e.get("state") for e in report.get("epochs", [])]
    detail["epoch_states"] = epoch_states
    checks["epoch_reclaimed_after_shutdown"] = epoch_states == ["RECLAIMED"]

    # 5. Coordinator purity
    log_text = ""
    for name in ("coordinator-run.log.txt", "coordinator-run.log"):
        candidate = evidence / name
        if candidate.exists():
            log_text = candidate.read_text()
            break
    checks["coordinator_torch_free_evidence"] = (
        "PyTorch was not found" in log_text
    )

    # 6. Coverage proof from the frozen chain plan
    blocks = chain_plan["blocks"]
    ranges = [(b["spec"]["start_layer"], b["spec"]["end_layer"]) for b in blocks]
    checks["stage_coverage_complete_contiguous"] = (
        len(blocks) == 3
        and ranges[0][0] == 0
        and ranges[-1][1] == 48
        and all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1))
    )
    shared = chain_plan.get("declared_shared_state", {})
    checks["shared_state_declared_tied_embedding"] = (
        shared.get("id") == "tied-embedding-lm-head"
        and shared.get("materialization_policy") == "duplicated-on-first-and-last-stage"
    )

    # 7. Selective materialization / accounting per stage (fail-closed).
    #    Every participating stage must have retained evidence; observed
    #    fetched bytes must equal planned owned bytes plus (for the
    #    embedding-owning/sharing stages) the declared shared bytes.
    stage_detail = []
    stage_checks_ok = True
    try:
        stages = stage_evidence(evidence, chain_plan, report)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        checks["stage_evidence_present_and_producer_bound"] = False
        checks["stage_fetch_matches_frozen_plan"] = False
        checks["stage_selective_load_clean"] = False
        checks["no_unexplained_host_mirror"] = False
        detail["stage_evidence_error"] = str(exc)
        stages = None
    if stages is not None:
        checks["stage_evidence_present_and_producer_bound"] = True
        shared_bytes = shared.get("bytes", 0)
        # The first stage's allowed keys ALREADY include the tied embedding
        # (block 0 owns it); only the LAST stage's fetch = owned + declared
        # shared bytes (tied lm_head re-materializes the same table).
        shared_on = {"last"}
        fetch_ok = True
        clean_ok = True
        mirror_ok = True
        for block, (role, observed) in zip(
            blocks, (("first", stages["first"]), ("middle", stages["middle"]),
                     ("last", stages["last"]))
        ):
            planned_owned = block["owned_checkpoint_bytes"]
            planned_total = planned_owned + (
                shared_bytes if role in shared_on else 0
            )
            observed_bytes = observed.get("fetched_bytes")
            role_ok = observed.get("role") == role
            layers_ok = observed.get("global_layer_ids") == list(
                range(block["spec"]["start_layer"], block["spec"]["end_layer"])
            )
            this_fetch = (
                role_ok
                and layers_ok
                and observed_bytes == planned_total
            )
            this_clean = (
                observed.get("unexpected_checkpoint_keys") == []
                and observed.get("whole_shard_sentinel_calls") == 0
            )
            this_mirror = (
                observed.get("host_staging_current_bytes") == 0
                and observed.get("unexplained_persistent_host_mirror_bytes") == 0
                and observed.get("resident_only") is True
            )
            fetch_ok = fetch_ok and this_fetch
            clean_ok = clean_ok and this_clean
            mirror_ok = mirror_ok and this_mirror
            stage_detail.append(
                {
                    "role": role,
                    "planned_owned_checkpoint_bytes": planned_owned,
                    "planned_total_with_shared": planned_total,
                    "observed_fetched_bytes": observed_bytes,
                    "role_matches": role_ok,
                    "global_layer_ids_match": layers_ok,
                    "unexpected_checkpoint_keys": observed.get(
                        "unexpected_checkpoint_keys"
                    ),
                    "whole_shard_sentinel_calls": observed.get(
                        "whole_shard_sentinel_calls"
                    ),
                    "host_staging_current_bytes": observed.get(
                        "host_staging_current_bytes"
                    ),
                    "unexplained_persistent_host_mirror_bytes": observed.get(
                        "unexplained_persistent_host_mirror_bytes"
                    ),
                    "resident_only": observed.get("resident_only"),
                    "resident_device_bytes": observed.get(
                        "resident_device_bytes"
                    ),
                    "vmstat_delta": observed.get("vmstat_delta"),
                }
            )
        checks["stage_fetch_matches_frozen_plan"] = fetch_ok
        checks["stage_selective_load_clean"] = clean_ok
        checks["no_unexplained_host_mirror"] = mirror_ok
        detail["stages"] = stage_detail
        # cross-check: retained last-stage report stats vs serving sessions.
        # Each generate() call crosses the wire as one prefill (prompt_len
        # rows) PLUS one speculative decode row (the controller's 2-token
        # generate), so expected rx rows = sum(prompt_len) + len(sessions).
        sessions = report["epochs"][0]["final_runtime_report"].get(
            "sessions", []
        )
        expected_rows = sum(
            s["prompt_len"] for s in sessions
        ) + len(sessions)
        last_stats = stages["_last_report"].get("stats", {})
        detail["last_stage_stats"] = last_stats
        detail["last_stage_rows_expected_from_sessions"] = expected_rows
        checks["last_stage_stats_consistent_with_sessions"] = (
            last_stats.get("activation_bytes_rx")
            == expected_rows * frozen_row_bytes(chain_plan)
        )

    # 8. Boundary geometry consistency (single plane)
    geometry = chain_plan.get("boundary_geometry", {})
    checks["boundary_geometry_single_plane"] = (
        geometry.get("planes") == 1
        and geometry.get("row_width") == 3840
        and geometry.get("dtype") == "bfloat16"
    )

    # 9. Secondary logit comparator (independently derived, fail-closed)
    try:
        comparator = evaluate_secondary_comparator(evidence)
        checks["secondary_logit_comparator_retained"] = True
    except (FileNotFoundError, ValueError) as exc:
        comparator = {
            "error": str(exc),
            "passes": False,
        }
        checks["secondary_logit_comparator_retained"] = False
    checks["secondary_logit_comparator_threshold"] = comparator.get("passes", False)
    detail["secondary_comparator"] = comparator

    failed = sorted(k for k, v in checks.items() if not v)
    gate_pass = not failed

    # Provenance identities stay distinct: canonical run producer, the
    # evidence-arm requalification producer, and the assembly HEAD.
    last_stage_path = evidence / "lifecycle/last-stage-final-report.json"
    comparator_path = evidence / "lifecycle/secondary-comparator.json"
    last_stage_producer = (
        load(last_stage_path)["producer_freetoken_sha"]
        if last_stage_path.exists() else None
    )
    comparator_arm_producer = None
    if comparator_path.exists():
        comparator_record = load(comparator_path)
        comparator_arm_producer = comparator_record["provenance"][
            "evidence_arm_producer"
        ]
    result = {
        "schema": "inferswarm.r6.result/2",
        "gate": "R6_DENSE_ARCHITECTURE_FALSIFICATION",
        "provenance": {
            "canonical_physical_producer": producer,
            "canonical_last_stage_final_report_producer": last_stage_producer,
            "secondary_comparator_arm_producer": comparator_arm_producer,
            "secondary_comparator_arm_mode": (
                load(comparator_path)["provenance"][
                    "evidence_arm_producer_check"
                ]["mode"] if comparator_path.exists() else None
            ),
            "repo_head_at_composition": head,
            "methodology": "docs/inferswarm_r6/METHODOLOGY.md (original freeze)",
            "methodology_amendment": (
                "docs/inferswarm_r6/METHODOLOGY-AMENDMENT-001.md"
                if amendment_present else None
            ),
        },
        "checks": checks,
        "failed_checks": failed,
        "detail": detail,
        "expected_generated_token_ids": EXPECTED_TOKENS,
    }
    if gate_pass:
        result["verdict"] = "R6_DENSE_ARCHITECTURE_FALSIFICATION_PASS_CANDIDATE"
    else:
        result["verdict"] = "R6_DENSE_ARCHITECTURE_FALSIFICATION_FAIL"
        result["verdict_note"] = (
            "Honest fail: the corrected composer does not pass every "
            "frozen condition from retained evidence. See failed_checks "
            "and detail.secondary_comparator."
        )
    (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"failed": failed, "passed": len(checks) - len(failed),
                      "total": len(checks), "verdict": result["verdict"]}))
    return 1 if failed else 0


def frozen_row_bytes(chain_plan: dict) -> int:
    geometry = chain_plan.get("boundary_geometry", {})
    return geometry.get("row_width", 3840) * 2


if __name__ == "__main__":
    raise SystemExit(main())
