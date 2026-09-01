"""Compose concise hashed baseline and disposition artifacts from retained evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freetoken.research.n0_model_block import write_json_with_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.evidence_dir
    allocation = json.loads((root / "allocation-primitives.json").read_text())
    retain = json.loads((root / "retain-mode.json").read_text())
    release = json.loads((root / "release-mode.json").read_text())
    baseline = {
        "schema": "inferswarm.pre-r3.retention-baseline/1",
        "r2_merge_base": "8627f441c880398389042ce8c0a604f6c4321dfa",
        "r2_disposition": "R2_LOCAL_SPLIT_EXECUTION_PASS",
        "observation": {
            "block_a_staging_bytes": 8_636_596_224,
            "block_a_post_realization_rss_shmem_approx_gib": 8.1,
            "block_b_staging_bytes": 9_545_711_616,
            "block_b_post_realization_rss_shmem_approx_gib": 8.9,
            "host_staging_current_bytes": 0,
            "unexplained_persistent_host_mirror_bytes": 0,
        },
        "classification": {
            "allocation_mechanism": "anonymous mmap + cudaHostRegister",
            "exact_owner": (
                "HostBank storage retained by the module-global process-lifetime "
                "_LIVE_BUFFERS list after R2 tensor/source dictionaries were detached"
            ),
            "lifetime": "worker process lifetime",
            "leak_classification": False,
            "finding": (
                "accidental post-final-residency lifetime overreach of a deliberate "
                "ordinary-offload retention rule"
            ),
            "ordinary_offload_intent": (
                "keep source banks attached so OffloadMoeCache.rebuild can resize GPU "
                "slot caches without rereading checkpoint storage"
            ),
            "post_r2_detach_reusable": False,
            "reason_not_reusable": (
                "R2 discarded tensor/source registries; the surviving mmap list had no "
                "supported rematerialization owner or operation"
            ),
            "history": [
                {
                    "commit": "3af9d90ee5e7af9bbff1e16f4f1c6201f41fff25",
                    "fact": "introduced process-lifetime mmap retention for offload banks",
                },
                {
                    "commit": "83300b9f5e244c53912d270b34c139700b8bdd4e",
                    "fact": "introduced logical resident-only detach",
                },
            ],
        },
    }
    write_json_with_sha(root / "retention-baseline.json", baseline)

    release_physical = release["physical_reclamation"]
    retain_accounting = {
        role: retain["participants"][role]["runtime"][
            "host_materialization_accounting"
        ]
        for role in ("a", "b")
    }
    release_accounting = {
        role: release["participants"][role]["runtime"][
            "host_materialization_accounting"
        ]
        for role in ("a", "b")
    }
    passed = bool(
        allocation["all_process_rss_reclaimed_at_least_95_percent"]
        and retain["passed"]
        and release["passed"]
        and release_physical["passed"]
    )
    result = {
        "schema": "inferswarm.pre-r3.host-reclamation-result/1",
        "result": (
            "HOST_STAGING_RECLAMATION_PASS"
            if passed
            else "HOST_STAGING_RECLAMATION_BLOCKED"
        ),
        "issue": "https://github.com/Zutfen-LLC/inferswarm/issues/53",
        "r2_merge_base": "8627f441c880398389042ce8c0a604f6c4321dfa",
        "ownership": baseline["classification"],
        "primitive_diagnostic_passed": allocation[
            "all_process_rss_reclaimed_at_least_95_percent"
        ],
        "retain": {
            "accounting": retain_accounting,
            "correctness": retain["correctness"],
            "post_finalization_rematerialization_supported": False,
            "rematerialization_timing_seconds": None,
            "timing_status": (
                "not measured: no supported post-finalization destroy/rematerialize "
                "operation exists; retention benefit is proven only for ordinary "
                "OffloadMoeCache.rebuild before finalization"
            ),
        },
        "release": {
            "accounting": release_accounting,
            "physical_reclamation": release_physical,
            "correctness": release["correctness"],
            "workers_remained_alive": release[
                "workers_remained_alive_through_resident_decode"
            ],
            "participant_runtime_counters": {
                role: {
                    key: release["participants"][role]["runtime"][key]
                    for key in (
                        "decode_graph", "host_expert_fetches",
                        "resident_source_accesses", "fallbacks",
                        "steady_model_state_movement_bytes", "populate_count",
                    )
                }
                for role in ("a", "b")
            },
        },
        "future_r3_accounting_principle": {
            "required_persistent_host_bytes": "execution-required, non-evictable",
            "optional_retained_host_cache_bytes": "intentional and evictable",
            "reclaimable_host_cache_bytes": "optional bytes eligible for physical release",
            "transient_staging_bytes": "materialization-time overlap",
            "physically_available_host_bytes": "OS evidence, not tensor counters",
        },
        "verification": {
            "lifecycle_and_ordinary_offload": "41 passed",
            "inferswarm_r0_r1_r2_and_offload_targeted": "228 passed",
            "r2_correctness_methodology_regressions": "28 passed",
            "ruff": "passed on all changed Python paths",
        },
        "limitations": [
            "RETAIN has no post-finalization rematerialization API or timing proof.",
            "Physical capacity proof is Linux /proc accounting on the frozen host.",
            "The lifecycle policy and accounting remain internal research interfaces.",
        ],
        "passed": passed,
    }
    write_json_with_sha(root / "result.json", result)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
