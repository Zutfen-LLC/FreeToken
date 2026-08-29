"""``python benchmarks/phase1_campaign.py`` -- the Phase-1 campaign CLI.

Subcommands:

    plan          emit the complete two-session campaign plan as JSON (no model start)
    validate      dry-run validation: provenance preflight, held-constant comparison,
                  intended-difference enumeration, counts, ordering, hashes
    run-session   execute one session (the P6 surface; refuses to start when the
                  preflight fails)

``--dev-smoke`` is the only way to alter the protocol (repetitions, warmups, class
selection); every alteration is recorded as a deviation, forces canonical=false, and
makes the artifact unusable by the canonical analysis. Without ``--dev-smoke``,
protocol changes are rejected.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from inferswarm_phase0 import CANONICAL_MODEL_REPOSITORY
from inferswarm_phase0 import provenance as prov

from . import CAMPAIGN_RUNNER_VERSION
from .campaign import (
    DEFAULT_TIMING_MAX_STEPS,
    CampaignDefinition,
    CampaignRefused,
    CampaignSettings,
    SessionExecution,
    plan_document,
    validation_document,
)
from .campaign_arms import (
    baseline_b1_arm,
    candidate_v2_arm,
    kv_matched_arm,
)
from .campaign_protocol import build_protocol


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="local checkpoint directory (or .ftw)")
    p.add_argument("--manifest", required=True, help="frozen workload manifest (JSON)")
    p.add_argument(
        "--model-repository",
        default=CANONICAL_MODEL_REPOSITORY,
        help="upstream model repository (criteria section 1.1)",
    )
    p.add_argument(
        "--model-revision",
        default=None,
        help="EXACT upstream commit SHA of the checkpoint (40 hex). Required for a canonical run.",
    )
    p.add_argument(
        "--placement",
        default=None,
        help=(
            "frozen placement artifact path; REQUIRED when the candidate arm runs. "
            "Its SHA-256 must equal the frozen canonical value before anything starts."
        ),
    )
    p.add_argument("--inferswarm-commit", default=None, help="InferSwarm commit this campaign belongs to")
    p.add_argument(
        "--correctness-prerequisites",
        default=None,
        help=(
            "JSON manifest naming the exact passing correctness-reference-v2, "
            "candidate C3, and P2/P3/P4 requalification artifacts plus the FreeToken "
            "runtime commit. REQUIRED for a canonical session."
        ),
    )
    p.add_argument("--out-root", default="phase1-campaign", help="where session directories are created")
    p.add_argument("--server-timeout", type=float, default=3600.0)
    p.add_argument(
        "--timing-max-steps",
        type=int,
        default=DEFAULT_TIMING_MAX_STEPS,
        help="MoE layer-timing retention capacity per measured block (decode steps)",
    )
    p.add_argument(
        "--kv-matched-tokens",
        type=int,
        default=None,
        help=(
            "add the supplementary KV-matched baseline arm with this --num-tokens "
            "(derive it from the candidate's resolved KV capacity; it never replaces "
            "the primary baseline)"
        ),
    )
    p.add_argument("--warmups", type=int, default=None, help="NON-CANONICAL override; needs --dev-smoke")
    p.add_argument("--repetitions", type=int, default=None, help="NON-CANONICAL override; needs --dev-smoke")
    p.add_argument(
        "--classes",
        default=None,
        help="comma-separated class subset; permitted only with --dev-smoke",
    )
    p.add_argument("--store-output-text", action="store_true", help="keep full generated text per generation")
    p.add_argument("--no-echo-server", action="store_true", help="do not mirror server output")
    p.add_argument(
        "--dev-smoke",
        action="store_true",
        help=(
            "developer smoke test: allows protocol overrides and a non-canonical "
            "manifest; stamps every artifact NON_CANONICAL_DEV_SMOKE"
        ),
    )
    p.add_argument("--output", default=None, help="write the JSON document here instead of stdout")


def _definition(args: argparse.Namespace) -> CampaignDefinition:
    canonical = not args.dev_smoke
    classes = (
        [c.strip() for c in args.classes.split(",") if c.strip()]
        if args.classes
        else None
    )
    protocol = build_protocol(
        warmups=args.warmups,
        repetitions=args.repetitions,
        classes=classes,
        dev_smoke=args.dev_smoke,
    )
    arms = [baseline_b1_arm(), candidate_v2_arm()]
    if args.kv_matched_tokens is not None:
        arms.append(kv_matched_arm(args.kv_matched_tokens))
    settings = CampaignSettings(
        model_path=args.model,
        manifest_path=args.manifest,
        model_repository=args.model_repository,
        model_revision=args.model_revision,
        placement_path=args.placement,
        inferswarm_commit=args.inferswarm_commit,
        out_root=Path(args.out_root),
        server_timeout=args.server_timeout,
        timing_max_steps=args.timing_max_steps,
        store_output_text=args.store_output_text,
        echo_server_output=not args.no_echo_server,
        prerequisites_path=args.correctness_prerequisites,
    )
    return CampaignDefinition(arms=arms, protocol=protocol, settings=settings, canonical=canonical)


def _emit(doc: dict, path: str | None) -> None:
    encoded = json.dumps(doc, indent=2) + "\n"
    if path:
        Path(path).write_text(encoded, encoding="utf-8")
        print(f"[phase1] wrote {path}", file=sys.stderr)
    else:
        print(encoded, end="")


def cmd_plan(args: argparse.Namespace) -> int:
    definition = _definition(args)
    prov.validate_revision(args.model_revision, canonical=not args.dev_smoke)
    doc = plan_document(definition, manifest=_load_manifest_for_plan(args))
    banner = (
        "CANONICAL Phase-1 campaign plan"
        if doc["canonical"]
        else "NON_CANONICAL_DEV_SMOKE plan - NOT a canonical campaign"
    )
    print(f"=== {banner} ===", file=sys.stderr)
    _emit(doc, args.output)
    return 0


def _load_manifest_for_plan(args: argparse.Namespace):
    from .campaign import load_campaign_manifest

    return load_campaign_manifest(args.manifest, canonical=not args.dev_smoke)


def cmd_validate(args: argparse.Namespace) -> int:
    definition = _definition(args)
    doc = validation_document(definition)
    banner = (
        "CANONICAL Phase-1 campaign validation"
        if doc["canonical"]
        else "VALIDATION FAILED / NON-CANONICAL - see preflight_refusals and blockers"
    )
    print(f"=== {banner} ===", file=sys.stderr)
    for blocker in doc["canonical_blockers"]:
        print(f"  - {blocker}", file=sys.stderr)
    for refusal in doc["preflight_refusals"]:
        print(f"  ! this campaign would be refused: {refusal}", file=sys.stderr)
    _emit(doc, args.output)
    return 0 if doc["canonical"] else 1


def cmd_run_session(args: argparse.Namespace) -> int:
    definition = _definition(args)
    if not definition.canonical and not args.dev_smoke:
        raise SystemExit("run-session requires --dev-smoke for a non-canonical run")
    session = SessionExecution(
        definition=definition,
        session_number=args.session,
        thermal_reset_attested=args.thermal_reset_attested,
    )
    doc = session.execute()
    status = doc["execution_status"]
    validity = doc["validity"]
    print(
        f"\n[phase1] session-{args.session}: {status} / {validity}: {doc['run_directory']}",
        flush=True,
    )
    if doc["stopped_early_reason"]:
        print(f"[phase1] STOPPED EARLY: {doc['stopped_early_reason']}", flush=True)
    for item in doc.get("campaign_invalidations") or []:
        where = "/".join(
            str(item[k]) for k in ("arm_id", "class_id") if item.get(k)
        ) or "campaign"
        print(f"[phase1]   INVALIDATING {item['code']} ({where}): {item['message']}", flush=True)
    _emit(doc, args.output)
    complete = status == "COMPLETE"
    return 0 if complete and validity != "INVALID" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phase1_campaign",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--version", action="version", version=f"inferswarm-phase1-campaign {CAMPAIGN_RUNNER_VERSION}"
    )
    sub = p.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="emit the complete two-session plan (no model start)")
    _add_common(plan)
    plan.set_defaults(func=cmd_plan)

    validate = sub.add_parser(
        "validate", help="dry-run validation without starting any model"
    )
    _add_common(validate)
    validate.set_defaults(func=cmd_validate)

    run = sub.add_parser(
        "run-session", help="execute one session (P6 surface; preflight-gated)"
    )
    _add_common(run)
    run.add_argument("--session", type=int, required=True, choices=(1, 2))
    run.add_argument(
        "--thermal-reset-attested",
        default=None,
        help=(
            "operator attestation that an independently cooled thermal reset was "
            "observed before this session; REQUIRED for session 2"
        ),
    )
    run.set_defaults(func=cmd_run_session)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return args.func(args)
    except (CampaignRefused, ValueError) as e:
        print(f"[phase1] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
