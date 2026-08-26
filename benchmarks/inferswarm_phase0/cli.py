"""``python benchmarks/phase0_baseline.py`` -- the Phase-0 baseline harness entry point.

Subcommands:

    sweep       run the B1-B5 performance sweep (criteria section 2.1)
    reference   run CORRECTNESS_REFERENCE (criteria section 2.4) and record its outputs
    profile     capture the hardware profile of the selected GPU
    hash        print the sha256 of a fixture, for freezing it into a manifest

``sweep`` and ``reference`` are separate subcommands on purpose. They answer different
questions and are never conflated: the sweep produces the input to selecting
``CANONICAL_PERFORMANCE_BASELINE`` (what InferSwarm must beat), while the reference is a
fixed configuration used only to decide whether a later distributed candidate computes the
intended result. The reference is never chosen by speed, and no ratio is ever computed
against it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

from . import HARNESS_VERSION
from .baselines import BASELINE_ARMS, BASELINE_ARMS_BY_ID, correctness_reference_arm
from .manifest import ManifestError, load_manifest, sha256_text
from .protocol import build_protocol
from .runner import Campaign, ServeSettings
from . import provenance as prov


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="local checkpoint directory (or .ftw)")
    p.add_argument(
        "--model-repository",
        default="nvidia/Qwen3.6-35B-A3B-NVFP4",
        help="upstream model repository (criteria section 1.1)",
    )
    p.add_argument(
        "--model-revision",
        default=None,
        help=(
            "EXACT upstream commit SHA of the checkpoint (40 hex). Required for a canonical "
            "run; branch names and 'main' are rejected. Never invent one."
        ),
    )
    p.add_argument(
        "--gpu",
        default=None,
        help=(
            "the physical GPU to serve on, as `ft serve --gpu`: a stable GPU UUID "
            "(preferred -- nvidia-smi indices move between boots) or an index. REQUIRED for "
            "a canonical run, and resolved to a UUID before anything starts so the server, "
            "`ft bench bw` and the microbenchmarks provably touch the same card"
        ),
    )
    p.add_argument("--manifest", required=True, help="frozen workload manifest (JSON)")
    p.add_argument("--out-root", default="phase0-runs", help="where run directories are created")
    p.add_argument("--short-name", default="rtx3060-baseline", help="run directory suffix")
    p.add_argument(
        "--session-id",
        default="session-1",
        help=(
            "distinct id per campaign session; criteria section 10 wants a second session on "
            "a different day and thermal state, with the order reversed"
        ),
    )
    p.add_argument(
        "--reverse-order",
        action="store_true",
        help="traverse arms and classes in reverse (the session-2 ordering)",
    )
    p.add_argument("--inferswarm-commit", default=None, help="InferSwarm commit this run belongs to")
    p.add_argument("--memory-ratio", type=float, default=0.9)
    p.add_argument("--kv-reserve-tokens", type=int, default=None)
    p.add_argument("--max-seq-len-override", type=int, default=None)
    p.add_argument("--server-timeout", type=float, default=1800.0)
    p.add_argument("--warmups", type=int, default=None, help="NON-CANONICAL override; needs --dev-smoke")
    p.add_argument("--repetitions", type=int, default=None, help="NON-CANONICAL override; needs --dev-smoke")
    p.add_argument(
        "--dev-smoke",
        action="store_true",
        help=(
            "developer smoke test: allows protocol overrides and a non-canonical manifest, "
            "and stamps the run NON-CANONICAL everywhere it is recorded"
        ),
    )
    p.add_argument(
        "--allow-missing-provenance",
        action="store_true",
        help="NON-CANONICAL: proceed even though required provenance could not be captured",
    )
    p.add_argument("--store-output-text", action="store_true", help="keep full generated text per rep")
    p.add_argument("--no-echo-server", action="store_true", help="do not mirror server output")
    p.add_argument("--dry-run", action="store_true", help="print the plan as JSON and exit")


def _settings(args: argparse.Namespace) -> ServeSettings:
    return ServeSettings(
        model_path=args.model,
        model_repository=args.model_repository,
        model_revision=args.model_revision,
        gpu=args.gpu,
        memory_ratio=args.memory_ratio,
        kv_reserve_tokens=args.kv_reserve_tokens,
        max_seq_len_override=args.max_seq_len_override,
        server_timeout=args.server_timeout,
    )


def _campaign(args: argparse.Namespace, arms) -> Campaign:
    canonical = not (args.dev_smoke or args.allow_missing_provenance)
    manifest = load_manifest(args.manifest, canonical=canonical)
    protocol = build_protocol(
        warmups=args.warmups,
        repetitions=args.repetitions,
        session_id=args.session_id,
        reverse_order=args.reverse_order,
        dev_smoke=args.dev_smoke,
    )
    prov.validate_revision(args.model_revision, canonical=canonical)
    prov.validate_inferswarm_commit(args.inferswarm_commit, canonical=canonical)
    if canonical and not args.gpu:
        raise ValueError(
            "--gpu is required for a canonical run: criteria section 2.1 fixes Phase 0 to "
            "ONE physical RTX 3060 -- the same card the Phase-1 candidate later uses as GPU "
            "0 -- and a run that lets the runtime choose cannot name which card it measured. "
            "Pass the stable GPU UUID from `nvidia-smi -L`."
        )
    return Campaign(
        arms=arms,
        manifest=manifest,
        protocol=protocol,
        settings=_settings(args),
        out_root=Path(args.out_root),
        short_name=args.short_name,
        inferswarm_commit=args.inferswarm_commit,
        canonical=canonical,
        store_output_text=args.store_output_text,
        echo_server_output=not args.no_echo_server,
    )


def _emit_dry_run(campaign: Campaign) -> int:
    doc = campaign.dry_run_document()
    refusals = doc["preflight_refusals"]
    if refusals:
        banner = "CANONICAL Phase-0 protocol - WOULD BE REFUSED, see below"
    elif doc["canonical"]:
        banner = "CANONICAL Phase-0 protocol"
    else:
        banner = "NON-CANONICAL developer smoke test - NOT a Phase-0 baseline"
    print(f"=== {banner} ===", file=sys.stderr)
    for blocker in doc["canonical_blockers"]:
        print(f"  - {blocker}", file=sys.stderr)
    for refusal in refusals:
        print(f"  ! this run would be refused: {refusal}", file=sys.stderr)
    print(json.dumps(doc, indent=2))
    return 0


def _exit_code(doc: dict) -> int:
    """0 only for a run that is both complete and not an invalid canonical attempt.

    An INVALID campaign exits non-zero even when every generation returned: the artifact is
    real and is kept, but nothing downstream should treat it as a baseline.
    """
    complete = doc.get("execution_status") == "COMPLETE"
    return 0 if complete and doc.get("validity") != "INVALID" else 1


def _report(doc: dict) -> None:
    print(f"\n[phase0] {doc['headline']}: {doc['run_directory']}", flush=True)
    for item in doc.get("campaign_invalidations") or []:
        where = "/".join(str(item[k]) for k in ("arm_id", "class_id") if item.get(k)) or "campaign"
        print(f"[phase0]   INVALIDATING {item['code']} ({where}): {item['message']}", flush=True)


def cmd_sweep(args: argparse.Namespace) -> int:
    ids = [a.strip() for a in args.arms.split(",") if a.strip()] if args.arms else [a.id for a in BASELINE_ARMS]
    unknown = [i for i in ids if i not in BASELINE_ARMS_BY_ID]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {sorted(BASELINE_ARMS_BY_ID)}")
    arms = [BASELINE_ARMS_BY_ID[i] for i in ids]
    campaign = _campaign(args, arms)
    if len(arms) < len(BASELINE_ARMS) and campaign.canonical:
        raise SystemExit(
            "a canonical sweep runs all of B1-B5 (criteria section 2.2 selects the winner "
            "from the full sweep); pass --dev-smoke to run a subset as a smoke test"
        )
    if args.no_bench_bw and campaign.canonical:
        raise SystemExit(
            "--no-bench-bw is refused for a canonical sweep. The `ft bench bw` refresh is a "
            "session-level prerequisite, not a B2 convenience: B2 resolves its per-step "
            "fetch split from the profile AND B3's `--moe-backend auto` reads the same "
            "profile to decide whether to upgrade offload to hybrid. Skipping it would let "
            "both arms resolve from a profile this campaign did not produce. Pass "
            "--dev-smoke to run a clearly-marked NON-CANONICAL developer smoke test."
        )
    campaign.refresh_bench_bw = not args.no_bench_bw
    campaign.bench_bw_dtype = args.bench_bw_dtype
    if args.dry_run:
        return _emit_dry_run(campaign)
    doc = campaign.execute()
    _report(doc)
    return _exit_code(doc)


def cmd_reference(args: argparse.Namespace) -> int:
    arm = correctness_reference_arm(args.nvfp4_backend, args.moe_cache_size)
    campaign = _campaign(args, [arm])
    # The reference is captured for reproducibility, so its text is always stored, and it is
    # run twice by default: criteria section 5.3 requires two independent reference runs of
    # every fixture to produce identical token sequences BEFORE any candidate is compared.
    #
    # Its request sampling is FORCED greedy (temperature 0.0 / top_p 1.0 / top_k -1) by the
    # runner, on the same frozen prompt fixture the sweep uses. `--sampling-defaults none`
    # alone is not enough: the manifest states sampling explicitly in every request body, and
    # a request-level value beats a server default -- so a manifest whose frozen performance
    # sampling is deliberately realistic would otherwise make the correctness reference
    # sampled, and FreeToken exposes no seed to make that reproducible (criteria section 5.3).
    campaign.store_output_text = True
    if args.dry_run:
        return _emit_dry_run(campaign)
    doc = campaign.execute()
    _report(doc)
    print(
        "[phase0] This is CORRECTNESS_REFERENCE material, not a performance baseline. "
        "Its self-consistency check (criteria section 5.3) compares two independent runs of "
        "this configuration -- run it again with a different --session-id and diff the "
        "recorded output_sha256 per class.",
        flush=True,
    )
    print(
        "[phase0] Note: identical HTTP text hashes are NOT the deeper C1/C2/C3 Phase-1 "
        "correctness instrumentation (per-layer outputs, router selections, step-0 logits). "
        "They are the reproducible fixture those gates will later be applied to.",
        flush=True,
    )
    return _exit_code(doc)


def cmd_profile(args: argparse.Namespace) -> int:
    from .hardware_profile import capture_profile

    doc = capture_profile(
        gpu=args.gpu,
        run_bench_bw=not args.no_bench_bw,
        dtype=args.dtype,
        expert_microbench=args.expert_microbench,
        hidden=args.hidden,
        intermediate=args.intermediate,
        top_k=args.top_k,
        include_grouped=not args.no_grouped_topk,
        device_bandwidth=args.device_bandwidth,
        device_bandwidth_bytes=args.device_bandwidth_mib << 20,
        device_bandwidth_reps=args.device_bandwidth_reps,
    )
    out = json.dumps(doc, indent=2)
    if args.out:
        Path(args.out).write_text(out + "\n")
        print(f"[phase0] wrote {args.out}", file=sys.stderr)
    else:
        print(out)
    return 0


def cmd_hash(args: argparse.Namespace) -> int:
    for path in args.paths:
        text = Path(path).read_bytes().decode("utf-8")
        print(f"{sha256_text(text)}  {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phase0_baseline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"inferswarm-phase0 {HARNESS_VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("sweep", help="the B1-B5 baseline sweep")
    _add_common(sweep)
    sweep.add_argument("--arms", default=None, help="comma list (default: all of B1-B5)")
    sweep.add_argument(
        "--no-bench-bw",
        action="store_true",
        help=(
            "skip the session-level `ft bench bw` refresh. REFUSED for a canonical sweep "
            "(B2 and B3 both read the profile); usable only with --dev-smoke"
        ),
    )
    sweep.add_argument(
        "--bench-bw-dtype",
        default="nvfp4",
        help="expert format the `ft bench bw` refresh benches (criteria section 2.1: this GPU + format)",
    )
    sweep.set_defaults(func=cmd_sweep)

    ref = sub.add_parser("reference", help="CORRECTNESS_REFERENCE capture")
    _add_common(ref)
    ref.add_argument(
        "--nvfp4-backend",
        required=True,
        choices=["marlin", "flashinfer", "triton"],
        help="the RESOLVED backend the candidate's remote expert GEMM uses (no 'auto')",
    )
    ref.add_argument(
        "--moe-cache-size",
        type=int,
        required=True,
        help="fixed cache slots: >= num_experts, and <= 992 under marlin",
    )
    ref.set_defaults(func=cmd_reference)

    profile = sub.add_parser("profile", help="hardware profile of the selected GPU")
    profile.add_argument("--gpu", default=None, help="GPU UUID or nvidia-smi index")
    profile.add_argument("--out", default=None, help="write JSON here instead of stdout")
    profile.add_argument("--dtype", default="nvfp4", help="expert format for `ft bench bw`")
    profile.add_argument("--no-bench-bw", action="store_true", help="skip `ft bench bw`")
    profile.add_argument(
        "--expert-microbench",
        action="store_true",
        help=(
            "also run the TRUE single-expert NVFP4 GEMV latency microbenchmark (top_k=1), "
            "plus a separately-labelled grouped top-k step diagnostic (diagnostic only)"
        ),
    )
    profile.add_argument("--hidden", type=int, default=2048, help="microbench hidden size H")
    profile.add_argument("--intermediate", type=int, default=512, help="microbench MoE intermediate I")
    profile.add_argument(
        "--top-k", type=int, default=8,
        help="routed experts per token for the GROUPED diagnostic; the single-expert "
             "measurement is always top_k=1",
    )
    profile.add_argument(
        "--no-grouped-topk", action="store_true",
        help="skip the grouped top-k step diagnostic and measure only the single expert",
    )
    profile.add_argument(
        "--device-bandwidth", action="store_true",
        help=(
            "also measure the selected GPU's device (VRAM) memory bandwidth -- issue #2's "
            "'memory bandwidth', which `ft bench bw` does NOT measure (it measures host DRAM "
            "and the PCIe link)"
        ),
    )
    profile.add_argument(
        "--device-bandwidth-mib", type=int, default=512,
        help="device-bandwidth working set per buffer, in MiB (must exceed L2 by a wide margin)",
    )
    profile.add_argument(
        "--device-bandwidth-reps", type=int, default=30,
        help="device-bandwidth measured repetitions (all are reported individually)",
    )
    profile.set_defaults(func=cmd_profile)

    h = sub.add_parser("hash", help="sha256 of fixture file(s), for freezing into a manifest")
    h.add_argument("paths", nargs="+")
    h.set_defaults(func=cmd_hash)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return args.func(args)
    except (ManifestError, ValueError) as e:
        print(f"[phase0] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
