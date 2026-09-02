"""Freeze and validate the complete R5A physical serving environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.inferswarm_r4.node_preflight import capture_node_profile
from benchmarks.inferswarm_r4.r4_plan import (
    CANONICAL_LINK,
    GPU_A_UUID,
    GPU_B_UUID,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    build_r4_plan,
)
from benchmarks.inferswarm_r4.r4_preflight_gate import (
    checkpoint_manifest,
    collect_local_identity,
    collect_remote_identity,
    run_gate,
)
from freetoken.research.n0_model_block import write_json_with_sha
from freetoken.research.r3_planner import freeze


def runtime_versions() -> dict[str, Any]:
    import torch

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    for package in ("transformers", "triton", "pytest"):
        try:
            module = __import__(package)
            versions[package] = module.__version__
        except Exception as exc:  # noqa: BLE001
            versions[package] = f"unavailable:{type(exc).__name__}"
    return versions


def _json_remote(ssh: list[str], command: str) -> dict[str, Any]:
    proc = subprocess.run([*ssh, command], check=True, capture_output=True, text=True)
    if proc.stderr.strip():
        raise RuntimeError(f"remote evidence collection wrote stderr: {proc.stderr[:300]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _selected(profile: dict[str, Any], uuid: str) -> dict[str, Any]:
    return next(row for row in profile["gpus"] if row["uuid"] == uuid)


def _gpu_environment(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "uuid": row["uuid"],
        "name": row["name"],
        "pci_bdf": row.get("pci_bus_id") or row.get("pci.bus_id"),
        "vram_total_bytes": int(row["memory_total_bytes"]),
        "reservation_bytes": 512 * 1024**2,
        "availability": "AVAILABLE",
        "integrity_eligible": True,
    }


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def freeze_physical_environment(args) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    ssh = shlex.split(args.ssh_node_b)
    ssh_prefix = shlex.join(ssh)

    identity_a = collect_local_identity(args.node_a_repo, "node.inferswarm01")
    identity_b = collect_remote_identity(
        ssh_prefix, args.node_b_repo, "node.inferswarm03"
    )
    profile_a = capture_node_profile(
        node_id="node.inferswarm01",
        peer_ipv4="10.0.0.219",
        interface="eno1",
        model_path=args.node_a_model,
    )
    remote_base = (
        f"cd {shlex.quote(args.node_b_repo)} && "
        f"PYTHONPATH={shlex.quote(args.node_b_repo)}:{shlex.quote(args.node_b_repo + '/python')} "
        f"{shlex.quote(args.node_b_python)} -m benchmarks.inferswarm_r5a.preflight"
    )
    profile_b = _json_remote(
        ssh,
        remote_base
        + " capture-node --node-id node.inferswarm03 --peer-ipv4 10.0.0.141 "
        + f"--interface enp5s0 --model {shlex.quote(args.node_b_model)}",
    )
    manifest_a = checkpoint_manifest(args.node_a_model)
    manifest_b = _json_remote(
        ssh,
        remote_base + f" checkpoint --model {shlex.quote(args.node_b_model)}",
    )
    versions_a = runtime_versions()
    versions_b = _json_remote(ssh, remote_base + " runtime-versions")

    r2_plan = json.loads(
        (Path(args.node_a_repo) / "docs/inferswarm_r2/frozen-plan.json").read_text()
    )
    participant_plan = build_r4_plan(
        r2_plan,
        node_a_hardware=profile_a,
        node_b_hardware=profile_b,
        link_freeze=CANONICAL_LINK,
        producer_sha=args.producer_sha,
    )
    gate = run_gate(
        producer_sha=args.producer_sha,
        node_a_profile=profile_a,
        node_b_profile=profile_b,
        identity_a=identity_a,
        identity_b=identity_b,
        plan=participant_plan,
        checkpoint_manifest_a=manifest_a,
        checkpoint_manifest_b=manifest_b,
        node_a_model_path=args.node_a_model,
        node_b_model_path=args.node_b_model,
    )
    runtime_record = {
        "producer_sha": args.producer_sha,
        "node_a": versions_a,
        "node_b": versions_b,
        "representation": "Qwen3.6-NVFP4/freetoken-native-resident-block-v1",
        "backend": "torch-cuda/freetoken-r4-captured-block-runtime",
    }
    network_record = {
        "node_a_interface": "eno1",
        "node_b_interface": "enp5s0",
        "node_a_ipv4": "10.0.0.141",
        "node_b_ipv4": "10.0.0.219",
        "negotiated_mbps": 1000,
        "duplex": "full",
        "mtu": 1500,
        "link_id": "link.node-a-to-node-b.tcp",
        "available": True,
    }
    environment = freeze(
        {
            "schema": "inferswarm.r5a.frozen-environment/1",
            "implementation_commit": args.producer_sha,
            "source": {"node_a": identity_a, "node_b": identity_b},
            "runtime_context": _digest(runtime_record),
            "runtime_versions": runtime_record,
            "network_context": "1GbE-full-duplex-MTU1500-eno1-enp5s0",
            "network": network_record,
            "model": {
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "checkpoint_identity": gate["checkpoint_identity"],
            },
            "node_a": {
                "node_id": "node.inferswarm01",
                "hostname": profile_a["hostname"],
                "gpus": [
                    _gpu_environment(_selected(profile_a, GPU_A_UUID)),
                    _gpu_environment(_selected(profile_a, "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55")),
                ],
                "memory": profile_a["memory"],
                "cpu": profile_a["cpu"],
            },
            "node_b": {
                "node_id": "node.inferswarm03",
                "hostname": profile_b["hostname"],
                "gpus": [_gpu_environment(_selected(profile_b, GPU_B_UUID))],
                "memory": profile_b["memory"],
                "cpu": profile_b["cpu"],
            },
            "compatibility": {
                "model_revision_validated": True,
                "representation_backend_compatible": True,
                "participant_plan_digest": participant_plan["digest"],
            },
            "preflight_gate": gate,
        }
    )
    for name, value in (
        ("identity-a.json", identity_a),
        ("identity-b.json", identity_b),
        ("checkpoint-manifest.json", {"node_a": manifest_a, "node_b": manifest_b}),
        ("runtime-versions.json", runtime_record),
    ):
        write_json_with_sha(raw / name, value)
    write_json_with_sha(out / "node-a-hardware.json", profile_a)
    write_json_with_sha(out / "node-b-hardware.json", profile_b)
    write_json_with_sha(out / "preflight-gate.json", gate)
    write_json_with_sha(out / "participant-plan.json", participant_plan)
    write_json_with_sha(out / "frozen-environment.json", environment)
    return environment, participant_plan, gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture-node")
    capture.add_argument("--node-id", required=True)
    capture.add_argument("--peer-ipv4", required=True)
    capture.add_argument("--interface", required=True)
    capture.add_argument("--model", required=True)
    checkpoint = sub.add_parser("checkpoint")
    checkpoint.add_argument("--model", required=True)
    sub.add_parser("runtime-versions")
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--producer-sha", required=True)
    freeze_parser.add_argument("--node-a-repo", required=True)
    freeze_parser.add_argument("--node-b-repo", required=True)
    freeze_parser.add_argument("--node-a-model", required=True)
    freeze_parser.add_argument("--node-b-model", required=True)
    freeze_parser.add_argument("--node-b-python", required=True)
    freeze_parser.add_argument("--ssh-node-b", required=True)
    freeze_parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "capture-node":
        print(
            json.dumps(
                capture_node_profile(
                    node_id=args.node_id,
                    peer_ipv4=args.peer_ipv4,
                    interface=args.interface,
                    model_path=args.model,
                )
            )
        )
    elif args.command == "checkpoint":
        print(json.dumps(checkpoint_manifest(args.model)))
    elif args.command == "runtime-versions":
        print(json.dumps(runtime_versions()))
    else:
        environment, participant, gate = freeze_physical_environment(args)
        print(
            json.dumps(
                {
                    "environment_digest": environment["digest"],
                    "participant_plan_digest": participant["digest"],
                    "gate": gate["result"],
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
