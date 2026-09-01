"""CLI for the mechanical fail-closed R4 preflight gate (runs on Node A).

Composes producer identity collection (local + SSH peer), hardware profile
capture on both Nodes, checkpoint manifests, and the full gate from
``r4_preflight_gate`` into one canonical preflight phase.  Exits non-zero
BEFORE any model realization when any frozen prerequisite drifts.  Retains:
node hardware profiles, identity proofs, checkpoint manifest, and the gate
record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

from benchmarks.inferswarm_r4.node_preflight import capture_node_profile
from benchmarks.inferswarm_r4.r4_preflight_gate import (
    checkpoint_manifest,
    collect_local_identity,
    collect_remote_identity,
    run_gate,
)
from freetoken.research.n0_model_block import write_json_with_sha

NODE_A_CONFIG = {
    "node_id": "node.inferswarm01",
    "peer_ipv4": "10.0.0.219",
    "interface": "eno1",
}
NODE_B_CONFIG = {
    "node_id": "node.inferswarm03",
    "peer_ipv4": "10.0.0.141",
    "interface": "enp5s0",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-sha", required=True)
    parser.add_argument("--node-a-repo", required=True)
    parser.add_argument("--node-b-repo", required=True)
    parser.add_argument("--ssh-node-b", required=True,
                        help="complete ssh prefix for Node B, quoted")
    parser.add_argument("--node-a-model", required=True)
    parser.add_argument("--node-b-model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)

    # 1. producer identity, both nodes, fail-closed collection
    identity_a = collect_local_identity(args.node_a_repo, NODE_A_CONFIG["node_id"])
    identity_b = collect_remote_identity(
        args.ssh_node_b, args.node_b_repo, NODE_B_CONFIG["node_id"]
    )
    write_json_with_sha(raw / "identity-a.json", identity_a)
    write_json_with_sha(raw / "identity-b.json", identity_b)

    # 2. hardware profiles (fresh capture, includes DMI RAM, CPU topology,
    #    normalized BDF keys, and the canonical link fields the gate checks)
    profile_a = capture_node_profile(
        node_id=NODE_A_CONFIG["node_id"],
        peer_ipv4=NODE_A_CONFIG["peer_ipv4"],
        interface=NODE_A_CONFIG["interface"],
        model_path=args.node_a_model,
    )
    write_json_with_sha(out / "node-a-hardware.json", profile_a)
    remote_script = (
        "cd /home/zutfen/FreeToken-r4 && TMPDIR=/var/tmp "
        "PYTHONPATH=/home/zutfen/FreeToken-r4/python "
        "/home/zutfen/FreeToken/.venv/bin/python -m "
        "benchmarks.inferswarm_r4.capture_profile --node node-b "
        "--out /tmp/r4-node-b-profile.json"
    )
    ssh_argv = shlex.split(args.ssh_node_b)
    subprocess.run(
        [*ssh_argv, remote_script],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        f"scp -q -i /home/zutfen/.ssh/id_r4_staging "
        f"zutfen@10.0.0.219:/tmp/r4-node-b-profile.json {out}/node-b-hardware.json",
        shell=True,
        check=True,
    )
    profile_b = json.loads((out / "node-b-hardware.json").read_text())
    (out / "node-b-hardware.json.sha256").write_text(
        f"{hashlib.sha256((out / 'node-b-hardware.json').read_bytes()).hexdigest()}"
        "  node-b-hardware.json\n"
    )

    # 3. checkpoint manifests (revision dirs already pinned by config)
    manifest_a = checkpoint_manifest(args.node_a_model)
    manifest_script = (
        "cd /home/zutfen/FreeToken-r4 && "
        "PYTHONPATH=/home/zutfen/FreeToken-r4/python "
        "/home/zutfen/FreeToken/.venv/bin/python -c '"
        "import json;from benchmarks.inferswarm_r4.r4_preflight_gate "
        "import checkpoint_manifest;"
        f'print(json.dumps(checkpoint_manifest("{args.node_b_model}")))\''
    )
    proc = subprocess.run(
        [*ssh_argv, manifest_script],
        capture_output=True,
        text=True,
        check=True,
    )
    manifest_b = json.loads(proc.stdout.strip().splitlines()[-1])
    write_json_with_sha(
        raw / "checkpoint-manifest.json",
        {
            "node_a": manifest_a,
            "node_b": manifest_b,
            "repository": "nvidia/Qwen3.6-35B-A3B-NVFP4",
            "revision": "491c2f1ea524c639598bf8fa787a93fed5a6fbce",
        },
    )

    # 4. the plan is needed for VRAM sizing; use the retained frozen plan if
    #    present, otherwise the R2-derived geometry constants
    plan_path = out / "r4-frozen-plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        plan = json.loads(
            (Path(__file__).resolve().parents[2] / "docs/inferswarm_r2/frozen-plan.json")
            .read_text()
        )
        # minimal shim: participant plans carry the materialization sizes
        plan = {"participant_r1_plans": plan.get("participant_r1_plans", {})}

    # 5. run the gate — any violation raises and this CLI exits non-zero
    gate = run_gate(
        producer_sha=args.producer_sha,
        node_a_profile=profile_a,
        node_b_profile=profile_b,
        identity_a=identity_a,
        identity_b=identity_b,
        plan=plan,
        checkpoint_manifest_a=manifest_a,
        checkpoint_manifest_b=manifest_b,
        node_a_model_path=args.node_a_model,
        node_b_model_path=args.node_b_model,
    )
    write_json_with_sha(out / "preflight-gate.json", gate)
    print(json.dumps({"gate": gate["result"], "producer": args.producer_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
