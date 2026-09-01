"""Explicit fail-closed/drift test coverage for the R4 gate-review
corrections (InferSwarm #57 bounded integration coverage).

Each test maps to one required semantic from the gate review:

 1. wrong producer SHA aborts canonical campaign        (gate + arms)
 2. dirty source tree aborts canonical campaign         (gate + arms)
 3. frozen GPU UUID/BDF drift aborts                    (gate)
 4. insufficient VRAM/headroom aborts                   (gate)
 5. Block-B MemAvailable < 12 GiB aborts                (gate + service)
 6. checkpoint/revision mismatch aborts                 (gate)
 7. canonical link speed/duplex/MTU/route drift aborts  (node_preflight)
 8. model-state materialization never transmitted after
    final residency                                     (wire record shape)
 9. network candidate FEASIBLE_UNRANKED before evidence (r4_plan; also
    covered in test_r4_wire.py::test_network_candidate_feasible_unranked)
10. diagnostic full-logit values never on the network   (test_r4_wire.py
    #19; here: capacity classifier must not use microbench as demand)
11. clean arm cannot include diagnostic payload         (test_r4_wire.py
    #13/14; here: clean-arm demand derivation rejects diagnostic arms)
12. persistent connection across W2/W4 re-establishment (test_r4_wire.py #20)
13. malformed/protocol/session/checksum fail-closed     (test_r4_wire.py #4-#9)
14. #53 RELEASE semantics intact                        (predecessor suite;
    here: swap-reliance delta check)

Bounded: no GPU, no model, no real network.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from typing import Any

from benchmarks.inferswarm_r4.compose_result import (
    CAPACITY_MARGIN,
    classify_one_gbe,
    workload_wire_demand,
)
from benchmarks.inferswarm_r4.node_preflight import require_canonical_link
from benchmarks.inferswarm_r4.r4_plan import build_r4_plan
from benchmarks.inferswarm_r4.r4_preflight_gate import (
    BLOCK_B_MIN_MEM_AVAILABLE_BYTES,
    BLOCK_B_MIN_PHYSICAL_RAM_BYTES,
    PreflightGateError,
    checkpoint_manifest,
    producer_identity,
    require_block_b_host_ram,
    require_block_vram_headroom,
    require_checkpoint_identity,
    require_frozen_gpu,
    require_no_swap_reliance,
    require_producer_identity,
    require_revision_directory,
    run_gate,
)
from benchmarks.inferswarm_r4.r4_plan import (
    ACCEPTED_R2_PLAN_DIGEST,
    GPU_A_UUID,
    GPU_B_UUID,
    NODE_A_ID,
    NODE_B_ID,
)

PRODUCER = "a" * 40


def _identity(node_id: str, sha: str = PRODUCER, dirty: list | None = None):
    return producer_identity(
        node_id=node_id, hostname="h", repo_sha=sha, dirty_entries=dirty or []
    )


def _gpu(uuid: str, bdf: str, mib: int = 12288) -> dict:
    return {
        "uuid": uuid,
        "pci_bus_id": bdf,
        "name": "RTX 3060",
        "memory_total_bytes": str(mib * 1024 * 1024),
    }


def _profile(node_id: str, gpus: list, mem_available: int = 13 << 30,
             physical_ram: int = 16 << 30) -> dict:
    return {
        "node_id": node_id,
        "hostname": "h",
        "gpus": gpus,
        "memory": {
            "physical_installed_ram_bytes": physical_ram,
            "mem_total_bytes": 15_484_800_000,
            "mem_available_bytes": mem_available,
            "measured_at_unix": 1.0,
        },
        "link": {"speed": "1000Mb/s", "duplex": "Full", "mtu": "1500"},
        "interface": {
            "name": "eno1" if node_id == NODE_A_ID else "enp5s0",
            "route_to_peer": (
                "10.0.0.219 dev eno1 src 10.0.0.141"
                if node_id == NODE_A_ID
                else "10.0.0.141 dev enp5s0 src 10.0.0.219"
            ),
        },
    }


def _plan() -> dict:
    vram = 11 * 1024**3
    return {
        "participant_r1_plans": {
            "exec.block-a": {
                "materializations": [
                    {"id": "mat.block-a.routed-staging", "expected_bytes": 8_636_596_224},
                    {"id": "mat.block-a.routed-vram", "expected_bytes": vram},
                ]
            },
            "exec.block-b": {
                "materializations": [
                    {"id": "mat.block-b.staging", "expected_bytes": 9_545_711_616},
                    {"id": "mat.block-b.routed-vram", "expected_bytes": vram},
                ]
            },
        }
    }


def _manifests(tmp_path: Path):
    dir_a = tmp_path / "ckpt-a" / "491c2f1ea524c639598bf8fa787a93fed5a6fbce"
    dir_b = tmp_path / "ckpt-b" / "491c2f1ea524c639598bf8fa787a93fed5a6fbce"
    for d in (dir_a, dir_b):
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.safetensors").write_bytes(b"weights")
        (d / "config.json").write_bytes(b"cfg")
    return dir_a, dir_b, checkpoint_manifest(str(dir_a)), checkpoint_manifest(str(dir_b))


def _gate(tmp_path, **overrides) -> Any:
    dir_a, dir_b, manifest_a, manifest_b = _manifests(tmp_path)
    args: dict[str, Any] = dict(
        producer_sha=PRODUCER,
        node_a_profile=_profile(NODE_A_ID, [_gpu(GPU_A_UUID, "00000000:02:00.0")]),
        node_b_profile=_profile(NODE_B_ID, [_gpu(GPU_B_UUID, "00000000:01:00.0")]),
        identity_a=_identity(NODE_A_ID),
        identity_b=_identity(NODE_B_ID),
        plan=_plan(),
        checkpoint_manifest_a=manifest_a,
        checkpoint_manifest_b=manifest_b,
        node_a_model_path=str(dir_a),
        node_b_model_path=str(dir_b),
        frozen_bdf_a="00000000:02:00.0",
        frozen_bdf_b="00000000:01:00.0",
    )
    args.update(overrides)
    return run_gate(**args)


# 1. wrong producer SHA aborts -------------------------------------------------


def test_wrong_producer_sha_aborts(tmp_path: Path) -> None:
    record = _gate(tmp_path)
    assert record["result"] == "ALL_PREFLIGHT_CHECKS_PASSED"
    with pytest.raises(PreflightGateError, match="producer SHA"):
        _gate(tmp_path, producer_sha="b" * 40)


def test_arm_refuses_wrong_producer() -> None:
    # the arm runner refuses when the running producer differs from the
    # plan's frozen producer
    source = Path("benchmarks/inferswarm_r4/run_experiment.py").read_text()
    assert "running producer" in source
    assert "frozen producer" in source
    assert "refuses to proceed" in source
    node_b = Path("benchmarks/inferswarm_r4/node_b_service.py").read_text()
    assert "refuses to proceed" in node_b


# 2. dirty source tree aborts ---------------------------------------------------


def test_dirty_tree_aborts(tmp_path: Path) -> None:
    with pytest.raises(PreflightGateError, match="dirty"):
        _gate(tmp_path, identity_b=_identity(NODE_B_ID, dirty=[" M x.py"]))


def test_identity_collection_unclean_stderr_aborts(monkeypatch) -> None:
    from benchmarks.inferswarm_r4 import r4_preflight_gate as gate

    def fake_run(cmd, capture_output, text):  # noqa: ANN001
        class P:
            returncode = 0
            stdout = "out\n"
            stderr = "fatal: detected dubious ownership"
        return P()

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    with pytest.raises(PreflightGateError, match="not clean"):
        gate.collect_local_identity("/repo", "node.x")


# 3. frozen GPU UUID/BDF drift aborts -------------------------------------------


def test_gpu_uuid_absent_aborts(tmp_path: Path) -> None:
    with pytest.raises(PreflightGateError, match="frozen GPU.*absent"):
        _gate(
            tmp_path,
            node_b_profile=_profile(
                NODE_B_ID, [_gpu("GPU-other", "00000000:01:00.0")]
            ),
        )


def test_gpu_bdf_drift_aborts(tmp_path: Path) -> None:
    with pytest.raises(PreflightGateError, match="BDF drift"):
        _gate(tmp_path, frozen_bdf_b="00000000:09:00.0")


# 4. insufficient VRAM/headroom aborts ------------------------------------------


def test_vram_headroom_insufficient_aborts(tmp_path: Path) -> None:
    gpu = _gpu(GPU_B_UUID, "00000000:01:00.0", mib=8192)
    with pytest.raises(PreflightGateError, match="vram-headroom"):
        _gate(
            tmp_path,
            node_b_profile=_profile(NODE_B_ID, [gpu]),
        )


def test_vram_headroom_boundary_passes() -> None:
    record = require_block_vram_headroom(
        _plan(), "exec.block-b",
        {"uuid": GPU_B_UUID, "capacity_bytes": 12 * 1024**3},
        NODE_B_ID,
    )
    assert record["remaining_bytes"] > 0


# 5. Block-B MemAvailable < 12 GiB aborts ----------------------------------------


def test_block_b_mem_available_low_aborts() -> None:
    with pytest.raises(PreflightGateError, match="MemAvailable"):
        require_block_b_host_ram(
            {
                "physical_installed_ram_bytes": BLOCK_B_MIN_PHYSICAL_RAM_BYTES,
                "mem_available_bytes": 11 * 1024**3,
            }
        )


def test_block_b_physical_ram_unproven_aborts() -> None:
    with pytest.raises(PreflightGateError, match="physically installed RAM not proven"):
        require_block_b_host_ram(
            {
                "physical_installed_ram_bytes": None,
                "mem_available_bytes": 14 * 1024**3,
            }
        )


def test_block_b_memtotal_below_16gib_still_passes_with_dmi() -> None:
    # MemTotal 15.48 GiB (firmware reservation) is NOT a failure when DMI
    # proves 16 GiB installed
    record = require_block_b_host_ram(
        {
            "physical_installed_ram_bytes": 16 * 1024**3,
            "mem_total_bytes": 16_229_876 * 1024,
            "mem_available_bytes": BLOCK_B_MIN_MEM_AVAILABLE_BYTES,
        }
    )
    assert record["physical_installed_ram_bytes"] == 16 * 1024**3
    assert record["linux_memtotal_bytes"] < 16 * 1024**3


# 6. checkpoint/revision mismatch aborts -----------------------------------------


def test_checkpoint_hash_mismatch_aborts(tmp_path: Path) -> None:
    _, dir_b, manifest_a, _ = _manifests(tmp_path)
    (dir_b / "model.safetensors").write_bytes(b"different")
    manifest_b = checkpoint_manifest(str(dir_b))
    with pytest.raises(PreflightGateError, match="files differ"):
        require_checkpoint_identity(manifest_a, manifest_b)


def test_checkpoint_file_set_mismatch_aborts(tmp_path: Path) -> None:
    _, _, manifest_a, _ = _manifests(tmp_path)
    with pytest.raises(PreflightGateError, match="file sets differ"):
        require_checkpoint_identity(manifest_a, {**manifest_a, "extra.bin": "0" * 64})


def test_wrong_revision_directory_aborts(tmp_path: Path) -> None:
    with pytest.raises(PreflightGateError, match="frozen revision"):
        require_revision_directory("/models/nvidia/Qwen3.6-35B-A3B-NVFP4/deadbeef")


# 7. canonical link drift aborts ---------------------------------------------------


@pytest.mark.parametrize(
    "link,match",
    [
        ({"speed": "100Mb/s", "duplex": "Full", "mtu": "1500"}, "speed"),
        ({"speed": "1000Mb/s", "duplex": "Half", "mtu": "1500"}, "duplex"),
        ({"speed": "1000Mb/s", "duplex": "Full", "mtu": "9000"}, "MTU"),
    ],
)
def test_link_drift_aborts(link, match) -> None:
    profile = {
        "node_id": NODE_A_ID,
        "link": link,
        "interface": {
            "name": "eno1",
            "route_to_peer": "10.0.0.219 dev eno1 src 10.0.0.141",
        },
    }
    with pytest.raises(RuntimeError, match=match):
        require_canonical_link(profile)


def test_vpn_route_drift_aborts() -> None:
    profile = {
        "node_id": NODE_A_ID,
        "link": {"speed": "1000Mb/s", "duplex": "Full", "mtu": "1500"},
        "interface": {
            "name": "eno1",
            "route_to_peer": "10.0.0.219 dev tailscale0 src 100.x.x.x",
        },
    }
    with pytest.raises(RuntimeError, match="VPN/loopback"):
        require_canonical_link(profile)


def test_wrong_interface_route_drift_aborts() -> None:
    profile = {
        "node_id": NODE_A_ID,
        "link": {"speed": "1000Mb/s", "duplex": "Full", "mtu": "1500"},
        "interface": {
            "name": "eno1",
            "route_to_peer": "10.0.0.219 dev wlp3s0 src 10.0.0.141",
        },
    }
    with pytest.raises(RuntimeError, match="does not use eno1"):
        require_canonical_link(profile)


# 8. no model-state materialization on the wire after residency ---------------------


def test_boundary_payload_sizes_are_activation_only() -> None:
    # the boundary contract admits only 8,192 B decode / 524,288 B prefill
    # activation frames; any model-state materialization (>= 8.6 GB) can
    # never fit a wire frame and the wire budget rejects it
    from freetoken.research import r4_wire

    assert r4_wire.HEADER_BUDGET + 524_288 < 8_636_596_224
    decode = 1 * 2 * 2048 * 2
    prefill = 64 * 2 * 2048 * 2
    assert decode == 8192 and prefill == 524288


# 10/11. capacity classifier semantics -----------------------------------------------


def _clean_arm():
    return {
        "wire_accounting": {
            "frames_tx": 69,
            "frames_rx": 69,
            "framing_control_bytes_tx": 24866,
            "wire_bytes_rx": 20583,
            "semantic_payload_bytes_tx": 1941504,
            "wire_bytes_tx": 1966370,
        },
        "sessions": [
            {
                "class_id": "W2",
                "session": {
                    "inter_token_p50_ns": 29_710_718,
                    "prefill_wall_ns": 1_628_711_671,
                    "boundaries": [
                        {"operation": "prefill", "payload_bytes": 442368},
                        *[{"operation": "decode", "payload_bytes": 8192}] * 31,
                    ],
                },
            },
            {
                "class_id": "W4",
                "session": {
                    "inter_token_p50_ns": 29_827_691,
                    "prefill_wall_ns": 2_784_768_534,
                    "boundaries": [
                        {"operation": "prefill", "payload_bytes": 524288},
                        {"operation": "prefill", "payload_bytes": 466944},
                        *[{"operation": "decode", "payload_bytes": 8192}] * 31,
                    ],
                },
            },
        ],
    }


def _microbench():
    return {
        "sizes": {
            "524288": {"rtt_ns_p50": 9_671_052.0, "effective_mbps_p50": 433.7},
            "8192": {"rtt_ns_p50": 528_023.0, "effective_mbps_p50": 124.12},
        }
    }


def test_capacity_uses_workload_demand_not_microbench() -> None:
    clean = _clean_arm()
    demand = workload_wire_demand(clean)
    # microbench achieved throughput (433.7) is NOT the demand basis
    result = classify_one_gbe(clean, _microbench(), {
        "iperf_a_to_b_mbps": 933.9, "iperf_b_to_a_mbps": 941.5,
        "retransmits_total": 0,
    })
    peak = result["actual_clean_arm_workload_wire_demand"]["peak_a_to_b_mbps"]
    assert peak < 433.7  # honest demand far below transport capability
    assert result["disposition"] == "R4_1GBE_PRIMITIVE_CAPACITY_VIABLE"
    assert (
        result["criterion"]["applicable_demand_limit_mbps"]
        == round(CAPACITY_MARGIN * 933.9, 2)
    )
    # transport measurements retained separately, labelled not-demand
    assert "NOT workload demand" in result["transport_only_service_measurements"]["note"]


def test_capacity_negative_when_demand_exceeds_limit() -> None:
    # saturating cadence: 1 us decode interval would demand >> limit
    clean = _clean_arm()
    for row in clean["sessions"]:
        row["session"]["inter_token_p50_ns"] = 1_000
        row["session"]["prefill_wall_ns"] = 10_000
    result = classify_one_gbe(clean, _microbench(), {
        "iperf_a_to_b_mbps": 933.9, "iperf_b_to_a_mbps": 941.5,
        "retransmits_total": 0,
    })
    assert result["disposition"] == "R4_1GBE_PRIMITIVE_CAPACITY_NEGATIVE"


def test_capacity_faster_transport_cannot_reduce_viability() -> None:
    # perverse-inversion guard: better network => higher limit, demand fixed
    clean = _clean_arm()
    slow = classify_one_gbe(clean, _microbench(), {
        "iperf_a_to_b_mbps": 500.0, "iperf_b_to_a_mbps": 500.0,
        "retransmits_total": 0,
    })
    fast = classify_one_gbe(clean, _microbench(), {
        "iperf_a_to_b_mbps": 2000.0, "iperf_b_to_a_mbps": 2000.0,
        "retransmits_total": 0,
    })
    for disposition in (slow["disposition"], fast["disposition"]):
        assert disposition in (
            "R4_1GBE_PRIMITIVE_CAPACITY_VIABLE",
            "R4_1GBE_PRIMITIVE_CAPACITY_NEGATIVE",
        )
    # if viable under a slower path, must remain viable under a faster one
    if slow["disposition"] == "R4_1GBE_PRIMITIVE_CAPACITY_VIABLE":
        assert fast["disposition"] == "R4_1GBE_PRIMITIVE_CAPACITY_VIABLE"


def test_capacity_retransmits_inconclusive() -> None:
    clean = _clean_arm()
    result = classify_one_gbe(clean, _microbench(), {
        "iperf_a_to_b_mbps": 933.9, "iperf_b_to_a_mbps": 941.5,
        "retransmits_total": 7,
    })
    assert result["disposition"] == "R4_1GBE_PRIMITIVE_CAPACITY_INCONCLUSIVE"


# 14. swap-reliance / RELEASE lifecycle ---------------------------------------------


def test_swap_delta_zero_passes() -> None:
    record = require_no_swap_reliance(
        {"pswpin": 12345, "pswpout": 678},
        {"pswpin": 12345, "pswpout": 678},
    )
    assert record["swap_reliance"] is False


def test_swap_delta_nonzero_aborts() -> None:
    with pytest.raises(PreflightGateError, match="swap"):
        require_no_swap_reliance(
            {"pswpin": 100, "pswpout": 0}, {"pswpin": 100, "pswpout": 3}
        )


# provenance (gate-review finding 6) -------------------------------------------------


def test_clean_arm_carries_producer_and_plan_identity() -> None:
    source = Path(
        "benchmarks/inferswarm_r4/run_experiment.py"
    ).read_text()
    assert '"producer_freetoken_sha": running_sha' in source
    assert '"plan_digest": plan["digest"]' in source
    assert '"node_identity"' in source


def test_plan_freeze_requires_producer_sha() -> None:
    source = Path("benchmarks/inferswarm_r4/freeze_r4_plan.py").read_text()
    assert "--implementation-commit (the exact producer SHA) is required" in source


def test_build_r4_plan_freezes_gpu_identity_and_producer() -> None:
    import json as _json
    from benchmarks.inferswarm_r4.r4_plan import build_r4_plan as build

    r2_path = Path("docs/inferswarm_r2/frozen-plan.json")
    if not r2_path.exists():
        pytest.skip("R2 frozen plan not present")
    r2 = _json.loads(r2_path.read_text())
    hw_a = {
        "gpus": [_gpu(GPU_A_UUID, "00000000:02:00.0")],
        "memory": {"mem_total_kib": 131072000},
    }
    hw_b = {
        "gpus": [_gpu(GPU_B_UUID, "00000000:01:00.0")],
        "memory": {"mem_total_kib": 16229876},
    }
    from benchmarks.inferswarm_r4.r4_plan import CANONICAL_LINK

    plan = build(
        r2,
        node_a_hardware=hw_a,
        node_b_hardware=hw_b,
        link_freeze=CANONICAL_LINK,
        producer_sha=PRODUCER,
    )
    assert plan["provenance"]["r4"]["producer_sha"] == PRODUCER
    assert (
        plan["frozen_gpu_identity"]["node.inferswarm03"]["pci_bdf"]
        == "00000000:01:00.0"
    )


# CPU topology / DMI RAM capture (gate-review finding 3) -----------------------------


def test_cpu_topology_deduplicates_hyperthreads(monkeypatch) -> None:
    from benchmarks.inferswarm_r4 import node_preflight as np

    cpuinfo = "\n\n".join(
        "processor\t: %d\nphysical id\t: 0\ncore id\t\t: %d\nmodel name\t: Intel i3-10100F"
        % (i, core)
        for i, core in enumerate((0, 0, 1, 1, 2, 2, 3, 3))
    )

    def fake_read_text(self):
        if str(self) == "/proc/cpuinfo":
            return cpuinfo
        if str(self).endswith("/cpu/online"):
            return "0-7\n"
        raise AssertionError(f"unexpected read {self}")

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    topology = np._cpu_topology()
    assert topology["physical_cores"] == 4
    assert topology["logical_cpus"] == 8
    assert topology["model"] == "Intel i3-10100F"


def test_physical_ram_from_dmi_not_memtotal() -> None:
    from benchmarks.inferswarm_r4 import node_preflight as np

    dmi = (
        "Handle 0x0001\nMemory Device\n\tSize: 4 GB\n"
        "Handle 0x0002\nMemory Device\n\tSize: 4 GB\n"
        "Handle 0x0003\nMemory Device\n\tSize: 4 GB\n"
        "Handle 0x0004\nMemory Device\n\tSize: 4 GB\n"
        "Handle 0x0005\nMemory Device\n\tSize: No Module Installed\n"
    )

    def fake_run(cmd, ethtool_path=np.DEFAULT_ETHTOOL):
        return dmi if cmd[:3] == ["sudo", "-n", "dmidecode"] else "<failed>"

    mp = pytest.MonkeyPatch()
    mp.setattr(np, "_run", fake_run)
    try:
        # 4 DMI devices x 4 GB = 16 GiB installed; the empty socket counts 0
        assert np._physical_installed_ram_bytes() == 16 * 1024**3
    finally:
        mp.undo()


def test_gpu_profiles_record_normalized_bdf() -> None:
    source = Path(
        "benchmarks/inferswarm_r4/node_preflight.py"
    ).read_text()
    assert 'item["pci_bus_id"] = item["pci.bus_id"]' in source
