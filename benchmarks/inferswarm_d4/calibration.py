"""Freeze D4 worker capability from isolated captured D3 physical paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

GPU_A = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
GPU_B = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replays", type=int, default=200)
    ns = parser.parse_args(); ns.output_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ns.repo, text=True).strip()
    rows = {}
    for shape in ("a", "b"):
        path = ns.output_dir / f"d4-worker-calibration-{shape}-raw.json"
        cmd = [str(ns.repo / ".venv/bin/python"), str(ns.repo / "benchmarks/inferswarm_d3/primitive.py"),
               "--model", ns.model, "--revision", ns.revision, "--placement", ns.placement,
               "--shape", shape, "--output", str(path), "--replays", str(ns.replays)]
        subprocess.run(cmd, cwd=ns.repo, check=True)
        row = json.loads(path.read_text())
        report = row["whole_model_graph"]
        checks = {"exact_sha": row["physical_tested_freetoken_commit"] == sha,
                  "graph_active": report["graph_active"] is True,
                  "captured_bs1": report["captured_batch_sizes"] == [1],
                  "isolated_worker_shape": report["active_workers"] == [shape],
                  "zero_fallback": report["fallback_count"] == 0,
                  "zero_failure": report["failure_count"] == 0,
                  "zero_recapture": report["graph_recapture_count"] == 0,
                  "zero_host_sync": report["steady_state_host_sync_count"] == 0,
                  "zero_weight_movement": report["steady_state_expert_weight_bytes_host_to_worker_a"] == 0
                                           and report["steady_state_expert_weight_bytes_host_to_worker_b"] == 0,
                  "paging_valid": row["paging_delta"]["pswpin"] == 0 and row["paging_delta"]["pswpout"] == 0,
                  "real_resident_kernel": row["one_layer"]["all_cases_passed"] is True,
                  "dynamic_payload_without_recapture": row["one_layer"]["captured_dynamic_payload"]["changed_payload_changes_output"] is True
                                                       and row["one_layer"]["captured_dynamic_payload"]["recapture_count_after_isolated_replay"] == 0}
        if not all(checks.values()):
            raise RuntimeError(f"D4 calibration {shape} invalid: {checks}")
        rows[shape] = {"raw_artifact": path.name, "raw_artifact_sha256": digest(path),
                       "physical_uuid": GPU_A if shape == "a" else GPU_B,
                       "service_time_distribution": row["one_layer"]["real_path_captured_replay_wall"],
                       "representative_layer_id": row["one_layer"]["layer_id"],
                       "fixed_width_route_geometry": row["one_layer"]["cases"][-1],
                       "startup_seconds": row["startup_seconds"], "paging_delta": row["paging_delta"],
                       "graph_state": report, "checks": checks}
    ca = Fraction(str(rows["a"]["service_time_distribution"]["median_us"]))
    cb = Fraction(str(rows["b"]["service_time_distribution"]["median_us"]))
    # (1/Ca) / (1/Ca + 1/Cb) simplifies exactly to Cb/(Ca+Cb).
    target_a, target_b = cb / (ca + cb), ca / (ca + cb)
    result = {"schema": "inferswarm.d4.worker-calibration/1", "status": "FROZEN_BEFORE_D4_PLACEMENT_AND_PERFORMANCE",
              "freetoken_sha": sha, "model": ns.model, "revision": ns.revision,
              "method": "existing proven D3 physical primitive; isolated active worker; representative first MoE layer; fixed-width mixed worker/local route geometry; real resident NVFP4 kernel; host-staged transport; captured replay",
              "workers": {"a": rows["a"], "b": rows["b"]},
              "service_medians_us": {"a": float(ca), "b": float(cb)},
              "normalized_capacity_targets": {
                  "a": {"numerator": target_a.numerator, "denominator": target_a.denominator, "value": float(target_a)},
                  "b": {"numerator": target_b.numerator, "denominator": target_b.denominator, "value": float(target_b)},
              }, "paging_valid": True, "graph_active": True, "fallback_count": 0, "failure_count": 0}
    ns.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
