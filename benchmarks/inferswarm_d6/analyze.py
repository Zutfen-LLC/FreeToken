"""Consolidate D6 host-local evidence without committing raw host artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(root: Path, name: str): return json.loads((root / name).read_text())


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--evidence", type=Path, required=True)
    ns = p.parse_args(); root = ns.evidence
    b, a, ab = (load(root, f"d6-critical-path-{shape}.json") for shape in ("b", "a", "ab"))
    primitive = load(root, "d6-transport-primitive.json"); serving = load(root, "d6-analysis.json")
    arms = {name: load(root, f"d6-{name.lower()}.json") for name in ("T0", "T1", "T2", "T3")}
    perturbation = {"schema": "inferswarm.d6.instrumentation-perturbation/1",
        "full_marker_b_only_rejected_ratio": .9180046868193685,
        "narrow": {shape: row["instrumentation_perturbation"] for shape, row in (("b", b), ("a", a), ("ab", ab))},
        "interpretation": "B-only trusted at >=0.97; A-only borderline below by 0.000303; AB component intervals potentially perturbing at 0.941119. Uninstrumented outer graph walls are authoritative."}
    (root / "d6-instrumentation-perturbation.json").write_text(json.dumps(perturbation, indent=2)+"\n")
    b_wall = b["instrumentation_perturbation"]["uninstrumented_wall"]["median_us"]
    ab_wall = ab["instrumentation_perturbation"]["uninstrumented_wall"]["median_us"]
    critical = {"schema": "inferswarm.d6.critical-path-analysis/1", "phase": "D6-A_FROZEN_BEFORE_D6-B",
        "representative_raw_payload_identical": True, "b_only": b["representative_case"],
        "a_only": a["representative_case"], "equal_ab": ab["representative_case"],
        "uninstrumented_b_layer_wall_us": b_wall, "uninstrumented_ab_layer_wall_us": ab_wall,
        "residual_ab_minus_b_us": ab_wall-b_wall,
        "concurrent_transport_retention": {
            "a_branch_ab_over_a_only": ab["representative_case"]["workers"]["a"]["complete_worker_branch"]["median_us"] / a["representative_case"]["workers"]["a"]["complete_worker_branch"]["median_us"],
            "b_branch_ab_over_b_only": ab["representative_case"]["workers"]["b"]["complete_worker_branch"]["median_us"] / b["representative_case"]["workers"]["b"]["complete_worker_branch"]["median_us"]},
        "attribution": "The added x1 A branch, dominated by fixed 32768-byte D2H, extends the join; B does not degrade. Second return H2D and scatter add GPU0 wall. Reduced local work offsets part of the tax.",
        "component_warning": "AB narrow event instrumentation is potentially perturbing; do not sum components."}
    (root / "d6-critical-path.json").write_text(json.dumps(critical, indent=2)+"\n")
    bytes_report = {"schema": "inferswarm.d6.transport-bytes/1", "per_worker_layer": {
        "d5_fixed": {"activation_h2d":4096,"route_metadata_h2d":68,"return_d2h":32768,
                     "return_h2d_gpu0":32768,"total":69700},
        "d6_count_0": {"activation_h2d":4096,"route_metadata_h2d":4,"return_d2h":0,
                       "return_h2d_gpu0":0,"total":4100,"expert_compute":False},
        "d6_count_4": {"activation_h2d":4096,"route_metadata_h2d":36,"return_d2h":16384,
                       "return_h2d_gpu0":16384,"total":36900,"saved_vs_d5":32800}},
        "t3_observed": {"d5_matched_capacity_bytes": 69700*2*arms["T3"]["ownership"]["layer_calls"],
                        "d6_actual_bytes": arms["T3"]["transport"]["total_worker_transport_bytes"]},
        "primitive": primitive["controlled_route_counts"]}
    bytes_report["t3_observed"]["bytes_saved"] = bytes_report["t3_observed"]["d5_matched_capacity_bytes"]-bytes_report["t3_observed"]["d6_actual_bytes"]
    (root / "d6-transport-bytes-baseline.json").write_text(json.dumps(bytes_report, indent=2)+"\n")
    report = f"""# D6 count-aware transport report

- Primitive: `{primitive['classification']}`
- Serving: `{serving['classification']}`
- Uninstrumented B-only / AB layer wall: {b_wall:.3f} / {ab_wall:.3f} us; residual {ab_wall-b_wall:.3f} us.
- Attribution: worker A's fixed x1 return extends fan-in; B retains {critical['concurrent_transport_retention']['b_branch_ab_over_b_only']:.6f} of its isolated branch wall ratio and A retains {critical['concurrent_transport_retention']['a_branch_ab_over_a_only']:.6f}.
- D5 fixed / D6 four-route bytes per worker/layer: 69,700 / 36,900; 32,800 saved.
- Zero route: 4,100 bytes (activation + count), no slots/weights, contribution payload, or expert work.
- T0: {arms['T0']['analysis']['decode_tok_s']['each_retained']} median {serving['median_decode_tok_s']['T0']:.6f} tok/s.
- T1: {arms['T1']['analysis']['decode_tok_s']['each_retained']} median {serving['median_decode_tok_s']['T1']:.6f} tok/s.
- T2: {arms['T2']['analysis']['decode_tok_s']['each_retained']} median {serving['median_decode_tok_s']['T2']:.6f} tok/s.
- T3: {arms['T3']['analysis']['decode_tok_s']['each_retained']} median {serving['median_decode_tok_s']['T3']:.6f} tok/s.
- B gain {serving['B_TRANSPORT_GAIN']:.9f}; AB gain {serving['AB_TRANSPORT_GAIN']:.9f}; E3 {serving['D6_E3']:.9f}; E3 relative improvement {serving['D6_VS_D5_E3']:.9f}.
- Marginal retention: `{serving['marginal_retention']}`. Fourth worker recommended: `{str(serving['fourth_worker_recommended']).lower()}`.
- No D7 or fourth-worker work was started.
"""
    (root / "d6-report.md").write_text(report)
    print(json.dumps({"critical_path": critical, "bytes": bytes_report["t3_observed"], "serving": serving}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
