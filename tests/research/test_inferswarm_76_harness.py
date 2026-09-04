"""CPU-only unit/source-contract tests for the #76 execution harness.

These run on any host (no torch import at collection): pure-reducer math,
case identity verification, envelope construction, and source-contract
checks on the frozen checkpoint map. Torch-dependent pieces are exercised
by node-side qualification.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))

from inferswarm_76 import (  # noqa: E402
    CAPTURE_POSITIONS,
    CHECKPOINT_FAMILY_MAP,
    ENVELOPES,
    ENVELOPE_CHECKPOINT_IDS,
    canonical_json_bytes,
    conservative_case_family,
    envelopes_from_case_metrics,
    load_corpus,
    nearest_rank_higher,
    sha256_bytes,
    tensor_metrics,
    verify_case_identity,
)


class FrozenContractTests(unittest.TestCase):
    def test_fifteen_envelopes(self):
        self.assertEqual(len(ENVELOPES), 15)
        self.assertEqual(len(ENVELOPE_CHECKPOINT_IDS), 10)

    def test_checkpoint_families_match_frozen_map(self):
        # Mirrors inferswarm@f394dc9 manifests/checkpoint-family-map.json
        self.assertEqual(
            CHECKPOINT_FAMILY_MAP["embedding-output"],
            ("local-bf16-backend-operation-output", "bfloat16"),
        )
        self.assertEqual(
            CHECKPOINT_FAMILY_MAP["full-final-row-fp32-consumer-logits"],
            ("fp32-consumer-logits", "float32"),
        )
        families = {f for f, _ in CHECKPOINT_FAMILY_MAP.values()}
        self.assertEqual(len(families), 5)

    def test_capture_positions_frozen(self):
        self.assertEqual(CAPTURE_POSITIONS, (0, 1, 3, 7))


class ReducerMathTests(unittest.TestCase):
    def test_metrics_known_values(self):
        m = tensor_metrics([1.0, 2.0, 3.0, 4.0], [1.5, 2.0, 2.5, 4.0])
        self.assertEqual(m["max-absolute-difference"], 0.5)
        expect_rms = math.sqrt(math.fsum([0.25, 0.25]) / 4)
        self.assertEqual(m["rms-difference"], expect_rms)
        # errors [0.5, 0, 0.5, 0] sorted [0, 0, 0.5, 0.5]; ceil(.99*4)=4 -> 0.5
        self.assertEqual(m["p99-absolute-error"], 0.5)

    def test_p99_nearest_rank_higher(self):
        # N=100: ceil(0.99*100)=99 -> 99th smallest (one-based) = second-
        # largest element of [1.0 x99, 2.0] is 1.0; the max alone is rank 100.
        self.assertEqual(nearest_rank_higher([1.0] * 99 + [2.0]), 1.0)
        self.assertEqual(nearest_rank_higher([2.0] + [1.0] * 99), 1.0)
        # N=100 with two distinct high values: rank 99 hits the 2nd-largest
        self.assertEqual(nearest_rank_higher([0.0] * 98 + [1.0, 2.0]), 1.0)
        self.assertEqual(nearest_rank_higher([0.0] * 100), 0.0)
        with self.assertRaises(ValueError):
            nearest_rank_higher([])
        with self.assertRaises(ValueError):
            nearest_rank_higher([1.0, -0.1])

    def test_case_family_max_per_metric(self):
        out = conservative_case_family([
            {"max-absolute-difference": 1.0, "rms-difference": 3.0,
             "p99-absolute-error": 2.0},
            {"max-absolute-difference": 4.0, "rms-difference": 1.0,
             "p99-absolute-error": 0.5},
        ])
        self.assertEqual(out["max-absolute-difference"], 4.0)
        self.assertEqual(out["rms-difference"], 3.0)
        self.assertEqual(out["p99-absolute-error"], 2.0)

    def test_envelopes_hex_serialization(self):
        rows = [
            {"max-absolute-difference": v, "rms-difference": v / 2,
             "p99-absolute-error": v / 4}
            for v in (0.5, 1.25, 0.125)
        ]
        out = envelopes_from_case_metrics({
            "embedding-output": rows[0],
            "layer-0-o-proj-input": rows[1],
            "layer-0-o-proj-output": rows[2],
            "global-layer-15-attention-o-proj-output": rows[0],
            "post-global-layer-15-residual": rows[1],
            "post-global-layer-31-residual": rows[2],
            "post-global-layer-47-residual": rows[0],
            "final-normalized-hidden-state": rows[1],
            "full-final-row-bf16-logits": rows[2],
            "full-final-row-fp32-consumer-logits": rows[0],
        })
        self.assertEqual(len(out), 15)
        for value in out.values():
            float.fromhex(value)  # exact round trip
        self.assertEqual(
            float.fromhex(out["local-bf16-backend-operation-output:"
                              "max-absolute-difference"]),
            1.25,
        )

    def test_missing_family_fails_closed(self):
        with self.assertRaises(ValueError):
            envelopes_from_case_metrics({
                "embedding-output": {
                    "max-absolute-difference": 1.0,
                    "rms-difference": 1.0,
                    "p99-absolute-error": 1.0,
                }
            })

    def test_domain_size_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            tensor_metrics([1.0, 2.0], [1.0])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            tensor_metrics([1.0], [float("nan")])


class CaseIdentityTests(unittest.TestCase):
    CASE = {
        "case_id": "c74-01-01-01",
        "case_sha256": "286556cf76004ad524933ead5d1768aae6d24aae6199db11500bf8315d8995fe",
        "content_class": "ordinary-prose",
        "length_regime": [4, 8],
        "prompt_sha256": "00985f1fa382d3485e136c3ff120fe1b8f2a169a56b97423d596569e0b96f1bb",
        "prompt_text": "wind bird stone wind",
        "token_count": 4,
        "token_ids": [15879, 8001, 10810, 6573],
        "token_ids_sha256": "7255f8af076acf1952ac8fa6125abe5370fe0529c3eea9f28e51749c9fba5cc6",
    }

    def test_valid_case_verifies(self):
        out = verify_case_identity(dict(self.CASE))
        self.assertEqual(out["case_id"], "c74-01-01-01")

    def test_mutated_prompt_fails(self):
        bad = dict(self.CASE)
        bad["prompt_text"] = "wind bird stone windx"
        with self.assertRaises(ValueError):
            verify_case_identity(bad)

    def test_mutated_tokens_fail(self):
        bad = dict(self.CASE)
        bad["token_ids"] = [15879, 8001, 10810, 6574]
        with self.assertRaises(ValueError):
            verify_case_identity(bad)

    def test_mutated_identity_fails(self):
        bad = dict(self.CASE)
        bad["content_class"] = "multilingual-text"
        with self.assertRaises(ValueError):
            verify_case_identity(bad)


class SourceContractTests(unittest.TestCase):
    """Structural contracts over the harness source (torch-free hosts)."""

    def test_margin_definition_documented_and_frozen(self):
        source = (REPO / "benchmarks/inferswarm_76/__init__.py").read_text()
        self.assertIn("matched-reference-top1-margin", source)

    def test_no_threshold_constants_in_harness(self):
        # The harness must not bake any numeric acceptance threshold; limits
        # arrive only from the derived frozen threshold manifest.
        for name in ("reference_runner.py", "chain_runner.py", "reducer.py"):
            source = (REPO / "benchmarks/inferswarm_76" / name).read_text()
            for token in ("0.25", "FROZEN_THRESHOLD"):
                self.assertNotIn(
                    token, source,
                    f"{name} must not contain threshold literal {token}")

    def test_row_pruning_preserves_final_row_domain(self):
        source = (REPO / "benchmarks/inferswarm_76/capture.py").read_text()
        self.assertIn("final_row_from_bf16_record", source)


if __name__ == "__main__":
    unittest.main()
