"""Issue #97 v4 physical producer contract tests."""
from __future__ import annotations

import inspect
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.inferswarm_97 import (
    CONTRACT_ID, EXPECTED_CHECKPOINT_SHA256, GENERATED_TOKENS,
    METHODOLOGY_COMMIT, MODEL_REVISION, V4_ISSUE, frozen_argmax_row,
    frozen_subject_record, require_producer_identity, validate_campaign_case_ids,
)


class Issue97ProducerIdentityTests(unittest.TestCase):
    def test_binds_accepted_v4_methodology_and_subject(self):
        self.assertEqual(V4_ISSUE, 97)
        self.assertEqual(METHODOLOGY_COMMIT, "e12a6e3d5589044bace0c9555c0d364fb57a6229")
        self.assertEqual(CONTRACT_ID, "inferswarm.gemma4-prediction-aligned-qualification/1")
        self.assertEqual(EXPECTED_CHECKPOINT_SHA256, "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d")
        self.assertEqual(MODEL_REVISION, "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7")
        self.assertEqual(GENERATED_TOKENS, 8)

    def test_argmax_rule_retains_lowest_exact_maximum(self):
        self.assertEqual(frozen_argmax_row([1.0, 3.0, 3.0, 2.0]), (1, 3.0))

    def test_campaign_rejects_holdout_and_historical_case_namespaces(self):
        accepted = [{"case_id": "c95-00-00-00"}, {"case_id": "p95-05-03-01"}]
        self.assertEqual(validate_campaign_case_ids(accepted), accepted)
        for forbidden in ("h95-00-00-00", "c86-00-00-00", "p86-00-00-00"):
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(ValueError):
                    validate_campaign_case_ids([{"case_id": forbidden}])

    def test_subject_record_rejects_substitution_and_binds_accepted_contract(self):
        record = frozen_subject_record(checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256, model_revision=MODEL_REVISION)
        self.assertEqual(record["contract_id"], CONTRACT_ID)
        self.assertEqual(record["methodology_commit"], METHODOLOGY_COMMIT)
        with self.assertRaises(ValueError):
            frozen_subject_record(checkpoint_sha256="0" * 64, model_revision=MODEL_REVISION)

    @patch("benchmarks.inferswarm_97.subprocess.check_output")
    def test_exact_clean_expected_producer_passes(self, output):
        output.side_effect = ["a" * 40 + "\n", ""]
        self.assertEqual(require_producer_identity(Path("/repo"), "a" * 40), {"commit": "a" * 40, "dirty": False, "expected_commit": "a" * 40})

    @patch("benchmarks.inferswarm_97.subprocess.check_output")
    def test_dirty_exact_producer_fails(self, output):
        output.side_effect = ["a" * 40 + "\n", " M file.py\n"]
        with self.assertRaisesRegex(ValueError, "dirty"):
            require_producer_identity(Path("/repo"), "a" * 40)

    @patch("benchmarks.inferswarm_97.subprocess.check_output")
    def test_clean_wrong_producer_fails(self, output):
        output.side_effect = ["b" * 40 + "\n", ""]
        with self.assertRaisesRegex(ValueError, "does not equal"):
            require_producer_identity(Path("/repo"), "a" * 40)

    def test_missing_or_malformed_expected_producer_fails(self):
        for value in (None, "", "not-a-sha", "a" * 39):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    require_producer_identity(Path("/repo"), value)

    def test_all_entrypoints_require_external_expected_producer_sha(self):
        root = Path(__file__).resolve().parents[2] / "benchmarks" / "inferswarm_97"
        for filename in ("reference_runner.py", "chain_runner.py", "last_stage_service.py"):
            with self.subTest(filename=filename):
                source = (root / filename).read_text()
                self.assertIn('\"--expected-producer-sha\", required=True', source)
                self.assertNotIn("--allow-producer", source)

    def test_last_stage_producer_and_subject_checks_precede_cuda_initialization(self):
        root = Path(__file__).resolve().parents[2] / "benchmarks" / "inferswarm_97"
        source = (root / "last_stage_service.py").read_text()
        serve = source[source.index("def serve("):source.index("def main(")]
        self.assertLess(serve.index("require_producer_identity"), serve.index("import torch"))
        self.assertLess(serve.index("frozen_subject_record"), serve.index("import torch"))
        self.assertLess(serve.index("require_producer_identity"), serve.index("set_rope_device"))


if __name__ == "__main__":
    unittest.main()
