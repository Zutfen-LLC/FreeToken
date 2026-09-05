"""Issue #97 v4 physical producer contract tests.

The producer remains the accepted #88 execution math, but its evidence identity
must bind the accepted issue #95 methodology before CUDA execution.
"""
from __future__ import annotations

import unittest

from benchmarks.inferswarm_97 import (
    CONTRACT_ID,
    EXPECTED_CHECKPOINT_SHA256,
    GENERATED_TOKENS,
    METHODOLOGY_COMMIT,
    MODEL_REVISION,
    V4_ISSUE,
    frozen_argmax_row,
    frozen_subject_record,
    validate_campaign_case_ids,
)


class Issue97ProducerIdentityTests(unittest.TestCase):
    def test_binds_accepted_v4_methodology_and_subject(self):
        self.assertEqual(V4_ISSUE, 97)
        self.assertEqual(METHODOLOGY_COMMIT, "e12a6e3d5589044bace0c9555c0d364fb57a6229")
        self.assertEqual(CONTRACT_ID, "inferswarm.gemma4-prediction-aligned-qualification/1")
        self.assertEqual(
            EXPECTED_CHECKPOINT_SHA256,
            "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d",
        )
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
        record = frozen_subject_record(
            checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
            model_revision=MODEL_REVISION,
        )
        self.assertEqual(record["contract_id"], CONTRACT_ID)
        self.assertEqual(record["methodology_commit"], METHODOLOGY_COMMIT)
        self.assertEqual(record["checkpoint_sha256"], EXPECTED_CHECKPOINT_SHA256)
        with self.assertRaises(ValueError):
            frozen_subject_record(checkpoint_sha256="0" * 64, model_revision=MODEL_REVISION)
        with self.assertRaises(ValueError):
            frozen_subject_record(
                checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
                model_revision="0" * 40,
            )


if __name__ == "__main__":
    unittest.main()
