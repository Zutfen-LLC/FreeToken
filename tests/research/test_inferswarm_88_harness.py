"""Issue #88 producer tests (Phase P: run BEFORE any physical execution).

CPU-pure tests over the #88 v3 semantic layer
(benchmarks/inferswarm_88). They freeze the producer contract required
by issue #88 Phase P:

- teacher-forced prefix identity (positive + negative);
- emitted winner follows the frozen ARGMAX_FIRST_MAX tie-break rule
  (including exact-equal-maxima ties resolving to the lowest token id);
- all 8 decision rows emitted exactly once per case;
- 15-envelope family completeness from per-checkpoint metrics;
- candidate data cannot influence D(r) (reference-only construction);
- output artifacts bind exact case/provenance identity;
- the semantic layer is torch-free (CPU/coordinator-pure).
"""

from __future__ import annotations

import json
import math
import unittest

from benchmarks.inferswarm_76 import (
    ENVELOPES,
    envelopes_from_case_metrics,
    tensor_metrics,
)
from benchmarks.inferswarm_88 import (
    ARGMAX_TIE_BREAK_IDENTITY,
    CAPTURE_POSITIONS,
    DECISION_DOMAIN_CONSTRUCTION,
    DECISION_DOMAIN_K,
    GENERATED_TOKENS,
    V3_METHODOLOGY_COMMIT,
    assert_teacher_forcing,
    build_chain_case_summary,
    build_reference_case_summary,
    canonical_prefix,
    decision_domain_row,
    executor_rule_proof,
    frozen_argmax_row,
    prefix_identity_proof,
    prefix_sha256,
)


def synthetic_row(seed: int, size: int = 4096) -> list[float]:
    """Deterministic pseudo-logits (v3-test style LCG)."""
    state = seed & 0xFFFFFFFF
    out = []
    for _ in range(size):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        out.append((state / 0x7FFFFFFF) * 2.0 - 1.0)
    return out


def reference_case_fixture(case_id: str = "c86-00-00-00"):
    case = {
        "case_id": case_id,
        "case_sha256": "a" * 64,
        "prompt_sha256": "b" * 64,
        "token_ids_sha256": "c" * 64,
        "token_ids": [1, 2, 3],
    }
    trajectory = [11, 22, 33, 44, 55, 66, 77, 88]
    decision_rows = []
    for step in range(GENERATED_TOKENS):
        prefix = list(case["token_ids"]) + trajectory[:step]
        domain = decision_domain_row(synthetic_row(step + 1))
        decision_rows.append({
            "decision_index": step,
            "prefix_len": len(prefix),
            "prefix_sha256": prefix_sha256(prefix),
            "domain_membership_sha256": domain["domain_membership_sha256"],
            "domain_size": domain["domain_size"],
            "domain_cutoff_hex": domain["cutoff_hex"],
            "emitted_token": trajectory[step],
            "emitted_rule": ARGMAX_TIE_BREAK_IDENTITY,
            "row_f32_sha256": f"{step:064d}",
            "row_element_count": 4096,
            "rule_proof": executor_rule_proof(
                synthetic_row(step + 1), trajectory[step]),
        })
    return case, trajectory, decision_rows


class FrozenArgmaxTests(unittest.TestCase):
    def test_rule_identity_frozen(self):
        self.assertEqual(
            ARGMAX_TIE_BREAK_IDENTITY,
            "ARGMAX_FIRST_MAX/lowest-token-id-among-exactly-equal-fp32-maxima",
        )

    def test_first_max_wins(self):
        row = [0.0, 5.0, 5.0, 1.0]
        index, value = frozen_argmax_row(row)
        self.assertEqual(index, 1)
        self.assertEqual(value, 5.0)

    def test_exact_ties_resolve_to_lowest_token_id(self):
        row = [3.0, 7.0, 7.0, 7.0, 2.0]
        index, _ = frozen_argmax_row(row)
        self.assertEqual(index, 1)

    def test_rule_proof_accepts_rule_winner(self):
        row = synthetic_row(7)
        winner, _ = frozen_argmax_row(row)
        proof = executor_rule_proof(row, winner)
        self.assertTrue(proof["rule_ok"])
        self.assertEqual(proof["lowest_index_among_equal_maxima"], winner)

    def test_rule_proof_rejects_wrong_token(self):
        row = synthetic_row(7)
        winner, _ = frozen_argmax_row(row)
        other = (winner + 1) % len(row)
        if row[other] == row[winner]:
            other = (winner + 2) % len(row)
        proof = executor_rule_proof(row, other)
        self.assertFalse(proof["rule_ok"])

    def test_tie_proof_reports_tie_count(self):
        row = [1.0, 9.0, 9.0, 0.0]
        winner, _ = frozen_argmax_row(row)
        proof = executor_rule_proof(row, winner)
        self.assertEqual(proof["tie_count"], 2)
        self.assertEqual(proof["lowest_index_among_equal_maxima"], 1)


class DecisionDomainTests(unittest.TestCase):
    def test_construction_identity_frozen(self):
        self.assertEqual(DECISION_DOMAIN_CONSTRUCTION,
                         "reference-top-1024-with-cutoff-ties/1")
        self.assertEqual(DECISION_DOMAIN_K, 1024)

    def test_domain_size_and_cutoff(self):
        row = synthetic_row(3, size=4096)
        info = decision_domain_row(row)
        self.assertGreaterEqual(info["domain_size"], 1024)
        # every member >= cutoff, every non-member < cutoff
        members = info["membership"]
        cutoff = float.fromhex(info["cutoff_hex"])
        for i in members[:50] + members[-50:]:
            self.assertGreaterEqual(row[i], cutoff)

    def test_cutoff_ties_included(self):
        row = [0.0] * 4096
        row[5] = 2.0
        row[4000] = 2.0
        info = decision_domain_row(row, k=8)
        # cutoff is 0.0 (8th-highest); ALL 4096 tokens tie at cutoff
        self.assertEqual(info["domain_size"], 4096)
        self.assertIn(5, info["membership"])
        self.assertIn(4000, info["membership"])

    def test_winner_in_domain_by_construction(self):
        row = synthetic_row(11)
        info = decision_domain_row(row)
        winner, best = frozen_argmax_row(row)
        self.assertIn(winner, info["membership"])
        self.assertEqual(float.fromhex(info["cutoff_hex"]) <= best, True)

    def test_membership_ordering_ascending(self):
        row = synthetic_row(13)
        info = decision_domain_row(row)
        self.assertEqual(info["membership"], sorted(info["membership"]))

    def test_candidate_cannot_influence_domain(self):
        # D(r) is computed from the reference row ONLY: the builder takes
        # no candidate input at all, and re-deriving from the reference
        # row is deterministic. A *non-uniformly* perturbed (candidate-
        # shaped) row yields a different membership, proving the hash
        # binds the exact reference values (a uniform shift would leave
        # membership invariant by construction, which is expected).
        row = synthetic_row(17)
        info = decision_domain_row(row)
        info2 = decision_domain_row(row)
        self.assertEqual(info, info2)
        candidate = list(row)
        nonmember = next(i for i in range(len(row))
                         if i not in set(info["membership"]))
        candidate[nonmember] = max(row) + 1.0  # inject a new winner
        self.assertNotEqual(
            info["domain_membership_sha256"],
            decision_domain_row(candidate)["domain_membership_sha256"],
        )

    def test_domain_membership_hash_canonical(self):
        # must equal sha256 over canonical JSON of the ascending id list
        import hashlib

        row = synthetic_row(19)
        info = decision_domain_row(row)
        expected = hashlib.sha256(
            (json.dumps(info["membership"], ensure_ascii=False,
                        sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
        self.assertEqual(info["domain_membership_sha256"], expected)


class TeacherForcingTests(unittest.TestCase):
    def _reference_decisions(self, token_ids, trajectory):
        rows = []
        for step in range(GENERATED_TOKENS):
            prefix = list(token_ids) + trajectory[:step]
            rows.append({
                "decision_index": step,
                "prefix_len": len(prefix),
                "prefix_sha256": prefix_sha256(prefix),
            })
        return rows

    def test_canonical_prefix_definition(self):
        token_ids = [1, 2, 3]
        trajectory = [11, 22, 33, 44, 55, 66, 77, 88]
        self.assertEqual(canonical_prefix(token_ids, trajectory, 0), [1, 2, 3])
        self.assertEqual(
            canonical_prefix(token_ids, trajectory, 3), [1, 2, 3, 11, 22, 33])
        self.assertEqual(
            canonical_prefix(token_ids, trajectory, 7),
            [1, 2, 3, 11, 22, 33, 44, 55, 66, 77],
        )

    def test_prefix_identity_proof_binds_bytes(self):
        proof = prefix_identity_proof([1, 2, 3])
        self.assertEqual(proof["prefix_len"], 3)
        self.assertEqual(proof["prefix_sha256"], prefix_sha256([1, 2, 3]))

    def test_teacher_forcing_accepts_exact_prefix(self):
        token_ids = [5, 6, 7, 8]
        trajectory = [31, 32, 33, 34, 35, 36, 37, 38]
        refs = self._reference_decisions(token_ids, trajectory)
        for step in range(GENERATED_TOKENS):
            prefix = canonical_prefix(token_ids, trajectory, step)
            assert_teacher_forcing(prefix=prefix, reference_decision=refs[step])

    def test_teacher_forcing_rejects_drifted_prefix(self):
        token_ids = [5, 6, 7, 8]
        trajectory = [31, 32, 33, 34, 35, 36, 37, 38]
        refs = self._reference_decisions(token_ids, trajectory)
        wrong = canonical_prefix(token_ids, trajectory, 4) + [99]
        with self.assertRaises(ValueError):
            assert_teacher_forcing(prefix=wrong, reference_decision=refs[4])
        swapped = list(canonical_prefix(token_ids, trajectory, 4))
        swapped[-1] = swapped[-1] + 1
        with self.assertRaises(ValueError):
            assert_teacher_forcing(prefix=swapped, reference_decision=refs[4])

    def test_teacher_forcing_rejects_length_mismatch(self):
        token_ids = [5]
        trajectory = list(range(41, 49))
        refs = self._reference_decisions(token_ids, trajectory)
        with self.assertRaises(ValueError):
            assert_teacher_forcing(prefix=[], reference_decision=refs[0])


class SummaryBuilderTests(unittest.TestCase):
    def test_reference_summary_binds_identity(self):
        case, trajectory, rows = reference_case_fixture()
        margins = [
            {"step": s, "margin_hex": (1.0 + s).hex()} for s in range(8)
        ]
        summary = build_reference_case_summary(
            case=case, generated=trajectory, margins=margins,
            decision_rows=rows, nan_inf_total=0,
            capture_manifest={"record_count": 40},
            producer={"commit": "e" * 40, "dirty": False},
            gpu_uuid="GPU-test", tag="t0", attempt_id="att-0",
            wall_seconds=1.0,
        )
        self.assertEqual(summary["case_id"], case["case_id"])
        self.assertEqual(summary["case_sha256"], case["case_sha256"])
        self.assertEqual(summary["producer"]["commit"], "e" * 40)
        self.assertEqual(summary["generated_token_ids"], trajectory)
        self.assertEqual(summary["argmax_tie_break"], ARGMAX_TIE_BREAK_IDENTITY)
        self.assertEqual(summary["decision_domain_construction"],
                         DECISION_DOMAIN_CONSTRUCTION)

    def test_reference_summary_requires_all_eight_rows(self):
        case, trajectory, rows = reference_case_fixture()
        margins = [{"step": s, "margin_hex": (1.0).hex()} for s in range(8)]
        with self.assertRaises(ValueError):
            build_reference_case_summary(
                case=case, generated=trajectory, margins=margins,
                decision_rows=rows[:7], nan_inf_total=0,
                capture_manifest={}, producer={"commit": "f" * 40, "dirty": False},
                gpu_uuid="g", tag="t", attempt_id="a", wall_seconds=0.0,
            )
        duplicated = rows[:7] + [dict(rows[3])]
        with self.assertRaises(ValueError):
            build_reference_case_summary(
                case=case, generated=trajectory, margins=margins,
                decision_rows=duplicated, nan_inf_total=0,
                capture_manifest={}, producer={"commit": "f" * 40, "dirty": False},
                gpu_uuid="g", tag="t", attempt_id="a", wall_seconds=0.0,
            )

    def test_chain_summary_binds_reference_prefixes(self):
        case, trajectory, ref_rows = reference_case_fixture()
        reference = {
            "case_id": case["case_id"],
            "case_sha256": case["case_sha256"],
            "generated_token_ids": trajectory,
            "decisions": ref_rows,
        }
        cand_rows = []
        for step in range(GENERATED_TOKENS):
            prefix = list(case["token_ids"]) + trajectory[:step]
            cand_rows.append({
                "decision_index": step,
                "prefix_len": len(prefix),
                "prefix_sha256": prefix_sha256(prefix),
                "emitted_token": trajectory[step],
                "emitted_rule": ARGMAX_TIE_BREAK_IDENTITY,
                "row_f32_sha256": f"{step:064x}",
                "row_element_count": 4096,
                "rule_proof": executor_rule_proof(
                    synthetic_row(step + 1), trajectory[step]),
                "row_retained_at": "last-stage-node",
            })
        summary = build_chain_case_summary(
            case=case, reference_case=reference, decision_rows=cand_rows,
            margins=[{"step": s, "margin_hex": (0.5).hex()} for s in range(8)],
            nan_inf_total=0, capture_manifests={"stage3": {"record_count": 20}},
            producer={"commit": "e" * 40, "dirty": False},
            tag="t", attempt_id="a", wall_seconds=0.0,
        )
        self.assertEqual(summary["reference_forced_trajectory"], trajectory)

    def test_chain_summary_rejects_prefix_mismatch(self):
        case, trajectory, ref_rows = reference_case_fixture()
        reference = {
            "case_id": case["case_id"],
            "case_sha256": case["case_sha256"],
            "generated_token_ids": trajectory,
            "decisions": ref_rows,
        }
        cand_rows = []
        for step in range(GENERATED_TOKENS):
            prefix = list(case["token_ids"]) + trajectory[:step]
            cand_rows.append({
                "decision_index": step,
                "prefix_len": len(prefix),
                "prefix_sha256": prefix_sha256(prefix),
                "emitted_token": trajectory[step],
                "emitted_rule": ARGMAX_TIE_BREAK_IDENTITY,
                "row_f32_sha256": f"{step:064x}",
                "row_element_count": 4096,
                "rule_proof": executor_rule_proof(
                    synthetic_row(step + 1), trajectory[step]),
                "row_retained_at": "last-stage-node",
            })
        # drift one candidate prefix
        bad_prefix = list(case["token_ids"]) + trajectory[:5] + [999]
        cand_rows[6]["prefix_sha256"] = prefix_sha256(bad_prefix)
        with self.assertRaises(ValueError):
            build_chain_case_summary(
                case=case, reference_case=reference, decision_rows=cand_rows,
                margins=[{"step": s, "margin_hex": (0.5).hex()} for s in range(8)],
                nan_inf_total=0, capture_manifests={},
                producer={"commit": "e" * 40, "dirty": False},
                tag="t", attempt_id="a", wall_seconds=0.0,
            )

    def test_chain_summary_rejects_case_substitution(self):
        case, trajectory, ref_rows = reference_case_fixture()
        other = dict(case)
        other["case_sha256"] = "d" * 64
        reference = {
            "case_id": case["case_id"],
            "case_sha256": case["case_sha256"],
            "generated_token_ids": trajectory,
            "decisions": ref_rows,
        }
        with self.assertRaises(ValueError):
            build_chain_case_summary(
                case=other, reference_case=reference,
                decision_rows=ref_rows,
                margins=[], nan_inf_total=0, capture_manifests={},
                producer={"commit": "e" * 40, "dirty": False},
                tag="t", attempt_id="a", wall_seconds=0.0,
            )


class EnvelopeCompletenessTests(unittest.TestCase):
    def test_fifteen_envelopes_complete(self):
        from benchmarks.inferswarm_76 import CHECKPOINT_FAMILY_MAP
        self.assertEqual(len(ENVELOPES), 15)
        metrics = tensor_metrics(synthetic_row(23), synthetic_row(24))
        per_checkpoint = {f"{cid}@{p}": metrics
                          for cid in CHECKPOINT_FAMILY_MAP
                          for p in CAPTURE_POSITIONS}
        envelopes = envelopes_from_case_metrics(per_checkpoint)
        self.assertEqual(set(envelopes), set(ENVELOPES))
        for value in envelopes.values():
            self.assertTrue(math.isfinite(float.fromhex(value)))

    def test_missing_family_rejected(self):
        from benchmarks.inferswarm_76 import FAMILIES, CHECKPOINT_FAMILY_MAP
        # one checkpoint per family except the last family is omitted
        present = {
            cid: tensor_metrics(synthetic_row(i), synthetic_row(i + 1))
            for i, cid in enumerate(
                cid for cid, (fam, _dt) in CHECKPOINT_FAMILY_MAP.items()
                if fam != FAMILIES[-1]
            )
        }
        with self.assertRaises(ValueError):
            envelopes_from_case_metrics(present)


class PurityTests(unittest.TestCase):
    def test_semantic_layer_is_torch_free(self):
        import inspect

        import benchmarks.inferswarm_88 as pkg
        source = inspect.getsource(pkg)
        self.assertNotIn("import torch", source)
        self.assertNotIn("cuda", source.replace("cuda:", "").lower()
                         .replace("gpu", ""))  # noqa: E501 (loose guard)

    def test_methodology_commit_pinned(self):
        self.assertEqual(
            V3_METHODOLOGY_COMMIT,
            "a8ec98a9fb9b673c93de5100d784ea772395efdb",
        )

    def test_generated_tokens_eight(self):
        self.assertEqual(GENERATED_TOKENS, 8)
        self.assertEqual(CAPTURE_POSITIONS, (0, 1, 3, 7))


if __name__ == "__main__":
    unittest.main()
