"""Issue #110 v5 holdout-only execution authority: tests.

All fixtures are SYNTHETIC and contain no holdout plaintext or secret
material. The h109 case IDs used in admission tests are the PUBLIC
committed case identities from the sealed-holdout commitment (public
metadata, not plaintext).

CPU-only: no torch, no CUDA, no model execution anywhere in this suite.

Covers:
1. The calibration runner/package files remain byte-identical to the
   calibration freeze (7e5c852) hashes — the frozen path is untouched.
2. Holdout admission accepts exactly the 24 committed h109 cases and
   rejects everything else (calibration IDs, historical namespaces,
   unknown/substituted/renumbered holdout IDs, duplicates).
3. Mechanical equivalence of the holdout entrypoints apart from
   namespace admission: each holdout entrypoint, executed as __main__,
   delegates to the frozen calibration ``main`` with ONLY the admission
   callable swapped, and exits with the frozen main's return code.
4. The calibration package itself keeps rejecting h109-* (admission swap
   is entrypoint-runtime-only, never installed package-wide).
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Calibration freeze (7e5c852) sha256 of every file the campaign froze.
CALIBRATION_FREEZE_SHA256 = {
    "benchmarks/inferswarm_110/__init__.py":
        "2e50c50a2e7ef7e202df618d4591a8c5f7d771be68d56727c11976f42e15ce72",
    "benchmarks/inferswarm_110/reference_runner.py":
        "0d4881771d26b05e2f84d97dda61ce2637c4b51a967600b5407e670f44e19aaa",
    "benchmarks/inferswarm_110/chain_runner.py":
        "3461abb56c8f47fc2dc64f8c267a2257ad59899f20a8d548418b596db41f4bae",
    "benchmarks/inferswarm_110/last_stage_service.py":
        "ee11a7321a467d7728c4d158abbaabbbb6a8b4ec620065231adfb2cae8b4187a",
}

for _p in (REPO, REPO / "python", REPO / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestCalibrationFilesUnchanged(unittest.TestCase):
    def test_every_frozen_calibration_file_is_byte_identical(self):
        for rel, expected in CALIBRATION_FREEZE_SHA256.items():
            with self.subTest(rel=rel):
                self.assertEqual(_sha256(REPO / rel), expected)


class TestHoldoutAdmission(unittest.TestCase):
    def setUp(self):
        from benchmarks.inferswarm_110b import validate_holdout_case_ids
        from benchmarks.inferswarm_110b import FROZEN_HOLDOUT_CASE_IDS
        self.validate = validate_holdout_case_ids
        self.frozen_ids = FROZEN_HOLDOUT_CASE_IDS

    @staticmethod
    def _case(case_id: str) -> dict:
        return {"case_id": case_id, "case_sha256": "0" * 64}

    def test_accepts_exactly_the_24_committed_cases(self):
        cases = [self._case(cid) for cid in self.frozen_ids]
        out = self.validate(cases)
        self.assertEqual([c["case_id"] for c in out], list(self.frozen_ids))

    def test_commitment_table_is_complete_and_unique(self):
        self.assertEqual(len(self.frozen_ids), 24)
        self.assertEqual(len(set(self.frozen_ids)), 24)
        for cid in self.frozen_ids:
            self.assertTrue(cid.startswith("h109-"))

    def test_rejects_calibration_and_historical_namespaces(self):
        for bad in ("c109-01-01-001", "p109-01-001", "h95-01-01-01",
                    "h86-01-01-01", "h74-01", "c95-03-02-01",
                    "c76-01-01-01", "unversioned", ""):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.validate([self._case(bad)])

    def test_rejects_unknown_substituted_holdout_ids(self):
        for bad in ("h109-01-01-004", "h109-99-99-999", "h109-01-04-001",
                    "h109-05-01-001"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.validate([self._case(bad)])

    def test_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            self.validate([self._case("h109-01-01-001"),
                           self._case("h109-01-01-001")])

    def test_rejects_non_dict_rows(self):
        with self.assertRaises(ValueError):
            self.validate(["h109-01-01-001"])

    def test_calibration_admission_still_rejects_h109(self):
        from benchmarks.inferswarm_110 import validate_campaign_case_ids
        with self.assertRaises(ValueError):
            validate_campaign_case_ids([self._case("h109-01-01-001")])

    def test_calibration_admission_still_accepts_calibration(self):
        from benchmarks.inferswarm_110 import validate_campaign_case_ids
        out = validate_campaign_case_ids(
            [self._case("c109-01-01-001"), self._case("p109-01-001")])
        self.assertEqual(len(out), 2)

    def test_importing_110b_does_not_touch_calibration_admission(self):
        frozen = importlib.import_module("benchmarks.inferswarm_110")
        before = frozen.validate_campaign_case_ids
        importlib.import_module("benchmarks.inferswarm_110b")
        self.assertIs(frozen.validate_campaign_case_ids, before)


class TestEntrypointEquivalence(unittest.TestCase):
    """Each holdout entrypoint = frozen main + ONLY the admission swap.

    Executes the real entrypoint source as __main__ with the frozen
    module's ``main`` monkeypatched to record (a) the admission callable
    active at delegation time and (b) that it was reached, then asserts
    the process exit code equals the frozen main's rc. The frozen
    module's original admission and main are restored in finally blocks.
    """

    def _run_entrypoint(self, entry_name: str, frozen_mod_name: str) -> None:
        frozen = importlib.import_module(frozen_mod_name)
        holdout = importlib.import_module("benchmarks.inferswarm_110b")

        observed = {}

        def fake_main(argv=None):
            observed["admission_at_delegation"] = \
                frozen.validate_campaign_case_ids
            observed["delegated_main"] = fake_main
            return 7  # distinctive sentinel rc

        original_main = frozen.main
        original_admission = frozen.validate_campaign_case_ids
        setattr(frozen, "main", fake_main)
        exit_code = None
        try:
            source = (REPO / "benchmarks" / "inferswarm_110b" /
                      entry_name).read_text()
            mod = types.ModuleType("holdout_entrypoint_under_test")
            mod.__dict__["__name__"] = "__main__"
            mod.__dict__["__file__"] = str(
                REPO / "benchmarks" / "inferswarm_110b" / entry_name)
            try:
                exec(compile(source, entry_name, "exec"), mod.__dict__)
            except SystemExit as exc:
                exit_code = exc.code
        finally:
            setattr(frozen, "main", original_main)
            setattr(frozen, "validate_campaign_case_ids",
                    original_admission)

        # Delegation reached the frozen main with the holdout admission
        # active, and NOTHING else in the frozen module changed.
        self.assertIn("admission_at_delegation", observed)
        self.assertIs(observed["admission_at_delegation"],
                      holdout.validate_holdout_case_ids)
        self.assertIsNot(observed["admission_at_delegation"],
                         original_admission)
        # sys.exit(frozen.main()) propagated the frozen rc verbatim.
        self.assertEqual(exit_code, 7)
        # Restoration left the frozen module exactly as it was.
        self.assertIs(frozen.validate_campaign_case_ids, original_admission)
        self.assertIs(frozen.main, original_main)

    def test_reference_entrypoint(self):
        self._run_entrypoint(
            "reference_runner_holdout.py",
            "benchmarks.inferswarm_110.reference_runner")

    def test_chain_entrypoint(self):
        self._run_entrypoint(
            "chain_runner_holdout.py",
            "benchmarks.inferswarm_110.chain_runner")

    def test_entrypoints_import_no_torch(self):
        import ast

        for name in ("reference_runner_holdout.py",
                     "chain_runner_holdout.py"):
            with self.subTest(name=name):
                tree = ast.parse(
                    (REPO / "benchmarks" / "inferswarm_110b" /
                     name).read_text())
                imported = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported.update(a.name for a in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported.add(node.module)
                self.assertFalse(
                    any(m == "torch" or m.startswith("torch.") or
                        m.startswith("freetoken") or
                        m.startswith("benchmarks.inferswarm_76") or
                        m.startswith("benchmarks.inferswarm_r6")
                        for m in imported),
                    f"{name} must import only admission + frozen runner")

    def test_holdout_package_contains_no_execution_math(self):
        """Source audit: 110b has no model/runtime/prefill calls."""
        import ast

        forbidden_calls = {
            "prefill", "generate", "forward", "reset_session_state",
            "arm_full_capture", "checkpoint_census", "freeze_dense_block_plan",
            "GemmaDenseStage", "I76StageClient", "I76LastStageClient",
        }
        for path in sorted((REPO / "benchmarks" / "inferswarm_110b")
                           .glob("*.py")):
            with self.subTest(path=path.name):
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        fn = node.func
                        name = getattr(fn, "id", None) or getattr(
                            fn, "attr", None)
                        self.assertNotIn(
                            name, forbidden_calls,
                            f"{path.name} must not call {name}")


if __name__ == "__main__":
    unittest.main()
