"""Issue #97 Phase B holdout-only execution authority: tests.

All fixtures are SYNTHETIC and contain no holdout plaintext or secret
material. The h95 case IDs used in admission tests are the PUBLIC
committed cell identities from the sealed-holdout commitment (public
metadata, not plaintext).

CPU-only: no torch, no CUDA, no model execution anywhere in this suite.

Covers:
1. Phase A runner/service/__init__ files remain byte-identical to the
   Phase A freeze (57dfcb7) hashes — the frozen path is untouched.
2. Holdout admission accepts exactly the 24 committed h95 cells and
   rejects everything else (calibration IDs, historical namespaces,
   unknown/substituted/renumbered holdout IDs, duplicates).
3. Mechanical equivalence of the holdout entrypoints apart from
   namespace admission: each holdout entrypoint, executed as __main__,
   delegates to the frozen Phase A ``main`` with ONLY the admission
   callable swapped, and exits with the frozen main's return code.
4. The Phase A package itself keeps rejecting h95-* (admission swap is
   entrypoint-runtime-only, never installed package-wide).
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Phase A freeze (57dfcb7) sha256 of every file the campaign froze.
PHASE_A_FREEZE_SHA256 = {
    "benchmarks/inferswarm_97/reference_runner.py":
        "f04b515c0a9b3dbccc52f8f2b0fe96c705637c0d128421e4c2437a6bbffde46d",
    "benchmarks/inferswarm_97/chain_runner.py":
        "480f01b6a456f948ec327c0afdbfd84237c6efb1e903b8a3f94a1672c89df921",
    "benchmarks/inferswarm_97/last_stage_service.py":
        "79bed35acd48873f4513c47fe9a8ab5243b42e9cb041a9d723a47688d2be308b",
    "benchmarks/inferswarm_97/__init__.py":
        "183d8824838f0ced70284a4c180f1ad83321375c8e15473c8071e44da20e2db3",
}

for _p in (REPO, REPO / "python", REPO / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestPhaseAFilesUnchanged(unittest.TestCase):
    def test_every_frozen_phase_a_file_is_byte_identical(self):
        for rel, expected in PHASE_A_FREEZE_SHA256.items():
            with self.subTest(rel=rel):
                self.assertEqual(_sha256(REPO / rel), expected)


class TestHoldoutAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pkg = importlib.import_module("benchmarks.inferswarm_97b")

    def _case(self, case_id: str) -> dict:
        return {"case_id": case_id}

    def test_accepts_exactly_the_24_committed_cells(self):
        cases = [self._case(cid) for cid in self.pkg.FROZEN_HOLDOUT_CASE_IDS]
        out = self.pkg.validate_holdout_case_ids(cases)
        self.assertEqual(len(out), 24)
        self.assertEqual(
            sorted(c["case_id"] for c in out),
            sorted(self.pkg.FROZEN_HOLDOUT_CASE_IDS))

    def test_rejects_calibration_and_historical_namespaces(self):
        for forbidden in ("c95-01-01-01", "p95-01-03-01", "c86-00-00-00",
                          "p86-05-03-01", "c74-00-00-00", "h86-03-05-01",
                          "p76-02-01-01", "unversioned", ""):
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(ValueError):
                    self.pkg.validate_holdout_case_ids([self._case(forbidden)])

    def test_rejects_unknown_substituted_holdout_ids(self):
        for forbidden in ("h95-01-01-02", "h95-99-99-99", "h95-01-01",
                          "h95-05-01-01", "h95-1-1-1"):
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(ValueError):
                    self.pkg.validate_holdout_case_ids([self._case(forbidden)])

    def test_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            self.pkg.validate_holdout_case_ids(
                [self._case("h95-01-01-01"), self._case("h95-01-01-01")])

    def test_rejects_non_dict_rows(self):
        with self.assertRaises(ValueError):
            self.pkg.validate_holdout_case_ids(["h95-01-01-01"])


class TestPhaseAPackageUnaffected(unittest.TestCase):
    def test_phase_a_admission_still_rejects_h95(self):
        frozen = importlib.import_module("benchmarks.inferswarm_97")
        with self.assertRaises(ValueError):
            frozen.validate_campaign_case_ids([{"case_id": "h95-01-01-01"}])

    def test_phase_a_admission_still_accepts_calibration(self):
        frozen = importlib.import_module("benchmarks.inferswarm_97")
        accepted = [{"case_id": "c95-01-01-01"}, {"case_id": "p95-05-03-01"}]
        self.assertEqual(frozen.validate_campaign_case_ids(accepted), accepted)

    def test_importing_97b_does_not_touch_phase_a_admission(self):
        frozen = importlib.import_module("benchmarks.inferswarm_97")
        before = frozen.validate_campaign_case_ids
        importlib.import_module("benchmarks.inferswarm_97b")
        self.assertIs(frozen.validate_campaign_case_ids, before)


class TestEntrypointEquivalence(unittest.TestCase):
    """Each holdout entrypoint = frozen main + ONLY the admission swap.

    Executes the real entrypoint source as __main__ with the frozen
    module's ``main`` monkeypatched to record (a) the admission callable
    active at delegation time and (b) the argv it would parse, then
    asserts the process exit code equals the frozen main's rc. The
    frozen module's original admission is restored in finally blocks.
    """

    def _run_entrypoint(self, entry_name: str, frozen_mod_name: str) -> None:
        frozen = importlib.import_module(frozen_mod_name)
        holdout = importlib.import_module("benchmarks.inferswarm_97b")

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
            source = (REPO / "benchmarks" / "inferswarm_97b" /
                      entry_name).read_text()
            mod = types.ModuleType("holdout_entrypoint_under_test")
            mod.__dict__["__name__"] = "__main__"
            mod.__dict__["__file__"] = str(
                REPO / "benchmarks" / "inferswarm_97b" / entry_name)
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
            "benchmarks.inferswarm_97.reference_runner")

    def test_chain_entrypoint(self):
        self._run_entrypoint(
            "chain_runner_holdout.py",
            "benchmarks.inferswarm_97.chain_runner")

    def test_entrypoints_import_no_torch(self):
        import ast

        for name in ("reference_runner_holdout.py", "chain_runner_holdout.py"):
            with self.subTest(name=name):
                tree = ast.parse(
                    (REPO / "benchmarks" / "inferswarm_97b" /
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
                        m.startswith("benchmarks.inferswarm_76")
                        for m in imported),
                    f"{name} must import only admission + frozen runner")

    def test_holdout_package_contains_no_execution_math(self):
        """Source audit: 97b has no model/runtime/prefill calls."""
        import ast

        forbidden_calls = {
            "prefill", "generate", "forward", "reset_session_state",
            "arm_full_capture", "checkpoint_census", "freeze_dense_block_plan",
            "GemmaDenseStage", "I76StageClient", "I76LastStageClient",
        }
        for path in sorted((REPO / "benchmarks" / "inferswarm_97b")
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
