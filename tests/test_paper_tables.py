"""Tests for the publication-table generator.

The generator exists to stop hand-transcription errors reaching print, so the
property that matters is that it FORMATS locked artifacts and never recomputes
anything. A generator that re-derived a metric could silently drift from the
value that was actually locked when a library is upgraded.

These also pin the numbers the paper will quote, so an artifact edited by
accident fails here rather than in review.
"""

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import generate_paper_tables as generator  # noqa: E402

TABLES = REPO_ROOT / "docs" / "PAPER_TABLES.md"
CONFIRMATORY = REPO_ROOT / "outputs" / "fusion" / "FINAL_confirmatory_primary_evaluation.json"


class GeneratorReadsRatherThanRecomputesTests(unittest.TestCase):
    def test_generator_imports_no_numeric_or_statistical_library(self):
        """The structural guarantee that nothing can be recomputed.

        Checking the IMPORTS rather than grepping for words: "mcnemar_table" is a
        function that READS a McNemar result, and a substring match on "mcnemar"
        would flag it. But an AUC, a bootstrap or a Wilson interval cannot be
        derived without numpy, scipy, sklearn, statistics or math -- so if none
        of those is imported, the module can only be formatting values it was
        given.
        """
        import ast
        tree = ast.parse(Path(generator.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for library in ("numpy", "scipy", "sklearn", "statistics", "math", "pandas"):
            self.assertNotIn(library, imported,
                             f"generator imports {library}; it could recompute a metric")

    def test_generator_defines_no_statistical_helper(self):
        """Only table formatters and loaders may exist in the module."""
        import ast
        tree = ast.parse(Path(generator.__file__).read_text(encoding="utf-8"))
        defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        allowed_suffixes = ("_table", "_appendix")
        allowed_exact = {"load", "fmt", "main", "interval"}
        for name in defined:
            self.assertTrue(
                name.endswith(allowed_suffixes) or name in allowed_exact,
                f"unexpected helper {name!r}: only formatters and loaders belong here",
            )

    def test_generator_never_opens_the_primary_split(self):
        source = Path(generator.__file__).read_text(encoding="utf-8")
        for artifact in ("primary_window_scores.csv", "rwf2000", "RWF-2000"):
            self.assertNotIn(artifact, source)

    def test_absent_artifact_is_declared_not_invented(self):
        lines = generator.confusion_table(None)
        self.assertIn(generator.MISSING, lines)
        self.assertIn("omitted rather than invented", generator.MISSING)


@unittest.skipUnless(CONFIRMATORY.is_file(), "confirmatory artifact absent")
class TablesMatchTheLockedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(CONFIRMATORY.read_text(encoding="utf-8"))

    def test_confusion_rows_match_the_artifact_exactly(self):
        rendered = "\n".join(generator.confusion_table(self.report))
        for key in ("B0_temporal_only", "F2_rank_sum"):
            entry = self.report["results"][key]
            row = f"| {entry['tp']} | {entry['fp']} | {entry['tn']} | {entry['fn']} "
            self.assertIn(row, rendered, key)

    def test_locked_headline_numbers_are_present(self):
        rendered = "\n".join(generator.confusion_table(self.report))
        self.assertIn("| 160 | 43 | 151 | 40 ", rendered)   # B0
        self.assertIn("| 160 | 46 | 148 | 40 ", rendered)   # F2

    def test_mcnemar_row_matches_the_artifact(self):
        rendered = "\n".join(generator.mcnemar_table(self.report))
        overall = self.report["paired_comparisons"]["F2_rank_sum"]["overall"]
        self.assertIn(f"| {overall['b']} | {overall['c']} |", rendered)
        self.assertIn(f"{overall['p_value']:.6f}", rendered)

    def test_boolean_gate_auc_is_a_dash_not_a_number(self):
        rendered = "\n".join(generator.confusion_table(self.report))
        for line in rendered.splitlines():
            if line.startswith("| F1 or-gate"):
                self.assertTrue(line.rstrip().endswith("| — | — |"),
                                "F1 has no continuous score; AUC must be a dash")
                return
        self.fail("F1 or-gate row not rendered")

    def test_source_clustered_interval_is_reported_beside_the_clip_one(self):
        rendered = "\n".join(generator.interval_table(self.report))
        self.assertIn("Source video", rendered)
        self.assertIn("not independent", rendered)


class GeneratedFileTests(unittest.TestCase):
    @unittest.skipUnless(TABLES.is_file(), "tables not generated")
    def test_provenance_appendix_lists_real_hashes(self):
        text = TABLES.read_text(encoding="utf-8")
        self.assertIn("Provenance appendix", text)
        digests = re.findall(r"\| `([0-9a-f]{64})` \|", text)
        self.assertGreaterEqual(len(digests), 4)

    @unittest.skipUnless(TABLES.is_file(), "tables not generated")
    def test_file_warns_against_hand_editing(self):
        text = TABLES.read_text(encoding="utf-8")
        self.assertIn("do not edit by hand", text.lower())
        self.assertIn("Nothing", text)

    @unittest.skipUnless(TABLES.is_file(), "tables not generated")
    def test_development_figures_are_labelled_as_such(self):
        """Carve numbers must never read as results."""
        text = TABLES.read_text(encoding="utf-8")
        self.assertIn("Not a performance claim", text)
        self.assertIn("the clips the thresholds were selected on", text)

    @unittest.skipUnless(TABLES.is_file(), "tables not generated")
    def test_regenerates_deterministically_apart_from_the_timestamp(self):
        before = TABLES.read_text(encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "generate_paper_tables.py")],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        after = TABLES.read_text(encoding="utf-8")
        strip = lambda text: [l for l in text.splitlines() if not l.startswith("Regenerated ")]
        self.assertEqual(strip(before), strip(after))


if __name__ == "__main__":
    unittest.main()
