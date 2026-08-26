"""Tests for the generated self-contained Kaggle bootstrap cell.

The cell is the only artefact that reaches Kaggle, so it must stay in
sync with the source tree and must remain independent of PyPI. These
tests pin both properties down without needing Kaggle.
"""

import ast
import base64
import io
import sys
import unittest
import zipfile
import zlib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from build_kaggle_cell import (  # noqa: E402
    ENTRY_MODULES,
    OUTPUT_FILE,
    build_cell,
    build_payload,
    required_modules,
)


class ModuleClosureTests(unittest.TestCase):
    """The embedded module list is derived, never hardcoded."""

    def test_closure_contains_both_entry_points(self):
        modules = required_modules()

        for entry in ENTRY_MODULES:
            self.assertIn(entry, modules)

    def test_closure_pulls_in_transitive_dependencies(self):
        modules = required_modules()

        for dependency in (
            "config",
            "dataset_inspection",
            "prepare_rwf2000_dataset",
            "rwf2000_config",
            "rwf2000_splits",
            "video_loader",
        ):
            self.assertIn(dependency, modules)

    def test_closure_excludes_the_heavy_model_stack(self):
        # The bootstrap must not need torch: it only prepares data.
        modules = required_modules()

        for excluded in (
            "temporal_training",
            "temporal_model",
            "temporal_preprocessing",
            "rwf2000_dataset",
        ):
            self.assertNotIn(excluded, modules)

    def test_closure_excludes_the_surveillance_pipeline(self):
        modules = required_modules()

        for excluded in ("tracker", "detection", "risk_assessment", "yolo_gradcam"):
            self.assertNotIn(excluded, modules)


class PayloadTests(unittest.TestCase):
    """The payload must round-trip to the exact module sources."""

    def test_payload_round_trips_every_module(self):
        modules = required_modules()

        blob = zlib.decompress(base64.b64decode(build_payload(modules)))
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            names = sorted(archive.namelist())
            self.assertEqual(names, sorted(f"{m}.py" for m in modules))
            for module in modules:
                self.assertEqual(
                    archive.read(f"{module}.py").decode("utf-8"),
                    (REPOSITORY_ROOT / "src" / f"{module}.py").read_text(encoding="utf-8"),
                )

    def test_every_embedded_module_is_valid_python(self):
        modules = required_modules()
        blob = zlib.decompress(base64.b64decode(build_payload(modules)))

        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            for name in archive.namelist():
                ast.parse(archive.read(name).decode("utf-8"), filename=name)


class GeneratedCellTests(unittest.TestCase):
    """Properties the shipped cell must hold."""

    @classmethod
    def setUpClass(cls):
        cls.cell = build_cell()

    def test_cell_is_valid_python(self):
        ast.parse(self.cell)

    def test_committed_cell_is_up_to_date(self):
        self.assertTrue(OUTPUT_FILE.is_file(), "kaggle_bootstrap_cell.py is missing")
        self.assertEqual(
            OUTPUT_FILE.read_text(encoding="utf-8"),
            self.cell,
            "kaggle_bootstrap_cell.py is stale; re-run tools/build_kaggle_cell.py",
        )

    def test_cell_never_installs_or_imports_py7zr(self):
        # The prose may mention py7zr (it explains why it is not used);
        # what must not appear is an install or an import of it.
        lowered = self.cell.lower()

        for forbidden in (
            "import py7zr",
            "install py7zr",
            "pip install",
            "pip3 install",
            "pip -q install",
        ):
            self.assertNotIn(forbidden, lowered)

    def test_embedded_modules_never_import_py7zr(self):
        blob = zlib.decompress(base64.b64decode(build_payload(required_modules())))

        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            for name in archive.namelist():
                source = archive.read(name).decode("utf-8")
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self.assertNotEqual(alias.name, "py7zr", name)
                    elif isinstance(node, ast.ImportFrom):
                        self.assertNotEqual(node.module, "py7zr", name)

    def test_cell_carries_no_placeholder(self):
        self.assertNotIn("__RWF2000_PAYLOAD__", self.cell)

    def test_cell_declares_the_kaggle_defaults(self):
        for expected in (
            "/kaggle/input",
            "/kaggle/working/rwf2000",
            "RESULT: READY",
            "RESULT: NOT READY",
        ):
            self.assertIn(expected, self.cell)

    def test_cell_requests_validation(self):
        # A bootstrap that skipped validation could report success on a
        # dataset that is not actually usable.
        self.assertIn("validate=True", self.cell)

    def test_cell_does_not_start_training(self):
        for forbidden in ("temporal_training", "run_training", "r3d_18", "--train"):
            self.assertNotIn(forbidden, self.cell)


if __name__ == "__main__":
    unittest.main()
