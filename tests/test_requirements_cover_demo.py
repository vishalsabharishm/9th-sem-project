"""Regression test: requirements must cover the demo entry point.

Found during the demo-readiness run. Every runtime import the dashboard needs
was installed on the development machine, so the demo worked -- but flask was
absent from requirements.txt, meaning a fresh environment could follow the
documented setup exactly and still fail at launch with ImportError.
"""
import re, sys, unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "requirements.txt"
SERVER = REPO_ROOT / "web" / "server.py"

# The interpreter's own list is the source of truth, so a newly used stdlib
# module can never be mistaken for an undeclared dependency. The literals below
# only supplement it (`__future__` is not in stdlib_module_names).
#
# This was a hand-maintained set, and it drifted: importing `uuid` in
# web/server.py failed a test whose actual purpose is to catch undeclared
# THIRD-PARTY imports. Deriving it removes that whole class of false positive
# without loosening what the test checks.
STDLIB_OR_LOCAL = set(sys.stdlib_module_names) | {"__future__"}


def declared():
    names = set()
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.add(re.split(r"[<>=!\[]", line)[0].strip().lower())
    return names


class RequirementsCoverTheDemoTests(unittest.TestCase):
    def test_flask_is_declared(self):
        self.assertIn("flask", declared(),
                      "web/server.py imports flask; a fresh env would fail at launch")

    def test_every_third_party_server_import_is_declared(self):
        import ast
        tree = ast.parse(SERVER.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])

        alias = {"cv2": "opencv-python", "PIL": "pillow", "yaml": "pyyaml"}
        local = {p.stem for p in (REPO_ROOT / "src").glob("*.py")}
        local |= {p.stem for p in (REPO_ROOT / "tools").glob("*.py")}

        names = declared()
        for module in sorted(imported):
            if module in STDLIB_OR_LOCAL or module in local:
                continue
            expected = alias.get(module, module).lower()
            self.assertIn(expected, names,
                          f"web/server.py imports {module!r} but {expected!r} "
                          "is not in requirements.txt")


if __name__ == "__main__":
    unittest.main()
