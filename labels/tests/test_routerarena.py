"""RouterArena @ the pinned revision is evaluation-only: loadable for eval and leakage, unreachable from training."""
import ast
import unittest
from pathlib import Path

from labels import routerarena as ra

ROOT = Path(__file__).resolve().parents[2]
TRAINING_TREES = ("train", "router", "atlas", "harness")


class RouterArena(unittest.TestCase):
    def test_the_splits_have_their_published_sizes(self):
        self.assertEqual({s: len(ra.load(s)) for s in ra.SPLITS}, {"full": 8400, "robustness": 420, "sub_10": 809})

    def test_question_text_is_context_plus_question_by_global_index(self):
        qs = ra.questions()
        self.assertEqual(len(qs), 8400 + 420)  # full and robustness; sub_10 is a subset of full
        row = ra.load("full")[0]
        self.assertIn(row["Question"], qs["routerarena/full/" + row["Global Index"]])

    def test_no_training_code_imports_routerarena(self):
        offenders = []
        for tree in TRAINING_TREES:
            for path in (ROOT / tree).rglob("*.py") if (ROOT / tree).exists() else ():
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                    if any("routerarena" in n for n in names) or (isinstance(node, ast.ImportFrom) and any(a.name == "routerarena" for a in node.names)):
                        offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
