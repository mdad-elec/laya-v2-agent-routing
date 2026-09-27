"""PREREG-g1 is append-only and every g1 result names the version it ran under."""
import json
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "PREREG-g1.md"


def prereg_blobs() -> set[str]:
    """Every blob PREREG-g1.md has had in history, plus the working copy's."""
    revs = subprocess.run(["git", "log", "--format=%H", "--", "PREREG-g1.md"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    blobs = {subprocess.run(["git", "rev-parse", f"{r}:PREREG-g1.md"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip() for r in revs}
    blobs.add(subprocess.run(["git", "hash-object", str(PREREG)], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip())
    return blobs


class PreregTest(unittest.TestCase):
    def test_every_result_names_a_prereg_version(self):
        blobs = prereg_blobs()
        results = sorted((ROOT / "results" / "g1").glob("*.json"))
        self.assertTrue(results)
        for path in results:
            with self.subTest(path.name):
                self.assertIn(json.loads(path.read_text())["prereg_sha"], blobs)

    def test_the_original_text_is_never_edited(self):
        """Amendments append: the first committed version is a prefix of the current file."""
        first = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H", "--", "PREREG-g1.md"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()[-1]
        original = subprocess.run(["git", "show", f"{first}:PREREG-g1.md"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        self.assertTrue(PREREG.read_text().startswith(original))

    def test_the_frontier_amendment_fixes_its_bars(self):
        text = PREREG.read_text()
        m = re.search(r"^## Amendment 1 \((\d{4}-\d{2}-\d{2})\)", text, re.M)
        self.assertIsNotNone(m, "Amendment 1 missing")
        amendment = text[m.start():]
        for needle in ("F1", "F2", "F3", "TwinRouterBench", "7cbb0deac8f697b5faa8489c309560e53d2ef088",
                       "RouterArena", "cff9659dd6a3a07815f4842fcdadea5a597a1df1", "77.63", "in-sample"):
            self.assertIn(needle, amendment)


if __name__ == "__main__":
    unittest.main()
