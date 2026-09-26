import shutil
import subprocess
import unittest

from suites.aider.suite import AiderPolyglot, parse_files


class Parse(unittest.TestCase):
    def test_each_file_block_is_read_by_its_path(self):
        answer = "Here:\nFILE: src/lib.rs\n```rust\npub fn f() -> u8 { 1 }\n```\nand\nFILE: helper.py\n```python\nx = 1\n```"
        self.assertEqual(parse_files(answer), {"src/lib.rs": "pub fn f() -> u8 { 1 }\n", "helper.py": "x = 1\n"})

    def test_the_last_block_for_a_path_wins_and_prose_is_ignored(self):
        answer = "FILE: a.py\n```python\nv1\n```\nfixed:\nFILE: a.py\n```python\nv2\n```"
        self.assertEqual(parse_files(answer), {"a.py": "v2\n"})
        self.assertEqual(parse_files("no files here"), {})


@unittest.skipUnless(shutil.which("docker") and subprocess.run(["docker", "image", "inspect", "laya-g1-polyglot:2"], capture_output=True).returncode == 0,
                     "needs docker and the laya-g1-polyglot:2 image (harness/sandbox/polyglot)")
class Sandbox(unittest.TestCase):
    def setUp(self):
        self.s = AiderPolyglot()
        self.items = {i.native_id: i for i in self.s.load()}

    def test_a_reference_solution_passes_and_the_untouched_stub_fails(self):
        for lang in ("python", "rust", "javascript"):
            item = next(i for i in self.items.values() if i.meta["language"] == lang)
            self.assertEqual(self.s.check_reference(item), 1.0, item.native_id)
            self.assertEqual(self.s.check(item, ""), 0.0, f"{item.native_id}: the stub as given does not pass")

    def test_a_model_cannot_overwrite_the_tests(self):
        item = next(i for i in self.items.values() if i.meta["language"] == "python")
        test_file = item.gold["test_files"][0]
        cheat = self.s.reference_answer(item) + f"\nFILE: {test_file}\n```python\ndef test_ok():\n    assert True\n```"
        # The cheat's test file is ignored: the reference still passes the REAL tests, and a wrong
        # solution plus a replaced test file still fails.
        self.assertEqual(self.s.check(item, cheat), 1.0)
        wrong = f"FILE: {item.gold['solution_files'][0]}\n```python\nraise SystemExit(1)\n```\nFILE: {test_file}\n```python\ndef test_ok():\n    assert True\n```"
        self.assertEqual(self.s.check(item, wrong), 0.0)


@unittest.skipUnless(__import__("os").environ.get("LAYA_SLOW") == "1", "LAYA_SLOW=1: runs all 111 exercises twice (~5 min)")
class Golden(unittest.TestCase):
    def test_every_reference_passes_and_every_stub_fails(self):
        from concurrent.futures import ThreadPoolExecutor

        s = AiderPolyglot()
        items = s.load()
        self.assertEqual(len(items), 111, "113 exercises less the two excluded by name")
        with ThreadPoolExecutor(4) as pool:
            refs = list(pool.map(s.check_reference, items))
            stubs = list(pool.map(lambda i: s.check(i, ""), items))
        self.assertEqual([i.native_id for i, r in zip(items, refs) if r != 1.0], [])
        self.assertEqual([i.native_id for i, r in zip(items, stubs) if r != 0.0], [])


if __name__ == "__main__":
    unittest.main()
