"""Aider polyglot (Exercism exercises, MIT): write code that passes an exercise's tests -- `code`.

Exercises come from github.com/Aider-AI/polyglot-benchmark at a pinned commit (Exercism's MIT
tracks; the benchmark repo itself carries no licence file, so this is on the licence sign-off).
Python, Rust and JavaScript: 16 per language by default.

The model gets the exercise's instructions and its stub file(s), and returns each solution file in
full after a `FILE: <path>` line. Only the exercise's declared SOLUTION files are written; a block
for a test file, or any other path, is ignored, so the tests a model is scored by are always the
exercise's own. The tests run in the laya-g1-polyglot sandbox image (harness/sandbox/polyglot)
with --network none and CPU, memory and time limits; skipped tests are un-skipped, as Aider's own
harness does.

Metric: pass@1, one attempt, no test output shown. Aider's leaderboard reports a second attempt
after the failing test output; this is deliberately the stricter single-shot number, named so.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from suites.base import CACHE, Item, Suite

REVISION = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
IMAGE = "laya-g1-polyglot:2"  # harness/sandbox/polyglot; :2 = Rust 1.89 + the track's crates prefetched
LANGUAGES = ("python", "rust", "javascript")
TIMEOUT_S = 180
# Measured 2026-09-26 on laya-g1-polyglot:2 over all 113 exercises: every other reference passes its
# own tests and every other stub fails them untouched.
EXCLUDED = {
    "rust/react": "the reference solution does not compile on a current toolchain (a doc comment documenting nothing)",
    "javascript/ledger": "a refactoring exercise: the stub already passes its tests untouched, so it cannot tell a solution from none",
}
_FILE = re.compile(r"FILE:\s*(\S+)\s*\n```[^\n]*\n(.*?)```", re.S)
_TEST_CMD = {
    "python": "python -m pytest -q -p no:cacheprovider {tests}",
    "rust": "CARGO_TARGET_DIR=/tmp/target cargo test --offline --quiet -- --include-ignored",
    "javascript": "ln -sf /opt/js/node_modules node_modules && node /opt/js/node_modules/.bin/jest --ci --rootDir /work {tests}",
}


def parse_files(answer: str) -> dict[str, str]:
    return {path: body for path, body in _FILE.findall(answer)}


def _unskip(language: str, text: str) -> str:
    if language == "javascript":
        return re.sub(r"\bxtest\(", "test(", text)
    if language == "python":
        return re.sub(r"^\s*@pytest\.mark\.skip.*$", "", text, flags=re.M)
    return text  # rust: `--include-ignored` runs #[ignore] tests


class AiderPolyglot(Suite):
    name, domain, licence = "aider_polyglot", "code", "MIT (Exercism tracks)"
    source = "github.com/Aider-AI/polyglot-benchmark"
    revision = REVISION
    notes = "pass@1, single attempt (Aider's leaderboard is a 2-attempt number); python/rust/javascript"

    def repo(self) -> Path:
        root = CACHE / self.name / self.revision / "repo"
        if not root.exists():
            import tarfile
            import urllib.request

            archive = root.parent / "src.tar.gz"
            root.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists():
                urllib.request.urlretrieve(f"https://github.com/Aider-AI/polyglot-benchmark/archive/{REVISION}.tar.gz", archive)
            with tarfile.open(archive) as tar:
                tar.extractall(root.parent / "_x")
            shutil.move(str(next((root.parent / "_x").iterdir())), root)
            shutil.rmtree(root.parent / "_x", ignore_errors=True)
        return root

    def load(self) -> list[Item]:
        items = []
        for language in LANGUAGES:
            for ex in sorted((self.repo() / language / "exercises" / "practice").iterdir()):
                config = json.loads((ex / ".meta" / "config.json").read_text())
                files = config["files"]
                docs = ex / ".docs"
                instructions = (docs / "instructions.md").read_text()
                if (docs / "instructions.append.md").exists():
                    instructions += "\n" + (docs / "instructions.append.md").read_text()
                stubs = "\n\n".join(f"FILE: {f}\n```\n{(ex / f).read_text()}```" for f in files["solution"])
                prompt = (f"Solve this {language} exercise.\n\n{instructions}\n\nThe starting file(s):\n\n{stubs}\n\n"
                          f"Return the COMPLETE contents of every file you change, each after a line `FILE: <path>` in a fenced "
                          f"code block. Keep the file names and the public interface the tests expect.")
                if f"{language}/{ex.name}" in EXCLUDED:
                    continue
                items.append(Item(suite=self.name, native_id=f"{language}/{ex.name}", messages=[{"role": "user", "content": prompt}],
                                  gold={"dir": str(ex), "solution_files": files["solution"], "test_files": files["test"],
                                        "example_files": files.get("example", [])},
                                  meta={"language": language}))
        return items

    def check_reference(self, item: Item) -> float:
        """The golden path: the exercise's own reference solution, WITH its Cargo-example.toml when
        it has one (a reference may use crates the stub's manifest does not list). A model's answer
        never gets this: check() writes only the declared solution files."""
        extra = {}
        manifest = Path(item.gold["dir"]) / ".meta" / "Cargo-example.toml"
        if item.meta["language"] == "rust" and manifest.exists():
            extra["Cargo.toml"] = manifest.read_text()
        return self._run(item, {**parse_files(self.reference_answer(item)), **extra}, set(item.gold["solution_files"]) | set(extra))

    def reference_answer(self, item: Item) -> str:
        """The exercise's own reference solution, in the answer format -- the golden check's input."""
        ex = Path(item.gold["dir"])
        examples = [ex / e for e in item.gold["example_files"]]
        blocks = []
        for sol, example in zip(item.gold["solution_files"], examples):
            blocks.append(f"FILE: {sol}\n```\n{example.read_text()}```")
        return "\n\n".join(blocks)

    def check(self, item: Item, answer: str) -> float:
        return self._run(item, parse_files(answer), set(item.gold["solution_files"]))

    def _run(self, item: Item, written: dict[str, str], allowed: set[str]) -> float:
        language = item.meta["language"]
        with tempfile.TemporaryDirectory(prefix="laya-g1-aider-") as tmp:
            work = Path(tmp) / "work"
            shutil.copytree(item.gold["dir"], work, ignore=shutil.ignore_patterns(".meta", ".docs", "node_modules"))
            for path, body in written.items():
                if path in allowed:
                    (work / path).parent.mkdir(parents=True, exist_ok=True)
                    (work / path).write_text(body)
            for test in item.gold["test_files"]:
                p = work / test
                if p.exists():
                    p.write_text(_unskip(language, p.read_text()))
            cmd = _TEST_CMD[language].format(tests=" ".join(item.gold["test_files"]))
            try:
                # As the host's own unprivileged uid, so the container can write the throwaway copy
                # (cargo writes Cargo.lock) and whatever it writes is the host's to clean up.
                import os

                done = subprocess.run(["docker", "run", "--rm", "--network", "none", "--cpus", "2", "--memory", "2g",
                                       "--pids-limit", "256", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
                                       "-v", f"{work}:/work", IMAGE, "sh", "-c", cmd],
                                      capture_output=True, text=True, timeout=TIMEOUT_S)
            except subprocess.TimeoutExpired:
                return 0.0
            return 1.0 if done.returncode == 0 else 0.0

    def stratum(self, item: Item) -> str:
        return item.meta["language"]
