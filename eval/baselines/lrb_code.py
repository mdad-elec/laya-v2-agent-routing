"""LLMRouterBench's CODE (ynulihao/LLMRouterBench), pinned by commit, for F3's baselines.

Fetched as the commit's tarball into the cache and put on sys.path, so the baselines run the
benchmark's own implementations. The repository carries no LICENSE file (checked 2026-09-27):
it is run locally for evaluation, never vendored or redistributed.
"""
from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

from suites.base import CACHE

REPO = "ynulihao/LLMRouterBench"
COMMIT = "c77cb0506949d8f959e97967d2fefca0e8ff1b05"


def checkout() -> Path:
    root = CACHE.parent / "public" / "llmrouterbench-src" / COMMIT
    if not (root / "baselines" / "AvengersPro" / "balance_cluster_router.py").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        archive = root.parent / f"{COMMIT}.tar.gz"
        if not archive.exists():
            urllib.request.urlretrieve(f"https://github.com/{REPO}/archive/{COMMIT}.tar.gz", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(root.parent / "_x", filter="data")
        next((root.parent / "_x").iterdir()).rename(root)
        (root.parent / "_x").rmdir()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
