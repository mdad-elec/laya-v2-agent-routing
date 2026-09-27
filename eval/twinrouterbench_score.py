"""F1's scorer: TwinRouterBench's own `main/eval/section11.compute_v2_scores`, at the pinned commit.

The source tree is fetched as the commit's tarball into the cache and imported from there, so
the numbers are the benchmark's, not a re-implementation. `score(rows, predictions)` builds the
per-row records exactly as its `main/eval/runner.py` does and refuses a row with no prediction.
"""
from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

from labels.twinrouterbench import COMMIT, REPO
from suites.base import CACHE


def checkout() -> Path:
    root = CACHE.parent / "public" / "twinrouterbench-src" / COMMIT
    if not (root / "main" / "eval" / "section11.py").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        archive = root.parent / f"{COMMIT}.tar.gz"
        if not archive.exists():
            urllib.request.urlretrieve(f"https://github.com/{REPO}/archive/{COMMIT}.tar.gz", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(root.parent / "_x", filter="data")
        next((root.parent / "_x").iterdir()).rename(root)
        (root.parent / "_x").rmdir()
    return root


def _section11():
    root = str(checkout())
    if root not in sys.path:
        sys.path.insert(0, root)
    from main.eval import section11

    return section11


def score(rows: list[dict], predictions: dict[str, int]) -> dict:
    records = []
    for r in rows:
        pred = predictions[r["id"]]
        gold = r["target_tier_id"]
        records.append({"id": r["id"], "benchmark": r["benchmark"], "gold_tier_id": gold, "pred_tier_id": int(pred),
                        "match": pred == gold, "passed": pred >= gold, "instance_id": r.get("instance_id", r["id"]),
                        "step_index": r.get("step_index", 1), "total_steps": r.get("total_steps", 1), "messages": r.get("messages", [])})
    return _section11().compute_v2_scores(records)
