"""RouterArena (RouteWorks/RouterArena, Apache-2.0): frontier bar F2. EVALUATION-ONLY.

Its authors reject any submission that trains, fits or tunes a router component on its data, so
this module exists for two callers only: the F2 evaluation, and the leakage index that removes
RouterArena questions from our training prompts (labels.contamination). A test forbids every
training tree (train/, router/, atlas/, harness/) from importing it.

Pinned by HF dataset revision and per-file sha256; cached, never committed.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from suites.base import CACHE

DATASET = "RouteWorks/RouterArena"
REVISION = "a4a062ce3313b56bb09c042e1bc37b61d34e3bd8"
SHA256 = {
    "full": "358a6bb4160320b5cb5b68d6d0a885c89b128b06d3dea267d77ed6daeebe4369",
    "robustness": "6012eeea11e2d7b2018155e65c35ad887e9daa70e45aa38a40321da74d0cc2e9",
    "sub_10": "e7b4b7579e98df0b83202324357d0b91af78666ddbf1038a513d19edaf58efd5",
}
SPLITS = tuple(SHA256)


def _path(split: str) -> Path:
    path = CACHE.parent / "public" / "routerarena" / REVISION / f"{split}.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{split}-00000-of-00001.parquet", tmp)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256[split]:
        raise ValueError(f"{path}: sha256 {digest} is not the pinned {SHA256[split]}")
    return path


def load(split: str) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(_path(split)).to_pylist()


def questions() -> dict[str, str]:
    """Every question a router could be scored on (full + robustness), as Context + Question, for
    the leakage index. sub_10 is a subset of full and adds nothing."""
    out = {}
    for split in ("full", "robustness"):
        for row in load(split):
            out[f"routerarena/{split}/{row['Global Index']}"] = f"{row.get('Context') or ''}\n{row['Question']}".strip()
    return out
