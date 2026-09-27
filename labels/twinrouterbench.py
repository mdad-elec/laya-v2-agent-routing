"""TwinRouterBench's static bank (CommonstackAI/TwinRouterBench, Apache-2.0): frontier bar F1.

970 router-visible prefixes from 520 agent trajectories (SWE-bench Verified, BFCL v4, mtRAG,
QMSum, PinchBench). Each row is the conversation up to one routed call, labelled with the
cheapest of four pool tiers (low / mid / mid_high / high) that still lets the trajectory pass,
verified by degradation search and mixed-model execution. Pinned by commit and file sha256;
fetched into the cache, never committed.

- Splits are by TRAJECTORY (PREREG-g1 Amendment 1), so a trajectory never straddles tune/held.
- `router_view(row)` is all a router may read: the messages (and functions, if any). The label,
  the benchmark's name and the row's provenance fields stay with the scorer.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from suites.base import CACHE, read_jsonl, split_of

REPO = "CommonstackAI/TwinRouterBench"
COMMIT = "7cbb0deac8f697b5faa8489c309560e53d2ef088"
BANK_SHA256 = "5b4f90c24643b214a9b0f26bf4e05afc742554262f4ef405e0b3b4a4cce503f4"
TIERS = ("low", "mid", "mid_high", "high")
ROUTER_FIELDS = ("messages", "functions")


def bank_path() -> Path:
    path = CACHE.parent / "public" / "twinrouterbench" / COMMIT / "question_bank.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/data/static/question_bank.jsonl", tmp)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != BANK_SHA256:
        raise ValueError(f"{path}: sha256 {digest} is not the pinned {BANK_SHA256}")
    return path


def load() -> list[dict]:
    return read_jsonl(bank_path())


def trajectory_key(row: dict) -> str:
    return f"twinrouterbench/{row['benchmark']}/{row['instance_id']}"


def split(row: dict) -> str:
    return split_of(trajectory_key(row))


def router_view(row: dict) -> dict:
    return {k: row[k] for k in ROUTER_FIELDS if k in row}
