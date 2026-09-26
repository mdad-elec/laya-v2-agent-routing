"""General routing labels from public item-level outcomes: the lowest ABILITY tier that solves an item.

Pre-registered (PREREG-g1):
- a model's ability on a dataset is its mean score over that dataset's TUNE items only;
- a dataset's models are split into thirds by ability (small / medium / powerful; ties by name);
- an item's label is the lowest tier whose models' MEAN score on it reaches TAU = 0.5;
- an item no tier reaches is `powerful` and flagged `unsolved`.
The soft label is the per-tier mean score. Ability, not price: public cost figures do not rank what
a model costs to USE (see labels/bfcl_result.py). Laya learns the capability a task needs; the Atlas
maps that onto the cheapest served cell with it.
"""
from __future__ import annotations

from collections import defaultdict

BANDS = ("small", "medium", "powerful")
TAU = 0.5


def ability_tiers(rows: list[dict], tune) -> dict[str, dict[str, str]]:
    """dataset -> {model: band}, from each model's mean score on the dataset's tune items."""
    sums: dict[tuple, list] = defaultdict(lambda: [0.0, 0])
    for r in rows:
        if tune(r["prompt_hash"]):
            acc = sums[(r["dataset"], r["model"])]
            acc[0] += r["score"]
            acc[1] += 1
    by_dataset: dict[str, list] = defaultdict(list)
    for (dataset, model), (total, n) in sums.items():
        by_dataset[dataset].append((total / n, model))
    tiers: dict[str, dict[str, str]] = {}
    for dataset, ranked in by_dataset.items():
        ranked.sort(key=lambda x: (x[0], x[1]))
        n = len(ranked)
        tiers[dataset] = {model: BANDS[min(2, (3 * i) // n)] for i, (_, model) in enumerate(ranked)}
    return tiers


def label_items(rows: list[dict], tiers: dict[str, dict[str, str]]) -> list[dict]:
    scores: dict[tuple, dict[str, list]] = defaultdict(lambda: {b: [] for b in BANDS})
    meta: dict[tuple, str] = {}
    for r in rows:
        band = tiers.get(r["dataset"], {}).get(r["model"])
        if band is None:
            continue
        key = (r["dataset"], r["native_id"])
        scores[key][band].append(r["score"])
        meta[key] = r["prompt_hash"]
    out = []
    for (dataset, native), per in scores.items():
        p = {b: (sum(v) / len(v) if v else None) for b, v in per.items()}
        band = next((b for b in BANDS if p[b] is not None and p[b] >= TAU), None)
        out.append({"dataset": dataset, "native_id": native, "prompt_hash": meta[(dataset, native)],
                    "band": band or "powerful", "unsolved": band is None, "p": p})
    return out
