"""SR-KNN, TwinRouterBench's reference router, reproduced from its adapter
(swerouter/routers/sr_knn_adapter.py @ the pinned commit): each prefix is flattened to
"role:\\ncontent" blocks, embedded with Qwen/Qwen3-Embedding-0.6B (unnormalised, 1024-d), and a
constant 14-way category one-hot ("other") is appended; neighbours are by cosine.

The paper's static 77.89 is "an in-sample 1-nearest-neighbor upper bound over question-bank
embeddings". `predict(..., leave_one_out=True)` is the in-sample reading that excludes the row
itself; the F1 baseline is the same router with its index built from the TUNE rows only.
A k-vote tie goes to the label of the nearest tied neighbour.
"""
from __future__ import annotations

import numpy as np

CATEGORIES = ("biology", "business", "chemistry", "computer science", "economics", "engineering", "health", "history", "law", "math", "other",
              "philosophy", "physics", "psychology")


def features(embeddings: np.ndarray, category: str = "other") -> np.ndarray:
    one_hot = np.zeros((embeddings.shape[0], len(CATEGORIES)), dtype=np.float32)
    one_hot[:, CATEGORIES.index(category)] = 1.0
    return np.concatenate([embeddings.astype(np.float32), one_hot], axis=1)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def predict(index_x: np.ndarray, index_y: np.ndarray, query_x: np.ndarray, k: int = 1, leave_one_out: bool = False) -> np.ndarray:
    sims = _unit(query_x) @ _unit(index_x).T
    if leave_one_out:
        np.fill_diagonal(sims, -np.inf)
    order = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    out = []
    for row in order:
        labels = [int(index_y[j]) for j in row]
        counts = {lab: labels.count(lab) for lab in labels}
        best = max(counts.values())
        out.append(next(lab for lab in labels if counts[lab] == best))
    return np.array(out)
