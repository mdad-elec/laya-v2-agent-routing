"""Avengers-Pro (F3 baseline): LLMRouterBench's `balance_cluster_router.py` @ c77cb050, unmodified.

PREREG-g1 Amendment 4: its balance_config values (25 clusters, top_k 3, β 9, seed 42, max_router 1),
performance_weight = α and cost_sensitivity = 1 − α. Its class-level query-embedding cache is
filled with OUR vectors and its network embedder is replaced by one that raises, so a query we
did not embed stops the run instead of silently calling a remote API.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from eval.baselines.lrb_code import checkout


class _NoNetwork:
    model_name = "offline"

    def generate_embedding(self, text):
        raise RuntimeError(f"Avengers-Pro asked for an embedding we did not provide: {text[:80]!r}")


def _records(items: list[dict]) -> list[dict]:
    return [{"query": it["query"], "dataset": it["dataset"], "index": i, "records": dict(it["scores"]),
             "usages": {m: {"cost": c, "prompt_tokens": 0, "completion_tokens": 0} for m, c in it["costs"].items()}}
            for i, it in enumerate(items)]


def route(train: list[dict], test: list[dict], embeddings: dict[str, np.ndarray], alpha: float, n_clusters: int = 25) -> dict[tuple, str]:
    checkout()
    from baselines.AvengersPro.balance_cluster_router import BalanceClusterRouter
    from baselines.AvengersPro.config import SimpleClusterConfig
    from baselines.AvengersPro.simple_cluster_router import SimpleClusterRouter

    missing = [it["query"] for it in train + test if it["query"] not in embeddings]
    if missing:
        raise RuntimeError(f"{len(missing)} queries have no embedding, e.g. {missing[0][:80]!r}")
    SimpleClusterRouter._global_embedding_cache.clear()
    SimpleClusterRouter._global_embedding_cache.update({q: np.asarray(v, dtype=float) for q, v in embeddings.items()})
    original = SimpleClusterRouter._create_embedding_generator
    SimpleClusterRouter._create_embedding_generator = lambda self: _NoNetwork()
    try:
        with tempfile.TemporaryDirectory(prefix="laya-g1-avengers-") as tmp:
            paths = {name: Path(tmp) / f"{name}.jsonl" for name in ("train", "test")}
            for name, rows in (("train", _records(train)), ("test", _records(test))):
                paths[name].write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            (Path(tmp) / "baseline_scores.json").write_text("{}", encoding="utf-8")
            cfg = SimpleClusterConfig(train_data_path=str(paths["train"]), test_data_path=str(paths["test"]), baseline_scores_path=str(Path(tmp) / "baseline_scores.json"),
                                      seed=42, n_clusters=n_clusters, max_router=1, top_k=3, beta=9.0, max_workers=1,
                                      embedding_model="Qwen3-Embedding-0.6B", performance_weight=alpha, cost_sensitivity=1 - alpha)
            router = BalanceClusterRouter(cfg)
            router.build_cluster_model(_records(train))
            picks = router.route_queries_batch([it["query"] for it in test])
    finally:
        SimpleClusterRouter._create_embedding_generator = original
    return {(it["dataset"], it["native_id"]): p[0] for it, p in zip(test, picks, strict=True)}
