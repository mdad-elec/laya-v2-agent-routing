"""Phase 1, F3's second ceiling read (PREREG-g1 Amendment 8): task-conditioned kNN over per-model
outcomes, shrunk toward the task row.

Same items, splits, baselines, formulas and gate as `eval/ceiling_f3.py` (which is imported, not
copied). The estimator for an item q with task t:

    P(model | q) = (Σ_{j ∈ N_k(q, t)} score_j(model) + m · row_t(model)) / (|N_k| + m)

N_k(q, t) = the k nearest HC-train items of the same task by cosine over the baselines' Qwen3
vectors. (k, m) is chosen on HC-train by 5-fold out-of-fold best AvgAcc from the Amendment 8 menu;
the cross-task variant (neighbours from every task) is reported and never chosen. HC-test is read
for the second time for F3; both reads are reported.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval import ceiling_f3 as c3
from eval import llmrouterbench_f3 as f3
from eval.baselines import avengers_pro, embedllm
from eval.baselines.f3_predict import GRID, load_vectors
from labels.gate_s3 import prereg_sha
from train.policy import cheapest_above

K_GRID = (8, 16, 32, 64)
M_GRID = (1.0, 4.0, 16.0)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class KnnTable:
    def __init__(self, k: int, m: float, by_task: bool) -> None:
        self.k, self.m, self.by_task = k, m, by_task

    def fit(self, items: list[dict], vec: dict[str, np.ndarray]) -> KnnTable:
        self.row = c3.ScoreTable(c3.GROUPINGS["task"], 0.0).fit(items)
        pools: dict[object, list[dict]] = defaultdict(list)
        for it in items:
            pools[it["dataset"] if self.by_task else None].append(it)
        self.pools = {}
        for key_, pool in pools.items():
            X = _unit(np.stack([vec[it["prompt_hash"]] for it in pool]))
            S = np.array([[it["scores"][mdl] for mdl in f3.MODELS] for it in pool])
            C = np.array([[it["costs"][mdl] for mdl in f3.MODELS] for it in pool])
            self.pools[key_] = (X, S, C)
        return self

    def predict(self, items: list[dict], vec: dict[str, np.ndarray]) -> list[tuple[dict, dict]]:
        out = []
        for it in items:
            row_p, row_c = self.row.predict(it)
            pool = self.pools.get(it["dataset"] if self.by_task else None)
            if pool is None:
                out.append((row_p, row_c))
                continue
            X, S, C = pool
            sims = X @ _unit(vec[it["prompt_hash"]][None, :])[0]
            nn = np.argsort(-sims, kind="stable")[: self.k]
            n = len(nn)
            p = {mdl: (S[nn, j].sum() + self.m * row_p[mdl]) / (n + self.m) for j, mdl in enumerate(f3.MODELS)}
            cost = {mdl: (C[nn, j].sum() + self.m * row_c[mdl]) / (n + self.m) for j, mdl in enumerate(f3.MODELS)}
            out.append((p, cost))
        return out


def configs(fit, query, vec, k: int, m: float, by_task: bool) -> dict[str, dict[tuple, str]]:
    preds = KnnTable(k, m, by_task).fit(fit, vec).predict(query, vec)
    return {f"tau={tau}": {c3.k(it): cheapest_above(p, c, tau) for it, (p, c) in zip(query, preds, strict=True)} for tau in GRID}


def choose(train: list[dict], vec, by_task: bool) -> tuple[int, float, dict]:
    fold = {c3.k(it): c3._pct("g1-hc-fold/", it["prompt_hash"], 5) for it in train}
    scores = {}
    for k in K_GRID:
        for m in M_GRID:
            merged: dict[str, dict] = defaultdict(dict)
            for f in range(5):
                q = [it for it in train if fold[c3.k(it)] == f]
                fit = [it for it in train if fold[c3.k(it)] != f]
                for cfg, ch in configs(fit, q, vec, k, m, by_task).items():
                    merged[cfg].update(ch)
            scores[f"k={k}/m={m}"] = round(c3.best_acc(train, merged), 4)
    best = max(scores, key=lambda s: (scores[s], s))
    k_, m_ = best.split("/")
    return int(k_[2:]), float(m_[2:]), scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/g1/ceiling-f3-knn.json")
    args = ap.parse_args()

    from labels import routerarena as ra
    from labels import twinrouterbench as trb
    from labels.leakage import build_eval_index
    from labels.leakage_report import suite_items

    index = build_eval_index(ra.questions(), trb.load(), suite_items())
    tune = [it for it in f3.items() if it["split"] == "tune"]
    usable = [it for it in tune if not c3.is_calibration(it) and not index.hits(it["query"])]
    train = [it for it in usable if not c3.is_hc_test(it)]
    test = [it for it in usable if c3.is_hc_test(it)]
    vectors = load_vectors(c3.EMBEDDINGS)

    k, m, menu = choose(train, vectors, by_task=True)
    kx, mx, menu_x = choose(train, vectors, by_task=False)
    name, m_row, _ = c3.choose_table(train)
    routers = {"ceiling_knn": configs(train, test, vectors, k, m, True),
               "ceiling_table (read 1)": c3.table_configs(train, test, c3.GROUPINGS[name], m_row),
               "knn_no_task (reported)": configs(train, test, vectors, kx, mx, False)}

    emb = {it["query"]: vectors[it["prompt_hash"]] for it in train + test}
    routers["avengers_pro"] = {f"alpha={a}": avengers_pro.route(train, test, emb, alpha=a) for a in GRID}
    epochs, _ = embedllm.choose_epochs(train, emb)
    model = embedllm.fit(train, emb, epochs=epochs)
    cost = embedllm.expected_costs(train)
    probs = model.predict([emb[it["query"]] for it in test])
    routers["embedllm"] = {f"tau={tau}": {c3.k(it): embedllm.pick(p, cost, tau) for it, p in zip(test, probs, strict=True)} for tau in GRID}

    singles = {mdl: {c3.k(it): mdl for it in test} for mdl in f3.MODELS}
    best_single = max(f3.MODELS, key=lambda mdl: (f3.avg_acc(train, {c3.k(it): mdl for it in train}), mdl))
    metrics = c3.evaluate(test, routers, singles, best_single)

    def theta_choice(r):
        return routers[r][metrics[r]["theta_star"].split("/", 1)[1]]

    boot = {b: c3.bootstrap(test, theta_choice("ceiling_knn"), theta_choice(b)) for b in ("avengers_pro", "embedllm")}
    met = all(boot[b][x]["ci90"][0] > 0 for b in boot for x in ("avg_acc_diff", "slice_acc_diff"))
    result = {"prereg_sha": prereg_sha(), "phase": "1, F3 second read (Amendment 8)", "hc_test_reads_for_f3": 2,
              "items": {"hc_train": len(train), "hc_test": len(test)}, "embedllm_epochs": epochs,
              "chosen_on_hc_train": {"k": k, "m": m}, "menu_hc_train_oof_best_acc": menu,
              "no_task_variant": {"k": kx, "m": mx, "menu": menu_x},
              "hc_test": metrics, "paired_bootstrap_ceiling_knn_minus": boot, "gate": "PASS" if met else "NOT MET", "held_items_read": 0}
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({x: result[x] for x in ("chosen_on_hc_train", "hc_test", "paired_bootstrap_ceiling_knn_minus", "gate")}, indent=1))


if __name__ == "__main__":
    main()
