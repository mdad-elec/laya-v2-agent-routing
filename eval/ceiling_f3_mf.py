"""Phase 1, F3's third ceiling read (PREREG-g1 Amendment 9): a learned per-item estimator on a
dataset-stratified HC-test.

Layer 2's P(solve) is EmbedLLM's own model (`eval/baselines/embedllm.py`, its epoch rule applied to
each fit set), used as `mf+task` (the task answer appended to its input as a one-hot) or
`blend(w)` (w · P_embedllm + (1 − w) · the task row). The policy and expected costs are the task
row's (cheapest above τ, 19-point grid). The variant is chosen on HC-train by 5-fold out-of-fold
best AvgAcc; HC-test is read once. Everything else (formulas, baselines, gate) is
`eval/ceiling_f3.py`'s, imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval import ceiling_f3 as c3
from eval import llmrouterbench_f3 as f3
from eval.baselines import avengers_pro, embedllm
from eval.baselines.f3_predict import GRID, load_vectors
from labels.gate_s3 import prereg_sha
from train.policy import cheapest_above

VARIANTS = ("mf+task", "blend(0.25)", "blend(0.5)", "blend(0.75)")


def stratified_split(usable: list[dict]) -> tuple[list[dict], list[dict]]:
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for it in usable:
        by_ds[it["dataset"]].append(it)
    train, test = [], []
    for items in by_ds.values():
        items = sorted(items, key=lambda it: hashlib.sha256(("g1-hc9/" + it["prompt_hash"]).encode()).hexdigest())
        n = len(items)
        cut = max(math.ceil(0.15 * n), min(math.ceil(n / 2), 20))
        test += items[:cut]
        train += items[cut:]
    return train, test


def with_task(vec: dict[str, np.ndarray], items: list[dict]) -> dict[str, np.ndarray]:
    onehot = {ds: np.eye(len(f3.DATASETS), dtype=np.float32)[i] for i, ds in enumerate(f3.DATASETS)}
    return {it["query"]: np.concatenate([vec[it["prompt_hash"]], onehot[it["dataset"]]]) for it in items}


def plain(vec: dict[str, np.ndarray], items: list[dict]) -> dict[str, np.ndarray]:
    return {it["query"]: vec[it["prompt_hash"]] for it in items}


def mf(fit: list[dict], query: list[dict], emb: dict[str, np.ndarray]) -> list[dict[str, float]]:
    epochs, _ = embedllm.choose_epochs(fit, emb)
    return embedllm.fit(fit, emb, epochs=epochs).predict([emb[it["query"]] for it in query])


def variant_configs(variant: str, fit: list[dict], query: list[dict], vec, mf_plain=None) -> dict[str, dict[tuple, str]]:
    row = c3.ScoreTable(c3.GROUPINGS["task"], 0.0).fit(fit)
    rows = [row.predict(it) for it in query]
    if variant == "mf+task":
        ps = mf(fit, query, with_task(vec, fit + query))
    else:
        w = float(variant[len("blend("):-1])
        base = mf_plain if mf_plain is not None else mf(fit, query, plain(vec, fit + query))
        ps = [{mdl: w * pe[mdl] + (1 - w) * rp[mdl] for mdl in f3.MODELS} for pe, (rp, _) in zip(base, rows, strict=True)]
    return {f"tau={tau}": {c3.k(it): cheapest_above(p, rc, tau) for it, p, (_, rc) in zip(query, ps, rows, strict=True)} for tau in GRID}


def choose(train: list[dict], vec) -> tuple[str, dict]:
    fold = {c3.k(it): c3._pct("g1-hc-fold/", it["prompt_hash"], 5) for it in train}
    merged: dict[str, dict[str, dict]] = {v: defaultdict(dict) for v in VARIANTS}
    for f in range(5):
        q = [it for it in train if fold[c3.k(it)] == f]
        fit = [it for it in train if fold[c3.k(it)] != f]
        base = mf(fit, q, plain(vec, fit + q))
        for v in VARIANTS:
            for cfg, ch in variant_configs(v, fit, q, vec, mf_plain=None if v == "mf+task" else base).items():
                merged[v][cfg].update(ch)
        print(f"fold {f} done", flush=True)
    scores = {v: round(c3.best_acc(train, merged[v]), 4) for v in VARIANTS}
    return max(VARIANTS, key=lambda v: (scores[v], v)), scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/g1/ceiling-f3-mf.json")
    args = ap.parse_args()

    from labels import routerarena as ra
    from labels import twinrouterbench as trb
    from labels.leakage import build_eval_index
    from labels.leakage_report import suite_items

    index = build_eval_index(ra.questions(), trb.load(), suite_items())
    tune = [it for it in f3.items() if it["split"] == "tune"]
    usable = [it for it in tune if not c3.is_calibration(it) and not index.hits(it["query"])]
    train, test = stratified_split(usable)
    vec = load_vectors(c3.EMBEDDINGS)

    chosen, menu = choose(train, vec)
    routers = {"ceiling_mf": variant_configs(chosen, train, test, vec)}
    routers["ceiling_table (reported)"] = c3.table_configs(train, test, c3.GROUPINGS["task"], 0.0)
    emb = plain(vec, train + test)
    routers["avengers_pro"] = {f"alpha={a}": avengers_pro.route(train, test, emb, alpha=a) for a in GRID}
    probs = mf(train, test, emb)
    cost = embedllm.expected_costs(train)
    routers["embedllm"] = {f"tau={tau}": {c3.k(it): embedllm.pick(p, cost, tau) for it, p in zip(test, probs, strict=True)} for tau in GRID}

    singles = {mdl: {c3.k(it): mdl for it in test} for mdl in f3.MODELS}
    best_single = max(f3.MODELS, key=lambda mdl: (f3.avg_acc(train, {c3.k(it): mdl for it in train}), mdl))
    metrics = c3.evaluate(test, routers, singles, best_single)

    def theta_choice(r):
        return routers[r][metrics[r]["theta_star"].split("/", 1)[1]]

    boot = {b: c3.bootstrap(test, theta_choice("ceiling_mf"), theta_choice(b)) for b in ("avengers_pro", "embedllm")}
    met = all(boot[b][x]["ci90"][0] > 0 for b in boot for x in ("avg_acc_diff", "slice_acc_diff"))
    result = {"prereg_sha": prereg_sha(), "phase": "1, F3 third read (Amendment 9, stratified HC-test)", "f3_ceiling_reads": 3,
              "items": {"hc_train": len(train), "hc_test": len(test),
                        "hc_test_by_dataset": dict(sorted({ds: sum(it["dataset"] == ds for it in test) for ds in f3.DATASETS}.items()))},
              "chosen_on_hc_train": chosen, "menu_hc_train_oof_best_acc": menu,
              "hc_test": metrics, "paired_bootstrap_ceiling_mf_minus": boot, "gate": "PASS" if met else "NOT MET", "held_items_read": 0}
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({x: result[x] for x in ("items", "chosen_on_hc_train", "menu_hc_train_oof_best_acc", "hc_test",
                                             "paired_bootstrap_ceiling_mf_minus", "gate")}, indent=1))


if __name__ == "__main__":
    main()
