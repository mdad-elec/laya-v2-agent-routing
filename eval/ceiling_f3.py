"""Phase 1 of recipe 2 (PREREG-g1 Amendment 7): the oracle-group ceiling on LLMRouterBench (F3).

Same question as `eval/ceiling.py`, for F3's performance-cost setting (Amendment 3): with the task
answered PERFECTLY (the item's own dataset), does a measured table + the cheapest-above-τ policy
beat Avengers-Pro and EmbedLLM, each fitted on the same items with the benchmark's own code?

Items: F3's tune items, minus the calibration split (`g1-cal/`, Amendment 5) and minus every
prompt that hits an evaluation set (`labels.leakage.build_eval_index`, as training does).
- HC-test = `sha256("g1-hc/" + prompt_hash) mod 100 < 15` (Amendment 7); HC-train = the rest.
- The table's grouping and shrinkage m are chosen on HC-train alone, by 5-fold out-of-fold AvgAcc
  at its best τ. HC-test is then read once, for every router, with F3's own formulas
  (`eval/llmrouterbench_f3.py`): θ* = the configuration with the best AvgAcc, ℬ = the best single
  model on HC-train, CostSave, the ≤3-correct recall slice at θ*, ParetoDist.
- The gate (F3's shape): the ceiling beats EACH baseline on AvgAcc(θ*) and on slice accuracy,
  paired bootstrap over HC-test items stratified by dataset, 10,000 resamples, θ* fixed, 90% CI
  lower bound > 0.

No held item is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval import llmrouterbench_f3 as f3
from eval.baselines import avengers_pro, embedllm
from eval.baselines.f3_predict import GRID, load_vectors
from labels.gate_s3 import prereg_sha
from train.policy import cheapest_above

M_GRID = (0.0, 1.0, 4.0, 16.0)
GROUPINGS = {"global": lambda it: (), "task": lambda it: (it["dataset"],)}
BOOTSTRAP = 10_000
SEED = 20260930
EMBEDDINGS = Path.home() / ".cache/laya-g1/public/llmrouterbench-f3-qwen3-embedding-0.6b.jsonl"


def _pct(prefix: str, key: str, mod: int) -> int:
    return int(hashlib.sha256(f"{prefix}{key}".encode()).hexdigest()[:8], 16) % mod


def is_calibration(it: dict) -> bool:
    return _pct("g1-cal/", it["prompt_hash"], 20) == 0


def is_hc_test(it: dict) -> bool:
    return int(hashlib.sha256(("g1-hc/" + it["prompt_hash"]).encode()).hexdigest(), 16) % 100 < 15


def k(it: dict) -> tuple:
    return (it["dataset"], it["native_id"])


class ScoreTable:
    """Per (group, model): mean score and mean cost, each shrunk toward the model's own fit-set mean."""

    def __init__(self, grouping, m: float) -> None:
        self.grouping, self.m = grouping, m

    def fit(self, items: list[dict]) -> ScoreTable:
        s, c, n = defaultdict(float), defaultdict(float), defaultdict(int)
        gs, gc, gn = defaultdict(float), defaultdict(float), defaultdict(int)
        for it in items:
            g = self.grouping(it)
            for mdl in f3.MODELS:
                s[g, mdl] += it["scores"][mdl]
                c[g, mdl] += it["costs"][mdl]
                n[g, mdl] += 1
                gs[mdl] += it["scores"][mdl]
                gc[mdl] += it["costs"][mdl]
                gn[mdl] += 1
        self.prior_s = {mdl: gs[mdl] / gn[mdl] for mdl in f3.MODELS}
        self.prior_c = {mdl: gc[mdl] / gn[mdl] for mdl in f3.MODELS}
        self.s, self.c, self.n = s, c, n
        return self

    def predict(self, it: dict) -> tuple[dict[str, float], dict[str, float]]:
        g = self.grouping(it)
        p, cost = {}, {}
        for mdl in f3.MODELS:
            n = self.n.get((g, mdl), 0)
            den = n + self.m
            p[mdl] = (self.s[g, mdl] + self.m * self.prior_s[mdl]) / den if den else self.prior_s[mdl]
            cost[mdl] = (self.c[g, mdl] + self.m * self.prior_c[mdl]) / den if den else self.prior_c[mdl]
        return p, cost


def table_configs(fit: list[dict], query: list[dict], grouping, m: float) -> dict[str, dict[tuple, str]]:
    t = ScoreTable(grouping, m).fit(fit)
    preds = [t.predict(it) for it in query]
    return {f"tau={tau}": {k(it): cheapest_above(p, c, tau) for it, (p, c) in zip(query, preds, strict=True)} for tau in GRID}


def best_acc(items: list[dict], configs: dict[str, dict[tuple, str]]) -> float:
    return max(f3.avg_acc(items, ch) for ch in configs.values())


def choose_table(train: list[dict]) -> tuple[str, float, dict]:
    fold = {k(it): _pct("g1-hc-fold/", it["prompt_hash"], 5) for it in train}
    scores = {}
    for name, grouping in GROUPINGS.items():
        for m in M_GRID:
            merged: dict[str, dict] = defaultdict(dict)
            for f in range(5):
                q = [it for it in train if fold[k(it)] == f]
                fit = [it for it in train if fold[k(it)] != f]
                for cfg, ch in table_configs(fit, q, grouping, m).items():
                    merged[cfg].update(ch)
            scores[f"{name}/m={m}"] = round(best_acc(train, merged), 4)
    best = max(scores, key=lambda s: (scores[s], s))
    name, m = best.split("/m=")
    return name, float(m), scores


def evaluate(items: list[dict], routers: dict[str, dict[str, dict[tuple, str]]], singles: dict[str, dict[tuple, str]], best_single: str) -> dict:
    configs = {f"{r}/{c}": (f3.avg_acc(items, ch), f3.total_cost(items, ch)) for r, cfgs in routers.items() for c, ch in cfgs.items()}
    configs.update({f"single/{m}": (f3.avg_acc(items, ch), f3.total_cost(items, ch)) for m, ch in singles.items()})
    slice_items = f3.recall_slice(items)
    out = {}
    for r, cfgs in routers.items():
        names = [f"{r}/{c}" for c in cfgs]
        theta = max(names, key=lambda n: (configs[n][0], -configs[n][1]))
        cs = f3.cost_save(configs, names, f"single/{best_single}")
        out[r] = {"theta_star": theta, "avg_acc": round(configs[theta][0], 4),
                  "perf_gain": round(f3.perf_gain(configs, names, f"single/{best_single}"), 4),
                  "cost_save": None if cs is None else round(cs, 4),
                  "slice_acc": round(sum(it["scores"][cfgs[theta.split("/", 1)[1]][k(it)]] for it in slice_items) / len(slice_items), 4),
                  "pareto_dist": round(f3.pareto_dist(configs, names), 4)}
    out["best_single"] = {"model": best_single, "avg_acc": round(configs[f"single/{best_single}"][0], 4)}
    out["slice_items"] = len(slice_items)
    return out


def bootstrap(items: list[dict], a: dict[tuple, str], b: dict[tuple, str]) -> dict:
    by_ds: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_ds[it["dataset"]].append(i)
    sa = np.array([it["scores"][a[k(it)]] for it in items])
    sb = np.array([it["scores"][b[k(it)]] for it in items])
    in_slice = np.array([sum(v >= f3.SOLVED for v in it["scores"].values()) <= f3.SLICE_MAX_CORRECT for it in items])
    rng = np.random.default_rng(SEED)
    acc_d, slice_d = [], []
    for _ in range(BOOTSTRAP):
        draw = {ds: rng.choice(idx, size=len(idx), replace=True) for ds, idx in by_ds.items()}
        acc_d.append(np.mean([sa[d].mean() - sb[d].mean() for d in draw.values()]))
        allidx = np.concatenate(list(draw.values()))
        sl = allidx[in_slice[allidx]]
        slice_d.append(sa[sl].mean() - sb[sl].mean() if len(sl) else 0.0)

    def ci(v):
        v = np.array(v)
        return {"mean": round(float(v.mean()), 4), "ci90": [round(float(np.percentile(v, 5)), 4), round(float(np.percentile(v, 95)), 4)]}

    return {"avg_acc_diff": ci(acc_d), "slice_acc_diff": ci(slice_d)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/g1/ceiling-f3.json")
    args = ap.parse_args()

    from labels import routerarena as ra
    from labels import twinrouterbench as trb
    from labels.leakage import build_eval_index
    from labels.leakage_report import suite_items

    index = build_eval_index(ra.questions(), trb.load(), suite_items())
    tune = [it for it in f3.items() if it["split"] == "tune"]
    usable = [it for it in tune if not is_calibration(it) and not index.hits(it["query"])]
    train = [it for it in usable if not is_hc_test(it)]
    test = [it for it in usable if is_hc_test(it)]

    name, m, menu = choose_table(train)
    routers = {"ceiling": table_configs(train, test, GROUPINGS[name], m)}

    vectors = load_vectors(EMBEDDINGS)
    emb = {it["query"]: vectors[it["prompt_hash"]] for it in train + test}
    routers["avengers_pro"] = {f"alpha={a}": avengers_pro.route(train, test, emb, alpha=a) for a in GRID}
    epochs, curve = embedllm.choose_epochs(train, emb)
    model = embedllm.fit(train, emb, epochs=epochs)
    cost = embedllm.expected_costs(train)
    probs = model.predict([emb[it["query"]] for it in test])
    routers["embedllm"] = {f"tau={tau}": {k(it): embedllm.pick(p, cost, tau) for it, p in zip(test, probs, strict=True)} for tau in GRID}

    singles = {mdl: {k(it): mdl for it in test} for mdl in f3.MODELS}
    best_single = max(f3.MODELS, key=lambda mdl: (f3.avg_acc(train, {k(it): mdl for it in train}), mdl))
    metrics = evaluate(test, routers, singles, best_single)

    def theta_choice(r):
        return routers[r][metrics[r]["theta_star"].split("/", 1)[1]]

    boot = {b: bootstrap(test, theta_choice("ceiling"), theta_choice(b)) for b in ("avengers_pro", "embedllm")}
    met = all(boot[b][m_]["ci90"][0] > 0 for b in boot for m_ in ("avg_acc_diff", "slice_acc_diff"))
    result = {"prereg_sha": prereg_sha(), "phase": "1 (oracle-group ceiling)", "bar": "F3 shape, on HC-test",
              "items": {"tune": len(tune), "usable": len(usable), "hc_train": len(train), "hc_test": len(test),
                        "hc_test_by_dataset": dict(sorted(defaultdict(int, {ds: sum(it["dataset"] == ds for it in test) for ds in f3.DATASETS}).items()))},
              "table_menu_hc_train_oof_best_acc": menu, "chosen_on_hc_train": {"grouping": name, "m": m},
              "embedllm_epochs": epochs, "embedllm_tune_val_curve": curve,
              "hc_test": metrics, "paired_bootstrap_ceiling_minus": boot, "gate": "PASS" if met else "NOT MET", "held_items_read": 0}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k_: result[k_] for k_ in ("items", "chosen_on_hc_train", "hc_test", "paired_bootstrap_ceiling_minus", "gate")}, indent=1))


if __name__ == "__main__":
    main()
