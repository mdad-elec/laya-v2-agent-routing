"""Phase 1 of recipe 2 (PREREG-g1 Amendment 7): the oracle-group ceiling on TwinRouterBench (F1).

The question it answers, before any classifier is trained: if the semantic questions were answered
PERFECTLY (the benchmark and scenario come from the row's own fields, standing in for `task` and
`domain`), does classifier → table → policy beat SR-KNN at all? If not, no brain can make this
design reach F1, and nothing is trained.

Everything else a group may use is what a router can really compute from the conversation
(`router/digest.py`): the step position (`turn`), whether the recent tool results show errors or a
test pass, and the context bucket. `total_steps` is NOT used: no router knows it.

Splits (Amendment 7): tune only, five trajectory folds by `sha256("g1-hc/" + trajectory_key) mod 5`.
- HC-train = folds 0–2. Scored out of fold: each fold by a table fitted on the other two.
- HC-test = folds 3–4. Scored once, by a table fitted on folds 0–2.

The menu below is fixed before this file first ran, and the configuration is chosen on HC-train
alone. HC-test is read as a number for the chosen configuration and for each group definition
(listed for transparency, never used to choose). SR-KNN (its published method, k = 1, Qwen3
embeddings) is fitted and scored on the same folds, and a paired bootstrap over HC-test
trajectories gives the gate: ceiling − SR-KNN, 90% CI lower bound > 0.

No held row is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eval.baselines.sr_knn import features, predict
from eval.baselines.sr_knn_twinrouterbench import load_embeddings
from eval.twinrouterbench_score import score
from labels import twinrouterbench as trb
from labels.gate_s3 import prereg_sha
from router import digest
from train.table import TierTable, cheapest_passing

HC_TRAIN_FOLDS = (0, 1, 2)
HC_TEST_FOLDS = (3, 4)
M_GRID = (0.0, 1.0, 4.0, 16.0)
TAU_GRID = tuple(round(0.5 + 0.02 * i, 2) for i in range(25))  # 0.50 … 0.98
BOOTSTRAP = 2000
SEED = 20260930
EMBEDDINGS = Path.home() / ".cache/laya-g1/public/twinrouterbench-emb/qwen3-embedding-0.6b.jsonl"


def hc_fold(row: dict) -> int:
    return int(hashlib.sha256(("g1-hc/" + trb.trajectory_key(row)).encode()).hexdigest(), 16) % 5


def turn_bucket(turn: int) -> str:
    return "1" if turn <= 1 else "2-3" if turn <= 3 else "4-7" if turn <= 7 else "8+"


def structural(row: dict) -> dict:
    d = digest.build(row["messages"], row.get("functions"))
    return {"stage": turn_bucket(d["turn"]), "errors": d["tool_rounds"]["errors"] > 0,
            "critical": d["tool_rounds"]["critical"], "tests": d["tool_rounds"]["tests_passed"], "context": d["context"]}


# Each group definition names the answers it keys on. "task" is the oracle semantic answer (the
# benchmark); "domain" the scenario; the rest are the computed digest fields.
GROUPINGS: dict[str, tuple[str, ...]] = {
    "global": (),
    "structure_only": ("stage", "errors", "context"),
    "task": ("task",),
    "task+stage": ("task", "stage"),
    "task+stage+errors": ("task", "stage", "errors"),
    "domain+stage+errors+tests+context": ("domain", "stage", "errors", "tests", "context"),
}


def answers(row: dict, s: dict) -> dict:
    return {"task": row["benchmark"], "domain": row["scenario"], **s}


def key(a: dict, fields: tuple[str, ...]) -> tuple:
    return tuple(a[f] for f in fields)


def route(fit: list[int], query: list[int], ans: list[dict], y: np.ndarray, fields: tuple[str, ...], m: float, tau: float) -> dict[int, int]:
    table = TierTable(m).fit([key(ans[i], fields) for i in fit], [int(y[i]) for i in fit])
    return {i: cheapest_passing(table.p_solve(key(ans[i], fields)), tau) for i in query}


def combined(rows: list[dict], idx: list[int], pred: dict[int, int]) -> float:
    return score([rows[i] for i in idx], {rows[i]["id"]: pred[i] for i in idx})["combined_score_percent"]


def out_of_fold(rows, folds_of, folds, ans, y, fields, m, tau) -> dict[int, int]:
    pred: dict[int, int] = {}
    for f in folds:
        query = [i for i, fo in folds_of.items() if fo == f]
        fit = [i for i, fo in folds_of.items() if fo in folds and fo != f]
        pred.update(route(fit, query, ans, y, fields, m, tau))
    return pred


def bootstrap(rows: list[dict], idx: list[int], a: dict[int, int], b: dict[int, int]) -> dict:
    """Paired, over HC-test trajectories; a trajectory drawn twice is renamed, so repeats never merge."""
    by_traj: dict[str, list[int]] = {}
    for i in idx:
        by_traj.setdefault(trb.trajectory_key(rows[i]), []).append(i)
    trajs = sorted(by_traj)
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(BOOTSTRAP):
        sample, pa, pb = [], {}, {}
        for n, t in enumerate(rng.choice(len(trajs), size=len(trajs), replace=True)):
            for i in by_traj[trajs[t]]:
                r = dict(rows[i])
                r["id"] = f"{r['id']}#{n}"
                r["instance_id"] = f"{r.get('instance_id', r['id'])}#{n}"
                sample.append(r)
                pa[r["id"]], pb[r["id"]] = a[i], b[i]
        diffs.append(score(sample, pa)["combined_score_percent"] - score(sample, pb)["combined_score_percent"])
    d = np.array(diffs)
    return {"resamples": BOOTSTRAP, "seed": SEED, "mean": round(float(d.mean()), 2),
            "ci90": [round(float(np.percentile(d, 5)), 2), round(float(np.percentile(d, 95)), 2)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/g1/ceiling-f1.json")
    args = ap.parse_args()

    rows = [r for r in trb.load() if trb.split(r) == "tune"]
    y = np.array([r["target_tier_id"] for r in rows])
    ans = [answers(r, structural(r)) for r in rows]
    folds_of = {i: hc_fold(r) for i, r in enumerate(rows)}
    train = [i for i in folds_of if folds_of[i] in HC_TRAIN_FOLDS]
    test = [i for i in folds_of if folds_of[i] in HC_TEST_FOLDS]

    # 1. choose (grouping, m, τ) on HC-train, out of fold. HC-test is not touched here.
    grid = []
    for name, fields in GROUPINGS.items():
        for m in M_GRID:
            for tau in TAU_GRID:
                pred = out_of_fold(rows, {i: folds_of[i] for i in train}, HC_TRAIN_FOLDS, ans, y, fields, m, tau)
                grid.append((combined(rows, train, pred), name, m, tau))
    best_per_grouping = {}
    for c, name, m, tau in grid:
        if name not in best_per_grouping or c > best_per_grouping[name][0]:
            best_per_grouping[name] = (c, m, tau)
    chosen_c, chosen_name, chosen_m, chosen_tau = max(grid, key=lambda g: g[0])

    # 2. HC-test, once per grouping at its HC-train-chosen (m, τ); the gate uses the chosen one.
    per_grouping = {}
    for name, (c_train, m, tau) in best_per_grouping.items():
        pred = route(train, test, ans, y, GROUPINGS[name], m, tau)
        per_grouping[name] = {"fields": list(GROUPINGS[name]), "m": m, "tau": tau,
                              "hc_train_oof": round(c_train, 2), "hc_test": round(combined(rows, test, pred), 2)}
    ceiling_pred = route(train, test, ans, y, GROUPINGS[chosen_name], chosen_m, chosen_tau)
    ceiling_scores = score([rows[i] for i in test], {rows[i]["id"]: ceiling_pred[i] for i in test})

    # 3. SR-KNN on the same folds: index = HC-train rows.
    emb = load_embeddings(EMBEDDINGS)
    X = features(np.stack([emb[r["id"]] for r in rows]))
    knn = predict(X[train], y[train], X[test], k=1)
    knn_pred = {i: int(p) for i, p in zip(test, knn, strict=True)}
    knn_scores = score([rows[i] for i in test], {rows[i]["id"]: knn_pred[i] for i in test})
    constants = {f"always_{trb.TIERS[t]}": round(combined(rows, test, dict.fromkeys(test, t)), 2) for t in range(4)}

    # 4. the gate.
    boot = bootstrap(rows, test, ceiling_pred, knn_pred)
    keys = ("case_pass_rate_percent", "case_exact_match_percent", "trajectory_pass_rate_percent", "cost_savings_score_percent", "combined_score_percent")
    result = {
        "prereg_sha": prereg_sha(), "phase": "1 (oracle-group ceiling)", "bar": "F1 part 1 shape, on HC-test",
        "rows": {"tune": len(rows), "hc_train": len(train), "hc_test": len(test),
                 "hc_test_trajectories": len({trb.trajectory_key(rows[i]) for i in test})},
        "menu": {"groupings": {k: list(v) for k, v in GROUPINGS.items()}, "m": list(M_GRID), "tau": list(TAU_GRID)},
        "chosen_on_hc_train": {"grouping": chosen_name, "m": chosen_m, "tau": chosen_tau, "hc_train_oof": round(chosen_c, 2)},
        "ceiling_hc_test": {k: round(ceiling_scores[k], 2) for k in keys},
        "sr_knn_hc_test": {k: round(knn_scores[k], 2) for k in keys},
        "constants_hc_test": constants,
        "per_grouping": per_grouping,
        "paired_bootstrap_ceiling_minus_sr_knn": boot,
        "gate": "PASS" if boot["ci90"][0] > 0 else "NOT MET",
        "held_rows_read": 0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("chosen_on_hc_train", "ceiling_hc_test", "sr_knn_hc_test", "constants_hc_test",
                                             "paired_bootstrap_ceiling_minus_sr_knn", "gate")}, indent=1))


if __name__ == "__main__":
    main()
