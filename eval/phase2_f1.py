"""Phase 2 of recipe 2 on F1 (PREREG-g1 Amendment 7): does a TEXT classifier recover the ceiling?

Phase 1 answered `task` with the benchmark's name. The brain cannot: it answers question set v0
(`train/questions.py`), in which BFCL and PinchBench are both `tool_agentic`. So this measures, on
the same HC folds as `eval/ceiling.py`:

1. **The v0 vocabulary ceiling.** v0's `task`/`domain` answered perfectly (the map), groupings and
   (m, τ) chosen on HC-train out of fold, HC-test read once. What the vocabulary itself costs.
2. **The realised router.** A proxy brain answers `task` and `domain` from text: logistic
   regression (C = 1, 2,000 iterations, fixed here) on the same Qwen3 vectors SR-KNN uses. The
   TABLE is fitted on true answers (training data has them); every query is keyed by the proxy's
   PREDICTED answers (production has only those), out of fold on HC-train, and fitted on all of
   HC-train for HC-test. Its answer accuracy on HC-test is reported.

The gate is Phase 1's: the realised router beats SR-KNN on HC-test, paired trajectory bootstrap,
90% CI lower bound > 0. A realised router that passes is what Phase 3 (fine-tuning Laya as the
brain) has to match or beat. No held row is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from eval import ceiling as c1
from eval.baselines.sr_knn import features, predict
from eval.baselines.sr_knn_twinrouterbench import load_embeddings
from eval.twinrouterbench_score import score
from labels import twinrouterbench as trb
from labels.gate_s3 import prereg_sha
from train.questions import benchmark_answers
from train.table import TierTable, cheapest_passing

GROUPINGS = {k: v for k, v in c1.GROUPINGS.items() if k != "global"}


def v0(row: dict, s: dict) -> dict:
    task, domain = benchmark_answers(row["benchmark"])
    return {"task": task, "domain": domain, **s}


def route(fit, query, fit_ans, query_ans, y, fields, m, tau) -> dict[int, int]:
    table = TierTable(m).fit([c1.key(fit_ans[i], fields) for i in fit], [int(y[i]) for i in fit])
    return {i: cheapest_passing(table.p_solve(c1.key(query_ans[i], fields)), tau) for i in query}


def classify(X, labels, fit, query) -> dict[int, str]:
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X[fit], [labels[i] for i in fit])
    return dict(zip(query, clf.predict(X[query]).tolist(), strict=True))


def predicted_answers(X, true_ans, structural, folds_of, train, test) -> list[dict]:
    """Proxy answers: out of fold inside HC-train; fitted on all of HC-train for HC-test."""
    pred = [dict(a) for a in true_ans]
    for field in ("task", "domain"):
        labels = [a[field] for a in true_ans]
        for f in c1.HC_TRAIN_FOLDS:
            q = [i for i in train if folds_of[i] == f]
            fit = [i for i in train if folds_of[i] != f]
            for i, v in classify(X, labels, fit, q).items():
                pred[i][field] = v
        for i, v in classify(X, labels, train, test).items():
            pred[i][field] = v
    return pred


def choose(rows, folds_of, train, fit_ans, query_ans, y):
    best = None
    for name, fields in GROUPINGS.items():
        for m in c1.M_GRID:
            for tau in c1.TAU_GRID:
                pred: dict[int, int] = {}
                for f in c1.HC_TRAIN_FOLDS:
                    q = [i for i in train if folds_of[i] == f]
                    fit = [i for i in train if folds_of[i] != f]
                    pred.update(route(fit, q, fit_ans, query_ans, y, fields, m, tau))
                c = c1.combined(rows, train, pred)
                if best is None or c > best[0]:
                    best = (c, name, m, tau)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/g1/phase2-f1.json")
    args = ap.parse_args()

    rows = [r for r in trb.load() if trb.split(r) == "tune"]
    y = np.array([r["target_tier_id"] for r in rows])
    struct = [c1.structural(r) for r in rows]
    true_ans = [v0(r, s) for r, s in zip(rows, struct, strict=True)]
    folds_of = {i: c1.hc_fold(r) for i, r in enumerate(rows)}
    train = [i for i in folds_of if folds_of[i] in c1.HC_TRAIN_FOLDS]
    test = [i for i in folds_of if folds_of[i] in c1.HC_TEST_FOLDS]

    emb = load_embeddings(c1.EMBEDDINGS)
    E = np.stack([emb[r["id"]] for r in rows])
    Xn = E / np.linalg.norm(E, axis=1, keepdims=True)
    pred_ans = predicted_answers(Xn, true_ans, struct, folds_of, train, test)
    accuracy = {f: round(float(np.mean([pred_ans[i][f] == true_ans[i][f] for i in test])), 4) for f in ("task", "domain")}

    out = {}
    for label, q_ans in (("v0_vocabulary_ceiling", true_ans), ("realised_proxy_brain", pred_ans)):
        c_train, name, m, tau = choose(rows, folds_of, train, true_ans, q_ans, y)
        p = route(train, test, true_ans, q_ans, y, GROUPINGS[name], m, tau)
        s = score([rows[i] for i in test], {rows[i]["id"]: p[i] for i in test})
        out[label] = {"chosen_on_hc_train": {"grouping": name, "m": m, "tau": tau, "hc_train_oof": round(c_train, 2)},
                      "hc_test": {k: round(s[k], 2) for k in ("case_pass_rate_percent", "trajectory_pass_rate_percent",
                                                                  "cost_savings_score_percent", "combined_score_percent")},
                      "pred": p}

    X = features(E)
    knn = predict(X[train], y[train], X[test], k=1)
    knn_pred = {i: int(v) for i, v in zip(test, knn, strict=True)}
    boot = c1.bootstrap(rows, test, out["realised_proxy_brain"]["pred"], knn_pred)
    for v in out.values():
        v.pop("pred")
    result = {"prereg_sha": prereg_sha(), "phase": "2 (text-classifier proxy), F1", "proxy": "logistic regression on Qwen3-Embedding-0.6B",
              "proxy_answer_accuracy_hc_test": accuracy, **out,
              "sr_knn_hc_test_combined": round(c1.combined(rows, test, knn_pred), 2),
              "paired_bootstrap_realised_minus_sr_knn": boot,
              "gate": "PASS" if boot["ci90"][0] > 0 else "NOT MET", "held_rows_read": 0}
    path = Path(args.out)
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
