"""Rung 2's combiner: a multinomial logistic regression over the brain's own probabilities (the band
question and the auxiliary ones, one call), fitted on TUNE, adopted only if 5-fold cross-validation
beats taking the band question's argmax by >= 0.03 (PREREG). The fit uses sklearn; serving uses
scripts/laya/combine.py's stdlib formula, pinned to reproduce predict_proba.

    python3 scripts/ab/combiner.py results/laya-v2/r2-v2schema-tier-240-tune.json results/laya-v2/r2-v2schema-cells-pilot-tune.json \\
        --out scripts/laya/combiner.json
"""
import argparse
import json
import sys

FEATURES = [["tier", "small"], ["tier", "medium"], ["tier", "powerful"],
            ["needs_tools", "yes"], ["needs_code", "yes"], ["long_output", "yes"]]


def vector(rec, features):
    return [float(((rec.get("probabilities") or {}).get(q) or {}).get(o, 0.0)) for q, o in features]


def _argmax_acc(recs):
    return sum(max(r["probabilities"]["tier"], key=r["probabilities"]["tier"].get) == r["gold"] for r in recs) / len(recs)


def fit(recs, features=FEATURES, seed=20260923):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    X = [vector(r, features) for r in recs]
    y = [r["gold"] for r in recs]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed % (2 ** 31))
    best = None
    for C in (0.1, 1.0, 10.0):
        score = cross_val_score(LogisticRegression(C=C, max_iter=2000), X, y, cv=folds).mean()
        if best is None or score > best[1]:
            best = (C, score)
    argmax = []
    for _, test in folds.split(X, y):
        argmax.append(_argmax_acc([recs[i] for i in test]))
    model = LogisticRegression(C=best[0], max_iter=2000).fit(X, y)
    body = {"target": "tier", "features": features, "classes": list(model.classes_),
            "coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(), "C": best[0],
            "cv_acc": round(best[1], 4), "argmax_cv_acc": round(sum(argmax) / len(argmax), 4), "n": len(recs)}
    body["cv_acc"], body["argmax_cv_acc"] = float(body["cv_acc"]), float(body["argmax_cv_acc"])
    body["classes"] = [str(c) for c in body["classes"]]
    body["adopted"] = bool(body["cv_acc"] - body["argmax_cv_acc"] >= 0.03)
    return body, model


def load(paths):
    recs = []
    for p in paths:
        d = json.load(open(p))
        if d.get("split") != "tune":
            sys.exit(f"{p} is not a tune result; the combiner is fitted on tune only")
        for k, v in d["records"].items():
            if k.startswith("borrowed/"):
                recs += v
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    body, _ = fit(load(args.results))
    print(json.dumps({k: v for k, v in body.items() if k not in ("coef", "intercept")}, indent=1))
    if args.out and body["adopted"]:
        json.dump(body, open(args.out, "w"), indent=1)
        print("adopted →", args.out)
    elif args.out:
        print("NOT adopted: cross-validated gain under 0.03 over the band question's argmax")
    return 0


if __name__ == "__main__":
    sys.exit(main())
