#!/usr/bin/env python3
"""Rung 2's wording search — on the TUNE split only, staged where Ship 0 showed the effects:

    stage A  every criteria set (the five measured wordings + scripts/laya/candidates/c*.json) in the
             measured order powerful, medium, small                                     → keep 3
    stage B  those × the six option orders                                             → keep 2
    stage C  those × three instruction sentences × three state keys                     → the winner

    python3 scripts/ab/wording_search.py --base-url http://127.0.0.1:8111 --out scripts/ab/results/laya-v2/r2-search

The metric is band accuracy POOLED over the headline tune rows (the borrowed tier rows and the pilot
asks; the unaudited DSS rows are scored and reported, never selected on), ties to macro-F1, then name.
Every configuration's per-row records are kept, so the search's own optimism is estimated offline:
5 seeded half-splits of the tune rows, the best configuration on one half scored on the other
(`selection_adjusted_acc`). The held-out read is one configuration, named here, read once elsewhere.
"""
import argparse
import glob
import itertools
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brain_bench  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = os.path.join(HERE, "..", "laya", "candidates")
INSTRUCTIONS = (
    "Which model tier should answer the user's `request`? Pick the cheapest tier that can do it well.",
    "Which model tier should answer the user's `request`? Pick the cheapest tier whose answer a careful reviewer would accept without escalating.",
    "How demanding is the user's `request` for a language model? Pick the tier that matches it.",
)
STATE_KEYS = ("request", "message", "task")
COSTLY_FIRST = ("powerful", "medium", "small")


def metric(records):
    head = [r for r in records if r.get("corpus") != "dss-requests"]
    pred, gold = [r["predicted"] for r in head], [r["gold"] for r in head]
    return {"acc": sum(p == g for p, g in zip(pred, gold)) / len(head), "f1": brain_bench.macro_f1(pred, gold), "n": len(head),
            "dss_acc": (lambda d: sum(r["predicted"] == r["gold"] for r in d) / len(d) if d else None)(
                [r for r in records if r.get("corpus") == "dss-requests"])}


def top(scores, k):
    return [name for name, _ in sorted(scores.items(), key=lambda kv: (-kv[1]["acc"], -kv[1]["f1"], kv[0]))[:k]]


def schema(criteria, order, instruction, key):
    return {"tier": {"type": "choice", "instructions": instruction.replace("`request`", f"`{key}`"),
                     "criteria": {t: criteria[t] for t in order}}}


def criteria_sets():
    sets = {name: dict(v) for name, v in brain_bench.WORDINGS.items()}
    for path in sorted(glob.glob(os.path.join(CANDIDATES, "c*.json"))):
        sets[os.path.splitext(os.path.basename(path))[0]] = json.load(open(path))
    return sets


def selection_adjusted(correct_by_config, ids, folds=5, seed=20260923):
    """Mean over seeded half-splits: pick the best configuration on half A, score it on half B."""
    scores = []
    for f in range(folds):
        rng = random.Random(f"{seed}-{f}")
        shuffled = sorted(ids)
        rng.shuffle(shuffled)
        a, b = shuffled[: len(shuffled) // 2], shuffled[len(shuffled) // 2:]
        for first, second in ((a, b), (b, a)):
            best = max(sorted(correct_by_config), key=lambda c: sum(correct_by_config[c][i] for i in first))
            scores.append(sum(correct_by_config[best][i] for i in second) / len(second))
    return sum(scores) / len(scores)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8111")
    ap.add_argument("--token", default=os.environ.get("LAYA_API_TOKEN"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rows = brain_bench.rows_for("tier-240", "tune") + brain_bench.rows_for("cells-pilot", "tune")
    health = brain_bench.health(args.base_url, args.token)
    sets = criteria_sets()
    evaluated = {}

    def evaluate(name, questions, key):
        path = os.path.join(args.out, f"{name}.json")
        if os.path.exists(path):
            recs = json.load(open(path))["records"]
        else:
            recs = brain_bench.run_wording(args.base_url, rows, None, args.token, state_key=key, schema=questions, record_probs=True)
            json.dump({"name": name, "schema": questions, "state_key": key, "brain": health, "records": recs}, open(path, "w"))
        evaluated[name] = {"records": recs, "schema": questions, "state_key": key, **metric(recs)}
        print(f"[search] {name:48s} acc {evaluated[name]['acc']:.3f} f1 {evaluated[name]['f1']:.3f}", flush=True)
        return evaluated[name]

    a = {f"A:{s}": evaluate(f"A-{s}", schema(sets[s], COSTLY_FIRST, INSTRUCTIONS[0], "request"), "request") for s in sets}
    keep_a = [n.split(":", 1)[1] for n in top(a, 3)]
    b = {}
    for s in keep_a:
        for order in itertools.permutations(COSTLY_FIRST):
            tag = "".join(t[0] for t in order)
            b[f"B:{s}/{tag}"] = evaluate(f"B-{s}-{tag}", schema(sets[s], order, INSTRUCTIONS[0], "request"), "request")
    keep_b = [n.split(":", 1)[1] for n in top(b, 2)]
    c = {}
    for sb in keep_b:
        s, tag = sb.split("/")
        order = tuple({"p": "powerful", "m": "medium", "s": "small"}[ch] for ch in tag)
        for i, ins in enumerate(INSTRUCTIONS):
            for key in STATE_KEYS:
                c[f"C:{s}/{tag}/i{i}/{key}"] = evaluate(f"C-{s}-{tag}-i{i}-{key}", schema(sets[s], order, ins, key), key)
    winner = top(c, 1)[0]
    ids = [r["id"] for r in rows if r.get("corpus") != "dss-requests"]
    correct = {n: {r["id"]: r["predicted"] == r["gold"] for r in v["records"] if r.get("corpus") != "dss-requests"}
               for n, v in evaluated.items()}
    adjusted = selection_adjusted(correct, ids)
    summary = {"brain": health, "tune_rows": len(ids), "configurations": len(evaluated),
               "stage_a": {n: {k: v[k] for k in ("acc", "f1", "dss_acc")} for n, v in a.items()}, "kept_a": keep_a,
               "stage_b": {n: {k: v[k] for k in ("acc", "f1", "dss_acc")} for n, v in b.items()}, "kept_b": keep_b,
               "stage_c": {n: {k: v[k] for k in ("acc", "f1", "dss_acc")} for n, v in c.items()},
               "winner": winner, "winner_tune_acc": c[winner]["acc"], "winner_schema": c[winner]["schema"],
               "winner_state_key": c[winner]["state_key"], "selection_adjusted_acc": adjusted}
    json.dump(summary, open(os.path.join(args.out, "..", "r2-search.json"), "w"), indent=1)
    print(f"[search] winner {winner}: tune {c[winner]['acc']:.3f}; selection-adjusted {adjusted:.3f} over {len(evaluated)} configurations", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
