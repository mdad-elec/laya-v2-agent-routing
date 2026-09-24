#!/usr/bin/env python3
"""The verified cascade (PREREG amendment 5): the cheap tier answers, a Laya verifier reads the request and the
answer, and the turn escalates one tier when P(correct) < τ. Replayed over graded cell answers; no model call
except the verifier's.

    python3 scripts/ab/cascade_bench.py rows --out ~/laya-data/verifier            # training rows from tune asks
    ~/venvs/laya/bin/python scripts/ab/cascade_bench.py verify --model <dir|hub id> --name A --out …/verify-A.jsonl
    python3 scripts/ab/cascade_bench.py report                                       # fit τ on tune, read held once
"""
import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "laya-v2")
CELLS = os.path.expanduser(os.environ.get("CELLS_DIR", "cells-120"))  # the graded 120-ask cell run
SOLVED = 0.8
GRID = [round(0.05 * i, 2) for i in range(1, 20)]
QUESTION = {"verdict": {
    "type": "choice",
    "instructions": "Does `answer` fully and correctly do what `request` asks?",
    "criteria": {
        "correct": "The answer does everything the request asks, and every fact, number and line of code in it is right.",
        "wrong": "The answer misses part of the request, gets something wrong, or answers a different question.",
    }}}


def replay(asks, tiers, outcomes, p_correct, tau, start=None):
    per = {}
    for ask in asks:
        i = (start or {}).get(ask, 0)
        cost = lat = tried = 0
        while True:
            cell = tiers[i]
            o = outcomes.get((ask, cell)) or {}
            cost += o.get("cost_microusd") or 0
            lat += o.get("latency_ms") or 0
            tried += 1
            last = i == len(tiers) - 1
            if last or (p_correct.get((ask, cell)) or 0.0) >= tau:
                per[ask] = {"cell": cell, "score": o.get("score"), "cost": cost, "latency": lat, "tried": tried}
                break
            i += 1
    scored = [v["score"] for v in per.values() if v["score"] is not None]
    return {"quality": statistics.fmean(scored) if scored else None, "n_scored": len(scored),
            "cost_mean": statistics.fmean(v["cost"] for v in per.values()),
            "latency_p50": statistics.median(v["latency"] for v in per.values()),
            "escalated": sum(v["tried"] > 1 for v in per.values()), "per_ask": per}


def fit_tau(asks, tiers, outcomes, p_correct, target, start=None):
    runs = [(tau, replay(asks, tiers, outcomes, p_correct, tau, start)) for tau in GRID]
    ok = [(r["cost_mean"], tau, r) for tau, r in runs if r["quality"] is not None and r["quality"] >= target - 1e-12]
    if ok:
        _, tau, r = min(ok, key=lambda x: (x[0], x[1]))
        return tau, r
    _, _, tau, r = min(((-r["quality"], r["cost_mean"], tau, r) for tau, r in runs), key=lambda x: (x[0], x[1], x[2]))
    return tau, r


def auroc(labels, probs):
    pos = [p for l, p in zip(labels, probs) if l]
    neg = [p for l, p in zip(labels, probs) if not l]
    if not pos or not neg:
        return None
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def training_rows(answers, outcomes, prompts, val_asks):
    rows = []
    for (ask, cell), text in sorted(answers.items()):
        score = (outcomes.get((ask, cell)) or {}).get("score")
        if score is None:
            continue
        ok = score >= SOLVED
        rows.append({"id": f"{ask}|{cell}", "state": json.dumps({"request": prompts[ask], "answer": text}, ensure_ascii=False),
                     "questions": json.dumps(QUESTION, ensure_ascii=False),
                     "gold": json.dumps({"verdict": {"label": "correct" if ok else "wrong",
                                                     "probabilities": {"correct": 0.9, "wrong": 0.1} if ok else {"correct": 0.1, "wrong": 0.9}}}),
                     "split": "val" if ask in val_asks else "train", "source": "cells-120", "variant": "verifier"})
    return rows


def paired(a, b, key):
    """Two per-ask maps as aligned lists. Scores pair only where both are graded; costs pair on every shared ask."""
    asks = sorted(set(a) & set(b))
    if key == "score":
        asks = [x for x in asks if a[x]["score"] is not None and b[x]["score"] is not None]
    return [a[x][key] for x in asks], [b[x][key] for x in asks]


# ---- data ---------------------------------------------------------------------------------------------

def load_all():
    sys.path.insert(0, HERE)
    import cell_bench
    import factorial
    import split_tools
    outcomes = factorial.load_outcomes(CELLS)
    raw = cell_bench.load_done(os.path.join(CELLS, "answers.jsonl"))
    answers = {}
    for key, a in raw.items():
        ask, cell, _ = key.split("|")
        if a.get("status") == 200 and a.get("answer"):
            answers[(ask, cell)] = a["answer"]
    prompts = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(os.path.join(HERE, "corpus", "cells-pilot.jsonl")) if l.strip()}
    frozen = split_tools.load("cells-pilot")
    menu = json.load(open(os.path.join(HERE, "corpus", "menus", "cells.json")))["menu"]
    return outcomes, answers, prompts, frozen, [menu["small"], menu["medium"], menu["powerful"]]


def cmd_rows(args):
    import random
    _, answers, prompts, frozen, _ = load_all()
    outcomes = load_all()[0]
    tune = sorted(frozen["tune"])
    rng = random.Random(20260924)
    val = set(rng.sample(tune, max(1, len(tune) // 5)))
    rows = training_rows({k: v for k, v in answers.items() if k[0] in set(tune)}, outcomes, prompts, val)
    os.makedirs(args.out, exist_ok=True)
    for split in ("train", "val"):
        with open(os.path.join(args.out, f"{split}.jsonl"), "w") as f:
            for r in rows:
                if r["split"] == split:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    held = set(frozen["held"])
    assert not any(r["id"].split("|")[0] in held for r in rows), "a held ask reached the verifier's training rows"
    n = {s: sum(r["split"] == s for r in rows) for s in ("train", "val")}
    pos = sum(json.loads(r["gold"])["verdict"]["label"] == "correct" for r in rows) / len(rows)
    print(json.dumps({"rows": n, "val_asks": sorted(val), "share_correct": round(pos, 3)}))


def cmd_export(args):
    """Every graded (ask, cell) answer as {key, request, answer}: the verifier's whole input, so a remote box needs
    nothing but this file, this script and laya."""
    _, answers, prompts, _, tiers = load_all()
    first = [k for k in sorted(answers) if k[1] in tiers] + [k for k in sorted(answers) if k[1] not in tiers]
    with open(args.out, "w") as f:
        for ask, cell in first:
            f.write(json.dumps({"key": f"{ask}|{cell}", "request": prompts[ask], "answer": answers[(ask, cell)]}, ensure_ascii=False) + "\n")
    print(len(first), "answers →", args.out, "(the three tier cells first)")


def cmd_verify(args):
    """P(correct) for every exported answer, from a Laya checkpoint loaded in-process. Resumable."""
    import laya
    agent = laya.load(args.model, device=args.device)
    run = getattr(agent, "system_one", None) or agent.predict
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["key"] for l in open(args.out)}
    rows = [json.loads(l) for l in open(getattr(args, "in"))]
    with open(args.out, "a") as f:
        for i, r in enumerate(rows):
            key = r["key"]
            if key in done:
                continue
            out = run({"request": r["request"], "answer": r["answer"]}, QUESTION)
            a = out["verdict"] if "verdict" in out else out["answers"]["verdict"]
            f.write(json.dumps({"key": key, "p_correct": a["probabilities"]["correct"]}) + "\n")
            if i % 100 == 0:
                f.flush()
                print(i, flush=True)
    print("done", args.out)


def load_p(path):
    return {tuple(json.loads(l)["key"].split("|")): json.loads(l)["p_correct"] for l in open(path) if l.strip()}


def cmd_report(args):
    sys.path.insert(0, HERE)
    import stats
    import factorial
    outcomes, _, _, frozen, tiers = load_all()
    tune, held = sorted(frozen["tune"]), sorted(frozen["held"])
    dec = {**factorial.load_decisions(os.path.join(RES, "decisions-laya-v2-policy-tune.jsonl")),
           **factorial.load_decisions(os.path.join(RES, "decisions-laya-v2-policy.jsonl"))}
    menu = dict(zip(("small", "medium", "powerful"), tiers))

    def policy(asks):
        per = {}
        for a in asks:
            cell = menu[dec[a]["band"]]
            o = outcomes.get((a, cell)) or {}
            per[a] = {"cell": cell, "score": o.get("score"), "cost": o.get("cost_microusd") or 0, "latency": o.get("latency_ms") or 0}
        sc = [v["score"] for v in per.values() if v["score"] is not None]
        return {"quality": statistics.fmean(sc), "cost_mean": statistics.fmean(v["cost"] for v in per.values()),
                "latency_p50": statistics.median(v["latency"] for v in per.values()), "per_ask": per}
    base_tune, base_held = policy(tune), policy(held)
    start = {a: ("small", "medium", "powerful").index(dec[a]["band"]) for a in tune + held}
    oracle = {k: (1.0 if (o.get("score") or 0) >= SOLVED else 0.0) for k, o in outcomes.items() if o.get("score") is not None}
    arms = {}
    for name, path, hybrid in (("A-cascade", "verify-A.jsonl", False), ("B-cascade", "verify-B.jsonl", False),
                               ("B-hybrid", "verify-B.jsonl", True), ("oracle-cascade", None, False)):
        full = os.path.join(RES, "cascade", path) if path else None
        if full and not os.path.exists(full):
            continue
        p = load_p(full) if full else oracle
        st = start if hybrid else None
        tau, tr = fit_tau(tune, tiers, outcomes, p, base_tune["quality"], st) if full else (0.5, replay(tune, tiers, outcomes, p, 0.5))
        hr = replay(held, tiers, outcomes, p, tau, st)
        qa, qb = paired(base_held["per_ask"], hr["per_ask"], "score")
        ca, cb_ = paired(base_held["per_ask"], hr["per_ask"], "cost")
        labels = [(o["score"] >= SOLVED, p.get(k)) for k, o in outcomes.items()
                  if k[0] in set(held) and o.get("score") is not None and p.get(k) is not None]
        arms[name] = {"tau": tau, "tune_quality": tr["quality"], "tune_cost_mean": tr["cost_mean"], "held_quality": hr["quality"],
                      "held_n_scored": hr["n_scored"], "held_cost_mean": hr["cost_mean"], "held_latency_p50": hr["latency_p50"],
                      "held_escalated": hr["escalated"], "quality_vs_policy_ci90": stats.paired_bootstrap_ci(qa, qb),
                      "cost_vs_policy_ci90": stats.paired_bootstrap_ci(ca, cb_),
                      "auroc_held_all_cells": auroc([l for l, _ in labels], [q for _, q in labels]) if full else None,
                      "claim_met": None}
        q_lo, c_hi = arms[name]["quality_vs_policy_ci90"][0], arms[name]["cost_vs_policy_ci90"][1]
        arms[name]["claim_met"] = bool(q_lo > -0.02 and c_hi < 0)
    out = {"tiers": tiers, "v2_policy": {"tune_quality": base_tune["quality"], "held_quality": base_held["quality"],
                                         "held_cost_mean": base_held["cost_mean"], "held_latency_p50": base_held["latency_p50"]},
           "arms": arms}
    with open(os.path.join(RES, "cascade.json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rows")
    p.add_argument("--out", required=True)
    p = sub.add_parser("export")
    p.add_argument("--out", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--model", required=True)
    p.add_argument("--in", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    sub.add_parser("report")
    args = ap.parse_args(argv)
    return {"rows": cmd_rows, "export": cmd_export, "verify": cmd_verify, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
