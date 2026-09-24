"""The 2×2 (and its ceilings): every DECIDER × every MENU, scored from the graded outcomes of the
23 measured cells — RouterBench's method — so a decider is judged on the same asks and the same
answers as every other, with no new model call.

    python3 scripts/ab/factorial.py --pilot ~/dss-ab-cells/20260924-cells-120 --split held \\
        --decisions judge=results/laya-v2/decisions-judge.jsonl --decisions laya-v2=results/laya-v2/decisions-laya-v2.jsonl \\
        --out scripts/ab/results/laya-v2/factorial

A decisions file is JSON lines `{id, band[, effort]}`. A menu maps a band to a cell
(`corpus/menus/cells.json`, `ladder.json`). The quality of (decider, menu) on an ask is the graded
score of the cell the menu resolves the decider's band to: a machine check if the ask has one, else
the blind panel. A FAILED answer scores 0 (the person got nothing); an UNSCORED one is counted and
left out of the mean — never zeroed. `oracle-cell` (the best graded cell per ask) bounds them all.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell_bench  # noqa: E402
import checks  # noqa: E402
import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = "local-27b"


def load_outcomes(pilot_dir):
    """{(ask, cell): {score, latency_ms, cost_microusd}} from a cell_bench run directory."""
    answers = cell_bench.load_done(os.path.join(pilot_dir, "answers.jsonl"))
    grades = cell_bench.load_done(os.path.join(pilot_dir, "grades.jsonl"))
    ledger = cell_bench.load_done(os.path.join(pilot_dir, "ledger.jsonl"))
    out = {}
    for key, a in answers.items():
        ask, cell, _ = key.split("|")
        g = grades.get(key) or {}
        if a.get("status") != 200:
            score = 0.0
        else:
            score = g.get("check") if g.get("check") is not None else g.get("panel_score")
        out[(ask, cell)] = {"score": score, "latency_ms": a.get("latency_ms"),
                            "cost_microusd": (ledger.get(key) or {}).get("cost_microusd")}
    return out


def _cell_for(menu, decision):
    return menu(decision["band"], decision.get("effort")) if callable(menu) else menu[decision["band"]]


def replay_cells(cell_by_ask, outcomes, asks):
    scores, unscored, lat, cost, off, missing = [], 0, [], [], 0, 0
    per_ask = {}
    for ask in asks:
        cell = cell_by_ask.get(ask)
        o = outcomes.get((ask, cell)) if cell else None
        if o is None:
            missing += 1
            continue
        if not cell.startswith(LOCAL + "/"):
            off += 1
        if o["latency_ms"] is not None:
            lat.append(o["latency_ms"])
        if o["cost_microusd"] is not None:
            cost.append(o["cost_microusd"])
        if o["score"] is None:
            unscored += 1
            continue
        scores.append(o["score"])
        per_ask[ask] = o["score"]
    return {"quality": statistics.fmean(scores) if scores else None, "n_scored": len(scores), "unscored": unscored,
            "missing": missing, "off_local": off, "latency_ms_p50": statistics.median(lat) if lat else None,
            "cost_microusd_p50": statistics.median(cost) if cost else None, "per_ask": per_ask}


def replay(decisions, menu, outcomes, asks):
    return replay_cells({a: _cell_for(menu, d) for a, d in decisions.items() if a in asks}, outcomes, asks)


def oracle_cell(outcomes, asks):
    """The best graded cell per ask; ties go to the cheaper cell (then the faster)."""
    best = {}
    for (ask, cell), o in outcomes.items():
        if ask not in asks or o["score"] is None:
            continue
        key = (-o["score"], o["cost_microusd"] if o["cost_microusd"] is not None else float("inf"),
               o["latency_ms"] if o["latency_ms"] is not None else float("inf"), cell)
        if ask not in best or key < best[ask][0]:
            best[ask] = (key, cell)
    return {a: c for a, (_, c) in best.items()}


def label_decisions(bands):
    return {a: {"band": checks.BAND_OF[b]} for a, b in bands.items()}


def h1(b_decisions, a_decisions, gold):
    """H1 for decider B against decider A on the same rows: the margin and the one-sided exact McNemar."""
    ids = sorted(set(gold) & set(a_decisions) & set(b_decisions))
    a_ok = [a_decisions[i]["band"] == gold[i] for i in ids]
    b_ok = [b_decisions[i]["band"] == gold[i] for i in ids]
    m = stats.mcnemar_exact(a_ok, b_ok)
    acc_a, acc_b = sum(a_ok) / len(ids), sum(b_ok) / len(ids)
    return {"n": len(ids), "acc_a": acc_a, "acc_b": acc_b, "margin": round(acc_b - acc_a, 6), **m,
            "met": (acc_b - acc_a) >= 0.05 and m["p_b_better"] < 0.05}


def load_decisions(path):
    return {r["id"]: r for r in (json.loads(l) for l in open(path) if l.strip())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--split", default="held", choices=["tune", "held", "all"])
    ap.add_argument("--decisions", action="append", default=[], help="name=path (JSON lines {id, band[, effort]})")
    ap.add_argument("--menus", default="cells,ladder")
    ap.add_argument("--baseline", default="judge")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    import split_tools
    corpus = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(HERE, "corpus", "cells-pilot.jsonl")) if l.strip()}
    frozen = split_tools.load("cells-pilot")
    asks = sorted(corpus) if args.split == "all" else sorted(set(frozen[args.split]) & set(corpus))
    outcomes = load_outcomes(args.pilot)
    menus = {m: json.load(open(os.path.join(HERE, "corpus", "menus", f"{m}.json")))["menu"] for m in args.menus.split(",")}
    deciders = {"oracle-label": label_decisions({a: corpus[a]["band"] for a in asks})}
    for spec in args.decisions:
        name, path = spec.split("=", 1)
        deciders[name] = load_decisions(path)
    rows = []
    for name, decisions in deciders.items():
        for mname, menu in menus.items():
            r = replay(decisions, menu, outcomes, asks)
            rows.append({"decider": name, "menu": mname, **r})
    oracle = replay_cells(oracle_cell(outcomes, asks), outcomes, asks)
    rows.append({"decider": "oracle-cell", "menu": "—", **oracle})
    base = {r["menu"]: r for r in rows if r["decider"] == args.baseline}
    for r in rows:
        b = base.get(r["menu"])
        if b and r is not b and r["decider"] != "oracle-cell":
            common = sorted(set(r["per_ask"]) & set(b["per_ask"]))
            if common:
                r["vs_baseline_ci90"] = stats.paired_bootstrap_ci([b["per_ask"][a] for a in common], [r["per_ask"][a] for a in common])
    q = lambda v, f="%.3f": "—" if v is None else f % v
    lines = [f"# Factorial — decider × menu ({args.split}, {len(asks)} asks)", "",
             "| decider | menu | quality | vs " + args.baseline + " (90% CI) | scored | unscored | off local | latency p50 | cost p50 (µUSD) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["menu"], -(r["quality"] or 0))):
        ci = r.get("vs_baseline_ci90")
        lines.append(f"| {r['decider']} | {r['menu']} | {q(r['quality'])} | {'—' if not ci else f'[{ci[0]:+.3f}, {ci[1]:+.3f}]'} | "
                     f"{r['n_scored']} | {r['unscored']} | {r['off_local']} | {q(r['latency_ms_p50'], '%.0f')} | {q(r['cost_microusd_p50'], '%.0f')} |")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.out + ".json", "w") as f:
        json.dump({"split": args.split, "asks": asks, "rows": rows}, f, indent=1)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
