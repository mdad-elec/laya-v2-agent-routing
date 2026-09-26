#!/usr/bin/env python3
"""The external leg: Laya as a router on RouterBench (Hu et al., ICML 2024; withmartian/routerbench),
placed on that benchmark's own cost-quality plane.

    python3 scripts/ab/routerbench_leg.py inspect                      # pin the columns, or stop
    python3 scripts/ab/routerbench_leg.py prepare                      # bands, menu, labels, splits → derived/
    ~/venvs/laya/bin/python scripts/ab/routerbench_leg.py bench --base-url http://127.0.0.1:8111 --schema FILE --name laya-v2
    python3 scripts/ab/routerbench_leg.py score                        # AIQ per task, the table and SVGs

Pre-registered (scripts/laya/PREREG.md, D5.4): the eleven models are banded by MEASURED mean cost,
four / four / three; each band's menu model is its most accurate one; a prompt's label is the
cheapest band whose menu model solves it (score 1 — a partial GSM8K score is not a solve), and a
prompt no menu model solves is `powerful` and flagged unsolvable. Tasks are the paper's six with
enough rows (MMLU's subjects pooled; MT-Bench excluded as in the paper), split 70/30 seeded, at most
500 train and 500 test prompts per task sampled stratified by label. A router's point is the mean
quality and mean cost of sending each test prompt to its band's menu model; sweeping the
low-confidence upgrade (P(band) < tau → one band up) traces its curve. AIQ is the area under the
non-decreasing convex hull over [c_min, c_max], divided by the width; zero below a curve's cheapest
point. The pickle is never redistributed — only derived labels and this code.
"""
import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.expanduser("~/routerbench/routerbench_0shot.pkl")
# The sampled prompts, kept beside the pickle and never committed: RouterBench's text is not ours to
# redistribute; the repo keeps only derived labels, ids and scores.
PROMPTS = os.path.expanduser("~/routerbench/prompts-sample.jsonl")
DERIVED = os.path.join(HERE, "results", "laya-v2", "routerbench")
MODELS = ["WizardLM/WizardLM-13B-V1.2", "claude-instant-v1", "claude-v1", "claude-v2", "gpt-3.5-turbo-1106",
          "gpt-4-1106-preview", "meta/code-llama-instruct-34b-chat", "meta/llama-2-70b-chat", "mistralai/mistral-7b-chat",
          "mistralai/mixtral-8x7b-chat", "zero-one-ai/Yi-34B-Chat"]
TASKS = {"mmlu": lambda n: n.startswith("mmlu"), "hellaswag": lambda n: n == "hellaswag",
         "gsm8k": lambda n: n == "grade-school-math", "arc": lambda n: n == "arc-challenge",
         "winogrande": lambda n: n == "winogrande", "mbpp": lambda n: n == "mbpp"}
BANDS = ("small", "medium", "powerful")
SEED = 20260923
PER_SPLIT = 500
TAUS = [0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 1.01]


def columns(path=PKL):
    import pandas as pd
    return list(pd.read_pickle(path).columns)


def bands_by_cost(mean_cost):
    order = sorted(mean_cost, key=mean_cost.get)
    return {m: ("small" if i < 4 else "medium" if i < 8 else "powerful") for i, m in enumerate(order)}


def label(scores, menu):
    for b in BANDS:
        if scores[menu[b]] >= 1.0:
            return b, False
    return "powerful", True


def ndch(points):
    """The non-decreasing convex hull (upper-left frontier) of (cost, quality) points, cost ascending."""
    pts = sorted(set(points))
    frontier = []
    for c, q in pts:
        if frontier and q <= frontier[-1][1]:
            continue                                  # dearer and no better: dominated
        frontier.append((c, q))
    hull = []
    for p in frontier:                                # upper concave hull of what remains
        while len(hull) >= 2:
            (c1, q1), (c2, q2) = hull[-2], hull[-1]
            if (q2 - q1) * (p[0] - c1) <= (p[1] - q1) * (c2 - c1):
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def aiq(points, cmin, cmax):
    """Area under the hull over [cmin, cmax] / (cmax - cmin); 0 below the curve's cheapest point,
    linear between hull points, flat after the last one."""
    hull = ndch(points)
    if cmax <= cmin or not hull:
        return 0.0

    def q_at(c):
        for (c1, q1), (c2, q2) in zip(hull, hull[1:]):
            if c1 <= c <= c2:
                return q1 + (q2 - q1) * (c - c1) / (c2 - c1)
        return hull[-1][1] if c >= hull[-1][0] else hull[0][1]

    start = max(cmin, hull[0][0])
    if start >= cmax:
        return 0.0
    xs = sorted({start, cmax} | {c for c, _ in hull if start < c < cmax})
    return sum((q_at(a) + q_at(b)) / 2 * (b - a) for a, b in zip(xs, xs[1:])) / (cmax - cmin)


def prepare():
    import pandas as pd
    df = pd.read_pickle(PKL)
    missing = [c for m in MODELS for c in (m, f"{m}|total_cost") if c not in df.columns]
    if missing or "sample_id" not in df.columns:
        sys.exit(f"RouterBench columns are not what this code reads: missing {missing[:5]}")
    df["task"] = None
    for t, f in TASKS.items():
        df.loc[df["eval_name"].map(f), "task"] = t
    df = df[df["task"].notna()]
    mean_cost = {m: float(df[f"{m}|total_cost"].mean()) for m in MODELS}
    bands = bands_by_cost(mean_cost)
    menu = {b: max((m for m in MODELS if bands[m] == b), key=lambda m: float(df[m].mean())) for b in BANDS}
    rows = []
    for t in TASKS:
        sub = df[df["task"] == t]
        ids = sorted(sub["sample_id"])
        rng = random.Random(f"{SEED}-{t}")
        rng.shuffle(ids)
        cut = int(len(ids) * 0.7)
        side = {i: ("train" if n < cut else "test") for n, i in enumerate(ids)}
        for _, r in sub.iterrows():
            lab, unsolvable = label({m: float(r[m]) for m in MODELS}, menu)
            rows.append({"sample_id": r["sample_id"], "task": t, "split": side[r["sample_id"]], "label": lab, "unsolvable": unsolvable,
                         "scores": {m: float(r[m]) for m in MODELS}, "costs": {m: float(r[f"{m}|total_cost"]) for m in MODELS},
                         "prompt_chars": len(r["prompt"])})
    sample = []
    for t in TASKS:
        for split in ("train", "test"):
            pool = [r for r in rows if r["task"] == t and r["split"] == split]
            by = {}
            for r in pool:
                by.setdefault(r["label"], []).append(r)
            rng = random.Random(f"{SEED}-{t}-{split}")
            k = min(PER_SPLIT, len(pool))
            picked = []
            for lab, rs in sorted(by.items()):
                n = max(1, round(k * len(rs) / len(pool)))
                picked += rng.sample(sorted(rs, key=lambda r: r["sample_id"]), min(n, len(rs)))
            sample += picked[:k]
    os.makedirs(DERIVED, exist_ok=True)
    json.dump({"mean_cost": mean_cost, "bands": bands, "menu": menu, "per_split": PER_SPLIT, "seed": SEED}, open(os.path.join(DERIVED, "bands.json"), "w"), indent=1)
    with open(os.path.join(DERIVED, "labels.jsonl"), "w") as f:
        for r in sorted(sample, key=lambda r: (r["task"], r["split"], r["sample_id"])):
            f.write(json.dumps(r) + "\n")
    text = df.set_index("sample_id")["prompt"]
    with open(PROMPTS, "w") as f:
        for r in sorted(sample, key=lambda r: r["sample_id"]):
            f.write(json.dumps({"sample_id": r["sample_id"], "prompt": text[r["sample_id"]]}) + "\n")
    counts = {}
    for r in sample:
        counts.setdefault(f"{r['task']}/{r['split']}", {}).setdefault(r["label"], 0)
        counts[f"{r['task']}/{r['split']}"][r["label"]] += 1
    print(json.dumps({"bands": bands, "menu": menu, "sampled": len(sample), "labels": counts}, indent=1))


def bench(base_url, schema_path, name, token=None, split="test"):
    sys.path.insert(0, HERE)
    import brain_bench
    prompts = {json.loads(l)["sample_id"]: json.loads(l)["prompt"] for l in open(PROMPTS)}
    schema = json.load(open(schema_path))
    out = os.path.join(DERIVED, f"decisions-{name}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["sample_id"] for l in open(out)}
    rows = [json.loads(l) for l in open(os.path.join(DERIVED, "labels.jsonl"))]
    rows = [r for r in rows if split == "all" or r["split"] == split]
    todo = [r for r in rows if r["sample_id"] not in done]
    print(f"[rb] {len(todo)} of {len(rows)} prompts to decide ({name})", flush=True)
    for n, r in enumerate(todo, 1):
        payload = brain_bench.ask(base_url, {"request": prompts[r["sample_id"]][:4000]}, schema, token)
        a = payload["answers"]
        rec = {"sample_id": r["sample_id"], "task": r["task"], "split": r["split"],
               "probabilities": {q: v.get("probabilities") for q, v in a.items()},
               "band": a["tier"]["choice"], "band_p": max(a["tier"]["probabilities"].values()),
               "ms": payload.get("_round_trip_ms"), "input_tokens": payload.get("usage", {}).get("input_tokens")}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if n % 200 == 0:
            print(f"[rb] {n}/{len(todo)}", flush=True)


def _point(rows, decide, menu):
    q = sum(r["scores"][menu[decide(r)]] for r in rows) / len(rows)
    c = sum(r["costs"][menu[decide(r)]] for r in rows) / len(rows)
    return c, q


def up(band):
    return BANDS[min(BANDS.index(band) + 1, 2)]


def score():
    cfg = json.load(open(os.path.join(DERIVED, "bands.json")))
    menu = cfg["menu"]
    rows = [json.loads(l) for l in open(os.path.join(DERIVED, "labels.jsonl"))]
    deciders = {}
    for path in sorted(os.listdir(DERIVED)):
        if path.startswith("decisions-") and path.endswith(".jsonl"):
            deciders[path[len("decisions-"):-len(".jsonl")]] = {json.loads(l)["sample_id"]: json.loads(l) for l in open(os.path.join(DERIVED, path))}
    report = {}
    for t in TASKS:
        test = [r for r in rows if r["task"] == t and r["split"] == "test"]
        singles = {m: (sum(r["costs"][m] for r in test) / len(test), sum(r["scores"][m] for r in test) / len(test)) for m in MODELS}
        cmin, cmax = min(c for c, _ in singles.values()), max(c for c, _ in singles.values())
        oracle = (sum(min((r["costs"][m] for m in MODELS if r["scores"][m] >= 1.0), default=min(r["costs"].values())) for r in test) / len(test),
                  sum(max(r["scores"].values()) for r in test) / len(test))
        entry = {"n_test": len(test), "singles": singles, "zero_router_aiq": aiq(list(singles.values()), cmin, cmax),
                 "oracle": oracle, "label_router": _point(test, lambda r: r["label"], menu), "routers": {}}
        for name, dec in deciders.items():
            have = [r for r in test if r["sample_id"] in dec]
            if len(have) < len(test):
                continue
            curve = [_point(have, (lambda tau: lambda r: up(dec[r["sample_id"]]["band"]) if dec[r["sample_id"]]["band_p"] < tau else dec[r["sample_id"]]["band"])(tau), menu) for tau in TAUS]
            acc = sum(dec[r["sample_id"]]["band"] == r["label"] for r in have) / len(have)
            entry["routers"][name] = {"band_acc": acc, "curve": curve, "aiq": aiq(curve, cmin, cmax)}
        report[t] = entry
    json.dump(report, open(os.path.join(DERIVED, "..", "routerbench.json"), "w"), indent=1)
    lines = ["| task | n test | zero-router AIQ | " + " | ".join(f"{n} AIQ (band acc)" for n in deciders) + " | label-router (cost, quality) | oracle (cost, quality) |",
             "|---|---|---|" + "---|" * len(deciders) + "---|---|"]
    for t, e in report.items():
        cells = " | ".join(f"{e['routers'][n]['aiq']:.3f} ({e['routers'][n]['band_acc']:.2f})" if n in e["routers"] else "—" for n in deciders)
        lines.append(f"| {t} | {e['n_test']} | {e['zero_router_aiq']:.3f} | {cells} | ({e['label_router'][0]:.4f}, {e['label_router'][1]:.3f}) | ({e['oracle'][0]:.4f}, {e['oracle'][1]:.3f}) |")
    open(os.path.join(DERIVED, "..", "routerbench.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["inspect", "prepare", "bench", "score"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8111")
    ap.add_argument("--schema", default=os.path.join(HERE, "..", "laya", "questions.json"))
    ap.add_argument("--name", default="laya")
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    args = ap.parse_args(argv)
    if args.cmd == "inspect":
        cols = columns()
        missing = [c for m in MODELS for c in (m, f"{m}|total_cost") if c not in cols]
        print("columns OK" if not missing else f"MISSING {missing}")
        return 1 if missing else 0
    if args.cmd == "prepare":
        prepare()
    elif args.cmd == "bench":
        bench(args.base_url, args.schema, args.name, os.environ.get("LAYA_API_TOKEN"), args.split)
    else:
        score()
    return 0


if __name__ == "__main__":
    sys.exit(main())
