#!/usr/bin/env python3
"""RouteLLM's released BERT router on this programme's corpora and on RouterBench (PREREG amendment 3).

    ~/venvs/laya/bin/python scripts/ab/routellm_bench.py score --corpus tier-240 --split all --out results/laya-v2/routellm-tier-240.jsonl
    ~/venvs/laya/bin/python scripts/ab/routellm_bench.py score-prompts --in ~/routerbench/prompts-sample.jsonl --out …/routellm-routerbench.jsonl
    python3 scripts/ab/routellm_bench.py report                       # fit on tune, read held once, write routellm.md/json

The score is RouteLLM's `BERTRouter.calculate_strong_win_rate` verbatim: softmax over the three logits,
1 − P(tie) − P(weak wins). Two thresholds fitted on tune map it onto small / medium / powerful.
Scoring needs torch + transformers (the laya venv); fitting and the report are stdlib.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "laya-v2")
CHECKPOINT = "routellm/bert_gpt4_augmented"
GRID = [round(i / 100, 2) for i in range(0, 102)]            # 1.01 lets the powerful band be empty


def strong_win_rate(logits):
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    z = sum(e)
    return 1 - (e[1] + e[2]) / z


def band_of(score, t_lo, t_hi):
    return "small" if score < t_lo else "medium" if score < t_hi else "powerful"


def fit_bands(rows):
    """(t_lo, t_hi, tune accuracy): the best pair on the grid; ties go to the lowest t_lo, then the lowest t_hi."""
    best = None
    for i, t_lo in enumerate(GRID):
        for t_hi in GRID[i:]:
            acc = sum(band_of(r["score"], t_lo, t_hi) == r["gold"] for r in rows) / len(rows)
            if best is None or acc > best[2]:
                best = (t_lo, t_hi, acc)
    return best


def write_decisions(path, rows, t_lo, t_hi):
    with open(path, "w") as f:
        for r in rows:
            if r["corpus"] == "cells-pilot":
                f.write(json.dumps({"id": r["id"], "band": band_of(r["score"], t_lo, t_hi)}) + "\n")


def binary_curve(rows, score, strong, weak):
    """RouteLLM's native setting: the asks scored highest go to the strong model first. One point per
    share routed strong, 0 … all, in order of the router's score (ties broken by id, deterministic)."""
    order = sorted(rows, key=lambda r: (-score[r["id"]], str(r["id"])))
    n = len(rows)
    c = sum(r["costs"][weak] for r in rows)
    q = sum(r["scores"][weak] for r in rows)
    pts = [(c / n, q / n)]
    for r in order:
        c += r["costs"][strong] - r["costs"][weak]
        q += r["scores"][strong] - r["scores"][weak]
        pts.append((c / n, q / n))
    return pts


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ---- scoring (venv) -----------------------------------------------------------------------------------

class Scorer:
    def __init__(self, checkpoint=CHECKPOINT, threads=4):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3).eval()
        self.tok = AutoTokenizer.from_pretrained(checkpoint)
        self.sha = _weights_sha(checkpoint)

    def __call__(self, prompt):
        inputs = self.tok(prompt, return_tensors="pt", padding=True, truncation=True)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0].tolist()
        return strong_win_rate(logits), logits


def _weights_sha(checkpoint):
    import hashlib
    from huggingface_hub import hf_hub_download
    h = hashlib.sha256()
    with open(hf_hub_download(checkpoint, "model.safetensors"), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_score(args):
    sys.path.insert(0, HERE)
    import brain_bench
    rows = brain_bench.rows_for(args.corpus, args.split)
    s = Scorer()
    with open(args.out, "w") as f:
        for r in rows:
            score, logits = s(r["message"])
            f.write(json.dumps({"id": r["id"], "gold": r["tier"], "corpus": r["corpus"], "score": score,
                                "logits": logits, "checkpoint": CHECKPOINT, "weights_sha256": s.sha}) + "\n")
    print(f"scored {len(rows)} rows → {args.out} ({CHECKPOINT} @ {s.sha[:12]})")


def cmd_score_prompts(args):
    s = Scorer()
    done = {r["sample_id"] for r in load_jsonl(args.out)} if os.path.exists(args.out) else set()
    rows = [r for r in load_jsonl(getattr(args, "in")) if r["sample_id"] not in done]
    with open(args.out, "a") as f:
        for i, r in enumerate(rows[: args.limit or None]):
            score, _ = s(r["prompt"])
            f.write(json.dumps({"sample_id": r["sample_id"], "score": score}) + "\n")
            if i % 200 == 0:
                f.flush()
                print(f"{i}/{len(rows)}", flush=True)
    print("done", args.out)


def split_rows(rows, frozen):
    """Headline rows of one frozen split: the unaudited DSS requests stay apart, as in every H1 read."""
    return [r for r in rows if r["id"] in frozen and r["corpus"] != "dss-requests"]


def cmd_report(args):
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, "..", "publish"))
    import split_tools
    import stats
    import render_figures as rf
    rows = load_jsonl(os.path.join(RES, "routellm-tier-240.jsonl")) + load_jsonl(os.path.join(RES, "routellm-cells-pilot.jsonl"))
    frozen = {k: set(split_tools.load("tier-240")[k]) | set(split_tools.load("cells-pilot")[k]) for k in ("tune", "held")}
    tune, held = split_rows(rows, frozen["tune"]), split_rows(rows, frozen["held"])
    t_lo, t_hi, tune_acc = fit_bands(tune)
    pred = {r["id"]: band_of(r["score"], t_lo, t_hi) for r in held}
    v2 = rf.held_records("r4-held-v2-policy")
    ids = {r["id"] for r in v2}
    if ids != set(pred):
        raise SystemExit(f"held rows differ: {len(ids ^ set(pred))} ids not shared")
    judge = {r["id"]: r["predicted"] for r in rf.judge_held(ids)}
    gold = {r["id"]: r["gold"] for r in v2}
    lv2 = {r["id"]: r["predicted"] for r in v2}
    corpus = {r["id"]: r["corpus"] for r in v2}
    order = sorted(ids)

    def acc(p, sel=lambda i: True):
        xs = [i for i in order if sel(i)]
        return sum(p[i] == gold[i] for i in xs) / len(xs), len(xs)

    def mc(a, b, sel=lambda i: True):
        xs = [i for i in order if sel(i)]
        return stats.mcnemar_exact([a[i] == gold[i] for i in xs], [b[i] == gold[i] for i in xs])

    R = {"small": 0, "medium": 1, "powerful": 2}
    out = {"checkpoint": CHECKPOINT, "weights_sha256": rows[0]["weights_sha256"], "t_lo": t_lo, "t_hi": t_hi,
           "tune_accuracy": tune_acc, "n_tune": len(tune), "n_held": len(order), "share": {}, "by_corpus": {}}
    out["share"] = {b: sum(pred[i] == b for i in order) / len(order) for b in R}
    for name, sel in (("all", lambda i: True), ("borrowed", lambda i: corpus[i] == "requests"),
                      ("pilot", lambda i: corpus[i] != "requests")):
        out["by_corpus"][name] = {
            "routellm": acc(pred, sel), "laya_v2": acc(lv2, sel), "judge": acc(judge, sel),
            "laya_v2_vs_routellm": mc(lv2, pred, sel), "routellm_vs_judge": mc(pred, judge, sel),
            "routellm_under": sum(R[pred[i]] < R[gold[i]] for i in order if sel(i)),
            "routellm_over": sum(R[pred[i]] > R[gold[i]] for i in order if sel(i))}
    with open(os.path.join(RES, "routellm.json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    write_decisions(os.path.join(RES, "decisions-routellm.jsonl"), rows, t_lo, t_hi)
    print(json.dumps(out, indent=1))


STRONG, WEAK = "gpt-4-1106-preview", "mistralai/mixtral-8x7b-chat"


def cmd_routerbench(args):
    """Amendment 3's RouterBench leg: GPT-4 against Mixtral-8x7B, each router swept over its own score."""
    sys.path.insert(0, HERE)
    import routerbench_leg as leg
    d = os.path.join(RES, "routerbench")
    rows = [r for r in load_jsonl(os.path.join(d, "labels.jsonl")) if r["split"] == "test"]
    for r in rows:
        r["id"] = r["sample_id"]
    scores = {"routellm-bert": {r["sample_id"]: r["score"] for r in load_jsonl(os.path.join(d, "routellm-scores.jsonl"))}}
    for name in ("laya-root", "laya-v2-policy"):
        scores[name] = {r["sample_id"]: r["probabilities"]["tier"]["powerful"] for r in load_jsonl(os.path.join(d, f"decisions-{name}.jsonl"))}
    report = {}
    for task in sorted({r["task"] for r in rows}):
        test = [r for r in rows if r["task"] == task]
        cw = sum(r["costs"][WEAK] for r in test) / len(test)
        cs = sum(r["costs"][STRONG] for r in test) / len(test)
        qw = sum(r["scores"][WEAK] for r in test) / len(test)
        qs = sum(r["scores"][STRONG] for r in test) / len(test)
        entry = {"n": len(test), "weak": [cw, qw], "strong": [cs, qs], "line_aiq": leg.aiq([(cw, qw), (cs, qs)], cw, cs), "routers": {}}
        for name, sc in scores.items():
            curve = binary_curve(test, sc, STRONG, WEAK)
            half = curve[len(test) // 2]
            entry["routers"][name] = {"aiq": leg.aiq(curve, cw, cs), "quality_at_half_strong": half[1]}
        report[task] = entry
    # Paired bootstrap over prompts, per task and for the mean over tasks: RouteLLM − Laya-v2 AIQ.
    import random
    rng = random.Random(20260924)
    tasks = sorted(report)
    by_task = {t: [r for r in rows if r["task"] == t] for t in tasks}
    diffs = {t: [] for t in tasks}
    means = []
    for _ in range(1000):
        per = []
        for t in tasks:
            test = by_task[t]
            smp = [test[rng.randrange(len(test))] for _ in test]
            cw = sum(r["costs"][WEAK] for r in smp) / len(smp)
            cs = sum(r["costs"][STRONG] for r in smp) / len(smp)
            a_ = leg.aiq(binary_curve(smp, scores["routellm-bert"], STRONG, WEAK), cw, cs)
            b_ = leg.aiq(binary_curve(smp, scores["laya-v2-policy"], STRONG, WEAK), cw, cs)
            diffs[t].append(a_ - b_)
            per.append(a_ - b_)
        means.append(sum(per) / len(per))

    def ci(xs):
        xs = sorted(xs)
        return [xs[int(0.05 * len(xs))], xs[int(0.95 * len(xs)) - 1]]
    for t in tasks:
        report[t]["routellm_minus_laya_v2_ci90"] = ci(diffs[t])
    point = sum(report[t]["routers"]["routellm-bert"]["aiq"] - report[t]["routers"]["laya-v2-policy"]["aiq"] for t in tasks) / len(tasks)
    report["_mean_over_tasks"] = {"routellm_minus_laya_v2": point, "ci90": ci(means)}
    with open(os.path.join(RES, "routellm-routerbench.json"), "w") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    for t, e in report.items():
        if t.startswith("_"):
            print(t, e)
            continue
        print(t, e["n"], "diff ci", [round(x, 3) for x in e["routellm_minus_laya_v2_ci90"]], "line %.3f" % e["line_aiq"], " ".join(f"{n} {v['aiq']:.3f}" for n, v in e["routers"].items()),
              "| q@50%% strong " + " ".join(f"{n} {v['quality_at_half_strong']:.3f}" for n, v in e["routers"].items()),
              "| weak %.3f strong %.3f" % (e["weak"][1], e["strong"][1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score")
    p.add_argument("--corpus", required=True)
    p.add_argument("--split", default="all")
    p.add_argument("--out", required=True)
    p = sub.add_parser("score-prompts")
    p.add_argument("--in", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    sub.add_parser("report")
    sub.add_parser("routerbench")
    args = ap.parse_args(argv)
    return {"score": cmd_score, "score-prompts": cmd_score_prompts, "report": cmd_report, "routerbench": cmd_routerbench}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
