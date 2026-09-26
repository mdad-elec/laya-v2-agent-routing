#!/usr/bin/env python3
"""vLLM Semantic Router's released intent classifier on this programme's corpora (PREREG amendment 4).

    ~/venvs/laya/bin/python scripts/ab/vllmsr_bench.py classify --corpus tier-240 --out results/laya-v2/vllmsr-tier-240.jsonl
    ~/venvs/laya/bin/python scripts/ab/vllmsr_bench.py classify --corpus cells-pilot --out results/laya-v2/vllmsr-cells-pilot.jsonl
    python3 scripts/ab/vllmsr_bench.py report          # fit the category → band table on tune, read held once

RouterArena's adapter for this router (router_inference/router/vllm_sr.py) asks the router's intent endpoint for a
category and looks it up in a category → model table. This runs the classifier behind that endpoint directly and
fits the table on tune. Classification needs torch + transformers (the laya venv); the report is stdlib.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "laya-v2")
CHECKPOINT = "llm-semantic-router/mmbert32k-intent-classifier-merged"
RANK = {"small": 0, "medium": 1, "powerful": 2}


def fit_table(rows):
    """category → the band most tune rows of that category carry (ties to the more capable band), and the default
    band for a category tune never saw (tune's majority over all rows, same tie rule)."""
    def best(counter):
        return max(counter, key=lambda b: (counter[b], RANK[b]))
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r["category"]][r["gold"]] += 1
    return {c: best(n) for c, n in by.items()}, best(collections.Counter(r["gold"] for r in rows))


def band_for(category, table, default):
    return table.get(category, default)


def cmd_classify(args):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    sys.path.insert(0, HERE)
    import brain_bench
    import routellm_bench
    torch.set_num_threads(4)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT).eval()
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    labels = json.load(open(hf_hub_download(CHECKPOINT, "category_mapping.json")))
    idx = labels.get("idx_to_category") or labels
    sha = routellm_bench._weights_sha(CHECKPOINT)
    rows = brain_bench.rows_for(args.corpus, "all")
    with open(args.out, "w") as f:
        for r in rows:
            inputs = tok(r["message"], return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                p = torch.softmax(model(**inputs).logits[0], -1)
            k = int(p.argmax())
            f.write(json.dumps({"id": r["id"], "gold": r["tier"], "corpus": r["corpus"], "category": idx[str(k)],
                                "p": round(float(p[k]), 4), "checkpoint": CHECKPOINT, "weights_sha256": sha}) + "\n")
    print(f"classified {len(rows)} rows → {args.out} ({CHECKPOINT} @ {sha[:12]})")


def cmd_report(args):
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, "..", "publish"))
    import render_figures as rf
    import routellm_bench as rlb
    import split_tools
    import stats
    rows = rlb.load_jsonl(os.path.join(RES, "vllmsr-tier-240.jsonl")) + rlb.load_jsonl(os.path.join(RES, "vllmsr-cells-pilot.jsonl"))
    frozen = {k: set(split_tools.load("tier-240")[k]) | set(split_tools.load("cells-pilot")[k]) for k in ("tune", "held")}
    tune, held = rlb.split_rows(rows, frozen["tune"]), rlb.split_rows(rows, frozen["held"])
    table, default = fit_table(tune)
    tune_acc = sum(band_for(r["category"], table, default) == r["gold"] for r in tune) / len(tune)
    pred = {r["id"]: band_for(r["category"], table, default) for r in held}
    v2 = rf.held_records("r4-held-v2-policy")
    ids = {r["id"] for r in v2}
    if ids != set(pred):
        raise SystemExit(f"held rows differ: {len(ids ^ set(pred))} ids not shared")
    judge = {r["id"]: r["predicted"] for r in rf.judge_held(ids)}
    gold = {r["id"]: r["gold"] for r in v2}
    lv2 = {r["id"]: r["predicted"] for r in v2}
    corpus = {r["id"]: r["corpus"] for r in v2}
    order = sorted(ids)
    out = {"checkpoint": CHECKPOINT, "weights_sha256": rows[0]["weights_sha256"], "table": table, "default": default,
           "tune_accuracy": tune_acc, "n_tune": len(tune), "n_held": len(order),
           "held_categories": dict(collections.Counter(r["category"] for r in held).most_common()),
           "share": {b: sum(pred[i] == b for i in order) / len(order) for b in RANK}, "by_corpus": {}}
    for name, sel in (("all", lambda i: True), ("borrowed", lambda i: corpus[i] == "requests"),
                      ("pilot", lambda i: corpus[i] != "requests")):
        xs = [i for i in order if sel(i)]
        acc = lambda p: sum(p[i] == gold[i] for i in xs) / len(xs)  # noqa: E731
        m = stats.mcnemar_exact([pred[i] == gold[i] for i in xs], [lv2[i] == gold[i] for i in xs])
        out["by_corpus"][name] = {"n": len(xs), "vllmsr": acc(pred), "laya_v2": acc(lv2), "judge": acc(judge),
                                  "vllmsr_only": m["a_only"], "laya_v2_only": m["b_only"],
                                  "p_laya_v2_better": m["p_b_better"],
                                  "under": sum(RANK[pred[i]] < RANK[gold[i]] for i in xs),
                                  "over": sum(RANK[pred[i]] > RANK[gold[i]] for i in xs)}
    with open(os.path.join(RES, "vllmsr.json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    with open(os.path.join(RES, "decisions-vllmsr.jsonl"), "w") as f:
        for r in rows:
            if r["corpus"] == "cells-pilot":
                f.write(json.dumps({"id": r["id"], "band": band_for(r["category"], table, default)}) + "\n")
    print(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("classify")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    sub.add_parser("report")
    args = ap.parse_args(argv)
    return {"classify": cmd_classify, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
