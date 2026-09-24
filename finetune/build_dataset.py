"""The fine-tune's rows (rung 4), in the shape upstream's RLCD recipe reads: `{id, state, questions,
gold, split, source, variant}` with state/questions/gold as JSON strings.

    python3 scripts/laya/build_dataset.py --variant v2-policy|v2-outcome --pilot DIR --out DIR [--paraphrases FILE]

  state      the SERVING shape: {"request": <message>} (PREREG amendment after rung 1)
  questions  the served schema (scripts/laya/questions.json), byte-for-byte in its own order
  gold       per question {label, probabilities}:
               tier  v2-policy:  the corpus label, smoothed toward the adjacent tier
                     v2-outcome: 0.5 × that + 0.5 × the MEASURED target — most mass on the cheapest
                                 band whose cells-menu cell answered the ask well (score >= 0.8)
               effort (pilot asks, when the schema asks it): the measured histogram (effort_labels.py)

Rows come from the TUNE split only (the borrowed tier rows and the pilot asks; unaudited DSS rows
wait for gate G3), cut 80/20 into train/val by a seeded draw — val picks the epoch and fits the
temperature, so held-out is never read. Every row within 0.6 similarity of ANY held-out ask of
EITHER corpus is dropped, and the drops are listed (cross-corpus near-duplicates exist: t04~sml-02).
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AB = os.path.join(HERE, "..", "ab")
sys.path.insert(0, AB)
sys.path.insert(0, os.path.join(AB, "corpus"))
import checks  # noqa: E402
import split_tools  # noqa: E402
import validate_corpus  # noqa: E402

BANDS = ("small", "medium", "powerful")
SMOOTH = {"small": {"small": 0.9, "medium": 0.1, "powerful": 0.0},
          "medium": {"small": 0.05, "medium": 0.9, "powerful": 0.05},
          "powerful": {"small": 0.0, "medium": 0.1, "powerful": 0.9}}
SOLVED = 0.8
SEED = 20260923


def smoothed(label):
    return dict(SMOOTH[label])


def band_target(q):
    """q: band → graded score of the cells-menu cell for that band on this ask (None = unscored)."""
    if any(q.get(b) is None for b in BANDS):
        return None
    passing = [b for b in BANDS if q[b] >= SOLVED]
    if not passing:
        return None
    cheapest, rest = passing[0], passing[1:]
    t = {b: 0.0 for b in BANDS}
    t[cheapest] = 0.6 if rest else 1.0
    total = sum(q[b] for b in rest)
    for b in rest:
        t[b] = 0.4 * q[b] / total
    return t


def blend(label_dist, measured, w=0.5):
    if measured is None:
        return dict(label_dist)
    return {b: (1 - w) * label_dist[b] + w * measured[b] for b in BANDS}


def drop_leaks(rows, held_messages):
    kept, dropped = [], []
    for r in rows:
        if r["id"] in held_messages:
            dropped.append({"id": r["id"], "why": "held out"})
            continue
        near = next(((hid, s) for hid, m in held_messages.items() if (s := validate_corpus.similar(r["message"], m)) >= validate_corpus.DUP), None)
        if near:
            dropped.append({"id": r["id"], "why": f"near duplicate of held {near[0]} ({near[1]:.2f})"})
            continue
        kept.append(r)
    return kept, dropped


def _gold(dist):
    return {"label": max(dist, key=dist.get), "probabilities": dist}


def row(rid, message, schema, gold_by_question, split, source, variant, parent=None):
    return {"id": rid, "state": json.dumps({"request": message}, ensure_ascii=False),
            "questions": json.dumps(schema, ensure_ascii=False),
            "gold": json.dumps({q: _gold(d) for q, d in gold_by_question.items()}, ensure_ascii=False),
            "split": split, "source": source, "variant": variant, **({"parent_id": parent} if parent else {})}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=["v2-policy", "v2-outcome"])
    ap.add_argument("--schema", default=os.path.join(HERE, "questions.json"))
    ap.add_argument("--pilot", required=True, help="a cell_bench run dir with graded answers (the 120-ask one)")
    ap.add_argument("--paraphrases")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    import factorial
    schema = json.load(open(args.schema))
    band_q = next(q for q, v in schema.items() if v.get("type") == "choice" and set(v.get("criteria") or {}) == set(BANDS))
    t240, pil = split_tools.load("tier-240"), split_tools.load("cells-pilot")
    borrowed = {r["id"]: r for r in validate_corpus.load(os.path.join(AB, "corpus", "requests.jsonl"))}
    dss = {r["id"]: r for r in validate_corpus.load(os.path.join(AB, "corpus", "dss-requests.jsonl"))}
    pilot = {r["id"]: r for r in validate_corpus.load(os.path.join(AB, "corpus", "cells-pilot.jsonl"))}
    held = {i: borrowed[i]["message"] for i in t240["held"] if i in borrowed}
    held |= {i: dss[i]["message"] for i in t240["held"] if i in dss}
    held |= {i: pilot[i]["prompt"] for i in pil["held"] if i in pilot}
    candidates = [{"id": i, "message": borrowed[i]["message"], "tier": borrowed[i]["tier"], "source": "requests"}
                  for i in t240["tune"] if i in borrowed]
    candidates += [{"id": i, "message": pilot[i]["prompt"], "tier": checks.BAND_OF[pilot[i]["band"]], "source": "cells-pilot"}
                   for i in pil["tune"] if i in pilot and not pilot[i].get("private")]
    kept, dropped = drop_leaks(candidates, held)
    rng = random.Random(f"{SEED}-val")
    val_ids = set()
    for src in ("requests", "cells-pilot"):
        ids = sorted(r["id"] for r in kept if r["source"] == src)
        val_ids |= set(rng.sample(ids, max(1, len(ids) // 5)))
    outcomes = factorial.load_outcomes(args.pilot) if args.variant == "v2-outcome" else {}
    menu = json.load(open(os.path.join(AB, "corpus", "menus", "cells.json")))["menu"]
    effort_gold = {}
    ep = os.path.join(AB, "corpus", "derived", "effort-gold.jsonl")
    if "effort" in schema and os.path.exists(ep):
        effort_gold = {r["id"]: r["gold"] for r in (json.loads(l) for l in open(ep))}
    out_rows, measured_n = [], 0
    for r in kept:
        dist = smoothed(r["tier"])
        if args.variant == "v2-outcome" and r["source"] == "cells-pilot":
            q = {b: (outcomes.get((r["id"], menu[b])) or {}).get("score") for b in BANDS}
            t = band_target(q)
            measured_n += t is not None
            dist = blend(dist, t)
        gold = {band_q: dist}
        if r["id"] in effort_gold:
            gold["effort"] = effort_gold[r["id"]]
        out_rows.append(row(r["id"], r["message"], schema, gold, "val" if r["id"] in val_ids else "train", r["source"], args.variant))
    if args.paraphrases:
        by_parent = {x["id"]: x for x in out_rows}
        paras = [p for p in (json.loads(l) for l in open(args.paraphrases) if l.strip()) if p.get("message")]
        para_kept, para_dropped = drop_leaks([{"id": p["id"], "message": p["message"], "parent_id": p["parent_id"]} for p in paras], held)
        dropped += para_dropped
        for p in para_kept:
            parent = by_parent.get(p["parent_id"])
            if parent is None or parent["split"] != "train":
                continue                      # only the train rows' paraphrases train; val stays unaugmented
            out_rows.append(dict(parent, id=p["id"], state=json.dumps({"request": p["message"]}, ensure_ascii=False), parent_id=p["parent_id"]))
    os.makedirs(args.out, exist_ok=True)
    for split in ("train", "val"):
        with open(os.path.join(args.out, f"{split}.jsonl"), "w") as f:
            f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_rows if x["split"] == split))
    report = {"variant": args.variant, "train": sum(x["split"] == "train" for x in out_rows),
              "val": sum(x["split"] == "val" for x in out_rows), "measured_targets": measured_n,
              "dropped": dropped, "schema_sha256": __import__("hashlib").sha256(open(args.schema, "rb").read()).hexdigest()}
    json.dump(report, open(os.path.join(args.out, "build_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "dropped"}), f"dropped {len(dropped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
