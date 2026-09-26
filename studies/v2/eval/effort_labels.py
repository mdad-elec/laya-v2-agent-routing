"""The effort an ask needs, read off the measured cells — not opined (scripts/laya/PREREG.md, H6).

    python3 scripts/ab/effort_labels.py ~/dss-ab-cells/20260923-pilot [--delta 0.05]
        → scripts/ab/corpus/derived/effort-gold.jsonl and the constant baselines on tune

For ask a and model m measured at two or more canonical levels (efforts.py: none|off|low → minimal,
high|on → standard, max → maximal), the level m needs is the LOWEST one whose graded quality is
within delta of m's best on a. The ask's gold is the histogram of that level over the models — a
soft target by construction — and its label the histogram's mode (ties go to the higher level, the
safe direction). Fewer than three such models: no label. An unscored answer is left out.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell_bench  # noqa: E402
import efforts  # noqa: E402
import split_tools  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "corpus", "derived", "effort-gold.jsonl")


def needed(quality_by_level, delta=0.05):
    """The lowest canonical level within delta of this model's best on this ask, or None."""
    if len(quality_by_level) < 2:
        return None
    best = max(quality_by_level.values())
    for level in efforts.LEVELS:
        if level in quality_by_level and quality_by_level[level] >= best - delta:
            return level
    return None


def _score(g):
    return g["check"] if g.get("check") is not None else g.get("panel_score")


def labels(grades, delta=0.05, min_models=3):
    by = {}
    for g in grades:
        s = _score(g)
        if s is None:
            continue
        ask, cell, _ = g["key"].split("|")
        model, _, effort = cell.rpartition("@")
        level = efforts.canonical(effort)
        if level is None:
            continue
        by.setdefault(ask, {}).setdefault(model, {})[level] = s
    out = {}
    for ask, models in by.items():
        votes = [n for n in (needed(q, delta) for q in models.values()) if n]
        if len(votes) < min_models:
            continue
        gold = {lv: votes.count(lv) / len(votes) for lv in efforts.LEVELS}
        label = max(reversed(efforts.LEVELS), key=lambda lv: gold[lv])
        out[ask] = {"id": ask, "gold": gold, "label": label, "n_models": len(votes)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    grades = list(cell_bench.load_done(os.path.join(args.dir, "grades.jsonl")).values())
    rows = labels(grades, args.delta)
    split = split_tools.load("cells-pilot")
    side = {i: "tune" for i in split["tune"]} | {i: "held" for i in split["held"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for ask in sorted(rows):
            f.write(json.dumps({**rows[ask], "split": side.get(ask)}) + "\n")
    tune = [r for a, r in rows.items() if side.get(a) == "tune"]
    print(f"wrote {args.out}: {len(rows)} labelled asks ({len(tune)} tune), delta {args.delta}")
    for level in efforts.LEVELS:
        acc = sum(r["label"] == level for r in tune) / len(tune) if tune else 0
        print(f"  constant `{level}` on tune: {acc:.3f}")
    by_band = {}
    for r in rows.values():
        by_band.setdefault(r["id"][0], []).append(r["label"])
    print("  labels by band: " + ", ".join(f"{b}: " + "/".join(f"{lv[:3]} {v.count(lv)}" for lv in efforts.LEVELS)
                                           for b, v in sorted(by_band.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
