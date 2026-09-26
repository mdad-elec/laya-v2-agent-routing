"""Today's judge, asked for a BAND, so it can be compared with Laya on the same three-way decision.

    python3 scripts/ab/judge_adapter.py fit    --in results/laya-v2/judge-tune.jsonl  --out results/laya-v2/judge-adapter.json
    python3 scripts/ab/judge_adapter.py decide --in results/laya-v2/judge-held.jsonl  --adapter results/laya-v2/judge-adapter.json

The judge (`CLASSIFIER_PROMPT` in dss-gateway/src/router.rs) answers p_solve and a capability
boundary, and `classify_tier` turns that into TWO tiers: efficient iff p_solve ≥ base + step ×
steps(boundary). The adapter keeps that rule exactly — capable is `powerful` — and splits efficient
in two with one parameter, `t_small`: a `supported` verdict at or above it is `small`, anything else
efficient is `medium`. `t_small` is fitted on the TUNE split, the same courtesy Laya's selection gets.
The base, step and prompt are read out of the Rust source, never retyped (rule 8).
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTER_RS = os.path.join(HERE, "..", "..", "backend", "crates", "dss-gateway", "src", "router.rs")


def _rust():
    with open(ROUTER_RS) as f:
        return f.read()


def rust_thresholds():
    src = _rust()
    base = float(re.search(r"pub const DEFAULT_BASE_THRESHOLD: f64 = ([0-9.]+);", src).group(1))
    step = float(re.search(r"pub const DEFAULT_THRESHOLD_STEP: f64 = ([0-9.]+);", src).group(1))
    return base, step


def rust_classifier_prompt():
    """The Rust string literal, decoded: a trailing `\\` joins lines and swallows the next line's
    leading whitespace, and `\\"` is a quote."""
    raw = re.search(r'pub const CLASSIFIER_PROMPT: &str = "(.*?)";\n', _rust(), re.S).group(1)
    return re.sub(r"\\\n\s*", "", raw).replace('\\"', '"')


BASE, STEP = rust_thresholds()


def band_of_verdict(p_solve, boundary, t_small, base=BASE, step=STEP):
    if p_solve is None:
        return "powerful"              # an unusable verdict routes capable (the gateway's fail-strong)
    steps = {"supported": 0, "unsupported": 2}.get(boundary, 1)
    if p_solve < min(1.0, base + step * steps):
        return "powerful"
    return "small" if boundary == "supported" and p_solve >= t_small else "medium"


GRID = [round(0.80 + 0.01 * i, 2) for i in range(20)] + [1.01]


def fit_t_small(rows):
    """The grid point with the best tune band accuracy; ties go to the lowest threshold."""
    best = None
    for t in GRID:
        acc = sum(band_of_verdict(r["p_solve"], r["boundary"], t) == r["gold"] for r in rows) / len(rows)
        if best is None or acc > best[1]:
            best = (t, acc)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "decide"])
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out")
    ap.add_argument("--adapter")
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    if args.cmd == "fit":
        t, acc = fit_t_small(rows)
        body = {"t_small": t, "tune_band_accuracy": acc, "n": len(rows), "base": BASE, "step": STEP, "fitted_on": os.path.basename(args.inp)}
        with open(args.out, "w") as f:
            json.dump(body, f, indent=1)
        print(f"t_small={t} tune band acc {acc:.3f} (n={len(rows)}) → {args.out}")
    else:
        t = json.load(open(args.adapter))["t_small"]
        for r in rows:
            print(json.dumps({"id": r["id"], "band": band_of_verdict(r["p_solve"], r["boundary"], t),
                              "p_solve": r["p_solve"], "boundary": r["boundary"], "gold": r.get("gold")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
