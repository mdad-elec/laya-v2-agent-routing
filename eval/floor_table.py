"""The confidence floor, re-measured on the scale the brain now answers on (typed.rs:
"re-measure this whenever the brain, the question wording or the state shape changes").

    python3 scripts/ab/floor_table.py RESULT.json [--wording W] [--thresholds 0.35,0.40,...]

A brain_bench report's records → rows below / accuracy below / accuracy above per threshold, and the
recommended DSS_ROUTER_MIN_CONFIDENCE: the HIGHEST threshold that fires on at most a fifth of turns
and whose turns below it are no better than a coin flip (accuracy < 0.50) — the most protection that
still leaves the brain deciding four turns in five. None meets both: say so, keep the shipped value.
"""
import argparse
import json
import sys

DEFAULT = [0.35, 0.37, 0.40, 0.42, 0.45, 0.50, 0.55, 0.60]


def table(records, thresholds):
    out = []
    for t in thresholds:
        below = [r["predicted"] == r["gold"] for r in records if r["confidence"] < t]
        above = [r["predicted"] == r["gold"] for r in records if r["confidence"] >= t]
        out.append({"threshold": t, "below": len(below), "share_below": len(below) / len(records),
                    "acc_below": sum(below) / len(below) if below else None,
                    "acc_above": sum(above) / len(above) if above else None})
    return out


def recommend(rows, max_share=0.20, coin=0.50):
    ok = [r for r in rows if r["share_below"] <= max_share and r["acc_below"] is not None and r["acc_below"] < coin]
    return max(ok, key=lambda r: r["threshold"]) if ok else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result")
    ap.add_argument("--wording")
    ap.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT))
    args = ap.parse_args(argv)
    body = json.load(open(args.result))
    blocks = [k for k in body["records"] if k.startswith("borrowed/") and (not args.wording or k.endswith("/" + args.wording))]
    if len(blocks) != 1:
        sys.exit(f"pick one record block with --wording; found {blocks}")
    rows = table(body["records"][blocks[0]], [float(t) for t in args.thresholds.split(",")])
    q = lambda v: "—" if v is None else f"{v:.3f}"
    print("| threshold | rows below | share | accuracy below | accuracy above |\n|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['threshold']:.2f} | {r['below']} | {r['share_below']:.2f} | {q(r['acc_below'])} | {q(r['acc_above'])} |")
    pick = recommend(rows)
    print(f"\nrecommended DSS_ROUTER_MIN_CONFIDENCE: {pick['threshold']:.2f}" if pick else
          "\nno threshold fires on <= 20% of turns with coin-flip accuracy below it; keep the shipped value")
    return 0


if __name__ == "__main__":
    sys.exit(main())
