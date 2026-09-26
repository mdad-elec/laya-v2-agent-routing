"""Fit one temperature per question on the TUNE split — the calibration rung (PREREG, R3).

    python3 scripts/laya/fit_temperature.py RESULT.json --question tier --t0 1.0 [--bucket choice:3-5]

RESULT.json is a brain_bench report run with --record-probs on the tune split. The sidecar serves
p = softmax(z / T0), so the logits are recovered up to a constant as z = T0 · log p; the fit finds
the T that minimizes the negative log-likelihood of the gold (a label, one-hot, or a distribution)
under softmax(z / T). T is the value to write into the served checkpoint's
`temperature_by_options[bucket]` (make_served_dir.py) — the sidecar applies it, the gateway reads
calibrated probabilities and stays policy-only. One-dimensional and convex in log T, so a golden-
section search in the standard library is exact enough and deterministic; torch is not needed.
A held-out row in the input is refused, by id.
"""
import argparse
import json
import math
import os
import sys

SDK_CLAMP = (0.5, 5.0)     # laya.common: temperatures outside are clamped when loaded
BOUNDS = (0.1, 10.0)


def target(row, options):
    gold = row["gold"]
    if isinstance(gold, str):
        return [1.0 if o == gold else 0.0 for o in options]
    total = sum(float(gold.get(o, 0.0)) for o in options)
    return [float(gold.get(o, 0.0)) / total for o in options]


def _nll(rows, t, t0):
    total = 0.0
    for row in rows:
        options = list(row["probabilities"])
        z = [t0 * math.log(max(row["probabilities"][o], 1e-12)) for o in options]
        top = max(v / t for v in z)
        log_norm = top + math.log(sum(math.exp(v / t - top) for v in z))
        y = target(row, options)
        total -= sum(yi * (zi / t - log_norm) for yi, zi in zip(y, z))
    return total / len(rows)


def fit(rows, t0=1.0, bounds=BOUNDS, iters=80):
    """The NLL-minimizing temperature, by golden-section search over log T."""
    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = _nll(rows, math.exp(a), t0), _nll(rows, math.exp(b), t0)
    for _ in range(iters):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = _nll(rows, math.exp(a), t0)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = _nll(rows, math.exp(b), t0)
    t = math.exp((lo + hi) / 2)
    return t, {"t0": t0, "t": t, "n": len(rows), "nll_before": _nll(rows, t0, t0), "nll_after": _nll(rows, t, t0)}


def outside_clamp(t):
    return not (SDK_CLAMP[0] <= t <= SDK_CLAMP[1])


def refuse_held(rows, held):
    bad = sorted({r["id"] for r in rows if r.get("id") in held})
    if bad:
        raise ValueError(f"calibration is fitted on tune only; these rows are held out: {', '.join(bad)}")
    return rows


def calibrated(probs, t_old, t_new):
    z = {o: t_old * math.log(max(p, 1e-12)) for o, p in probs.items()}
    top = max(v / t_new for v in z.values())
    e = {o: math.exp(v / t_new - top) for o, v in z.items()}
    s = sum(e.values())
    return {o: v / s for o, v in e.items()}


def ece(rows, bins=10):
    pairs = []
    for r in rows:
        p = r["probabilities"]
        choice = max(p, key=p.get)
        gold = r["gold"] if isinstance(r["gold"], str) else max(r["gold"], key=r["gold"].get)
        pairs.append((p[choice], choice == gold))
    total, err = len(pairs), 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = [(c, ok) for c, ok in pairs if (c > lo or (i == 0 and c >= lo)) and c <= hi]
        if m:
            err += len(m) / total * abs(sum(ok for _, ok in m) / len(m) - sum(c for c, _ in m) / len(m))
    return err, sum(c for c, _ in pairs) / total - sum(ok for _, ok in pairs) / total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result")
    ap.add_argument("--question", default="tier")
    ap.add_argument("--wording", help="which record block of the report (default: the only one)")
    ap.add_argument("--t0", type=float, required=True, help="the temperature the sidecar applied (its /health)")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "ab"))
    import split_tools
    held = set(split_tools.load("tier-240")["held"]) | set(split_tools.load("cells-pilot")["held"])
    body = json.load(open(args.result))
    blocks = [k for k in body["records"] if k.startswith("borrowed/") and (not args.wording or k.endswith("/" + args.wording))]
    if len(blocks) != 1:
        sys.exit(f"pick one record block with --wording; found {blocks}")
    recs = [r for r in body["records"][blocks[0]] if (r.get("probabilities") or {}).get(args.question)]
    rows = refuse_held([{"id": r["id"], "probabilities": r["probabilities"][args.question], "gold": r["gold"]} for r in recs], held)
    t, report = fit(rows, args.t0)
    before = ece(rows)
    after = ece([dict(r, probabilities=calibrated(r["probabilities"], args.t0, t)) for r in rows])
    report.update({"question": args.question, "block": blocks[0], "ece_before": before[0], "signed_gap_before": before[1],
                   "ece_after": after[0], "signed_gap_after": after[1], "outside_sdk_clamp": outside_clamp(t),
                   "source": os.path.basename(args.result)})
    print(json.dumps(report, indent=1))
    if report["outside_sdk_clamp"]:
        print(f"WARNING: T={t:.3f} is outside the SDK's clamp {SDK_CLAMP}; the sidecar would apply the clamped value, "
              "so this calibration cannot be served as fitted", file=sys.stderr)
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
