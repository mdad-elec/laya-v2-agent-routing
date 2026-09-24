"""The gateway's cell chooser (`backend/crates/dss-gateway/src/cells.rs`), in python, so a decider
can be scored against a MENU from precomputed outcomes without running a gateway — RouterBench's
method. Proven against the picks the gateway itself made in the recorded take (`test_menu.py`); a
port that disagrees with the Rust is not a port.

    python3 scripts/ab/menu.py emit [--measured FILE] [--out scripts/ab/corpus/menus/cells.json]

A cell is `provider/model@effort`; its stats come from a `measured.json` (propose_cells.py).
Quality is the lower confidence bound shrunk toward the benchmark prior while thin; the frontier
drops every cell another beats on quality, latency AND cost; the band's profile weighs the
normalized terms; an empty band steps UP, never down.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BANDS = ("small", "medium", "powerful")
PRIOR_WEIGHT_N0 = 4.0
PRIOR_ONLY_DISCOUNT = 0.8
# cells.rs Policy::default — the spec's starting profiles (quality, time, cost).
PROFILES = {"small": (0.2, 0.6, 0.2), "medium": (0.4, 0.3, 0.3), "powerful": (0.7, 0.2, 0.1)}
PROVIDERS = os.path.join(HERE, "..", "..", "fleet", "providers.json")


def load(path):
    with open(path) as f:
        return json.load(f)


def _ladder_key(providers_path=PROVIDERS):
    """The gateway's rung order, for ties only: blended list price (3 input + 1 output), then the
    provider id, then declaration order of models and efforts (ladder.rs `ladder`)."""
    order, recs = {}, load(providers_path)["providers"]
    k = 0
    for r in recs:
        for m in r.get("models", []):
            pr = m.get("pricing") or {}
            blended = ((pr.get("input_microusd_per_million_tokens") or 0) * 3 + (pr.get("output_microusd_per_million_tokens") or 0)) / 4
            for e in [x["id"] for x in (m.get("reasoning") or {}).get("efforts", [])] or [None]:
                order[f"{r['provider_id']}/{m['model_id']}@{e}"] = (blended, r["provider_id"], k)
                k += 1
    return order


def cells_from_measured(measured, providers_path=PROVIDERS):
    """Every measured cell as {cell, provider, model, effort, q[3], latency, cost, prior}, in ladder order."""
    key = _ladder_key(providers_path)
    out = []
    for prov_model, entries in measured.items():
        provider, model = prov_model.split("/", 1)
        for m in entries:
            name = f"{prov_model}@{m['effort']}"
            out.append({"cell": name, "provider": provider, "model": model, "effort": m["effort"],
                        "q": [m.get(b) for b in BANDS], "latency": max(1, m["latency_ms_p50"]),
                        "cost": m["cost_microusd_p50"], "prior": m.get("prior")})
    out.sort(key=lambda c: key.get(c["cell"], (math.inf, c["provider"], 0)))
    return out


def effective_quality(cell, band):
    s, p = cell["q"][BANDS.index(band)], cell.get("prior")
    if s and p is not None:
        return (s["n"] * s["lcb"] + PRIOR_WEIGHT_N0 * p) / (s["n"] + PRIOR_WEIGHT_N0)
    if s:
        return s["lcb"]
    if p is not None:
        return p * PRIOR_ONLY_DISCOUNT
    return None


def _point(cell, band):
    q = effective_quality(cell, band)
    return None if q is None else (q, float(cell["latency"]), float(cell["cost"]))


def frontier(cells, band):
    pts = [(c, _point(c, band)) for c in cells]
    pts = [(c, p) for c, p in pts if p is not None]
    return [c for c, a in pts if not any(b[0] >= a[0] and b[1] <= a[1] and b[2] <= a[2] and
                                          (b[0] > a[0] or b[1] < a[1] or b[2] < a[2]) for _, b in pts)]


def _norm(vals, higher_is_better):
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-12:
        return [0.5] * len(vals)
    return [((v - lo) if higher_is_better else (hi - v)) / (hi - lo) for v in vals]


def score(cells, band, profile=None):
    """Frontier cells best first, each with its normalized terms and score (ties keep ladder order)."""
    wq, wt, wc = profile or PROFILES[band]
    f = frontier(cells, band)
    if not f:
        return []
    pts = [_point(c, band) for c in f]
    q = _norm([p[0] for p in pts], True)
    t = _norm([math.log(p[1]) for p in pts], False)
    c = _norm([math.log(p[2] + 1.0) for p in pts], False)
    scored = [{"cell": cell["cell"], "quality": q[k], "time": t[k], "cost": c[k],
               "score": wq * q[k] + wt * t[k] + wc * c[k], "frontier": len(f), "_i": k} for k, cell in enumerate(f)]
    scored.sort(key=lambda s: (-s["score"], s["_i"]))
    return scored


def pick(cells, band, profiles=None):
    """The best cell for `band`, stepping UP when the band has nothing ranked; None when nothing is."""
    profiles = profiles or PROFILES
    for current in BANDS[BANDS.index(band):]:
        ranked = score(cells, current, profiles[current])
        if ranked:
            best = dict(ranked[0])
            best.pop("_i")
            best["band"], best["stepped_to"] = band, (current if current != band else None)
            return best
    return None


def emit(measured_path, out_path):
    cells = cells_from_measured(load(measured_path))
    body = {"measured": os.path.relpath(measured_path, os.path.join(HERE, "..", "..")),
            "profiles": PROFILES,
            "menu": {b: pick(cells, b)["cell"] for b in BANDS},
            "frontier": {b: [c["cell"] for c in frontier(cells, b)] for b in BANDS}}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps(body, indent=1) + "\n")
    print(f"wrote {out_path}: " + ", ".join(f"{b} → {body['menu'][b]}" for b in BANDS))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["emit"])
    ap.add_argument("--measured", default=os.path.join(HERE, "results", "20260923-cells-pilot", "measured.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "corpus", "menus", "cells.json"))
    args = ap.parse_args(argv)
    emit(args.measured, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
