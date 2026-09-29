"""Fix τ before any held read (PREREG-g1 Amendment 5), from tune-side calibration rows only.

τ_F1: TwinRouterBench calibration trajectories (20% of tune, by `g1-cal/` hash). The policy picks
the lowest tier whose pool model has P ≥ τ, and the benchmark's own scorer gives Combined for
each grid τ. Writes results/g1/tau.json with the prereg sha and the checkpoint's step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.twinrouterbench_score import score
from labels import twinrouterbench as trb
from labels.gate_s3 import prereg_sha
from train.data import TIERS
from train.g1 import is_calibration
from train.policy import GRID, cheapest_above, choose_tau


def f1_rows_and_trajectories():
    rows = [r for r in trb.load() if trb.split(r) == "tune"]
    trajectory_of = {r["id"]: trb.trajectory_key(r) for r in rows}
    cal = [r for r in rows if is_calibration({"source": "twinrouterbench", "native_id": r["id"]}, trajectory_of)]
    return cal


def tier_choice(p_row: dict[str, float], tier_map: dict[str, str], tau: float) -> int:
    p = {t: p_row[tier_map[t]] for t in TIERS}
    return TIERS.index(cheapest_above(p, {t: i for i, t in enumerate(TIERS)}, tau))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probs", required=True, help="train.predict output over the calibration rows' digests")
    ap.add_argument("--out", default="results/g1/tau.json")
    args = ap.parse_args()
    probs = json.loads(Path(args.probs).read_text())
    cells = probs["cells"]
    tier_map = json.loads((Path(__file__).resolve().parent / "twinrouterbench_pool.json").read_text())["tier_to_model"]
    cal = f1_rows_and_trajectories()
    p_of = {rid: dict(zip(cells, p, strict=True)) for rid, p in probs["p"].items()}
    curve = {}

    def objective(tau):
        s = score(cal, {r["id"]: tier_choice(p_of[r["id"]], tier_map, tau) for r in cal})
        curve[tau] = round(s["combined_score_percent"], 3)
        return s["combined_score_percent"]

    tau = choose_tau(objective)
    result = {"prereg_sha": prereg_sha(), "ckpt_step": probs["ckpt_step"], "tau_f1": tau, "f1_calibration_rows": len(cal),
              "f1_calibration_combined_by_tau": {str(t): curve[t] for t in GRID}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(result)
    out.write_text(json.dumps(existing, indent=1))
    print(json.dumps({"tau_f1": tau, "calibration_rows": len(cal), "best_combined": curve[tau]}))


if __name__ == "__main__":
    main()
