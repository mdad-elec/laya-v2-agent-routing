"""F1 hill-climb rounds on session POLICY (PREREG-g1 Amendment 7's protocol; "a rule on/off").

Baseline: Phase 2's router (task+stage+errors, m 0, τ 0.76) with v0 answers. Phase 2 showed the
proxy brain's answers route identically on HC-test, so v0's true answers stand in for speed.
Each round changes ONE thing, re-chooses τ on HC-train out of fold, and is kept only when HC-train
AND HC-test both improve past the noise floor, a paired-bootstrap CI half-width measured first.
Reported beside Combined: SWE-bench-only Combined, the one benchmark whose labels spread across
all four tiers (the Phase 1 hand-check found every other benchmark routed to `low`).

- R1 carry-forward: a step never routes below this trajectory's previous ROUTED step (G0: never
  downgrade mid-session). Only the router's own earlier decisions are used, never a label.
- R2 escalate-after-error: one tier up when the digest shows an error in the recent tool rounds.
- R3: R1 + R2.

Appends one line per round to results/g1/hillclimb.jsonl. No held row is read.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eval import ceiling as c1
from eval.phase2_f1 import v0
from eval.twinrouterbench_score import score
from labels import twinrouterbench as trb
from labels.gate_s3 import prereg_sha
from train.table import TierTable, cheapest_passing

FIELDS = ("task", "stage", "errors")
M = 0.0
BOOT = 1000
LEDGER = Path("results/g1/hillclimb.jsonl")


def table_choice(fit, query, ans, y, tau):
    t = TierTable(M).fit([c1.key(ans[i], FIELDS) for i in fit], [int(y[i]) for i in fit])
    return {i: cheapest_passing(t.p_solve(c1.key(ans[i], FIELDS)), tau) for i in query}


def apply_policy(rows, base: dict[int, int], ans, carry: bool, escalate: bool) -> dict[int, int]:
    out = {}
    by_traj = defaultdict(list)
    for i in base:
        by_traj[trb.trajectory_key(rows[i])].append(i)
    for idx in by_traj.values():
        prev = None
        for i in sorted(idx, key=lambda j: rows[j].get("step_index", 1)):
            t = base[i]
            if escalate and ans[i]["errors"]:
                t = min(t + 1, 3)
            if carry and prev is not None:
                t = max(t, prev)
            out[i] = prev = t
    return out


def route(rows, fit, query, ans, y, tau, carry, escalate):
    return apply_policy(rows, table_choice(fit, query, ans, y, tau), ans, carry, escalate)


def oof(rows, folds, train, ans, y, tau, carry, escalate):
    pred = {}
    for f in c1.HC_TRAIN_FOLDS:
        q = [i for i in train if folds[i] == f]
        fit = [i for i in train if folds[i] != f]
        pred.update(route(rows, fit, q, ans, y, tau, carry, escalate))
    return pred


def comb(rows, idx, pred):
    return score([rows[i] for i in idx], {rows[i]["id"]: pred[i] for i in idx})["combined_score_percent"]


def main() -> None:
    rows = [r for r in trb.load() if trb.split(r) == "tune"]
    y = np.array([r["target_tier_id"] for r in rows])
    ans = [v0(r, c1.structural(r)) for r in rows]
    folds = {i: c1.hc_fold(r) for i, r in enumerate(rows)}
    train = [i for i in folds if folds[i] in c1.HC_TRAIN_FOLDS]
    test = [i for i in folds if folds[i] in c1.HC_TEST_FOLDS]
    swe_train = [i for i in train if rows[i]["benchmark"] == "swebench"]
    swe_test = [i for i in test if rows[i]["benchmark"] == "swebench"]

    def evaluate(carry, escalate):
        tau = max(c1.TAU_GRID, key=lambda t: (comb(rows, train, oof(rows, folds, train, ans, y, t, carry, escalate)), -t))
        p_tr = oof(rows, folds, train, ans, y, tau, carry, escalate)
        p_te = route(rows, train, test, ans, y, tau, carry, escalate)
        return {"tau": tau, "hc_train": round(comb(rows, train, p_tr), 2), "hc_test": round(comb(rows, test, p_te), 2),
                "swe_hc_train": round(comb(rows, swe_train, p_tr), 2), "swe_hc_test": round(comb(rows, swe_test, p_te), 2)}, p_te

    base, base_pred = evaluate(False, False)
    rounds = {"R1 carry-forward": (True, False), "R2 escalate-after-error": (False, True), "R3 carry+escalate": (True, True)}
    results = {}
    for name, (carry, esc) in rounds.items():
        res, pred = evaluate(carry, esc)
        boot = c1.bootstrap(rows, test, pred, base_pred) if pred != base_pred else {"mean": 0.0, "ci90": [0.0, 0.0]}
        results[name] = (res, boot)

    # noise floor: the widest paired CI half-width among the rounds (a round identical to baseline has 0)
    floor = max(0.5, max((b["ci90"][1] - b["ci90"][0]) / 2 for _, b in results.values()))
    lines = []
    for name, (res, boot) in results.items():
        d_train, d_test = res["hc_train"] - base["hc_train"], res["hc_test"] - base["hc_test"]
        keep = d_train > 0 and d_test > floor
        decision = "keep" if keep else ("revert: train-only gain (overfit)" if d_train > 0 and d_test <= 0 else "revert: inside the noise floor" if d_test > 0 else "revert")
        lines.append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "prereg_sha": prereg_sha(), "bar": "F1",
                      "change": name, "baseline": base, "round": res, "paired_bootstrap_round_minus_baseline_hc_test": boot,
                      "noise_floor_combined_points": round(floor, 2), "decision": decision})
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")
    print(json.dumps({"baseline": base, "noise_floor": round(floor, 2),
                      "rounds": {ln["change"]: {**ln["round"], "boot": ln["paired_bootstrap_round_minus_baseline_hc_test"], "decision": ln["decision"]} for ln in lines}}, indent=1))


if __name__ == "__main__":
    main()
