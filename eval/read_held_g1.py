"""The held read of F1 and F3 (PREREG-g1 Amendments 6, 7 and 10), run once.

F1 (TwinRouterBench held rows):
- Scored router (Amendment 10): Phase 2's proxy brain (logistic regression on Qwen3 vectors,
  fitted on all tune rows) answers v0 task/domain; the task + stage + errors table, m 0, τ 0.76,
  fitted on all tune rows with their true answers.
- Part 1: g1 − SR-KNN (k = 1, index = all tune rows), paired bootstrap over held trajectories,
  2,000 resamples, seed 20260927 (Amendment 6); met when the CI90 lower bound > 0.
- Part 2 (Amendment 7): g1's IN-SAMPLE Combined, with the table fitted on the scored held rows
  themselves (as SR-KNN's 77.89 index holds them), must exceed 77.89. Labelled in-sample.
- Reported, never selected: R4 (stage from tool rounds, τ 0.78), task-only, structure-only,
  per-benchmark Combined.

F3 (LLMRouterBench held items, Amendment 10's restated bar):
- g1 = `blend(0.5)`; every router is fitted on usable tune items (tune − calibration −
  contaminated prompts).
- Met when, against EACH of Avengers-Pro and EmbedLLM: AvgAcc(θ*) difference CI90 lower > −0.02,
  and recall-slice difference CI90 lower > 0. Bootstrap stratified by dataset, 10,000 resamples,
  seed 20260927, θ* fixed first.

Writes results/g1/verdict.json and one heldout-ledger.jsonl line per bar.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from eval import ceiling as c1
from eval import ceiling_f3 as c3
from eval import ceiling_f3_mf as c3mf
from eval import hillclimb_f1 as hc
from eval import llmrouterbench_f3 as f3
from eval.baselines import avengers_pro, embedllm
from eval.baselines.f3_predict import GRID, load_vectors
from eval.baselines.sr_knn import features, predict
from eval.baselines.sr_knn_twinrouterbench import load_embeddings
from eval.phase2_f1 import v0
from eval.twinrouterbench_score import score
from labels import twinrouterbench as trb
from labels.gate_s3 import prereg_sha
from train.table import TierTable, cheapest_passing

SEED = 20260927
F1_FIELDS, F1_M, F1_TAU = ("task", "stage", "errors"), 0.0, 0.76
PUBLISHED = 77.89
KEYS = ("case_pass_rate_percent", "case_exact_match_percent", "trajectory_pass_rate_percent", "cost_savings_score_percent", "combined_score_percent")


def f1_read() -> dict:
    rows = trb.load()
    tune = [i for i, r in enumerate(rows) if trb.split(r) == "tune"]
    held = [i for i, r in enumerate(rows) if trb.split(r) == "held"]
    y = np.array([r["target_tier_id"] for r in rows])
    true = [v0(r, c1.structural(r)) for r in rows]
    emb = load_embeddings(c1.EMBEDDINGS)
    E = np.stack([emb[r["id"]] for r in rows])
    Xn = E / np.linalg.norm(E, axis=1, keepdims=True)
    pred_ans = [dict(a) for a in true]
    accuracy = {}
    for field in ("task", "domain"):
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xn[tune], [true[i][field] for i in tune])
        for i, v in zip(held, clf.predict(Xn[held]).tolist(), strict=True):
            pred_ans[i][field] = v
        accuracy[field] = round(float(np.mean([pred_ans[i][field] == true[i][field] for i in held])), 4)

    def route(fit, query, fields, m, tau, q_ans, fit_ans=true):
        t = TierTable(m).fit([c1.key(fit_ans[i], fields) for i in fit], [int(y[i]) for i in fit])
        return {i: cheapest_passing(t.p_solve(c1.key(q_ans[i], fields)), tau) for i in query}

    def s(idx, p):
        out = score([rows[i] for i in idx], {rows[i]["id"]: p[i] for i in idx})
        return {k: round(out[k], 2) for k in KEYS}

    g1 = route(tune, held, F1_FIELDS, F1_M, F1_TAU, pred_ans)
    knn = dict(zip(held, predict(features(E)[tune], y[tune], features(E)[held], k=1).tolist(), strict=True))
    c1.SEED, c1.BOOTSTRAP = SEED, 2000
    boot = c1.bootstrap(rows, held, g1, knn)
    in_sample = s(held, route(held, held, F1_FIELDS, F1_M, F1_TAU, pred_ans))

    r4_ans = [hc.with_round_stage(r, a) for r, a in zip(rows, pred_ans, strict=True)]
    r4_true = [hc.with_round_stage(r, a) for r, a in zip(rows, true, strict=True)]
    reported = {
        "R4 stage from tool rounds (tau 0.78)": s(held, route(tune, held, F1_FIELDS, F1_M, 0.78, r4_ans, r4_true)),
        "task only": s(held, route(tune, held, ("task",), F1_M, F1_TAU, pred_ans)),
        "structure only": s(held, route(tune, held, ("stage", "errors", "context"), F1_M, 0.92, pred_ans)),
        "constants": {f"always_{trb.TIERS[t]}": s(held, dict.fromkeys(held, t))["combined_score_percent"] for t in range(4)},
        "per_benchmark_combined": {b: {"g1": s(idx, g1)["combined_score_percent"], "sr_knn": s(idx, knn)["combined_score_percent"]}
                                   for b in sorted({rows[i]["benchmark"] for i in held})
                                   for idx in [[i for i in held if rows[i]["benchmark"] == b]]},
    }
    part1 = boot["ci90"][0] > 0
    part2 = in_sample["combined_score_percent"] > PUBLISHED
    return {"held_rows": len(held), "held_trajectories": len({trb.trajectory_key(rows[i]) for i in held}),
            "proxy_answer_accuracy_held": accuracy, "g1_held": s(held, g1), "sr_knn_held": s(held, knn),
            "part1_paired_bootstrap_g1_minus_sr_knn": boot, "part1": "MET" if part1 else "NOT MET",
            "part2_g1_in_sample_held": in_sample, "part2_published_sr_knn_in_sample": PUBLISHED, "part2": "MET" if part2 else "NOT MET",
            "F1": "MET" if part1 and part2 else "NOT MET", "reported_not_selected": reported}


def f3_read() -> dict:
    from labels import routerarena as ra
    from labels.leakage import build_eval_index
    from labels.leakage_report import suite_items

    index = build_eval_index(ra.questions(), trb.load(), suite_items())
    items = f3.items()
    fit = [it for it in items if it["split"] == "tune" and not c3.is_calibration(it) and not index.hits(it["query"])]
    held = [it for it in items if it["split"] == "held"]
    vec = load_vectors(c3.EMBEDDINGS)
    routers = {"g1": c3mf.variant_configs("blend(0.5)", fit, held, vec)}
    emb = c3mf.plain(vec, fit + held)
    routers["avengers_pro"] = {f"alpha={a}": avengers_pro.route(fit, held, emb, alpha=a) for a in GRID}
    probs = c3mf.mf(fit, held, emb)
    cost = embedllm.expected_costs(fit)
    routers["embedllm"] = {f"tau={tau}": {c3.k(it): embedllm.pick(p, cost, tau) for it, p in zip(held, probs, strict=True)} for tau in GRID}
    singles = {mdl: {c3.k(it): mdl for it in held} for mdl in f3.MODELS}
    best_single = max(f3.MODELS, key=lambda mdl: (f3.avg_acc(fit, {c3.k(it): mdl for it in fit}), mdl))
    metrics = c3.evaluate(held, routers, singles, best_single)

    def theta(r):
        return routers[r][metrics[r]["theta_star"].split("/", 1)[1]]

    c3.SEED, c3.BOOTSTRAP = SEED, 10_000
    boot = {b: c3.bootstrap(held, theta("g1"), theta(b)) for b in ("avengers_pro", "embedllm")}
    checks = {b: {"avg_acc_non_inferior": boot[b]["avg_acc_diff"]["ci90"][0] > -0.02,
                  "slice_superior": boot[b]["slice_acc_diff"]["ci90"][0] > 0} for b in boot}
    met = all(v for c in checks.values() for v in c.values())
    return {"held_items": len(held), "fit_items": len(fit), "metrics": metrics, "paired_bootstrap_g1_minus": boot,
            "checks": checks, "F3": "MET" if met else "NOT MET"}


def main() -> None:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    ledger = Path("results/g1/heldout-ledger.jsonl")
    result = {"prereg_sha": prereg_sha(), "code": head, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    result["F1"] = f1_read()
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": result["at"], "prereg_sha": result["prereg_sha"], "code": head, "what": "F1 held read (Amendments 6, 7, 10), read 1",
                             "held_rows_scored_individually": result["F1"]["held_rows"], "verdict": result["F1"]["F1"]}) + "\n")
    result["F3"] = f3_read()
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "prereg_sha": result["prereg_sha"], "code": head,
                             "what": "F3 held read (Amendments 3, 6, 9, 10), read 1", "held_items_scored": result["F3"]["held_items"],
                             "verdict": result["F3"]["F3"]}) + "\n")
    result["F2"] = "not run: needs the owner's RouterArena spend approval"
    Path("results/g1/verdict.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"F1": {k: result["F1"][k] for k in ("g1_held", "sr_knn_held", "part1_paired_bootstrap_g1_minus_sr_knn", "part1",
                                                          "part2_g1_in_sample_held", "part2", "F1", "proxy_answer_accuracy_held")},
                      "F3": {k: result["F3"][k] for k in ("metrics", "paired_bootstrap_g1_minus", "checks", "F3")}}, indent=1))


if __name__ == "__main__":
    main()
