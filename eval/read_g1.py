"""The A6 read: F1 and F3 scored ONCE, exactly as PREREG-g1 Amendments 1-6 fix them.

Inputs, all computed before this runs and none of them reading a held label:
- g1's P(solve) on F1 held rows and F3 held items (train.predict), and τ_F1 (train.calibrate);
- F3 baseline choices (eval/baselines/f3_predict.py) for each embedder;
- the Qwen3 embeddings of the TwinRouterBench bank (SR-KNN).

It appends one line to results/g1/heldout-ledger.jsonl and writes results/g1/verdict.json. A bar
not met is written as not met.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eval import llmrouterbench_f3 as f3
from eval.baselines.sr_knn import features, predict
from eval.baselines.sr_knn_twinrouterbench import load_embeddings
from eval.twinrouterbench_score import score
from labels import twinrouterbench as trb
from labels.gate_s3 import prereg_sha
from train.calibrate import tier_choice
from train.policy import GRID, cheapest_above

F1_PUBLISHED = 77.89
F1_RESAMPLES = 2000
F3_RESAMPLES = 10000
SEED = 20260927


def _ci(diffs: np.ndarray) -> list[float]:
    return [round(float(np.percentile(diffs, 5)), 4), round(float(np.percentile(diffs, 95)), 4)]


def read_f1(probs: dict, tau: float, embeddings: Path) -> dict:
    rows = trb.load()
    held = [r for r in rows if trb.split(r) == "held"]
    tune = [r for r in rows if trb.split(r) == "tune"]
    pool = json.loads((Path(__file__).resolve().parents[1] / "train" / "twinrouterbench_pool.json").read_text())["tier_to_model"]
    p_of = {rid: dict(zip(probs["cells"], p, strict=True)) for rid, p in probs["p"].items()}
    g1 = {r["id"]: tier_choice(p_of[r["id"]], pool, tau) for r in held}
    emb = load_embeddings(embeddings)
    knn = predict(features(np.stack([emb[r["id"]] for r in tune])), np.array([r["target_tier_id"] for r in tune]),
                  features(np.stack([emb[r["id"]] for r in held])), k=1)
    srknn = {r["id"]: int(v) for r, v in zip(held, knn, strict=True)}
    point = {"g1": score(held, g1), "sr_knn_tune_fit": score(held, srknn)}
    by_traj: dict[str, list[dict]] = {}
    for r in held:
        by_traj.setdefault(trb.trajectory_key(r), []).append(r)
    keys = sorted(by_traj)
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(F1_RESAMPLES):
        sample = []
        for i, k in enumerate(rng.choice(len(keys), len(keys), replace=True)):
            for r in by_traj[keys[k]]:  # a resampled trajectory is a new trajectory: rename so repeats do not merge
                sample.append({**r, "id": f"{r['id']}#{i}", "instance_id": f"{r['instance_id']}#{i}"})
        pg = {s["id"]: g1[s["id"].split("#")[0]] for s in sample}
        pk = {s["id"]: srknn[s["id"].split("#")[0]] for s in sample}
        diffs.append(score(sample, pg)["combined_score_percent"] - score(sample, pk)["combined_score_percent"])
    diffs = np.array(diffs)
    lo, hi = _ci(diffs)
    g1_combined = point["g1"]["combined_score_percent"]
    keys5 = ("case_pass_rate_percent", "case_exact_match_percent", "trajectory_pass_rate_percent", "cost_savings_score_percent", "combined_score_percent")
    return {"tau_f1": tau, "held_rows": len(held), "held_trajectories": len(keys),
            "scores": {k: {m: round(v[m], 2) for m in keys5} for k, v in point.items()},
            "paired_bootstrap": {"resamples": F1_RESAMPLES, "unit": "trajectory", "combined_diff_ci90": [lo, hi]},
            "part1_met": lo > 0, "part2_met": g1_combined > F1_PUBLISHED, "met": lo > 0 and g1_combined > F1_PUBLISHED}


def _configs_accuracy_cost(items, choices_by_config):
    return {c: (f3.avg_acc(items, ch), f3.total_cost(items, ch)) for c, ch in choices_by_config.items()}


def _item_vectors(items, choice):
    return np.array([it["scores"][choice[(it["dataset"], it["native_id"])]] for it in items])


def _macro_diff_ci(items, a, b, resamples, rng, subset=None) -> list[float]:
    """Paired, stratified by dataset: macro AvgAcc(a) − AvgAcc(b) over resampled held items."""
    idx = [i for i in range(len(items)) if subset is None or i in subset]
    by_ds: dict[str, list[int]] = {}
    for i in idx:
        by_ds.setdefault(items[i]["dataset"], []).append(i)
    va, vb = _item_vectors(items, a), _item_vectors(items, b)
    out = np.zeros(resamples)
    for ds_idx in by_ds.values():
        ds_idx = np.array(ds_idx)
        draw = ds_idx[rng.integers(0, len(ds_idx), size=(resamples, len(ds_idx)))]
        out += (va[draw].mean(1) - vb[draw].mean(1))
    return _ci(out / len(by_ds))


def read_f3(probs: dict, predictions_dir: Path) -> dict:
    items = f3.items()
    tune = [it for it in items if it["split"] == "tune"]
    held = [it for it in items if it["split"] == "held"]
    key = lambda it: (it["dataset"], it["native_id"])  # noqa: E731
    expected_cost = {m: sum(it["costs"][m] for it in tune) / len(tune) for m in f3.MODELS}
    p_of = {k: dict(zip(probs["cells"], p, strict=True)) for k, p in probs["p"].items()}
    routers = {"g1": {f"tau={t}": {key(it): cheapest_above(p_of[f"{it['dataset']}/{it['native_id']}"], expected_cost, t) for it in held} for t in GRID}}
    for name in ("avengers_pro", "embedllm"):
        blob = json.loads((predictions_dir / f"{name}.json").read_text())
        order = [tuple(k.split("/", 1)) for k in blob["items"]]
        routers[name] = {c: dict(zip(order, ch, strict=True)) for c, ch in blob["configs"].items()}
    singles = {m: {key(it): m for it in held} for m in f3.MODELS}
    tune_acc = {m: f3.avg_acc(tune, {key(it): m for it in tune}) for m in f3.MODELS}
    best = max(tune_acc, key=lambda m: (tune_acc[m], m))
    configs = {f"single/{m}": v for m, v in _configs_accuracy_cost(held, singles).items()}
    for r, chs in routers.items():
        configs.update({f"{r}/{c}": v for c, v in _configs_accuracy_cost(held, chs).items()})
    names = {r: [f"{r}/{c}" for c in chs] for r, chs in routers.items()}
    best_key = f"single/{best}"
    theta_star = {r: max(names[r], key=lambda c: configs[c][0]) for r in routers}
    slice_idx = {i for i, it in enumerate(held) if it in f3.recall_slice([it])}
    summary = {}
    for r in routers:
        star = routers[r][theta_star[r].split("/", 1)[1]]
        summary[r] = {"perf_gain": round(f3.perf_gain(configs, names[r], best_key), 4), "cost_save": f3.cost_save(configs, names[r], best_key),
                      "pareto_dist": round(f3.pareto_dist(configs, names[r]), 4), "theta_star": theta_star[r],
                      "avg_acc_theta_star": round(configs[theta_star[r]][0], 4),
                      "slice_acc_theta_star": round(float(_item_vectors(held, star)[sorted(slice_idx)].mean()), 4)}
    rng = np.random.default_rng(SEED)
    g1_star = routers["g1"][theta_star["g1"].split("/", 1)[1]]
    tests = {}
    for base in ("avengers_pro", "embedllm"):
        b_star = routers[base][theta_star[base].split("/", 1)[1]]
        acc_ci = _macro_diff_ci(held, g1_star, b_star, F3_RESAMPLES, rng)
        slice_ci = _macro_diff_ci(held, g1_star, b_star, F3_RESAMPLES, rng, subset=slice_idx)
        cs_g, cs_b = summary["g1"]["cost_save"], summary[base]["cost_save"]
        tests[base] = {"avg_acc_diff_ci90": acc_ci, "slice_acc_diff_ci90": slice_ci,
                       "perf_gain_met": acc_ci[0] > 0, "slice_met": slice_ci[0] > 0,
                       "cost_save_met": cs_g is not None and (cs_b is None or cs_g > cs_b)}
    met = all(t["perf_gain_met"] and t["slice_met"] and t["cost_save_met"] for t in tests.values())
    return {"best_single": best, "held_items": len(held), "slice_items": len(slice_idx), "routers": summary, "tests": tests, "met": met}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--f1-probs", required=True)
    ap.add_argument("--f3-probs", required=True)
    ap.add_argument("--tau", default="results/g1/tau.json")
    ap.add_argument("--trb-embeddings", required=True)
    ap.add_argument("--f3-predictions", required=True, help="the qwen3-embedding-0.6b predictions directory")
    ap.add_argument("--f3-predictions-minilm", help="the sensitivity embedder's predictions directory")
    ap.add_argument("--out", default="results/g1/verdict.json")
    args = ap.parse_args()
    tau = json.loads(Path(args.tau).read_text())
    f1p, f3p = json.loads(Path(args.f1_probs).read_text()), json.loads(Path(args.f3_probs).read_text())
    if f1p["ckpt_step"] != tau["ckpt_step"] or f3p["ckpt_step"] != tau["ckpt_step"]:
        raise SystemExit("the probabilities and τ come from different checkpoints")
    result = {"prereg_sha": prereg_sha(), "ckpt_step": tau["ckpt_step"],
              "F1": read_f1(f1p, tau["tau_f1"], Path(args.trb_embeddings)), "F3": read_f3(f3p, Path(args.f3_predictions))}
    if args.f3_predictions_minilm:
        result["F3_sensitivity_minilm"] = read_f3(f3p, Path(args.f3_predictions_minilm))
    result["F2"] = "not run: RouterArena needs an inference run (an owner spend gate)"
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=1))
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    with (out.parent / "heldout-ledger.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "prereg_sha": result["prereg_sha"], "code": head,
                             "ckpt_step": result["ckpt_step"], "what": "A6 read: F1 and F3 held, scored once",
                             "F1_met": result["F1"]["met"], "F3_met": result["F3"]["met"]}) + "\n")
    print(json.dumps({"F1": {k: result["F1"][k] for k in ("scores", "paired_bootstrap", "met")}, "F3": {k: result["F3"][k] for k in ("routers", "tests", "met")}}, indent=1))


if __name__ == "__main__":
    main()
