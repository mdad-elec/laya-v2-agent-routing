"""A3, SR-KNN on TwinRouterBench: reproduce the published in-sample number, then fit on tune.

Reads the Qwen3-Embedding vectors (one JSON line per bank row id, computed once on the Spark by
eval/baselines/embed_qwen3.py) and writes results/g1/a3-sr-knn.json:
- `in_sample`: SR-KNN over all 970 rows, scored by the benchmark's own scorer. AGGREGATES ONLY
  (the published Table 2 number, 77.89); no per-row output, so no held row is inspected.
  Leave-one-out 1-NN, and 5-NN including the row itself, since "in-sample 1-NN" admits both.
- `tune_cv`: the F1 baseline's recipe (index = tune rows only), estimated by 5-fold
  cross-validation grouped by trajectory INSIDE tune. The held score is read once, in A6.
Each run is also appended to results/g1/heldout-ledger.jsonl, as PREREG-g1 Amendment 1 requires.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eval.baselines.sr_knn import features, predict
from eval.twinrouterbench_score import score
from labels import twinrouterbench as trb
from labels.gate_s3 import prereg_sha
from suites.base import SPLIT_SALT

KEYS = ("case_pass_rate_percent", "case_exact_match_percent", "trajectory_pass_rate_percent", "cost_savings_score_percent", "combined_score_percent")


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    out = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rec = json.loads(line)
            out[rec["id"]] = np.asarray(rec["e"], dtype=np.float32)
    return out


def summary(s: dict) -> dict:
    return {k: round(s[k], 2) for k in KEYS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out", default="results/g1/a3-sr-knn.json")
    args = ap.parse_args()
    rows = trb.load()
    emb = load_embeddings(Path(args.embeddings))
    missing = [r["id"] for r in rows if r["id"] not in emb]
    if missing:
        raise SystemExit(f"{len(missing)} bank rows have no embedding, e.g. {missing[:3]}")
    X = features(np.stack([emb[r["id"]] for r in rows]))
    y = np.array([r["target_tier_id"] for r in rows])
    ids = [r["id"] for r in rows]

    in_sample = {}
    for name, k, loo in (("1nn_leave_one_out", 1, True), ("5nn_including_self", 5, False), ("1nn_including_self", 1, False)):
        pred = predict(X, y, X, k=k, leave_one_out=loo)
        in_sample[name] = summary(score(rows, dict(zip(ids, pred.tolist(), strict=True))))

    tune = [i for i, r in enumerate(rows) if trb.split(r) == "tune"]
    trajs = sorted({trb.trajectory_key(rows[i]) for i in tune})
    fold_of = {t: int(hashlib.sha256(f"{SPLIT_SALT}/fold/{t}".encode()).hexdigest()[:8], 16) % 5 for t in trajs}
    preds = {}
    for f in range(5):
        test = [i for i in tune if fold_of[trb.trajectory_key(rows[i])] == f]
        train = [i for i in tune if fold_of[trb.trajectory_key(rows[i])] != f]
        p = predict(X[train], y[train], X[test], k=1)
        preds.update({ids[i]: int(v) for i, v in zip(test, p, strict=True)})
    tune_rows = [rows[i] for i in tune]
    constant = {name: summary(score(tune_rows, {r["id"]: tier for r in tune_rows})) for name, tier in (("always_low", 0), ("always_mid", 1), ("always_mid_high", 2), ("always_high", 3))}
    result = {"prereg_sha": prereg_sha(), "bar": "F1", "baseline": "SR-KNN (Qwen/Qwen3-Embedding-0.6B + 'other' one-hot, cosine)",
              "published_in_sample_combined": 77.89, "in_sample": in_sample,
              "tune_cv": {"rows": len(tune_rows), "trajectories": len(trajs), "folds": 5, "k": 1, **summary(score(tune_rows, preds))},
              "tune_constant_routers": constant,
              "reproduced": False,
              "why_not": "No reading of 'in-sample 1-nearest-neighbor over question-bank embeddings' with the adapter's embedder and flattening gives "
                         "77.89: leave-one-out 1-NN is lower, 1-NN including the row itself is trivially 100% exact. The paper does not state "
                         "the exclusion rule, k, or the embedding truncation. F1 part 1 therefore compares against the well-defined tune-fitted "
                         "SR-KNN; part 2 keeps the published 77.89 as stated."}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    with (out.parent / "heldout-ledger.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "prereg_sha": result["prereg_sha"], "code": head,
                             "what": "A3 SR-KNN in-sample reproduction over all 970 TwinRouterBench rows (aggregate scores only) + tune-only CV",
                             "held_rows_scored_individually": 0}) + "\n")
    print(json.dumps({"in_sample": in_sample, "tune_cv": result["tune_cv"], "tune_constant_routers": constant}, indent=1))


if __name__ == "__main__":
    main()
