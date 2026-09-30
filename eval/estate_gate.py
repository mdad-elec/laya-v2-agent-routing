"""S8's estate gate (O1): does recipe 2 route THIS estate's asks at least as well as Laya-v2?

Owner decision 2026-09-30: the production table may be fitted on the estate's TUNE outcomes; held
asks stay held. This is the dss/auto table, not the published checkpoint, which stays public-only.

- **Asks:** the Laya-v2 study's frozen splits (`studies/v2/dataset/data/splits.json`): the pilot
  asks (`cases.jsonl`) and the borrowed tier-labelled requests (`requests_tier.jsonl`). The
  private DSS rows are not in the dataset and are left out, as v2's headline left them out.
- **Label:** each ask's band (small 0, medium 1, powerful 2), the estate's own "cheapest band that
  does it well".
- **Brain:** the proxy of Phase 2, a logistic regression answering v0's `task`, trained on PUBLIC
  texts only (the g1-v1 digests, MiniLM vectors, mapped by `train/questions.py`), applied to each
  ask's digest.
- **Menu, fixed before the read, chosen on tune by 5-fold out-of-fold band accuracy:**
  - `table`: the band table keyed by the predicted task, m ∈ {0, 1, 4};
  - `knn`: the k nearest tune asks by cosine (k ∈ {3, 5, 9}), shrunk toward the task row
    (m ∈ {1, 4}).
  τ runs over 0.50–0.98 and the policy is the cheapest band with P ≥ τ.
- **Gate:** held band accuracy ≥ 0.771, which is v2's 0.801 minus the 0.03 margin (point
  estimate), with a Wilson 90% interval and a by-corpus breakdown reported. Read once.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from labels import twinrouterbench as trb
from labels.gate_s3 import prereg_sha
from router.digest import build, render
from train.questions import benchmark_answers, dataset_answers
from train.table import TierTable, cheapest_passing

V2 = Path("studies/v2/dataset/data")
G1 = Path.home() / ".cache/laya-g1/train/g1-v1"
BANDS = {"small": 0, "medium": 1, "powerful": 2}
TAU = tuple(round(0.5 + 0.02 * i, 2) for i in range(25))
BAR = 0.801 - 0.03


def estate_asks() -> tuple[list[dict], list[dict]]:
    splits = json.loads((V2 / "splits.json").read_text())
    text = {}
    for line in (V2 / "cases.jsonl").open():
        r = json.loads(line)
        text[r["id"]] = (r["prompt"], r["tier"], "pilot")
    for line in (V2 / "requests_tier.jsonl").open():
        r = json.loads(line)
        text[r["id"]] = (r["message"], r["tier"], "borrowed")
    out = {"tune": [], "held": []}
    for part in ("cells-pilot", "tier-240"):
        for split in ("tune", "held"):
            for i in splits[part][split]:
                if i in text:
                    prompt, tier, corpus = text[i]
                    out[split].append({"id": i, "prompt": prompt, "band": BANDS[tier], "corpus": corpus})
    return out["tune"], out["held"]


def public_task_classifier(enc) -> LogisticRegression:
    vec = {}
    for line in (G1 / "digest-minilm.jsonl").open():
        r = json.loads(line)
        vec[r["id"]] = r["e"]
    bench = {r["id"]: r["benchmark"] for r in trb.load()}
    labels = {}
    for line in (G1 / "examples.jsonl").open():
        e = json.loads(line)
        if e["text"] in labels or e["text"] not in vec:
            continue
        if e["source"] == "twinrouterbench":
            labels[e["text"]] = benchmark_answers(bench[e["native_id"]])[0]
        else:
            labels[e["text"]] = dataset_answers(e["dataset"])[0]
    ids = sorted(labels)
    X = np.array([vec[i] for i in ids], dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return LogisticRegression(C=1.0, max_iter=2000).fit(X, [labels[i] for i in ids])


def fold(ask_id: str) -> int:
    return int(hashlib.sha256(f"estate-gate/{ask_id}".encode()).hexdigest()[:8], 16) % 5


def predict_bands(fit, query, X, task, cfg) -> list[int]:
    kind, k, m, tau = cfg
    row = TierTable(m).fit([task[i] for i in fit], [asks_band[i] for i in fit])
    out = []
    for q in query:
        p_row = row.p_solve(task[q])
        if kind == "table":
            p = p_row
        else:
            sims = X[fit] @ X[q]
            nn = [fit[j] for j in np.argsort(-sims, kind="stable")[:k]]
            p = [(sum(asks_band[j] <= b for j in nn) + m * p_row[b]) / (len(nn) + m) for b in range(4)]
        out.append(cheapest_passing(p, tau))
    return out


asks_band: list[int] = []


def main() -> None:
    from sentence_transformers import SentenceTransformer

    tune, held = estate_asks()
    rows = tune + held
    asks_band.extend(r["band"] for r in rows)
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    digests = [render(build([{"role": "user", "content": r["prompt"]}])) for r in rows]
    X = enc.encode(digests, batch_size=64, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    task = list(public_task_classifier(enc).predict(X))
    tune_idx = list(range(len(tune)))
    held_idx = list(range(len(tune), len(rows)))

    menu = [("table", 0, m, t) for m in (0.0, 1.0, 4.0) for t in TAU] + [("knn", k, m, t) for k in (3, 5, 9) for m in (1.0, 4.0) for t in TAU]

    def oof_acc(cfg) -> float:
        hits = 0
        for f in range(5):
            q = [i for i in tune_idx if fold(rows[i]["id"]) == f]
            fit = [i for i in tune_idx if fold(rows[i]["id"]) != f]
            hits += sum(p == asks_band[i] for p, i in zip(predict_bands(fit, q, X, task, cfg), q, strict=True))
        return hits / len(tune_idx)

    scored = sorted(((oof_acc(c), c) for c in menu), key=lambda s: (-s[0], s[1][0] != "table", -s[1][3]))
    best_acc, best = scored[0]
    pred = predict_bands(tune_idx, held_idx, X, task, best)
    hit = [p == asks_band[i] for p, i in zip(pred, held_idx, strict=True)]
    acc = sum(hit) / len(hit)

    def wilson(k, n, z=1.645):
        c = (k + z * z / 2) / (n + z * z)
        h = z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
        return [round(c - h, 4), round(c + h, 4)]

    by = defaultdict(list)
    for h_, i in zip(hit, held_idx, strict=True):
        by[rows[i]["corpus"]].append(h_)
    confusion = defaultdict(int)
    for p, i in zip(pred, held_idx, strict=True):
        confusion[f"{asks_band[i]}->{p}"] += 1
    result = {"prereg_sha": prereg_sha(), "gate": "S8 estate gate O1 (read once)", "n_tune": len(tune), "n_held": len(held),
              "chosen_on_tune": {"kind": best[0], "k": best[1], "m": best[2], "tau": best[3], "tune_oof_accuracy": round(best_acc, 4)},
              "held_band_accuracy": round(acc, 4), "wilson90": wilson(sum(hit), len(hit)),
              "by_corpus": {c: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for c, v in by.items()},
              "gold->pred": dict(sorted(confusion.items())),
              "v2_reference": {"pooled_151": 0.801, "borrowed_91": 0.857, "pilot_60": 0.717},
              "bar": round(BAR, 3), "verdict": "MET" if acc >= BAR else "NOT MET",
              "note": "held counts come from the published dataset (private DSS rows excluded, as v2's headline excluded them)"}
    out = Path.home() / ".cache/laya-g1/estate-gate.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
