"""Gate S3 (PREREG-g1): is the routing band learnable from text at all? Tune items only.

For each dataset: TF-IDF (word 1-2 grams) + kNN (k=15, cosine), 5-fold CV grouped by prompt hash,
against the dataset's majority band. PASS = kNN - majority >= 0.05 on >= 3 datasets AND pooled
kNN - pooled majority >= 0.03. A dataset with > 85% of its labels in one band is reported and
left out of training. The result carries PREREG-g1's blob sha.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from labels.leakage import held_hashes, is_trainable
from labels.tiers import BANDS, ability_tiers, label_items
from suites.base import CACHE, split_of

K = 15
FOLDS = 5
MIN_GAIN, MIN_DATASETS, MIN_POOLED, MAX_ONE_BAND = 0.05, 3, 0.03, 0.85


def prereg_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD:PREREG-g1.md"], capture_output=True, text=True, check=True).stdout.strip()


def suite_held_hashes() -> set[str]:
    from suites.bfcl.suite import BFCL
    from suites.bird.suite import BIRDMiniDev
    from suites.chat.suite import ArenaHard
    from suites.ifeval.suite import IFEval
    from suites.mmlu_pro.suite import MMLUPro
    from suites.tatqa.suite import TATQA

    items = []
    for suite in (MMLUPro(), TATQA(), IFEval(), BIRDMiniDev(), ArenaHard(), BFCL()):
        items += suite.load()
    return held_hashes(items)


def load_rows() -> tuple[list[dict], dict[str, str]]:
    base = CACHE.parent / "public" / "llmrouterbench"
    rows = [json.loads(line) for line in (base / "outcomes.jsonl").open(encoding="utf-8")]
    prompts = json.loads((base / "prompts.json").read_text(encoding="utf-8"))
    # BFCL's episodes: the label row keys on the episode's first user turn, as the suite does.
    from labels.bfcl_result import load_all
    from labels.llmrouterbench import prompt_hash
    from suites.bfcl.suite import BFCL

    first_turn = {i.native_id: i.meta["source_text"] for i in BFCL().load()}
    bfcl_rows, _ = load_all()
    for r in bfcl_rows:
        text = first_turn[r["native_id"]]
        h = prompt_hash(text)
        prompts.setdefault(h, text)
        rows.append({"source": r["source"], "dataset": "bfcl_multi_turn_base", "native_id": r["native_id"], "model": r["model"],
                     "score": 1.0 if r["pass"] else 0.0, "prompt_hash": h})
    return rows, prompts


def knn_vs_majority(texts: list[str], labels: list[str]) -> tuple[float, float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsClassifier

    n = len(texts)
    folds = min(FOLDS, n)
    correct_knn = correct_major = 0
    for train, test in GroupKFold(n_splits=folds).split(texts, labels, groups=list(range(n))):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        X = vec.fit_transform([texts[i] for i in train])
        y = [labels[i] for i in train]
        knn = KNeighborsClassifier(n_neighbors=min(K, len(train)), metric="cosine").fit(X, y)
        pred = knn.predict(vec.transform([texts[i] for i in test]))
        majority = Counter(y).most_common(1)[0][0]
        correct_knn += sum(1 for i, p in zip(test, pred, strict=True) if p == labels[i])
        correct_major += sum(1 for i in test if labels[i] == majority)
    return correct_knn / n, correct_major / n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/g1/s3-gate.json")
    args = ap.parse_args()
    rows, prompts = load_rows()
    held = suite_held_hashes()
    tune = lambda h: split_of(h) == "tune" and is_trainable(h, held)  # noqa: E731
    tiers = ability_tiers(rows, tune=tune)
    labels = [x for x in label_items(rows, tiers) if tune(x["prompt_hash"])]
    by_dataset: dict[str, list] = defaultdict(list)
    for x in labels:
        by_dataset[x["dataset"]].append(x)
    report, pooled_knn, pooled_major, pooled_n, passing = {}, 0.0, 0.0, 0, 0
    for dataset, xs in sorted(by_dataset.items()):
        bands = Counter(x["band"] for x in xs)
        top_share = bands.most_common(1)[0][1] / len(xs)
        entry = {"items": len(xs), "bands": {b: bands.get(b, 0) for b in BANDS}, "unsolved": sum(x["unsolved"] for x in xs),
                 "models": len(tiers.get(dataset, {})), "top_band_share": round(top_share, 3)}
        if top_share > MAX_ONE_BAND or len(xs) < 2 * FOLDS:
            entry["left_out"] = f"{top_share:.0%} of labels in one band" if top_share > MAX_ONE_BAND else "too few items"
        else:
            knn, major = knn_vs_majority([prompts[x["prompt_hash"]] for x in xs], [x["band"] for x in xs])
            entry.update(knn=round(knn, 4), majority=round(major, 4), gain=round(knn - major, 4))
            passing += knn - major >= MIN_GAIN
            pooled_knn += knn * len(xs)
            pooled_major += major * len(xs)
            pooled_n += len(xs)
        report[dataset] = entry
    pooled_gain = (pooled_knn - pooled_major) / pooled_n if pooled_n else 0.0
    verdict = "PASS" if passing >= MIN_DATASETS and pooled_gain >= MIN_POOLED else "FAIL"
    result = {"prereg_sha": prereg_sha(), "gate": "S3", "verdict": verdict, "datasets_passing": passing,
              "pooled": {"knn": round(pooled_knn / pooled_n, 4) if pooled_n else None, "majority": round(pooled_major / pooled_n, 4) if pooled_n else None,
                         "gain": round(pooled_gain, 4), "items": pooled_n},
              "held_suite_prompts_excluded": len(held), "per_dataset": report}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("verdict", "datasets_passing", "pooled", "held_suite_prompts_excluded")}))


if __name__ == "__main__":
    main()
