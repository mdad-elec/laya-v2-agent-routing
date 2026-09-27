"""How many public training prompts contain an evaluation question (PREREG-g1 Amendment 1 leakage).

Writes results/g1/leakage.json: the eval index's size, and the contaminated training prompts by
eval source and by training dataset. Every one of them is excluded from training.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from labels import routerarena as ra
from labels import twinrouterbench as trb
from labels.gate_s3 import load_rows, prereg_sha
from labels.leakage import build_eval_index


def suite_items() -> list:
    from suites.bfcl.suite import BFCL
    from suites.bird.suite import BIRDMiniDev
    from suites.chat.suite import ArenaHard
    from suites.ifeval.suite import IFEval
    from suites.mmlu_pro.suite import MMLUPro
    from suites.tatqa.suite import TATQA

    items = []
    for suite in (MMLUPro(), TATQA(), IFEval(), BIRDMiniDev(), ArenaHard(), BFCL()):
        items += suite.load()
    return items


def source_of(qid: str) -> str:
    parts = qid.split("/")
    return "/".join(parts[:2]) if parts[0] in ("suite", "routerarena") else parts[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/g1/leakage.json")
    args = ap.parse_args()
    index = build_eval_index(ra.questions(), trb.load(), suite_items())
    rows, prompts = load_rows()
    datasets: dict[str, set] = defaultdict(set)
    for r in rows:
        datasets[r["prompt_hash"]].add(r["dataset"])
    by_source, by_dataset, contaminated = Counter(), Counter(), 0
    for h, text in prompts.items():
        hits = index.hits(text)
        if hits:
            contaminated += 1
            by_source.update({source_of(q) for q in hits})
            by_dataset.update(datasets[h])
    result = {"prereg_sha": prereg_sha(), "eval_questions_indexed": len(index), "eval_questions_too_short_to_match": len(index.skipped),
              "training_prompts": len(prompts), "contaminated_prompts": contaminated,
              "by_eval_source": dict(by_source.most_common()), "by_training_dataset": dict(by_dataset.most_common())}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("eval_questions_indexed", "training_prompts", "contaminated_prompts")}))


if __name__ == "__main__":
    main()
