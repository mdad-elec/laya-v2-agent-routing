"""MMLU-Pro (TIGER-Lab, MIT): 10-option multiple choice across 14 subjects -- the `knowledge` domain.

Fetched at a pinned dataset revision from huggingface.co/datasets/TIGER-Lab/MMLU-Pro. The prompt
and the answer extraction follow the benchmark's own convention ("the answer is (X)"), with a
final "Answer: X" line also accepted; only the LAST stated answer counts, so a model that reasons
through a wrong option and settles on the right one is scored on what it settled on.
"""
from __future__ import annotations

import re
import urllib.request

from suites.base import Item, Suite

LETTERS = "ABCDEFGHIJ"
_FORMS = (re.compile(r"answer is \(?([A-J])\)?", re.I), re.compile(r"\banswer\s*[:：]\s*\(?([A-J])\)?", re.I))


def extract_letter(answer: str) -> str | None:
    found = []
    for pattern in _FORMS:
        found.extend((m.start(), m.group(1).upper()) for m in pattern.finditer(answer))
    return max(found)[1] if found else None


def to_item(row: dict) -> Item:
    options = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(row["options"]))
    prompt = (f"The following is a multiple choice question about {row['category']}. Think step by step, "
              f"then finish with \"the answer is (X)\" where X is the letter of the correct option.\n\n"
              f"Question: {row['question']}\n\nOptions:\n{options}")
    return Item(suite="mmlu_pro", native_id=str(row["question_id"]), messages=[{"role": "user", "content": prompt}],
                gold=row["answer"], meta={"category": row["category"]})


class MMLUPro(Suite):
    name, domain, licence = "mmlu_pro", "knowledge", "MIT"
    source = "huggingface.co/datasets/TIGER-Lab/MMLU-Pro"
    revision = "b189ec765aa7ed75c8acfea42df31fdae71f97be"

    def load(self) -> list[Item]:
        import pyarrow.parquet as pq

        path = self.cache_dir() / "test.parquet"
        if not path.exists():
            url = f"https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/{self.revision}/data/test-00000-of-00001.parquet"
            urllib.request.urlretrieve(url, path)
        return [to_item(r) for r in pq.read_table(path).to_pylist()]

    def check(self, item: Item, answer: str) -> float:
        return 1.0 if extract_letter(answer) == item.gold else 0.0

    def stratum(self, item: Item) -> str:
        return item.meta["category"]
