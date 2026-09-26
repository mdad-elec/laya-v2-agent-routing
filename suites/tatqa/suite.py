"""TAT-QA (NExT++, CC BY 4.0): numeric questions over a financial report's table and text -- `fin_table`.

Fetched at a pinned commit of github.com/NExTplusplus/TAT-QA (dev split). Only `arithmetic`
questions become items: they have one numeric gold answer a checker can read without a judge.

Pre-registered tolerance: a prediction counts when it equals the gold within 0.5% of the gold
(or exactly at 2 decimals), in the table's unit OR converted out of it (the gold 200 with scale
"million" also accepts 200,000,000), and a percentage also accepts its fraction (20 -> 0.2). The
model is asked for a bare number on a final "Answer:" line; the LAST such line counts.
"""
from __future__ import annotations

import json
import re
import urllib.request

from suites.base import Item, Suite

SCALE = {"": 1.0, "thousand": 1e3, "million": 1e6, "billion": 1e9, "percent": 1.0}
_ANSWER = re.compile(r"\banswer\s*[:：]\s*([^\n]+)", re.I)
_NUMBER = re.compile(r"\(?-?\$?\s*\d[\d,]*(?:\.\d+)?\)?")


def extract_number(answer: str) -> float | None:
    lines = _ANSWER.findall(answer)
    if not lines:
        return None
    m = _NUMBER.search(lines[-1])
    if not m:
        return None
    token = m.group(0)
    negative = token.startswith("(") and token.endswith(")") or "-" in token
    digits = re.sub(r"[^\d.]", "", token)
    try:
        value = float(digits)
    except ValueError:
        return None
    return -value if negative else value


def _table(rows: list[list[str]]) -> str:
    return "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)


def items_from(contexts: list[dict]) -> list[Item]:
    items = []
    for ctx in contexts:
        paragraphs = "\n".join(p["text"] for p in sorted(ctx["paragraphs"], key=lambda p: p["order"]))
        for q in ctx["questions"]:
            if q["answer_type"] != "arithmetic":
                continue
            prompt = (f"Answer the question from this financial report's table and text.\n\nTable:\n{_table(ctx['table']['table'])}\n\n"
                      f"Text:\n{paragraphs}\n\nQuestion: {q['question']}\n\n"
                      f"Work it out, then give the result as a bare number (no units; a percentage as e.g. 12.5) on a final line \"Answer: <number>\".")
            items.append(Item(suite="tatqa", native_id=q["uid"], messages=[{"role": "user", "content": prompt}],
                              gold={"value": float(q["answer"]), "scale": q.get("scale") or ""}, meta={"scale": q.get("scale") or ""}))
    return items


def _close(pred: float, gold: float) -> bool:
    return round(pred, 2) == round(gold, 2) or abs(pred - gold) <= 0.005 * max(1.0, abs(gold))


class TATQA(Suite):
    name, domain, licence = "tatqa", "fin_table", "CC-BY-4.0"
    source = "github.com/NExTplusplus/TAT-QA (dataset_raw/tatqa_dataset_dev.json)"
    revision = "870accc41953dcde885aabeb963d94aabdc0fbc3"

    def load(self) -> list[Item]:
        path = self.cache_dir() / "dev.json"
        if not path.exists():
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/NExTplusplus/TAT-QA/{self.revision}/dataset_raw/tatqa_dataset_dev.json", path)
        return items_from(json.loads(path.read_text()))

    def check(self, item: Item, answer: str) -> float:
        pred = extract_number(answer)
        if pred is None:
            return 0.0
        gold, scale = item.gold["value"], item.gold["scale"]
        candidates = [gold, gold * SCALE.get(scale, 1.0)]
        if scale == "percent":
            candidates.append(gold / 100.0)
        return 1.0 if any(_close(pred, c) for c in candidates) else 0.0

    def stratum(self, item: Item) -> str:
        return item.meta["scale"]
