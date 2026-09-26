"""Leakage: public label items that are also HELD items of this repository's suites never train Laya.

Our suites and the public label sources overlap (Arena-Hard, MMLU-Pro, ...). An item held out to
test g1 must not have been seen in training, so a label row whose prompt hash equals the hash of a
held suite item's raw source text (`meta["source_text"]`, the question before our prompt template)
is excluded from tune. Whitespace and case do not hide a match (labels.llmrouterbench.prompt_hash).
"""
from __future__ import annotations

from labels.llmrouterbench import prompt_hash
from suites.base import Item, split_of


def held_hashes(items: list[Item]) -> set[str]:
    return {prompt_hash(i.meta["source_text"]) for i in items if i.meta.get("source_text") and split_of(i.item_id) == "held"}


def is_trainable(label_prompt_hash: str, held: set[str]) -> bool:
    return label_prompt_hash not in held
