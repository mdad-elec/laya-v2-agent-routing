"""Leakage: public label items that are also HELD items of this repository's suites never train Laya.

Our suites and the public label sources overlap (Arena-Hard, MMLU-Pro, ...). An item held out to
test g1 must not have been seen in training, so a label row whose prompt hash equals the hash of a
held suite item's raw source text (`meta["source_text"]`, the question before our prompt template)
is excluded from tune. Whitespace and case do not hide a match (labels.llmrouterbench.prompt_hash).

The exact hash misses the same question under another template, so every training prompt is also
checked against one `EvalIndex` (labels.contamination) of everything any g1 bar or held read
scores (PREREG-g1 Amendment 1): all of RouterArena, TwinRouterBench's HELD trajectories (their
task, i.e. the first user message), and our suites' held items.
"""
from __future__ import annotations

from labels.contamination import EvalIndex
from labels.llmrouterbench import prompt_hash
from suites.base import Item, split_of


def held_hashes(items: list[Item]) -> set[str]:
    return {prompt_hash(i.meta["source_text"]) for i in items if i.meta.get("source_text") and split_of(i.item_id) == "held"}


def is_trainable(label_prompt_hash: str, held: set[str]) -> bool:
    return label_prompt_hash not in held


def _task(row: dict) -> str:
    return next((m.get("content") or "" for m in row["messages"] if m.get("role") == "user"), "")


def build_eval_index(routerarena: dict[str, str], twinrouterbench_rows: list[dict], suite_items: list[Item]) -> EvalIndex:
    from labels import twinrouterbench as trb

    questions = dict(routerarena)
    for row in twinrouterbench_rows:
        if trb.split(row) == "held":
            questions.setdefault(trb.trajectory_key(row), str(_task(row)))
    for item in suite_items:
        if item.meta.get("source_text") and split_of(item.item_id) == "held":
            questions[f"suite/{item.suite}/{item.native_id}"] = item.meta["source_text"]
    return EvalIndex(questions)
