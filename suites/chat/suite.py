"""Arena-Hard-Auto v2.0 (LMArena, Apache-2.0): open-ended prompts, judged pairwise against a baseline -- `chat`.

750 prompts (500 hard: coding and math; 250 creative writing) and the baseline answers
(o3-mini-2025-01-31 for hard prompts, gemini-2.0-flash-001 for creative writing) come from
github.com/lmarena/arena-hard-auto at a pinned commit, with Arena-Hard's own judge prompts, A/B
template and verdict labels.

Each judgment is two games with the positions swapped, as Arena-Hard runs them, so a judge that
prefers whichever answer comes first cancels out. Per game the candidate scores 1 (better), 0.5
(tie) or 0 (worse). The item score is the mean over games and over the judges in the panel. The
harness supplies judges from another model family than the candidate: no model grades its own
family. An unreadable verdict makes the item UNSCORED, never 0. "Solved" in the Atlas means
score >= 0.5: at least as good as the benchmark's baseline.
"""
from __future__ import annotations

import ast
import json
import re
import urllib.request

from suites.base import Item, Suite, read_jsonl

REVISION = "196f6b826783b3da7310e361a805fa36f0be83f3"
BASE = f"https://raw.githubusercontent.com/lmarena/arena-hard-auto/{REVISION}/data/arena-hard-v2.0"
BASELINE = {"hard_prompt": "o3-mini-2025-01-31", "creative_writing": "gemini-2.0-flash-001"}
TEMPLATE = ("<|User Prompt|>\n{QUESTION}\n\n<|The Start of Assistant A's Answer|>\n{ANSWER_A}\n<|The End of Assistant A's Answer|>"
            "\n\n<|The Start of Assistant B's Answer|>\n{ANSWER_B}\n<|The End of Assistant B's Answer|>")
_VERDICT = [re.compile(r"\[\[([AB<>=]+)\]\]"), re.compile(r"\[([AB<>=]+)\]")]
# Arena-Hard's utils/judge_utils.py at the pinned commit, verbatim.
_TAIL = ("After providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n"
         "1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n"
         "4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\".")
_HEAD = ("Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. "
         "You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\n")
_MIDDLE = ("Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or "
           "follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to "
           "ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the "
           "response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\n"
           "Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in "
           "the assistants' answers that would be beneficial to include when responding to the user prompt.\n\n")
SYSTEM = {
    "hard_prompt": _HEAD + "Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\n"
    "When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\n"
    + _MIDDLE + _TAIL,
    "creative_writing": _HEAD + "When evaluating the assistants' answers, compare both assistants' answers. You must identify and correct any mistakes or inaccurate information.\n\n"
    + _MIDDLE + _TAIL,
}
# The candidate's score per game, keyed by the verdict and by which side the candidate sat on.
_POINTS = {"A>>B": (1.0, 0.0), "A>B": (1.0, 0.0), "A=B": (0.5, 0.5), "B>A": (0.0, 1.0), "B>>A": (0.0, 1.0)}


class Unscored(Exception):
    """No judge gave a readable verdict: the item has no score, which is not the same as 0."""


def parse_verdict(text: str) -> str | None:
    for pattern in _VERDICT:
        found = pattern.findall(text)
        if found:
            label = found[-1].strip()
            return label if label in _POINTS else None
    return None


def _answer_text(record: dict) -> str:
    messages = record["messages"]
    if isinstance(messages, str):
        messages = ast.literal_eval(messages)
    content = messages[-1]["content"]
    if isinstance(content, dict):
        content = content.get("answer", "")
    return str(content)


class ArenaHard(Suite):
    name, domain, licence = "arena_hard_v2", "chat", "Apache-2.0"
    source = "github.com/lmarena/arena-hard-auto (data/arena-hard-v2.0)"
    revision = REVISION
    notes = "pairwise vs Arena-Hard's baselines; two games per judge, positions swapped; cross-family panel"

    def __init__(self, judges: list | None = None):
        self.judges = judges or []
        self.excluded: dict[str, str] = {}

    def _jsonl(self, name: str) -> list[dict]:
        path = self.cache_dir() / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            urllib.request.urlretrieve(f"{BASE}/{name}", path)
        return read_jsonl(path)

    def load(self) -> list[Item]:
        questions = self._jsonl("question.jsonl")
        baselines = {kind: {r["uid"]: _answer_text(r) for r in self._jsonl(f"model_answer/{model}.jsonl")} for kind, model in BASELINE.items()}
        items, self.excluded = [], {}
        for q in questions:
            kind = "creative_writing" if q["category"] == "creative_writing" else "hard_prompt"
            baseline = baselines[kind].get(q["uid"])
            if not baseline:
                self.excluded[q["uid"]] = f"no {BASELINE[kind]} baseline answer at the pinned commit"
                continue
            items.append(Item(suite=self.name, native_id=q["uid"], messages=[{"role": "user", "content": q["prompt"]}],
                              gold={"baseline": baseline, "category": kind, "prompt": q["prompt"]},
                              meta={"category": q.get("subcategory") or q["category"]}))
        return items

    def check(self, item: Item, answer: str) -> float:
        if not self.judges:
            raise Unscored("no judge panel was supplied")
        system = SYSTEM[item.gold["category"]]
        scores = []
        for judge in self.judges:
            for candidate_is_a in (False, True):
                a, b = (answer, item.gold["baseline"]) if candidate_is_a else (item.gold["baseline"], answer)
                user = TEMPLATE.format(QUESTION=item.gold["prompt"], ANSWER_A=a, ANSWER_B=b)
                verdict = parse_verdict(judge([{"role": "system", "content": system}, {"role": "user", "content": user}]))
                if verdict is not None:
                    scores.append(_POINTS[verdict][0 if candidate_is_a else 1])
        if not scores:
            raise Unscored("no judge returned a readable verdict")
        return sum(scores) / len(scores)

    def stratum(self, item: Item) -> str:
        return item.meta["category"]
