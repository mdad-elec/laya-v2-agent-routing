"""Contamination: does a training prompt contain an evaluation question, whatever template wraps it?

An exact prompt hash (labels.leakage) only catches a byte-identical prompt after whitespace and
case. The eval sources wrap the SAME questions in their own templates: RouterArena's "Please solve
the following mathematical problem step by step... Question: ...", LLMRouterBench's own prompt,
our suites' prompts. So a question is matched on its words:
- a question of >= N words is in a prompt when at least half of its word N-grams occur there;
- a shorter question is in a prompt when its word sequence occurs there contiguously;
- a question of fewer than MIN_WORDS words cannot be identified at all: it is listed in `skipped`.
Only the eval QUESTION is indexed (never its template), so sharing template boilerplate is not a match.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

N = 8
MIN_SHARE = 0.5
MIN_WORDS = 4
_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return _WORD.findall(str(text).lower())


def shingles(ws: list[str]) -> set[str]:
    return {" ".join(ws[i:i + N]) for i in range(len(ws) - N + 1)}


class EvalIndex:
    """Eval questions by id; `hits(prompt)` names every eval question the prompt contains."""

    def __init__(self, questions: dict[str, str]):
        self._by_shingle: dict[str, list[str]] = defaultdict(list)
        self._need: dict[str, int] = {}
        self._short: dict[str, str] = {}
        self.skipped: list[str] = []
        for qid, text in questions.items():
            ws = words(text)
            if len(ws) < MIN_WORDS:
                self.skipped.append(qid)
                continue
            if len(ws) < N:
                self._short[qid] = " " + " ".join(ws) + " "
                continue
            sh = shingles(ws)
            self._need[qid] = math.ceil(MIN_SHARE * len(sh))
            for s in sh:
                self._by_shingle[s].append(qid)

    def __len__(self) -> int:
        return len(self._need) + len(self._short)

    def hits(self, prompt: str) -> set[str]:
        ws = words(prompt)
        counts: Counter[str] = Counter()
        for s in shingles(ws):
            for qid in self._by_shingle.get(s, ()):
                counts[qid] += 1
        found = {qid for qid, c in counts.items() if c >= self._need[qid]}
        if self._short:
            joined = " " + " ".join(ws) + " "
            found |= {qid for qid, seq in self._short.items() if seq in joined}
        return found
