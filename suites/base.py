"""The contract every public benchmark suite implements.

A suite fetches its items at a PINNED revision into a cache outside the repo (item text is the
benchmark authors', not ours to redistribute), and scores an answer in [0, 1] with the suite's own
checker. The repo commits only item ids, which are hashes of (suite, native id), never of the text.

Splits are a pure function of the item id, so they are frozen before anything is measured and a
suite that grows never moves an item from tune to held (the defect `auto_run.split` had in v2).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

DOMAINS = ("code", "sql", "fin_table", "instruct", "knowledge", "long_ctx", "chat", "tools_multiturn")
SPLIT_SALT = "laya-g1/2026-09-26"  # changing this re-draws every split: never, after S2 starts
CACHE = Path(os.environ.get("LAYA_SUITE_CACHE", Path.home() / ".cache" / "laya-g1" / "suites"))


@dataclass
class Item:
    suite: str
    native_id: str
    messages: list  # the OpenAI-shape conversation the model is sent
    gold: object  # whatever the checker needs; never committed
    meta: dict = field(default_factory=dict)

    @property
    def item_id(self) -> str:
        return hashlib.sha256(f"{self.suite}\x00{self.native_id}".encode()).hexdigest()[:16]


def split_of(item_id: str) -> str:
    """'tune' or 'held', about half each, fixed forever by the id and the salt."""
    h = int(hashlib.sha256(f"{SPLIT_SALT}\x00{item_id}".encode()).hexdigest()[:8], 16)
    return "held" if h % 2 else "tune"


class Suite:
    name: str
    domain: str
    licence: str
    source: str  # where it is fetched from
    revision: str  # the pinned commit / dataset sha
    notes: str = ""

    def load(self) -> list[Item]:
        raise NotImplementedError

    def check(self, item: Item, answer: str) -> float:
        raise NotImplementedError

    def run(self, item: Item, ask) -> tuple[float, list]:
        """Score one item with `ask(messages) -> text`, the model under test. Most suites are one
        ask and one check; an agentic suite (BFCL) overrides this with its own loop. Returns the
        score and the transcript of what was asked and answered."""
        answer = ask(item.messages)
        return self.check(item, answer), [*item.messages, {"role": "assistant", "content": answer}]

    def card(self) -> dict:
        assert self.domain in DOMAINS, self.domain
        return {"suite": self.name, "domain": self.domain, "licence": self.licence, "source": self.source,
                "revision": self.revision, "notes": self.notes}

    def cache_dir(self) -> Path:
        d = CACHE / self.name / self.revision
        d.mkdir(parents=True, exist_ok=True)
        return d


def select(items: list[Item], n: int, stratum=lambda item: "") -> list[Item]:
    """n items, spread as evenly as possible over strata, chosen by id hash -- the same n on every
    machine, independent of file order, and adding items to a suite never reshuffles the chosen set
    more than the new items themselves."""
    groups: dict[str, list[Item]] = {}
    for item in sorted(items, key=lambda i: hashlib.sha256(f"select\x00{i.item_id}".encode()).hexdigest()):
        groups.setdefault(stratum(item), []).append(item)
    chosen: list[Item] = []
    keys = sorted(groups)
    while len(chosen) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(chosen) < n:
                chosen.append(groups[k].pop(0))
    return chosen
