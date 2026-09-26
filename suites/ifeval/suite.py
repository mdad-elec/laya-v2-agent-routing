"""IFEval (Google Research, Apache-2.0): verifiable instructions -- the `instruct` domain.

Items and checker come from google-research/instruction_following_eval at a pinned commit; the
checker is the benchmark's own code, vendored unmodified (see vendor/NOTICE), so scores mean what
the published IFEval numbers mean. Score = prompt-level STRICT accuracy: 1 only when every
instruction in the prompt is followed by the response as written.
"""
from __future__ import annotations

import json
import urllib.request

from suites.base import Item, Suite

REVISION = "e6890f85757dd84e27ca6df2dd30651dafad28e0"


def to_item(row: dict) -> Item:
    return Item(suite="ifeval", native_id=str(row["key"]), messages=[{"role": "user", "content": row["prompt"]}],
                gold={"ids": row["instruction_id_list"], "kwargs": row["kwargs"], "prompt": row["prompt"]},
                meta={"family": row["instruction_id_list"][0].split(":")[0]})


class IFEval(Suite):
    name, domain, licence = "ifeval", "instruct", "Apache-2.0"
    source = "github.com/google-research/google-research/instruction_following_eval (data/input_data.jsonl)"
    revision = REVISION

    def _nltk_ready(self) -> None:
        """The checker's sentence counts need NLTK's `punkt_tab`; provisioned into the suite's own
        cache, never assumed to be on the machine (measured: without it 60 of 480 GPT-4 responses
        raised, and the score read 79.38% against the paper's 76.89%)."""
        import langdetect
        import nltk

        # langdetect is randomly seeded by default, so the language instructions scored the same
        # response differently run to run. Pinned: a checker gives one answer per response.
        langdetect.DetectorFactory.seed = 0
        target = str(self.cache_dir() / "nltk_data")
        if target not in nltk.data.path:
            nltk.data.path.insert(0, target)
        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            if not nltk.download("punkt_tab", download_dir=target, quiet=True):
                raise RuntimeError("could not provision NLTK punkt_tab for the IFEval checker")

    def load(self) -> list[Item]:
        self._nltk_ready()
        path = self.cache_dir() / "input_data.jsonl"
        if not path.exists():
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/google-research/google-research/{REVISION}/instruction_following_eval/data/input_data.jsonl", path)
        return [to_item(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]

    def check(self, item: Item, answer: str) -> float:
        from suites.ifeval.vendor import instructions_registry

        self._nltk_ready()

        if not answer.strip():
            return 0.0
        for index, instruction_id in enumerate(item.gold["ids"]):
            instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
            kwargs = {k: v for k, v in (item.gold["kwargs"][index] or {}).items() if v is not None}
            instruction.build_description(**kwargs)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=item.gold["prompt"])
            if not instruction.check_following(answer):
                return 0.0
        return 1.0

    def stratum(self, item: Item) -> str:
        return item.meta["family"]
