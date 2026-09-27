"""The digest fits Laya's 512-token window on REAL sessions, measured in Laya's own tokenizer."""
import unittest
from pathlib import Path

from router.digest import BUDGET_TOKENS, build, render

TOKENIZER = Path(__file__).resolve().parents[2] / "studies" / "v2" / "model" / "tokenizer" / "tokenizer.json"


def token_counts(texts):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER))
    return sorted(len(tok.encode(t).ids) for t in texts)


class Budget(unittest.TestCase):
    def check(self, counts):
        p99 = counts[min(len(counts) - 1, int(0.99 * len(counts)))]
        self.assertLessEqual(p99, BUDGET_TOKENS)
        self.assertLess(counts[-1], 512, "even the longest digest must fit the window")

    def test_twinrouterbench_agent_prefixes(self):
        from labels import twinrouterbench as trb

        self.check(token_counts(render(build(r["messages"])) for r in trb.load()))

    def test_public_single_turn_prompts(self):
        from labels.gate_s3 import load_rows
        from suites.base import CACHE

        if not (CACHE.parent / "public" / "llmrouterbench" / "prompts.json").exists():
            self.skipTest("LLMRouterBench prompt cache absent (1.28 GB archive; built by labels.llmrouterbench)")
        _, prompts = load_rows()
        self.check(token_counts(render(build([{"role": "user", "content": t}])) for t in prompts.values()))


if __name__ == "__main__":
    unittest.main()
