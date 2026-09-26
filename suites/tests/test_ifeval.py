import unittest

from suites.ifeval.suite import IFEval, to_item

ROW = {"key": 1000, "prompt": "Write a short note about rivers. Your entire response should be in English, and in all lowercase letters. No capital letters are allowed. Answer with at least 20 words.",
       "instruction_id_list": ["change_case:english_lowercase", "length_constraints:number_words"],
       "kwargs": [{}, {"relation": "at least", "num_words": 20}]}


class IFEvalSuite(unittest.TestCase):
    def test_prompt_level_strict_all_instructions_must_hold(self):
        s, item = IFEval(), to_item(ROW)
        ok = "rivers carve valleys over long ages and carry water, silt and life from the high mountains down to the wide sea where they finally rest."
        self.assertEqual(s.check(item, ok), 1.0)
        self.assertEqual(s.check(item, ok.capitalize()), 0.0, "one capital letter breaks the lowercase instruction")
        self.assertEqual(s.check(item, "rivers flow."), 0.0, "too short")

    def test_the_item_carries_the_prompt_verbatim(self):
        item = to_item(ROW)
        self.assertEqual(item.messages, [{"role": "user", "content": ROW["prompt"]}])
        self.assertEqual(item.native_id, "1000")


if __name__ == "__main__":
    unittest.main()


class Golden(unittest.TestCase):
    """The checker reproduces the published benchmark: GPT-4's own responses, shipped with IFEval at
    the pinned commit, score 76.89% prompt-level strict in Zhou et al. (2023) Table 2."""

    def test_gpt4_responses_reproduce_the_paper_number(self):
        import json
        import os
        import urllib.request

        if os.environ.get("LAYA_OFFLINE") == "1":
            self.skipTest("offline")
        s = IFEval()
        path = s.cache_dir() / "gpt4_responses.jsonl"
        if not path.exists():
            urllib.request.urlretrieve("https://raw.githubusercontent.com/google-research/google-research/"
                                       f"{s.revision}/instruction_following_eval/data/input_response_data_gpt4_20231107_145030.jsonl", path)
        items = {i.gold["prompt"]: i for i in s.load()}
        responses = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        # A response whose prompt is worded differently from the input file cannot be scored; the
        # paper's denominator is all 541 prompts, so it counts against the total.
        passed = sum(s.check(items[r["prompt"]], r["response"]) for r in responses if r["prompt"] in items)
        self.assertEqual(len(responses), 541)
        self.assertAlmostEqual(100 * passed / 541, 76.89, places=2)
