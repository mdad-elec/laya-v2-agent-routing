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

    def test_the_benchmark_s_randomised_item_scores_the_same_whatever_the_random_state(self):
        # IFEval item 1122 asks for the letter '#'; the upstream checker swaps a non-letter for a
        # random letter, which made this item a coin flip before the per-item seed.
        import random

        row = {"key": 1122, "prompt": "Write a riddle. Use lowercase only, and use the letter '#' at least 4 times.",
               "instruction_id_list": ["change_case:english_lowercase", "keywords:letter_frequency"],
               "kwargs": [{}, {"let_relation": "at least", "letter": "#", "let_frequency": 4}]}
        s, item = IFEval(), to_item(row)
        answer = "what has keys but opens no locks, a space but no room? a keyboard, of course."
        results = set()
        for seed in range(12):
            random.seed(seed)
            results.add(s.check(item, answer))
        self.assertEqual(len(results), 1, results)

    def test_the_item_carries_the_prompt_verbatim(self):
        item = to_item(ROW)
        self.assertEqual(item.messages, [{"role": "user", "content": ROW["prompt"]}])
        self.assertEqual(item.native_id, "1000")



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
        runs = [sum(s.check(items[r["prompt"]], r["response"]) for r in responses if r["prompt"] in items) for _ in range(2)]
        self.assertEqual(len(responses), 541)
        self.assertEqual(runs[0], runs[1], "the checker gives one answer per response")
        # The paper's 76.89% = 416/541 was one draw of the benchmark's single randomised item
        # (1122, letter '#'); the seeded checker lands on one side of it, never further.
        self.assertIn(runs[0], (416, 417), f"{runs[0]}/541 = {100 * runs[0] / 541:.2f}%")


if __name__ == "__main__":
    unittest.main()
