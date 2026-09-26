import unittest

from suites.ruler.suite import RULER, generate, string_match_all


class RULERSuite(unittest.TestCase):
    def test_generation_is_deterministic_by_seed_and_hits_the_length(self):
        a, b = generate("niah_multikey", seed=3, target_tokens=4000), generate("niah_multikey", seed=3, target_tokens=4000)
        self.assertEqual(a.messages, b.messages)
        self.assertNotEqual(a.messages, generate("niah_multikey", seed=4, target_tokens=4000).messages)
        for task in ("niah_multikey", "variable_tracking", "common_words"):
            chars = len(generate(task, seed=3, target_tokens=32_000).messages[0]["content"])
            self.assertTrue(30_000 * 4 <= chars <= 33_000 * 4, (task, chars))

    def test_the_answers_are_really_in_the_context_and_decoys_are_not_the_answer(self):
        for task in ("niah_multikey", "variable_tracking", "common_words"):
            item = generate(task, seed=11, target_tokens=6000)
            text = item.messages[0]["content"]
            self.assertTrue(item.gold, task)
            if task != "common_words":
                self.assertTrue(all(g in text for g in item.gold), task)

    def test_string_match_all_is_the_fraction_of_references_found(self):
        self.assertEqual(string_match_all("the values are 123 and 456", ["123", "456"]), 1.0)
        self.assertEqual(string_match_all("only 123", ["123", "456"]), 0.5)
        self.assertEqual(string_match_all("", ["123"]), 0.0)
        self.assertEqual(string_match_all("VAR ABC and var xyz", ["abc", "XYZ"]), 1.0, "case-insensitive, as RULER")

    def test_the_suite_scores_through_the_same_metric(self):
        item = generate("niah_multikey", seed=5, target_tokens=3000)
        self.assertEqual(RULER().check(item, " ".join(item.gold)), 1.0)
        self.assertEqual(RULER().check(item, "I could not find it"), 0.0)


if __name__ == "__main__":
    unittest.main()
