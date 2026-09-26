import unittest

from suites.base import Item
from suites.chat.suite import ArenaHard, Unscored, parse_verdict


def item(category="hard_prompt"):
    return Item(suite="arena_hard_v2", native_id="u1", messages=[{"role": "user", "content": "Explain X."}],
                gold={"baseline": "baseline answer", "category": category, "prompt": "Explain X."}, meta={"category": category})


class Verdicts(unittest.TestCase):
    def test_the_last_bracketed_label_is_the_verdict(self):
        self.assertEqual(parse_verdict("thinking [[A>B]] ... My final verdict is: [[B>>A]]"), "B>>A")
        self.assertEqual(parse_verdict("verdict [A=B]"), "A=B")
        self.assertIsNone(parse_verdict("no verdict here"))


class Pairwise(unittest.TestCase):
    def test_two_games_with_positions_swapped_and_the_candidate_scored_per_game(self):
        seen = []

        def judge_prefers_the_candidate(messages):
            seen.append(messages)
            text = messages[-1]["content"]
            a = text.split("<|The Start of Assistant A's Answer|>\n")[1].split("\n<|The End")[0]
            return "[[A>B]]" if a == "CANDIDATE" else "[[B>A]]"

        s = ArenaHard(judges=[judge_prefers_the_candidate])
        self.assertEqual(s.check(item(), "CANDIDATE"), 1.0)
        self.assertEqual(len(seen), 2, "one game each way")
        first, second = (m[-1]["content"] for m in seen)
        self.assertIn("Assistant A's Answer|>\nbaseline answer", first)
        self.assertIn("Assistant A's Answer|>\nCANDIDATE", second)
        self.assertTrue(seen[0][0]["content"].startswith("Please act as an impartial judge"))

    def test_ties_average_and_a_panel_is_averaged(self):
        tie = lambda m: "[[A=B]]"
        always_a = lambda m: "[[A>>B]]"   # position bias: prefers whoever is A
        self.assertEqual(ArenaHard(judges=[tie]).check(item(), "x"), 0.5)
        self.assertEqual(ArenaHard(judges=[always_a]).check(item(), "x"), 0.5, "a position-biased judge cancels out")

    def test_an_unreadable_verdict_leaves_the_item_unscored_never_zero(self):
        with self.assertRaises(Unscored):
            ArenaHard(judges=[lambda m: "I refuse"]).check(item(), "x")

    def test_creative_prompts_use_their_own_judge_prompt(self):
        prompts = []
        ArenaHard(judges=[lambda m: prompts.append(m[0]["content"]) or "[[A=B]]"]).check(item("creative_writing"), "x")
        self.assertNotIn("generating your own answer", prompts[0])


if __name__ == "__main__":
    unittest.main()
