"""Contamination: an eval question wrapped in a different template still blocks the training prompt."""
import unittest

from labels.contamination import EvalIndex

Q = "From an unlimited supply of 1-cent coins, 10-cent coins, and 25-cent coins, Silas wants to find a collection of coins that has a total value of N cents"
SHORT = "What is the capital of Australia?"


class Contamination(unittest.TestCase):
    def setUp(self):
        self.index = EvalIndex({"routerarena/AIME_112": Q, "routerarena/GEO_1": SHORT})

    def test_the_same_question_under_another_template_is_contaminated(self):
        wrapped = f"Please solve the following mathematical problem step by step.\n\nQuestion: {Q.upper()}\n\nPut the final answer in \\boxed{{}}."
        self.assertEqual(self.index.hits(wrapped), {"routerarena/AIME_112"})

    def test_a_short_question_is_matched_by_containment(self):
        self.assertEqual(self.index.hits("Answer briefly.  what is the capital   of australia?  Options: A) Sydney"), {"routerarena/GEO_1"})

    def test_sharing_only_template_words_is_not_contamination(self):
        other = "Please solve the following mathematical problem step by step. Question: a train leaves the station at 3 pm travelling 60 miles per hour, when does it arrive"
        self.assertEqual(self.index.hits(other), set())

    def test_a_question_too_short_to_identify_matches_nothing(self):
        index = EvalIndex({"x/1": "Why not?"})
        self.assertEqual(index.hits("why not? because it rained"), set())
        self.assertEqual(index.skipped, ["x/1"])

    def test_half_the_question_is_enough(self):
        words = Q.split()
        self.assertEqual(self.index.hits(" ".join(words[: len(words) // 2 + 4])), {"routerarena/AIME_112"})
        self.assertEqual(self.index.hits(" ".join(words[:6])), set())


if __name__ == "__main__":
    unittest.main()
