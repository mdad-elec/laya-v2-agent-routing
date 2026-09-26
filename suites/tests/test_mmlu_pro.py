import unittest

from suites.base import Item, select
from suites.mmlu_pro.suite import MMLUPro, extract_letter, to_item


class MMLUProSuite(unittest.TestCase):
    row = {"question_id": 70, "question": "Typical advertising regulatory bodies suggest?", "options": ["a", "b", "c", "d"],
           "answer": "C", "answer_index": 2, "category": "business", "src": "ori_mmlu-business"}

    def test_a_row_becomes_a_lettered_multiple_choice_prompt(self):
        item = to_item(self.row)
        text = item.messages[0]["content"]
        self.assertIn("A. a", text); self.assertIn("D. d", text)
        self.assertIn("the answer is (X)", text)
        self.assertEqual((item.gold, item.meta["category"], item.native_id), ("C", "business", "70"))

    def test_the_checker_reads_the_standard_forms_and_only_the_last_answer(self):
        item = to_item(self.row)
        s = MMLUPro()
        self.assertEqual(s.check(item, "Reasoning... the answer is (C)"), 1.0)
        self.assertEqual(s.check(item, "Answer: C"), 1.0)
        self.assertEqual(s.check(item, "first I thought the answer is (A) but the answer is (C)"), 1.0)
        self.assertEqual(s.check(item, "the answer is (B)"), 0.0)
        self.assertEqual(s.check(item, "I cannot tell"), 0.0)
        self.assertEqual(extract_letter("The answer is C."), "C")

    def test_selection_is_stratified_by_category(self):
        rows = [dict(self.row, question_id=i, category=["math", "law", "business"][i % 3]) for i in range(60)]
        chosen = select([to_item(r) for r in rows], 9, lambda it: it.meta["category"])
        self.assertEqual(sorted(it.meta["category"] for it in chosen).count("law"), 3)


if __name__ == "__main__":
    unittest.main()
