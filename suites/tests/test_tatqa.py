import unittest

from suites.tatqa.suite import TATQA, extract_number, items_from

CONTEXT = [{
    "table": {"table": [["", "2019", "2018"], ["Revenue", "1,200", "1,000"], ["Cost", "(300)", "250"]]},
    "paragraphs": [{"order": 1, "text": "All amounts are in millions."}],
    "questions": [
        {"uid": "q-arith", "order": 1, "question": "What is the change in revenue?", "answer": 200, "answer_type": "arithmetic", "scale": "million", "derivation": "1,200-1,000"},
        {"uid": "q-pct", "order": 2, "question": "What is the percentage change?", "answer": 20, "answer_type": "arithmetic", "scale": "percent", "derivation": ""},
        {"uid": "q-span", "order": 3, "question": "What is the unit?", "answer": ["millions"], "answer_type": "span", "scale": "", "derivation": ""},
    ],
}]


class TATQASuite(unittest.TestCase):
    def test_only_arithmetic_questions_become_items_with_table_and_text_in_the_prompt(self):
        items = items_from(CONTEXT)
        self.assertEqual([i.native_id for i in items], ["q-arith", "q-pct"])
        text = items[0].messages[0]["content"]
        self.assertIn("| Revenue | 1,200 | 1,000 |", text)
        self.assertIn("All amounts are in millions.", text)
        self.assertIn("Answer:", text)

    def test_the_number_is_read_from_the_last_answer_line(self):
        self.assertEqual(extract_number("step 1 ... Answer: 200"), 200.0)
        self.assertEqual(extract_number("Answer: $1,234.50 million"), 1234.5)
        self.assertEqual(extract_number("Answer: (300)"), -300.0)
        self.assertEqual(extract_number("Answer: -12.5%"), -12.5)
        self.assertIsNone(extract_number("I think it grew"))

    def test_the_tolerance_is_the_preregistered_one_and_either_scale_counts(self):
        s, items = TATQA(), items_from(CONTEXT)
        arith, pct = items
        self.assertEqual(s.check(arith, "Answer: 200"), 1.0)
        self.assertEqual(s.check(arith, "Answer: 200000000"), 1.0, "the same value converted out of millions")
        self.assertEqual(s.check(arith, "Answer: 200.9"), 1.0, "within 0.5%")
        self.assertEqual(s.check(arith, "Answer: 202"), 0.0)
        self.assertEqual(s.check(pct, "Answer: 20.0%"), 1.0)
        self.assertEqual(s.check(pct, "Answer: 0.2"), 1.0, "a percentage given as a fraction")
        self.assertEqual(s.check(pct, "no number"), 0.0)


if __name__ == "__main__":
    unittest.main()
