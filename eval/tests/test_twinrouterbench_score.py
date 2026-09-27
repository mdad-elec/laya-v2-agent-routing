"""F1's scorer is TwinRouterBench's own section11 at the pinned commit; it reproduces the paper's always-high row."""
import unittest

from eval.twinrouterbench_score import score
from labels import twinrouterbench as trb


class Scorer(unittest.TestCase):
    def test_always_high_reproduces_table_2(self):
        rows = trb.load()
        s = score(rows, {r["id"]: 3 for r in rows})
        self.assertEqual(round(s["case_pass_rate_percent"], 2), 100.0)
        self.assertEqual(round(s["case_exact_match_percent"], 2), 17.53)
        self.assertEqual(round(s["trajectory_pass_rate_percent"], 2), 100.0)
        self.assertEqual(round(s["cost_savings_score_percent"], 2), 0.0)
        self.assertEqual(round(s["combined_score_percent"], 2), 54.38)

    def test_memoised_token_counts_change_no_number(self):
        import random

        from eval import twinrouterbench_score as ts

        rows = trb.load()[:80]
        rng = random.Random(3)
        preds = {r["id"]: rng.randint(0, 3) for r in rows}
        ts.MEMO["on"] = False
        try:
            plain = score(rows, preds)
        finally:
            ts.MEMO["on"] = True
        self.assertEqual(score(rows, preds), plain)
        self.assertEqual(score(rows, preds), plain, "a second, fully cached pass too")

    def test_a_missing_prediction_is_refused_not_scored(self):
        rows = trb.load()[:3]
        with self.assertRaises(KeyError):
            score(rows, {rows[0]["id"]: 0})

    def test_the_checkout_is_the_pinned_commit(self):
        from eval.twinrouterbench_score import checkout

        self.assertTrue((checkout() / "main" / "eval" / "section11.py").exists())


if __name__ == "__main__":
    unittest.main()
