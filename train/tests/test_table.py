import unittest

from train.table import TierTable, cheapest_passing


class TierTableTest(unittest.TestCase):
    def test_a_group_is_its_cumulative_label_distribution_when_unshrunk(self):
        t = TierTable(m=0).fit(["a", "a", "a", "a", "b"], [0, 0, 1, 3, 3])
        self.assertEqual(t.p_solve("a"), [0.5, 0.75, 0.75, 1.0])
        self.assertEqual(t.p_solve("b"), [0.0, 0.0, 0.0, 1.0])

    def test_an_unseen_group_routes_like_the_whole_fit_set(self):
        t = TierTable(m=4).fit(["a", "a", "b", "b"], [0, 0, 3, 3])
        self.assertEqual(t.p_solve("never seen"), [0.5, 0.5, 0.5, 1.0])

    def test_shrinkage_pulls_a_small_group_toward_the_prior(self):
        t = TierTable(m=2).fit(["a", "b", "b", "b"], [0, 3, 3, 3])
        # prior P(tier 0 passes) = 1/4; group a has 1 of 1 → (1 + 2·0.25) / (1 + 2)
        self.assertAlmostEqual(t.p_solve("a")[0], 0.5)
        self.assertEqual(t.p_solve("a")[3], 1.0)

    def test_bad_input_is_refused_not_absorbed(self):
        with self.assertRaises(ValueError):
            TierTable(m=0).fit(["a"], [4])
        with self.assertRaises(ValueError):
            TierTable(m=0).fit([], [])
        with self.assertRaises(ValueError):
            TierTable(m=-1)

    def test_the_policy_takes_the_cheapest_tier_that_clears_tau_else_the_top(self):
        self.assertEqual(cheapest_passing([0.5, 0.8, 0.9, 1.0], 0.75), 1)
        self.assertEqual(cheapest_passing([0.9, 0.9, 0.9, 1.0], 0.5), 0)
        self.assertEqual(cheapest_passing([0.1, 0.2, 0.3, 0.4], 0.99), 3)


if __name__ == "__main__":
    unittest.main()
