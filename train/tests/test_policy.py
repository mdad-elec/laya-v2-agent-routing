"""The pre-registered policy (Amendment 5): the cheapest candidate with P >= tau, else the argmax."""
import unittest

from train.policy import GRID, cheapest_above, choose_tau


class Policy(unittest.TestCase):
    def test_cheapest_above_tau_else_argmax(self):
        p = {"low": 0.2, "mid": 0.7, "high": 0.9}
        cost = {"low": 1.0, "mid": 2.0, "high": 3.0}
        self.assertEqual(cheapest_above(p, cost, 0.5), "mid")
        self.assertEqual(cheapest_above(p, cost, 0.1), "low")
        self.assertEqual(cheapest_above(p, cost, 0.95), "high", "none clears tau: the argmax")

    def test_tau_is_chosen_by_the_given_objective_on_the_given_rows_only(self):
        calls = []

        def objective(tau):
            calls.append(tau)
            return -abs(tau - 0.35)

        self.assertEqual(choose_tau(objective), 0.35)
        self.assertEqual(calls, GRID)
        self.assertEqual(len(GRID), 19)


if __name__ == "__main__":
    unittest.main()
