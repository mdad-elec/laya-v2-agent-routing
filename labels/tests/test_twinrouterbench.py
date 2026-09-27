"""TwinRouterBench static bank @ the pinned commit: shape, split by trajectory, and a router view without labels."""
import unittest
from collections import Counter

from labels import twinrouterbench as trb


class TwinRouterBench(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = trb.load()

    def test_the_bank_matches_its_manifest(self):
        self.assertEqual(len(self.rows), 970)
        self.assertEqual(Counter(r["benchmark"] for r in self.rows), {"swebench": 336, "bfcl": 248, "mtrag": 193, "qmsum": 145, "pinchbench": 48})
        self.assertEqual(len({trb.trajectory_key(r) for r in self.rows}), 520)
        self.assertEqual(Counter(r["target_tier"] for r in self.rows), {"low": 689, "high": 170, "mid": 62, "mid_high": 49})

    def test_a_trajectory_never_straddles_the_split(self):
        by_traj: dict[str, set] = {}
        for r in self.rows:
            by_traj.setdefault(trb.trajectory_key(r), set()).add(trb.split(r))
        self.assertTrue(all(len(s) == 1 for s in by_traj.values()))
        held = sum(1 for r in self.rows if trb.split(r) == "held")
        self.assertTrue(350 < held < 620, held)

    def test_the_router_view_carries_no_label_or_source_identity(self):
        for r in self.rows:
            view = trb.router_view(r)
            self.assertLessEqual(set(view), {"messages", "functions"})
            self.assertEqual(view["messages"], r["messages"])

    def test_rows_keep_the_scorer_fields(self):
        r = self.rows[0]
        for k in ("id", "benchmark", "instance_id", "step_index", "total_steps", "messages", "target_tier_id"):
            self.assertIn(k, r)


if __name__ == "__main__":
    unittest.main()
