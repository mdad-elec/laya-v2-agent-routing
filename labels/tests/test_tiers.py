import unittest

from labels.tiers import TAU, ability_tiers, label_items


def rows(table):
    """table: {model: {item: score}} -> outcome rows of one dataset."""
    return [{"dataset": "d", "native_id": i, "model": m, "score": s, "prompt_hash": f"h{i}"} for m, items in table.items() for i, s in items.items()]


class Tiers(unittest.TestCase):
    table = {
        # weak models solve only the easy item
        "w1": {"easy": 1, "mid": 0, "hard": 0, "held": 1}, "w2": {"easy": 1, "mid": 0, "hard": 0, "held": 0},
        # middle models solve easy and mid
        "m1": {"easy": 1, "mid": 1, "hard": 0, "held": 0}, "m2": {"easy": 1, "mid": 1, "hard": 0, "held": 0},
        # strong models solve everything but the unsolvable
        "s1": {"easy": 1, "mid": 1, "hard": 1, "held": 0}, "s2": {"easy": 1, "mid": 1, "hard": 1, "held": 0},
    }

    def test_ability_is_measured_on_tune_items_only_and_splits_into_thirds(self):
        tiers = ability_tiers(rows(self.table), tune=lambda h: h != "hheld")
        self.assertEqual(tiers["d"], {"w1": "small", "w2": "small", "m1": "medium", "m2": "medium", "s1": "powerful", "s2": "powerful"})
        # a held item that only a weak model solved does not lift that model's ability
        swapped = dict(self.table, w1={"easy": 1, "mid": 0, "hard": 0, "held": 1})
        self.assertEqual(ability_tiers(rows(swapped), tune=lambda h: h != "hheld")["d"]["w1"], "small")

    def test_the_label_is_the_lowest_tier_that_reliably_solves_the_item(self):
        tiers = ability_tiers(rows(self.table), tune=lambda h: h != "hheld")
        labels = {x["native_id"]: x for x in label_items(rows(self.table), tiers)}
        self.assertEqual(labels["easy"]["band"], "small")
        self.assertEqual(labels["mid"]["band"], "medium")
        self.assertEqual(labels["hard"]["band"], "powerful")
        self.assertFalse(labels["hard"]["unsolved"])
        self.assertEqual(labels["easy"]["p"], {"small": 1.0, "medium": 1.0, "powerful": 1.0})
        self.assertEqual(TAU, 0.5)

    def test_an_item_no_tier_solves_is_powerful_and_flagged(self):
        table = {m: {"x": 0.0} for m in ("a", "b", "c", "d", "e", "f")}
        tiers = ability_tiers(rows(table), tune=lambda h: True)
        [label] = label_items(rows(table), tiers)
        self.assertEqual((label["band"], label["unsolved"]), ("powerful", True))

    def test_fractional_scores_are_averaged_against_tau(self):
        table = {"a": {"x": 0.6}, "b": {"x": 0.3}, "c": {"x": 0.5}, "d": {"x": 0.5}, "e": {"x": 1.0}, "f": {"x": 1.0}}
        tiers = ability_tiers(rows(table), tune=lambda h: True)
        [label] = label_items(rows(table), tiers)
        self.assertEqual(label["band"], "medium", "small averages 0.45 < 0.5; medium averages 0.55")


if __name__ == "__main__":
    unittest.main()
