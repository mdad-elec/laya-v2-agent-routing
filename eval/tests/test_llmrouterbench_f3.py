"""F3 (PREREG-g1 Amendment 3): the setting, and the paper's formulas on toy numbers."""
import unittest

from eval import llmrouterbench_f3 as f3


def item(ds, i, scores, costs=None, split="held"):
    return {"dataset": ds, "native_id": str(i), "split": split, "scores": scores, "costs": costs or {m: 1.0 for m in scores}}


class Formulas(unittest.TestCase):
    def test_avg_acc_is_macro_over_datasets(self):
        items = [item("a", 0, {"m": 1.0}), item("a", 1, {"m": 1.0}), item("a", 2, {"m": 1.0}), item("b", 0, {"m": 0.0})]
        self.assertEqual(f3.avg_acc(items, {("a", "0"): "m", ("a", "1"): "m", ("a", "2"): "m", ("b", "0"): "m"}), 0.5)

    def test_perf_gain_and_cost_save(self):
        configs = {"best": (0.60, 100.0), "r1": (0.66, 120.0), "r2": (0.61, 70.0), "r3": (0.50, 10.0)}
        self.assertAlmostEqual(f3.perf_gain(configs, ["r1", "r2", "r3"], best="best"), 0.10)
        self.assertAlmostEqual(f3.cost_save(configs, ["r1", "r2", "r3"], best="best"), 0.30)
        self.assertIsNone(f3.cost_save(configs, ["r3"], best="best"), "no configuration reaches Best Single")

    def test_pareto_dist_is_zero_on_the_frontier_and_positive_off_it(self):
        configs = {"a": (0.9, 100.0), "b": (0.5, 1.0), "c": (0.5, 100.0)}
        self.assertEqual(f3.pareto_dist(configs, ["a", "b"]), 0.0)
        self.assertGreater(f3.pareto_dist(configs, ["c"]), 0.0)

    def test_the_recall_slice(self):
        few = item("a", 0, {f"m{i}": (1.0 if i < 3 else 0.0) for i in range(13)})
        many = item("a", 1, {f"m{i}": (1.0 if i < 4 else 0.0) for i in range(13)})
        self.assertEqual([x["native_id"] for x in f3.recall_slice([few, many])], ["0"])


class Setting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from suites.base import CACHE

        if not (CACHE.parent / "public" / "llmrouterbench" / "outcomes.jsonl").exists():
            raise unittest.SkipTest("LLMRouterBench outcome cache absent (1.28 GB archive)")
        cls.items = f3.items()

    def test_every_item_has_all_13_models_and_tau2_keeps_its_278_tasks(self):
        self.assertTrue(all(set(i["scores"]) == set(f3.MODELS) for i in self.items))
        self.assertEqual(sum(1 for i in self.items if i["dataset"] == "tau2"), 278)
        self.assertEqual(sorted({i["dataset"] for i in self.items}), sorted(f3.DATASETS))

    def test_a_missing_model_scores_zero_as_the_benchmark_fills_it(self):
        tau = [i for i in self.items if i["dataset"] == "tau2"]
        self.assertTrue(all(i["scores"]["gpt-5-chat"] == 0.0 and i["filled"] == ["gpt-5-chat"] for i in tau))

    def test_identical_prompts_share_a_split(self):
        by_prompt = {}
        for i in self.items:
            by_prompt.setdefault((i["dataset"], i["prompt_hash"]), set()).add(i["split"])
        self.assertTrue(all(len(s) == 1 for s in by_prompt.values()))


if __name__ == "__main__":
    unittest.main()
