"""Cell descriptors: a model is DESCRIBED from public data (never an id), and every name joins or is reported."""
import unittest

from atlas import descriptors as d


class NameParsing(unittest.TestCase):
    def test_parameters_effort_and_native_tools_come_from_the_name(self):
        self.assertEqual(d.parse_name("Qwen_Qwen3-8B-FC"), {"params_b": 8.0, "active_b": None, "reasoning": None, "native_tools": True})
        self.assertEqual(d.parse_name("qwen3-235b-a22b-thinking-2507"), {"params_b": 235.0, "active_b": 22.0, "reasoning": True, "native_tools": False})
        self.assertEqual(d.parse_name("qwen3-235b-a22b-no-thinking")["reasoning"], False)
        self.assertEqual(d.parse_name("MadeAgents_Hammer2.1-0.5b")["params_b"], 0.5)
        self.assertEqual(d.parse_name("deepseek-r1-0528")["reasoning"], True)
        self.assertIsNone(d.parse_name("gpt-5")["params_b"])

    def test_keys_strip_orgs_tool_modes_and_dates(self):
        self.assertEqual(d.key("anthropic/claude-opus-4.6"), "claudeopus46")
        self.assertEqual(d.key("claude-sonnet-4-20250514"), "claudesonnet4")
        self.assertEqual(d.key("gpt-5-2025-08-07"), "gpt5")
        self.assertEqual(d.key("Qwen_Qwen3-32B-FC"), "qwen332b")


class Table(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = d.EpochTable.load()

    def test_the_selected_benchmarks_are_data_driven_and_exclude_superseded_files(self):
        self.assertEqual(len(self.table.benchmarks), d.N_BENCH)
        self.assertEqual(d.N_BENCH, 24)
        self.assertNotIn("frontiermath.csv", self.table.benchmarks)

    def test_the_twinrouterbench_pool_is_fully_described(self):
        for name in ("deepseek/deepseek-v3.2", "minimax/minimax-m2.7", "google/gemini-3-flash-preview", "anthropic/claude-opus-4.6"):
            with self.subTest(name):
                desc = d.describe(name, self.table)
                self.assertIsNotNone(desc["epoch_version"], name)
                self.assertTrue(desc["eci"] is not None or sum(desc["public_mask"]) >= 2, "a capability index or two benchmarks")

    def test_every_alias_lands_on_a_described_version(self):
        for name, target in d.ALIASES.items():
            with self.subTest(name):
                self.assertIn(d.key(target), self.table.scores)
                self.assertIsNotNone(d.describe(name, self.table)["epoch_version"])

    def test_an_unknown_model_is_described_by_its_name_alone_and_says_so(self):
        desc = d.describe("MadeAgents_Hammer2.1-0.5b", self.table)
        self.assertIsNone(desc["epoch_version"])
        self.assertEqual(sum(desc["public_mask"]), 0)
        self.assertEqual(desc["name"]["params_b"], 0.5)

    def test_vectors_have_one_fixed_width(self):
        a = d.vector(d.describe("gpt-4.1", self.table))
        b = d.vector(d.describe("nothing-like-this-1b", self.table))
        self.assertEqual(len(a), len(b))
        self.assertEqual(len(a), d.WIDTH)


if __name__ == "__main__":
    unittest.main()
