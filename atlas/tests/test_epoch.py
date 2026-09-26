"""Epoch AI benchmark CSVs -> Atlas priors, normalised by each benchmark's own baseline and ceiling."""
import json
import unittest
from pathlib import Path

from atlas.importers.epoch import BENCHMARK_DOMAIN, match, normalise, profiles_from
from atlas.schema import DOMAINS, validate_profile

CATALOGUE = json.loads((Path(__file__).resolve().parents[1] / "catalogue.json").read_text())
META = {"gpqa_diamond.csv": {"benchmark": "GPQA diamond", "score_column": "Best score (across scorers)", "scale": "1.0",
                            "random_baseline": "0.25", "score_ceiling": "1.0", "release_date": "2023-11-20"}}


class Epoch(unittest.TestCase):
    def test_model_underscore_effort_maps_to_its_cell_and_nothing_else_is_guessed(self):
        self.assertEqual(match("gpt-5.6-sol_low", CATALOGUE), ("codex/gpt-5.6-sol", "low"))
        self.assertEqual(match("claude-fable-5-1_xhigh", CATALOGUE), ("claude-code/claude-fable-5-1", "xhigh"))
        self.assertEqual(match("gpt-5.6-luna_none", CATALOGUE), ("codex/gpt-5.6-luna", "none"))
        for name in ("claude-opus-5", "gpt-5.6-sol_unknown", "gpt-5.6-sol_promax", "claude-sonnet-5_16K", "nemotron-3-ultra"):
            self.assertIsNone(match(name, CATALOGUE), name)

    def test_scores_are_normalised_so_random_chance_is_zero_and_the_ceiling_is_one(self):
        m = META["gpqa_diamond.csv"]
        self.assertAlmostEqual(normalise("0.25", m), 0.0)
        self.assertAlmostEqual(normalise("1.0", m), 1.0)
        self.assertAlmostEqual(normalise("0.625", m), 0.5)
        self.assertEqual(normalise("0.1", m), 0.0, "below chance is clamped, not negative")
        self.assertIsNone(normalise("", m))

    def test_rows_become_valid_priors_and_the_rest_is_reported(self):
        rows = [{"Model version": "gpt-5.6-sol_low", "Best score (across scorers)": "0.8990", "Release date": "2026-07-09"},
                {"Model version": "gpt-5.6-sol_unknown", "Best score (across scorers)": "0.9", "Release date": "2026-07-09"},
                {"Model version": "gpt-5.6-sol_high", "Best score (across scorers)": "", "Release date": "2026-07-09"}]
        profiles, report = profiles_from("gpqa_diamond.csv", rows, META, CATALOGUE, updated="2026-09-26")
        self.assertEqual(len(profiles), 1)
        p = profiles[0]
        self.assertEqual(validate_profile(p), [], p)
        self.assertEqual((p["model"], p["effort"], p["domain"], p["source"]), ("codex/gpt-5.6-sol", "low", "knowledge", "imported"))
        self.assertIn("Epoch AI", p["attribution"])
        self.assertIn("gpt-5.6-sol_unknown", report["unmatched"])
        self.assertIn("gpt-5.6-sol_high", report["no_score"])

    def test_an_unmapped_benchmark_imports_nothing(self):
        profiles, report = profiles_from("lmca_external.csv", [{"Model version": "gpt-5.6-sol_low", "Score": "50"}], {}, CATALOGUE, updated="2026-09-26")
        self.assertEqual(profiles, [])
        self.assertEqual(report["skipped"], "lmca_external.csv has no domain mapping")

    def test_the_map_targets_only_atlas_domains(self):
        self.assertTrue(set(BENCHMARK_DOMAIN.values()) <= set(DOMAINS))


if __name__ == "__main__":
    unittest.main()
