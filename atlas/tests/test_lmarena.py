"""LMArena rows -> Atlas priors. The matcher is where a silent mistake would live, so it is pinned
on real names from the 2026-09-25 publication."""
import json
import unittest
from pathlib import Path

from atlas.importers.lmarena import CATEGORY_DOMAIN_FLAT as CATEGORY_DOMAIN, match, profiles_from
from atlas.schema import validate_profile

CATALOGUE = json.loads((Path(__file__).resolve().parents[1] / "catalogue.json").read_text())


class Matcher(unittest.TestCase):
    def test_an_effort_suffixed_name_maps_to_its_cell(self):
        self.assertEqual(match("claude-opus-5.5-high", CATALOGUE), ("claude-code/claude-opus-5-5", "high"))
        self.assertEqual(match("gpt-5.6-sol-xhigh", CATALOGUE), ("codex/gpt-5.6-sol", "xhigh"))
        self.assertEqual(match("gpt-6-astra-max", CATALOGUE), ("codex/gpt-6-astra", "max"))
        self.assertEqual(match("claude-fable-5.1-max", CATALOGUE), ("claude-code/claude-fable-5-1", "max"))

    def test_the_agent_arena_display_names_map_the_same_way(self):
        self.assertEqual(match("Claude Fable 5.1 (Max)", CATALOGUE), ("claude-code/claude-fable-5-1", "max"))
        self.assertEqual(match("Claude Opus 5 (High)", CATALOGUE), ("claude-code/claude-opus-5", "high"))

    def test_a_name_that_does_not_say_its_effort_or_model_is_not_guessed(self):
        self.assertIsNone(match("claude-opus-5", CATALOGUE), "a bare name does not say which effort")
        self.assertIsNone(match("qwen3.8-max", CATALOGUE), "Qwen3.8 Max is not the 27B we serve")
        self.assertIsNone(match("claude-opus-5-ultra", CATALOGUE), "an effort the model does not declare")
        self.assertIsNone(match("claude-opus-4-6-high", CATALOGUE), "not in the catalogue")


class Rows(unittest.TestCase):
    def rows(self):
        base = dict(organization="anthropic", license="Proprietary", variance=1.0, rank=1, leaderboard_publish_date="2026-09-25")
        return [
            dict(base, model_name="claude-opus-5.5-high", rating=1517.8, rating_lower=1505.6, rating_upper=1530.0, vote_count=2307, category="coding"),
            dict(base, model_name="gpt-5.6-sol-xhigh", rating=1455.6, rating_lower=1451.0, rating_upper=1460.1, vote_count=34258, category="coding"),
            dict(base, model_name="some-other-model", rating=1300.0, rating_lower=1290.0, rating_upper=1310.0, vote_count=900, category="coding"),
            dict(base, model_name="claude-opus-5", rating=1500.0, rating_lower=1495.0, rating_upper=1505.0, vote_count=900, category="coding"),
            dict(base, model_name="claude-opus-5.5-high", rating=1500.0, rating_lower=1490.0, rating_upper=1510.0, vote_count=900, category="german"),
        ]

    def test_ratings_become_valid_imported_priors_normalised_within_their_category(self):
        profiles, report = profiles_from("text", self.rows(), CATALOGUE, revision="a4e245e5")
        self.assertEqual(len(profiles), 2, "only matched rows in a mapped category")
        for p in profiles:
            self.assertEqual(validate_profile(p), [], p)
            self.assertEqual((p["source"], p["domain"], p["licence"]), ("imported", "code", "CC-BY-4.0"))
            self.assertIn("lmarena-ai/leaderboard-dataset@a4e245e5", p["attribution"])
        top = next(p for p in profiles if p["effort"] == "high")
        self.assertEqual(top["score"], 1.0, "min-max over every row of the category, matched or not")
        low = next(p for p in profiles if p["effort"] == "xhigh")
        self.assertAlmostEqual(low["score"], (1455.6 - 1300.0) / (1517.8 - 1300.0), places=6)

    def test_what_was_not_imported_is_reported_by_name_never_dropped_silently(self):
        _, report = profiles_from("text", self.rows(), CATALOGUE, revision="a4e245e5")
        self.assertIn("some-other-model", report["unmatched"])
        self.assertIn("claude-opus-5", report["unmatched"])
        self.assertIn("german", report["unmapped_categories"])

    def test_the_category_map_only_targets_atlas_domains(self):
        from atlas.schema import DOMAINS
        self.assertTrue(set(CATEGORY_DOMAIN.values()) <= set(DOMAINS))


if __name__ == "__main__":
    unittest.main()
