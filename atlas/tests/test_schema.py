"""The Atlas row contract: every later stage (import, measure, calibrate, route) writes to it."""
import unittest

from atlas.schema import DOMAINS, Profile, validate_profile


def good(**over):
    row = dict(model="claude-code/claude-opus-5-5", effort="high", domain="code", score=0.81, ci90=[0.72, 0.88], n=48,
               source="measured", benchmark="aider-polyglot", version="2025-06-rust-js-py", metric="pass@1",
               date="2026-09-30", licence="Apache-2.0", attribution="Aider-AI/aider polyglot benchmark",
               seat_status="servable")
    row.update(over)
    return row


class ProfileContract(unittest.TestCase):
    def test_a_measured_row_round_trips(self):
        row = good()
        self.assertEqual(validate_profile(row), [])
        self.assertEqual(Profile.from_row(row).to_row(), row)

    def test_the_domains_are_a_closed_set(self):
        self.assertEqual(DOMAINS, ("code", "sql", "fin_table", "instruct", "knowledge", "long_ctx", "chat", "tools_multiturn"))
        self.assertIn("domain 'erp' is not one of", validate_profile(good(domain="erp"))[0])

    def test_a_measured_row_carries_its_sample_and_interval(self):
        errors = validate_profile(good(n=0, ci90=None))
        self.assertTrue(any("n" in e for e in errors) and any("ci90" in e for e in errors), errors)
        self.assertTrue(validate_profile(good(ci90=[0.9, 0.7])), "a low bound above the high bound is refused")
        self.assertTrue(validate_profile(good(score=1.2)), "a score outside [0, 1] is refused")

    def test_an_imported_row_names_where_it_came_from(self):
        imported = good(source="imported", n=None, ci90=None, benchmark="lmarena-text", metric="elo-normalised")
        self.assertEqual(validate_profile(imported), [])
        self.assertTrue(any("attribution" in e for e in validate_profile(dict(imported, attribution=""))))
        self.assertTrue(any("licence" in e for e in validate_profile(dict(imported, licence=""))))

    def test_a_withdrawn_cell_carries_no_measured_number(self):
        withdrawn = good(model="codex/gpt-6-sol", seat_status="withdrawn@2026-09-26")
        self.assertIn("withdrawn", " ".join(validate_profile(withdrawn)))
        self.assertEqual(validate_profile(dict(withdrawn, source="imported", n=None, ci90=None)), [])

    def test_the_effort_is_explicit_even_when_a_model_has_none(self):
        self.assertEqual(validate_profile(good(effort="none")), [])
        self.assertTrue(validate_profile(good(effort="")), "an empty effort is refused; use 'none'")


if __name__ == "__main__":
    unittest.main()
