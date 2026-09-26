"""The campaign runner: items x cells -> outcome rows, resumable, one lane per seat, loud on seat limits."""
import json
import tempfile
import unittest
from pathlib import Path

from harness.runner import SeatLimit, run_campaign
from suites.base import Item, Suite


class Echo(Suite):
    name, domain, licence, source, revision = "echo", "knowledge", "MIT", "test", "r1"

    def load(self):
        return [Item(suite="echo", native_id=str(i), messages=[{"role": "user", "content": str(i)}], gold=str(i)) for i in range(4)]

    def check(self, item, answer):
        return 1.0 if answer == item.gold else 0.0


class Backend:
    def __init__(self, reply, limit_after=None):
        self.reply, self.limit_after, self.calls = reply, limit_after, 0

    def ask(self, cell, messages):
        self.calls += 1
        if self.limit_after is not None and self.calls > self.limit_after:
            raise SeatLimit("usage limit reached for this seat")
        return self.reply(cell, messages), {"latency_ms": 5, "input_tokens": 10, "output_tokens": 2, "cost_microusd": 3}


class Runner(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp()) / "outcomes.jsonl"

    def rows(self):
        return [json.loads(line) for line in self.out.read_text().split("\n") if line.strip()]

    def test_every_item_x_cell_becomes_an_outcome_row_with_provenance(self):
        good = Backend(lambda cell, m: m[-1]["content"] if cell.endswith("@high") else "wrong")
        report = run_campaign([Echo()], ["p/m@high", "p/m@low"], good, self.out, harness_sha="abc")
        rows = self.rows()
        self.assertEqual(len(rows), 8)
        self.assertEqual({r["cell"] for r in rows}, {"p/m@high", "p/m@low"})
        high = [r["score"] for r in rows if r["cell"] == "p/m@high"]
        self.assertEqual(high, [1.0] * 4)
        r = rows[0]
        for field in ("item_id", "suite", "split", "cell", "rep", "score", "latency_ms", "input_tokens", "output_tokens",
                      "cost_microusd_list", "harness_sha", "measured_at", "suite_revision", "status"):
            self.assertIn(field, r)
        self.assertEqual(report["answered"], 8)

    def test_a_rerun_measures_only_what_is_missing(self):
        b = Backend(lambda cell, m: "x")
        run_campaign([Echo()], ["p/m@high"], b, self.out, harness_sha="abc")
        first = b.calls
        run_campaign([Echo()], ["p/m@high", "p/m@low"], b, self.out, harness_sha="abc")
        self.assertEqual(first, 4)
        self.assertEqual(b.calls, 8, "only the new cell's four items")
        self.assertEqual(len(self.rows()), 8)

    def test_a_seat_limit_stops_that_seat_and_records_nothing_false(self):
        b = Backend(lambda cell, m: m[-1]["content"], limit_after=2)
        report = run_campaign([Echo()], ["p/m@high"], b, self.out, harness_sha="abc")
        self.assertEqual(len(self.rows()), 2, "the refused calls are not rows")
        self.assertEqual(report["limited"], {"p": "usage limit reached for this seat"})

    def test_an_unscorable_item_is_a_row_marked_unscored_not_a_zero(self):
        class Unjudgeable(Echo):
            def check(self, item, answer):
                from suites.chat.suite import Unscored
                raise Unscored("no judge")
        run_campaign([Unjudgeable()], ["p/m@high"], Backend(lambda c, m: "x"), self.out, harness_sha="abc")
        rows = self.rows()
        self.assertTrue(all(r["status"] == "unscored" and r["score"] is None for r in rows), rows[0])

    def test_a_backend_error_is_recorded_as_such_with_its_text(self):
        def broken(cell, m):
            raise RuntimeError("HTTP 502: seat crashed")
        run_campaign([Echo()], ["p/m@high"], Backend(broken), self.out, harness_sha="abc")
        rows = self.rows()
        self.assertTrue(all(r["status"] == "error" and "502" in r["error"] for r in rows))
        # An errored item is retried on the next run, never counted as a measured 0.
        run_campaign([Echo()], ["p/m@high"], Backend(lambda c, m: m[-1]["content"]), self.out, harness_sha="abc")
        latest = {}
        for r in self.rows():
            latest[(r["item_id"], r["cell"])] = r
        self.assertTrue(all(r["status"] == "ok" and r["score"] == 1.0 for r in latest.values()))


if __name__ == "__main__":
    unittest.main()
