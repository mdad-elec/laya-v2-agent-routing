"""The suite contract every public benchmark implements."""
import unittest

from suites.base import Item, Suite, read_jsonl, split_of


class Toy(Suite):
    name, domain, licence, source, revision = "toy", "knowledge", "MIT", "example/toy", "abc123"

    def load(self):
        return [Item(suite="toy", native_id=str(i), messages=[{"role": "user", "content": f"what is {i}+1?"}], gold=str(i + 1)) for i in range(40)]

    def check(self, item, answer):
        return 1.0 if answer.strip() == item.gold else 0.0


class Contract(unittest.TestCase):
    def test_the_item_id_is_a_stable_hash_of_suite_and_native_id_never_the_text(self):
        a = Item(suite="toy", native_id="7", messages=[{"role": "user", "content": "x"}], gold="8")
        b = Item(suite="toy", native_id="7", messages=[{"role": "user", "content": "different text"}], gold="8")
        self.assertEqual(a.item_id, b.item_id)
        self.assertEqual(len(a.item_id), 16)
        self.assertNotIn("x", a.item_id.replace("x", "") if False else "")

    def test_the_split_is_a_pure_function_of_the_id_and_does_not_move_when_items_are_added(self):
        items = Toy().load()
        first = {i.item_id: split_of(i.item_id) for i in items}
        more = items + [Item(suite="toy", native_id="new-1", messages=[], gold="")]
        again = {i.item_id: split_of(i.item_id) for i in more}
        self.assertTrue(all(again[k] == v for k, v in first.items()), "adding items never moves one")
        held = sum(1 for v in first.values() if v == "held")
        self.assertTrue(10 <= held <= 30, f"roughly half held: {held}/40")

    def test_a_suite_declares_its_provenance(self):
        t = Toy()
        self.assertEqual(t.card()["licence"], "MIT")
        self.assertEqual(t.card()["revision"], "abc123")
        self.assertIn(t.card()["domain"], ("code", "sql", "fin_table", "instruct", "knowledge", "long_ctx", "chat", "tools_multiturn"))

    def test_jsonl_is_split_on_newlines_only_never_on_unicode_line_separators(self):
        # str.splitlines() also splits on U+2028/U+2029/U+0085, which a JSON string may contain
        # raw; measured: Arena-Hard's question file failed to parse ("Unterminated string").
        import json
        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkdtemp()) / "x.jsonl"
        path.write_text(json.dumps({"a": "line one\u2028still one"}, ensure_ascii=False) + "\n" + json.dumps({"b": 2}) + "\n\n")
        self.assertEqual(read_jsonl(path), [{"a": "line one\u2028still one"}, {"b": 2}])

    def test_the_checker_scores_in_zero_one(self):
        t = Toy()
        item = t.load()[3]
        self.assertEqual(t.check(item, "4"), 1.0)
        self.assertEqual(t.check(item, "5"), 0.0)


if __name__ == "__main__":
    unittest.main()
