import sqlite3
import tempfile
import unittest
from pathlib import Path

from suites.bird.suite import BIRDMiniDev, execute, extract_sql, schema_of, to_item


class BIRDSuite(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        db = self.dir / "shop" / "shop.sqlite"
        db.parent.mkdir()
        with sqlite3.connect(db) as c:
            c.executescript("CREATE TABLE customers (id INTEGER PRIMARY KEY, currency TEXT);"
                            "INSERT INTO customers (currency) VALUES ('EUR'),('EUR'),('CZK'),('USD');")
        self.row = {"question_id": "7", "db_id": "shop", "question": "How many customers pay in EUR?",
                    "evidence": "EUR is a currency", "SQL": "SELECT COUNT(*) FROM customers WHERE currency = 'EUR'", "difficulty": "simple"}
        self.suite = BIRDMiniDev(db_root=self.dir)

    def test_the_prompt_carries_the_schema_evidence_and_question(self):
        item = to_item(self.row, schema_of(self.dir / "shop" / "shop.sqlite"))
        text = item.messages[0]["content"]
        self.assertIn("CREATE TABLE customers", text)
        self.assertIn("EUR is a currency", text)
        self.assertIn("```sql", text)

    def test_execution_accuracy_compares_result_sets(self):
        item = to_item(self.row, schema_of(self.dir / "shop" / "shop.sqlite"))
        self.assertEqual(self.suite.check(item, "```sql\nSELECT count(id) FROM customers WHERE currency='EUR';\n```"), 1.0)
        self.assertEqual(self.suite.check(item, "```sql\nSELECT COUNT(*) FROM customers\n```"), 0.0)
        self.assertEqual(self.suite.check(item, "```sql\nSELECT nonsense FROM nowhere\n```"), 0.0, "an error scores 0")
        self.assertEqual(self.suite.check(item, "no sql here"), 0.0)

    def test_the_last_sql_block_counts_and_the_database_is_read_only(self):
        self.assertEqual(extract_sql("```sql\nSELECT 1\n```\nbetter:\n```sql\nSELECT 2;\n```"), "SELECT 2")
        db = self.dir / "shop" / "shop.sqlite"
        with self.assertRaises(sqlite3.OperationalError):
            execute(db, "DELETE FROM customers")
        self.assertEqual(execute(db, "SELECT COUNT(*) FROM customers"), [(4,)], "the delete never landed")

    def test_a_runaway_query_is_stopped_by_the_deadline(self):
        db = self.dir / "shop" / "shop.sqlite"
        slow = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM r) SELECT COUNT(*) FROM r"
        with self.assertRaises(sqlite3.OperationalError):
            execute(db, slow, deadline_s=0.5)



class Golden(unittest.TestCase):
    """Every usable gold query scores 1.0 on its own database, and the two whose gold needs longer
    than the model's deadline are excluded by id (measured 2026-09-26: #518 70 s, #701 > 180 s)."""

    def test_every_gold_query_passes_its_own_check(self):
        s = BIRDMiniDev()
        if not (s.cache_dir() / "dev_databases").exists():
            self.skipTest("BIRD databases not fetched (346 MB)")
        items = s.load()
        self.assertEqual(set(s.excluded), {"518", "701"})
        self.assertEqual(len(items), 498)
        failures = [i.native_id for i in items if s.check(i, "```sql\n" + i.gold["sql"] + "\n```") != 1.0]
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
