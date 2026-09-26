"""BIRD Mini-Dev (CC BY-SA 4.0): text-to-SQL over real databases, execution accuracy -- `sql`.

Questions and gold SQL: huggingface.co/datasets/birdsql/bird_mini_dev at a pinned revision (the
SQLite split, 500 pairs over 11 databases). The databases are BIRD's dev databases (the same 11),
fetched from BIRD's mirror and pinned by sha256. CC BY-SA: only derived scores are published,
never modified items.

Scoring is BIRD's own execution accuracy: the model's LAST ```sql block and the gold SQL both run
READ-ONLY on the database, and the answer counts when the two result sets are equal as sets. A
query that errors, writes, or runs past the deadline scores 0.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
import urllib.request
import zipfile
from pathlib import Path

from suites.base import Item, Suite

REVISION = "f65faf4ae3b638c1fa6df1d3370c8d92c8366301"
DEV_ZIP = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
DEV_ZIP_SHA256 = "cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630"  # fetched 2026-09-26
DEADLINE_S = 30.0
_BLOCK = re.compile(r"```(?:sql)?\s*\n(.*?)```", re.S | re.I)


def extract_sql(answer: str) -> str | None:
    blocks = _BLOCK.findall(answer)
    if not blocks:
        return None
    return blocks[-1].strip().rstrip(";").strip() or None


def execute(db: Path, sql: str, deadline_s: float = DEADLINE_S) -> list[tuple]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
    start = time.monotonic()
    conn.set_progress_handler(lambda: 1 if time.monotonic() - start > deadline_s else 0, 10_000)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def schema_of(db: Path) -> str:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name").fetchall()
    finally:
        conn.close()
    return "\n\n".join(r[0].strip() + ";" for r in rows)


def to_item(row: dict, schema: str) -> Item:
    prompt = (f"Write one SQLite query that answers the question.\n\nDatabase schema:\n{schema}\n\n"
              f"Evidence: {row.get('evidence') or '(none)'}\n\nQuestion: {row['question']}\n\n"
              f"Give the final query in a ```sql code block.")
    return Item(suite="bird_mini_dev", native_id=str(row["question_id"]), messages=[{"role": "user", "content": prompt}],
                gold={"db_id": row["db_id"], "sql": row["SQL"]}, meta={"difficulty": row["difficulty"], "db_id": row["db_id"]})


class BIRDMiniDev(Suite):
    name, domain, licence = "bird_mini_dev", "sql", "CC-BY-SA-4.0"
    source = "huggingface.co/datasets/birdsql/bird_mini_dev (sqlite) + BIRD dev databases"
    revision = REVISION
    notes = "share-alike: publish derived scores only"

    def __init__(self, db_root: Path | None = None):
        self._db_root = db_root
        self._gold: dict[str, set] = {}
        self.excluded: dict[str, str] = {}

    def db_root(self) -> Path:
        if self._db_root is None:
            self._db_root = self._fetch_databases()
        return self._db_root

    def _fetch_databases(self) -> Path:
        root = self.cache_dir() / "dev_databases"
        if root.exists() and any(root.iterdir()):
            return root
        outer = self.cache_dir() / "dev.zip"
        if not outer.exists():
            urllib.request.urlretrieve(DEV_ZIP, outer)
        import hashlib

        digest = hashlib.sha256(outer.read_bytes()).hexdigest()
        if digest != DEV_ZIP_SHA256:
            raise RuntimeError(f"BIRD dev.zip sha256 {digest} is not the pinned {DEV_ZIP_SHA256}: the mirror changed")
        with zipfile.ZipFile(outer) as z:
            inner = next(n for n in z.namelist() if n.endswith("dev_databases.zip"))
            z.extract(inner, self.cache_dir())
        with zipfile.ZipFile(self.cache_dir() / inner) as z:
            z.extractall(self.cache_dir())
        return root

    def db(self, db_id: str) -> Path:
        return self.db_root() / db_id / f"{db_id}.sqlite"

    def load(self) -> list[Item]:
        path = self.cache_dir() / "mini_dev_sqlite.json"
        if not path.exists():
            urllib.request.urlretrieve(f"https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/{REVISION}/data/mini_dev_sqlite-00000-of-00001.json", path)
        text = path.read_text().strip()
        rows = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
        schemas: dict[str, str] = {}
        gold = self._gold_results(rows)
        items, self.excluded = [], {}
        for row in rows:
            if row["db_id"] not in schemas:
                schemas[row["db_id"]] = schema_of(self.db(row["db_id"]))
            item = to_item(row, schemas[row["db_id"]])
            if isinstance(gold[item.native_id], str):
                # A gold query that errors or needs longer than the model gets is not a fair item:
                # the model's equally correct query would time out too. Excluded and reported.
                self.excluded[item.native_id] = gold[item.native_id]
                continue
            self._gold[item.item_id] = set(gold[item.native_id])
            items.append(item)
        return items

    def _gold_results(self, rows: list[dict]) -> dict:
        """native id -> the gold result as a list of row reprs, or the reason it has none. Computed
        once (two gold queries take 70 s and more than 180 s) and cached beside the databases."""
        path = self.cache_dir() / f"gold_results_deadline{int(DEADLINE_S)}.json"
        if path.exists():
            return json.loads(path.read_text())
        out = {}
        for row in rows:
            try:
                out[str(row["question_id"])] = sorted({repr(r) for r in execute(self.db(row["db_id"]), row["SQL"])})
            except sqlite3.Error as e:
                out[str(row["question_id"])] = f"gold query failed within {DEADLINE_S:.0f}s: {e}"
        path.write_text(json.dumps(out))
        return out

    def check(self, item: Item, answer: str) -> float:
        sql = extract_sql(answer)
        if sql is None:
            return 0.0
        db = self.db(item.gold["db_id"])
        if item.item_id not in self._gold:
            self._gold[item.item_id] = {repr(r) for r in execute(db, item.gold["sql"])}
        try:
            predicted = {repr(r) for r in execute(db, sql)}
        except sqlite3.Error:
            return 0.0
        return 1.0 if predicted == self._gold[item.item_id] else 0.0

    def stratum(self, item: Item) -> str:
        return item.meta["difficulty"]
