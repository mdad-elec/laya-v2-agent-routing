"""BFCL's published per-entry results (HuanzhiMao/BFCL-Result, Apache-2.0) as public item-level outcomes.

Each model's score file for a category is a summary line ({accuracy, correct_count, total_count})
followed by one line per FAILED entry. An entry not listed passed. The file's own summary is the
check: a file whose listed failures disagree with its correct_count is refused, never guessed at.

Labels are built from ABILITY, not price (labels/tiers.py). BFCL's "Total Cost ($)" is the
evaluators' own GPU bill for open-weight models (Falcon3-3B $36.70 against a 253B model's $0.72,
2025-12-16), so it cannot rank models by what they cost to use.
"""
from __future__ import annotations

import json
import urllib.request

from suites.base import CACHE

REVISION = "cadd76a8dc62984e7bf15ecaf358f070bae745cc"
SNAPSHOT = "2025-12-16"
SOURCE = f"bfcl-result@{REVISION[:8]}/{SNAPSHOT}"
CATEGORY_FILE = "multi_turn/BFCL_v4_multi_turn_base_score.json"
N_ENTRIES = 200


def outcomes_from_score_file(text: str, model: str, n_entries: int = N_ENTRIES) -> list[dict]:
    lines = [json.loads(line) for line in text.split("\n") if line.strip()]
    summary, failures = lines[0], lines[1:]
    failed = {f["id"]: ((f.get("error") or {}).get("error_type") or "invalid") for f in failures}
    if len(failed) != len(failures):
        raise ValueError(f"{model}: an entry is listed as failed twice")
    if summary.get("total_count") != n_entries:
        raise ValueError(f"{model}: {summary.get('total_count')} entries, expected {n_entries}")
    passed = n_entries - len(failed)
    if passed != summary.get("correct_count"):
        raise ValueError(f"{model}: {len(failed)} failures listed means {passed} correct, but the file says {summary.get('correct_count')}")
    rows = []
    for i in range(n_entries):
        native = f"multi_turn_base_{i}"
        rows.append({"source": SOURCE, "suite": "bfcl_multi_turn", "native_id": native, "model": model,
                     "pass": native not in failed, "reason": failed.get(native)})
    return rows


def fetch(model_dirs: list[str]) -> dict[str, str]:
    """model dir -> its score file text, cached at the pinned revision."""
    cache = CACHE.parent / "public" / "bfcl-result" / REVISION / SNAPSHOT
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for model in model_dirs:
        path = cache / f"{model}.json"
        if not path.exists():
            url = f"https://raw.githubusercontent.com/HuanzhiMao/BFCL-Result/{REVISION}/{SNAPSHOT}/score/{model}/{CATEGORY_FILE}"
            urllib.request.urlretrieve(url, path)
        out[model] = path.read_text(encoding="utf-8")
    return out


def model_dirs() -> list[str]:
    url = f"https://api.github.com/repos/HuanzhiMao/BFCL-Result/contents/{SNAPSHOT}/score?ref={REVISION}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return sorted(x["name"] for x in json.loads(r.read()) if x["type"] == "dir")


def load_all() -> tuple[list[dict], dict]:
    rows, report = [], {"models": 0, "refused": {}}
    for model, text in fetch(model_dirs()).items():
        try:
            rows += outcomes_from_score_file(text, model)
            report["models"] += 1
        except (ValueError, json.JSONDecodeError, IndexError) as e:
            report["refused"][model] = str(e)
    return rows, report
