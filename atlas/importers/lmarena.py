"""LMArena's public leaderboard (CC BY 4.0) as Atlas priors.

Source: huggingface.co/datasets/lmarena-ai/leaderboard-dataset, pinned by revision. Arena ratings
are human pairwise preferences, so they enter the Atlas as IMPORTED priors, never as measurements:
calibration later maps each (arena category -> our domain) onto what we measure on public suites,
and a mapping whose fit is poor is shown for information only.

The name matcher is deliberately strict. An arena name says its effort (`claude-opus-5.5-high`,
`Claude Fable 5.1 (Max)`) or it is not imported: a bare `claude-opus-5` does not say which effort
was rated, and guessing would put one effort's number on another's cell. Everything not imported
is reported by name.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LICENCE = "CC-BY-4.0"
DATASET = "lmarena-ai/leaderboard-dataset"

# Arena category -> Atlas domain. Only these become priors; the rest are reported as unmapped.
CATEGORY_DOMAIN = {
    ("text", "coding"): "code",
    ("text", "instruction_following"): "instruct",
    ("text", "industry_business_and_management_and_financial_operations"): "fin_table",
    ("text", "expert"): "knowledge",
    ("text", "longer_query"): "long_ctx",
    ("text", "overall"): "chat",
    ("agent", "overall"): "tools_multiturn",
}
# The flat view the tests read: which domains the map can reach.
CATEGORY_DOMAIN_FLAT = {f"{k[0]}:{k[1]}": v for k, v in CATEGORY_DOMAIN.items()}


def _normal(name: str) -> str:
    """`Claude Fable 5.1 (Max)` and `claude-fable-5.1-max` both become `claude-fable-5-1-max`."""
    s = name.strip().lower()
    s = re.sub(r"\s*\(([^)]*)\)\s*$", r"-\1", s)
    s = re.sub(r"[\s._]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def match(name: str, catalogue: dict) -> tuple[str, str] | None:
    """(catalogue model, effort) for an arena name that states both, else None."""
    norm = _normal(name)
    for entry in catalogue["models"]:
        model_id = entry["model"].split("/", 1)[1]
        stem = _normal(model_id.split("/")[-1])
        for effort in entry["efforts"]:
            if effort != "none" and norm == f"{stem}-{effort}":
                return entry["model"], effort
    return None


def profiles_from(arena: str, rows: list[dict], catalogue: dict, revision: str) -> tuple[list[dict], dict]:
    """Imported profiles for one arena file, and a report of everything that was not imported."""
    status = {m["model"]: m["seat_status"] for m in catalogue["models"]}
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)
    value = "rating" if rows and "rating" in rows[0] else "score"
    profiles, unmatched, unmapped = [], set(), set()
    for category, group in sorted(by_category.items()):
        domain = CATEGORY_DOMAIN.get((arena, category))
        if domain is None:
            unmapped.add(category)
            continue
        lo, hi = min(r[value] for r in group), max(r[value] for r in group)
        span = hi - lo or 1.0
        for row in group:
            hit = match(row["model_name"], catalogue)
            if hit is None:
                unmatched.add(row["model_name"])
                continue
            model, effort = hit
            profiles.append({
                "model": model, "effort": effort, "domain": domain,
                # Min-max over EVERY model in the category, matched or not, so a score means the
                # same thing whichever of our models happens to be listed.
                "score": round((row[value] - lo) / span, 6),
                "ci90": None, "n": None, "source": "imported",
                "benchmark": f"lmarena-{arena}-{category}", "version": str(row["leaderboard_publish_date"]),
                "metric": f"arena-{value}-minmax", "date": str(row["leaderboard_publish_date"]),
                "licence": LICENCE,
                "attribution": f"LMArena, {DATASET}@{revision} ({arena}/{category}), CC BY 4.0",
                "seat_status": status[model],
            })
    return profiles, {"unmatched": sorted(unmatched), "unmapped_categories": sorted(unmapped)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="directory holding <arena>.parquet files fetched at --revision")
    ap.add_argument("--revision", required=True)
    ap.add_argument("--catalogue", default=str(Path(__file__).resolve().parents[1] / "catalogue.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import pyarrow.parquet as pq  # only the CLI needs it; the tests and the matcher are stdlib

    catalogue = json.loads(Path(args.catalogue).read_text())
    out, reports = [], {}
    for arena in ("text", "agent"):
        path = Path(args.dir) / f"{arena}.parquet"
        rows = pq.read_table(path).to_pylist()
        profiles, reports[arena] = profiles_from(arena, rows, catalogue, args.revision)
        out.extend(profiles)
    Path(args.out).write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in out))
    print(json.dumps({"profiles": len(out), "report": {k: {"unmatched": len(v["unmatched"]), "unmapped_categories": len(v["unmapped_categories"])} for k, v in reports.items()}}))


if __name__ == "__main__":
    main()
