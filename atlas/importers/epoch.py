"""Epoch AI's Benchmarking Hub (CC BY 4.0) as Atlas priors.

Source: https://epoch.ai/data/benchmark_data.zip (the "use this data" download), pinned by its
sha256 and the date it was fetched. Epoch runs many benchmarks itself and records the reasoning
effort in the model name (`gpt-5.6-sol_low`, `claude-fable-5-1_xhigh`), which is exactly the Atlas
cell. Other files are compiled from external leaderboards; Epoch's licence covers its data, and
external data keeps its own terms, so each row names the source file it came from.

A benchmark enters only through BENCHMARK_DOMAIN, and only if Epoch's own metadata names its score
column. Scores are normalised by that metadata so random chance is 0 and the ceiling is 1. Names
without a declared effort (`_unknown`, `_promax`, a bare model) are reported, never guessed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

LICENCE = "CC-BY-4.0"

# Epoch source file -> Atlas domain. Unlisted files are reported as skipped.
BENCHMARK_DOMAIN = {
    "swe_bench_verified.csv": "code",
    "mirrorcode.csv": "code",
    "aider_polyglot_external.csv": "code",
    "deepswe_external.csv": "code",
    "frontiercode_external.csv": "code",
    "gpqa_diamond.csv": "knowledge",
    "otis_mock_aime_2024_2025.csv": "knowledge",
    "simpleqa_verified.csv": "knowledge",
    "hle_external.csv": "knowledge",
    "proofbench_external.csv": "knowledge",
    "simplebench_external.csv": "knowledge",
    "arc_agi_2_external.csv": "knowledge",
    "terminalbench_external.csv": "tools_multiturn",
    "osworld_2_external.csv": "tools_multiturn",
    "apex_agents_external.csv": "tools_multiturn",
    "balrog_external.csv": "tools_multiturn",
    "fictionlivebench_external.csv": "long_ctx",
}


def match(name: str, catalogue: dict) -> tuple[str, str] | None:
    """(catalogue model, effort) for `<model>_<effort>` when the model declares that effort."""
    if "_" not in name:
        return None
    stem, effort = name.rsplit("_", 1)
    for entry in catalogue["models"]:
        if entry["model"].split("/", 1)[1].split("/")[-1] == stem and effort in entry["efforts"]:
            return entry["model"], effort
    return None


def normalise(raw: str, meta: dict) -> float | None:
    """raw * scale, mapped so the benchmark's random baseline is 0 and its ceiling is 1."""
    try:
        value = float(raw) * float(meta["scale"])
    except (TypeError, ValueError):
        return None
    base, ceiling = float(meta.get("random_baseline") or 0.0), float(meta.get("score_ceiling") or 1.0)
    if ceiling <= base:
        return None
    return round(min(1.0, max(0.0, (value - base) / (ceiling - base))), 6)


def profiles_from(source_file: str, rows: list[dict], metadata: dict, catalogue: dict, updated: str) -> tuple[list[dict], dict]:
    domain = BENCHMARK_DOMAIN.get(source_file)
    meta = metadata.get(source_file)
    if domain is None:
        return [], {"skipped": f"{source_file} has no domain mapping"}
    if not meta or not meta.get("score_column"):
        return [], {"skipped": f"{source_file}: Epoch's metadata names no score column"}
    status = {m["model"]: m["seat_status"] for m in catalogue["models"]}
    profiles, unmatched, no_score = [], set(), set()
    for row in rows:
        name = row.get("Model version", "")
        hit = match(name, catalogue)
        if hit is None:
            unmatched.add(name)
            continue
        score = normalise(row.get(meta["score_column"], ""), meta)
        if score is None:
            no_score.add(name)
            continue
        model, effort = hit
        profiles.append({
            "model": model, "effort": effort, "domain": domain, "score": score,
            "ci90": None, "n": None, "source": "imported",
            "benchmark": f"epoch-{source_file.removesuffix('.csv')}", "version": updated,
            "metric": f"{meta['score_column']} normalised (baseline {meta.get('random_baseline') or 0}, ceiling {meta.get('score_ceiling') or 1})",
            "date": (row.get("Release date") or updated)[:10] or updated,
            "licence": LICENCE,
            "attribution": f"Epoch AI, Benchmarking Hub ({meta['benchmark']}, {source_file}, fetched {updated}), CC BY 4.0",
            "seat_status": status[model],
        })
    return profiles, {"unmatched": sorted(unmatched), "no_score": sorted(no_score)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zip", required=True, help="benchmark_data.zip as downloaded")
    ap.add_argument("--updated", required=True, help="the date the zip was fetched (YYYY-MM-DD)")
    ap.add_argument("--catalogue", default=str(Path(__file__).resolve().parents[1] / "catalogue.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    blob = Path(args.zip).read_bytes()
    z = zipfile.ZipFile(io.BytesIO(blob))
    read = lambda name: list(csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8")))
    metadata = {m["source_file"]: m for m in read("benchmark_metadata.csv") if m.get("source_file")}
    catalogue = json.loads(Path(args.catalogue).read_text())
    out, reports = [], {}
    for source_file in sorted(BENCHMARK_DOMAIN):
        rows = read(source_file) if source_file in z.namelist() else []
        profiles, reports[source_file] = profiles_from(source_file, rows, metadata, catalogue, args.updated)
        out.extend(profiles)
    Path(args.out).write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in out))
    print(json.dumps({"zip_sha256": hashlib.sha256(blob).hexdigest(), "profiles": len(out),
                      "per_file": {k: len([p for p in out if p["benchmark"] == f"epoch-{k.removesuffix('.csv')}"]) for k in sorted(BENCHMARK_DOMAIN)},
                      "skipped": {k: v["skipped"] for k, v in reports.items() if "skipped" in v}}, indent=1))


if __name__ == "__main__":
    main()
