"""Cell descriptors: the catalogue tower's input (PREREG-g1 Amendment 1).

A model is DESCRIBED, never given a learned id, so a model Laya-g1 was not trained on is routable
with no retraining. The public block comes from Epoch AI's Benchmarking Hub (CC BY 4.0 for Epoch's
own runs; external leaderboards keep their terms, and every value names its file):
- **Benchmarks.** Every benchmark file whose metadata declares a score column and that covers
  ≥ MIN_MODELS models, minus superseded files. Each is normalised by Epoch's own metadata, so random
  chance is 0 and the ceiling is 1.
- **Epoch Capabilities Index**, when the model has one.
- **Metadata:** open weights, release date, and training compute.

Parsed from the NAME alone, so present for every model: parameters (total, and active for MoE),
a reasoning flag, and native function calling (`-FC`).

**The join** (rule 8: identities must match across systems, checked by code). `key()` strips
organisation prefixes, tool-mode and effort suffixes, and dates. Epoch versions sharing a key are
merged, taking each benchmark's best score, i.e. the model's ceiling across efforts. A model that
matches nothing keeps an all-zero public mask, and `describe()` says so.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from atlas.importers.epoch import normalise
from suites.base import CACHE

EPOCH_ZIP = "benchmark_data-2026-09-27.zip"
EPOCH_ZIP_SHA256 = "a5255aaa9e5d13e1fa38c2378196f6bc708b535e3b2427b9e99fb08818b6a1a3"
MIN_MODELS = 60
# Renames the key rule cannot see, each certain and each checked against the pinned zip by a test.
ALIASES = {
    "qwen3-235b-a22b-2507": "Qwen3-235B-A22B-Instruct-2507",  # the 2507 non-thinking release is published as Instruct-2507
    "qwen3-235b-a22b-no-thinking": "Qwen3-235B-A22B",  # the original hybrid release, run with thinking off
    "qwen3-235b-a22b-thinking": "Qwen3-235B-A22B",  # the original hybrid release, run with thinking on
}
_EFFORT_SUFFIX = re.compile(r"_(high|low|medium|minimal|xhigh|max|none|unknown|thinking|promax|\d+k)$", re.I)
_DATE = re.compile(r"-(\d{8}|\d{4}-\d{2}-\d{2})$")


def key(name: str) -> str:
    s = name.split("/")[-1]
    s = re.sub(r"[-_]fc$", "", s, flags=re.I)
    s = _EFFORT_SUFFIX.sub("", s)
    if "_" in s:
        org, rest = s.split("_", 1)
        if not re.search(r"\d", org):
            s = rest
    s = _DATE.sub("", s)
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def parse_name(name: str) -> dict:
    low = name.lower()
    params = re.search(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)b(?![a-z])", low)
    active = re.search(r"-a(\d+(?:\.\d+)?)b(?![a-z])", low)
    if re.search(r"no-?thinking|non-?thinking|nothink", low):
        reasoning = False
    elif re.search(r"thinking|reasoning|(?<![a-z])r1(?![0-9])|(?<![a-z])z1(?![0-9])|openthinker", low):
        reasoning = True
    else:
        reasoning = None
    native = bool(re.search(r"[-_]fc$", name, re.I) or "-fc-" in low)
    return {"params_b": float(params.group(1)) if params else None, "active_b": float(active.group(1)) if active else None,
            "reasoning": reasoning, "native_tools": native}


def _read(z: zipfile.ZipFile, name: str) -> list[dict]:
    return list(csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8")))


@dataclass
class EpochTable:
    benchmarks: list[str]
    scores: dict[str, dict[str, float]]  # key -> benchmark file -> best normalised score
    versions: dict[str, list[str]]  # key -> the Epoch model versions merged into it
    eci: dict[str, float]
    meta: dict[str, dict]

    @classmethod
    def load(cls, path: Path | None = None) -> EpochTable:
        path = path or CACHE.parent / "public" / "epoch" / EPOCH_ZIP
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != EPOCH_ZIP_SHA256:
            raise ValueError(f"{path}: sha256 {digest} is not the pinned {EPOCH_ZIP_SHA256}")
        z = zipfile.ZipFile(path)
        benchmarks, scores, versions = [], {}, {}
        for m in _read(z, "benchmark_metadata.csv"):
            f, col = m.get("source_file"), m.get("score_column")
            if not f or not col or m.get("superseded_by") or f not in z.namelist():
                continue
            rows = [r for r in _read(z, f) if r.get("Model version") and r.get(col) not in (None, "")]
            if len({r["Model version"] for r in rows}) < MIN_MODELS:
                continue
            benchmarks.append(f)
            for r in rows:
                v = normalise(r[col], m)
                if v is None:
                    continue
                k = key(r["Model version"])
                versions.setdefault(k, [])
                if r["Model version"] not in versions[k]:
                    versions[k].append(r["Model version"])
                scores.setdefault(k, {})
                scores[k][f] = max(v, scores[k].get(f, 0.0))
        eci = {}
        for r in _read(z, "epoch_capabilities_index/eci_scores.csv"):
            for name in [r["Model"], *[v for v in (r.get("model_versions") or "").split(",") if v.strip()]]:
                if r.get("eci"):
                    eci[key(name.strip())] = max(float(r["eci"]), eci.get(key(name.strip()), -1e9))
        meta = {}
        for r in _read(z, "model_metadata.csv"):
            meta.setdefault(key(r["model_version"]), r)
        return cls(sorted(benchmarks), scores, versions, eci, meta)


def _years(d: str | None) -> float | None:
    try:
        return (date.fromisoformat(d[:10]) - date(2023, 1, 1)).days / 365.25
    except (TypeError, ValueError):
        return None


def describe(name: str, table: EpochTable) -> dict:
    k = key(ALIASES.get(name, name))
    found = k in table.scores or k in table.eci or k in table.meta
    s = table.scores.get(k, {})
    m = table.meta.get(k, {})
    compute = None
    try:
        compute = math.log10(float(m["training_compute_flop"]))
    except (KeyError, TypeError, ValueError):
        pass
    access = (m.get("accessibility") or "").lower()
    return {"model": name, "key": k, "epoch_version": (table.versions.get(k) or [k])[0] if found else None,
            "public": [s.get(b) for b in table.benchmarks], "public_mask": [int(b in s) for b in table.benchmarks],
            "eci": table.eci.get(k), "open_weights": ("open" in access) if access else None,
            "date_years": _years(m.get("date")), "log10_compute": compute, "name": parse_name(name)}


N_BENCH = 24  # measured: the rule selects 24 files from the pinned zip (terminalbench has 59 named models)
WIDTH = 2 * N_BENCH + 2 * 4 + 2 * 2 + 2 + 1  # benchmarks+masks, 4 scalars+masks, 2 sizes+masks, reasoning+mask, native tools


def _pair(x, scale=lambda v: v):
    return [0.0, 0.0] if x is None else [float(scale(x)), 1.0]


def vector(desc: dict) -> list[float]:
    if len(desc["public"]) != N_BENCH:
        raise ValueError(f"descriptor has {len(desc['public'])} benchmarks, WIDTH assumes {N_BENCH}")
    n = desc["name"]
    out = [v or 0.0 for v in desc["public"]] + [float(x) for x in desc["public_mask"]]
    out += _pair(desc["eci"], lambda v: (v - 100.0) / 100.0) + _pair(desc["open_weights"], float)
    out += _pair(desc["date_years"], lambda v: v / 4.0) + _pair(desc["log10_compute"], lambda v: (v - 22.0) / 5.0)
    out += _pair(n["params_b"], lambda v: math.log10(v) / 3.0) + _pair(n["active_b"], lambda v: math.log10(v) / 3.0)
    out += _pair(n["reasoning"], float) + [float(n["native_tools"])]
    return out
