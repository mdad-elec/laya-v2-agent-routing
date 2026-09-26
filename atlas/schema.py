"""The Model Atlas row: one (model, effort, domain) profile, measured or imported.

The Atlas is what a router knows about a model beyond its price. Every stage writes this one shape:
importers (public leaderboards, as priors), the measurement campaign (our own runs on public
suites, as evidence), calibration, and the router that reads it. A row that breaks the contract is
refused by name, before it can reach anything that ranks models (rule 2: validate before commit).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# Closed on purpose: a new domain is a schema change and a Decision, not a typo that silently
# becomes a column nobody reads.
DOMAINS = ("code", "sql", "fin_table", "instruct", "knowledge", "long_ctx", "chat", "tools_multiturn")
SOURCES = ("measured", "imported")
_SEAT = re.compile(r"^(servable|withdrawn@\d{4}-\d{2}-\d{2})$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MODEL = re.compile(r"^[a-z0-9][a-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._:/-]*$")


@dataclass(frozen=True)
class Profile:
    model: str  # provider/model, the catalogue's own spelling
    effort: str  # an effort id the model declares, or "none" for a model that declares none
    domain: str
    score: float  # in [0, 1]; for imported rows, the source's metric normalised onto [0, 1]
    ci90: list | None  # [low, high], required when measured
    n: int | None  # items behind the score, required when measured
    source: str
    benchmark: str
    version: str
    metric: str
    date: str  # when measured, or when the source published it
    licence: str
    attribution: str
    seat_status: str

    @classmethod
    def from_row(cls, row: dict) -> "Profile":
        errors = validate_profile(row)
        if errors:
            raise ValueError("; ".join(errors))
        return cls(**{k: row[k] for k in cls.__dataclass_fields__})

    def to_row(self) -> dict:
        return asdict(self)


def validate_profile(row: dict) -> list[str]:
    """Every rule this row breaks, each naming the field -- empty when the row is good."""
    errors: list[str] = []
    missing = [k for k in Profile.__dataclass_fields__ if k not in row]
    if missing:
        return [f"missing fields: {', '.join(missing)}"]
    unknown = sorted(set(row) - set(Profile.__dataclass_fields__))
    if unknown:
        errors.append(f"unknown fields: {', '.join(unknown)}")
    if not _MODEL.match(str(row["model"])):
        errors.append(f"model {row['model']!r} is not provider/model")
    if not isinstance(row["effort"], str) or not row["effort"]:
        errors.append("effort must be the model's effort id, or 'none' for a model that declares none")
    if row["domain"] not in DOMAINS:
        errors.append(f"domain {row['domain']!r} is not one of {', '.join(DOMAINS)}")
    if row["source"] not in SOURCES:
        errors.append(f"source {row['source']!r} is not one of {', '.join(SOURCES)}")
    score = row["score"]
    if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
        errors.append(f"score {score!r} is not in [0, 1]")
    if not _DATE.match(str(row["date"])):
        errors.append(f"date {row['date']!r} is not YYYY-MM-DD")
    if not _SEAT.match(str(row["seat_status"])):
        errors.append(f"seat_status {row['seat_status']!r} is not 'servable' or 'withdrawn@YYYY-MM-DD'")
    for field in ("benchmark", "version", "metric", "licence", "attribution"):
        if not str(row[field]).strip():
            errors.append(f"{field} is empty: a number without its provenance is not evidence")

    if row["source"] == "measured":
        n, ci = row["n"], row["ci90"]
        if not isinstance(n, int) or n < 1:
            errors.append("a measured row needs n >= 1 (the items behind the score)")
        if not (isinstance(ci, list) and len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci)):
            errors.append("a measured row needs ci90 = [low, high]")
        elif not 0.0 <= ci[0] <= ci[1] <= 1.0:
            errors.append(f"ci90 {ci} is not an interval inside [0, 1]")
        if str(row["seat_status"]).startswith("withdrawn"):
            errors.append("a withdrawn cell carries no measured number: its seat cannot serve it, so it was not measured there")
    return errors
