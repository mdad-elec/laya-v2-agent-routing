#!/usr/bin/env python3
"""The brain's question schema: one file, two byte-identical copies, one set of rules (rules 5, 8).

    python3 scripts/laya/check_questions.py            # the copies match and the schema is valid
    python3 scripts/laya/check_questions.py --served URL  # and the served checkpoint was built for it

`scripts/laya/questions.json` is what the bench measures; `backend/crates/dss-providers/content/
laya-questions.json` is what the gateway embeds as the shipped default (the image build copies only
`backend/`, so the crate cannot read `scripts/`). `brain_bench.py --emit-questions` writes both.
KEY ORDER IS MEANING: laya lays a `choice` out positionally, so the two copies are compared byte for
byte, never after sorting. The rules below are the ones `dss_providers::brain::Questions::validate`
enforces at admission — a schema this passes, the gateway admits.
"""
import argparse
import filecmp
import hashlib
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SCHEMA = os.path.join(HERE, "questions.json")
CRATE_COPY = os.path.join(REPO, "backend", "crates", "dss-providers", "content", "laya-questions.json")
BANDS = {"small", "medium", "powerful"}
EFFORT_LEVELS = ("minimal", "standard", "maximal")
MAX_OPTIONS, MAX_QUESTIONS = 10, 8


def same_bytes(a, b):
    return os.path.exists(a) and os.path.exists(b) and filecmp.cmp(a, b, shallow=False)


def validate(questions):
    errs = []
    if not questions:
        return ["the schema asks no question"]
    if len(questions) > MAX_QUESTIONS:
        errs.append(f"the schema asks {len(questions)} questions; more than {MAX_QUESTIONS} stops being free next to the turn")
    for qid, q in questions.items():
        kind = q.get("type")
        if kind not in ("choice", "score", "noul"):
            errs.append(f"question `{qid}`: type {kind!r} is not choice, score or noul")
            continue
        if not str(q.get("instructions", "")).strip():
            errs.append(f"question `{qid}`: instructions are empty")
        if kind in ("choice", "score"):
            n = len(q.get("criteria") or [])
            if n < 2:
                errs.append(f"question `{qid}`: a question offers at least two options, not {n}")
            if n > MAX_OPTIONS:
                errs.append(f"question `{qid}`: offers {n} options; Laya's calibration holds only up to {MAX_OPTIONS}")
    bands = [qid for qid, q in questions.items() if q.get("type") == "choice" and set(q.get("criteria") or {}) == BANDS]
    if not bands:
        errs.append("no question offers exactly the three bands small, medium, powerful — the router has no band to read")
    elif len(bands) > 1:
        errs.append(f"questions `{bands[0]}` and `{bands[1]}` both offer the three bands; the band is decided once")
    effort = questions.get("effort")
    if effort is not None:
        if effort.get("type") != "choice":
            errs.append(f"question `effort` must be a `choice` over the effort levels; it is a `{effort.get('type')}`")
        elif set(effort.get("criteria") or {}) != set(EFFORT_LEVELS):
            errs.append(f"question `effort` must offer exactly the levels {', '.join(EFFORT_LEVELS)}; it offers "
                        f"{', '.join(effort.get('criteria') or {})}")
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", default=SCHEMA)
    ap.add_argument("--served", help="a sidecar base URL (…:8099) whose /health manifest must name this schema")
    args = ap.parse_args(argv)
    with open(args.schema) as f:
        questions = json.load(f)
    failures = validate(questions)
    if not same_bytes(args.schema, CRATE_COPY):
        failures.append(f"the crate copy {os.path.relpath(CRATE_COPY, REPO)} differs from {os.path.relpath(args.schema, REPO)} "
                        "— re-emit with brain_bench.py --emit-questions (it writes both)")
    if args.served:
        want = hashlib.sha256(open(args.schema, "rb").read()).hexdigest()
        health = json.load(urllib.request.urlopen(args.served.rstrip("/") + "/health", timeout=10))
        got = ((health.get("manifest") or {}).get("questions_sha256"))
        if got != want:
            failures.append(f"the served checkpoint {health.get('model')!r} was built for schema {got}, not this one ({want[:12]}…)")
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        return 1
    band = next(qid for qid, q in questions.items() if q.get("type") == "choice" and set(q.get("criteria") or {}) == BANDS)
    print(f"OK: {len(questions)} question(s); band question `{band}` options {','.join(questions[band]['criteria'])}; "
          "crate copy byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
