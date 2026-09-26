"""The tune/held splits every Laya-v2 measurement reads, as committed files.

    python3 scripts/ab/split_tools.py freeze        # the pilot split, exactly as auto_run drew it on 2026-09-23
    python3 scripts/ab/split_tools.py freeze-tier   # the 240 tier-labelled rows, stratified by (corpus, tier)
    python3 scripts/ab/split_tools.py regrow        # after the pilot grows: frozen ids keep their side

A split drawn at run time from whatever the corpus holds that day is a leakage trap: auto_run's
draw samples per band from the sorted id list, so adding asks re-draws the sample and moves
yesterday's held-out asks into tuning. The files are the split; the draw only ever places ids the
files have never seen.
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus")
SPLITS = os.path.join(CORPUS, "splits")
SEED = 20260923
BANDS = ("trivial", "easy", "medium", "hard")
TIERS = ("small", "medium", "powerful")


def path(name):
    return os.path.join(SPLITS, f"{name}.split.json")


def load(name):
    return json.load(open(path(name)))


def rows(filename):
    return [json.loads(line) for line in open(os.path.join(CORPUS, filename)) if line.strip()]


def draw(ids, salt):
    """Half of `ids` (floor) to tune, seeded by `salt`; the order of `ids` does not matter."""
    ids = sorted(ids)
    picked = set(random.Random(f"{SEED}-{salt}").sample(ids, len(ids) // 2))
    return [i for i in ids if i in picked], [i for i in ids if i not in picked]


def pilot_split(asks, frozen=None):
    """Frozen ids keep their side; ids the file has never seen split half per band, seeded."""
    frozen = frozen or {"tune": [], "held": []}
    known = set(frozen["tune"]) | set(frozen["held"])
    tune, held = [i for i in frozen["tune"]], [i for i in frozen["held"]]
    present = {a["id"] for a in asks}
    tune, held = [i for i in tune if i in present], [i for i in held if i in present]
    for band in BANDS:
        new = [a["id"] for a in asks if a["band"] == band and a["id"] not in known]
        t, h = draw(new, f"{band}-v2")
        tune += t
        held += h
    return tune, held


def tier_split(rows_):
    tune, held = [], []
    for corpus in sorted({r["corpus"] for r in rows_}):
        for tier in TIERS:
            t, h = draw([r["id"] for r in rows_ if r["corpus"] == corpus and r["tier"] == tier], f"{corpus}-{tier}")
            tune += t
            held += h
    return sorted(tune), sorted(held)


def write(name, corpus, tune, held):
    os.makedirs(SPLITS, exist_ok=True)
    body = {"seed": SEED, "corpus": corpus, "tune": tune, "held": held}
    with open(path(name), "w") as f:
        f.write(json.dumps(body, indent=1) + "\n")
    print(f"wrote {os.path.relpath(path(name))} ({len(tune)} tune / {len(held)} held)")


def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "freeze":
        # The draw auto_run made for the 2026-09-23 three-arm run, reproduced once and then frozen.
        asks = rows("cells-pilot.jsonl")
        tune, held = [], []
        for band in BANDS:
            mine = sorted((a for a in asks if a["band"] == band), key=lambda a: a["id"])
            picked = set(a["id"] for a in random.Random(f"{SEED}-{band}").sample(mine, len(mine) // 2))
            tune += [a["id"] for a in mine if a["id"] in picked]
            held += [a["id"] for a in mine if a["id"] not in picked]
        write("cells-pilot", "cells-pilot.jsonl", tune, held)
    elif cmd == "freeze-tier":
        both = [dict(r, corpus="requests") for r in rows("requests.jsonl")] + \
               [dict(r, corpus="dss-requests") for r in rows("dss-requests.jsonl")]
        tune, held = tier_split(both)
        write("tier-240", "requests.jsonl+dss-requests.jsonl", tune, held)
    elif cmd == "regrow":
        tune, held = pilot_split(rows("cells-pilot.jsonl"), load("cells-pilot"))
        write("cells-pilot", "cells-pilot.jsonl", tune, held)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
