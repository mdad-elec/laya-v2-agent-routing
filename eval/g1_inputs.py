"""The digests g1 scores for calibration and for the A6 read. Texts only: no label is read here.

Writes to <out>/:
- f1_cal.json:  TwinRouterBench tune calibration rows (τ_F1 is fixed on these);
- f1_held.json: TwinRouterBench held rows;
- f3_held.json: F3's held items (single-turn queries);
- f3_cells.json: F3's 13 models; f1_cells.json: the 4 pool models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval import llmrouterbench_f3 as f3
from labels import twinrouterbench as trb
from router.digest import build, render
from train.calibrate import f1_rows_and_trajectories


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    pool = json.loads((Path(__file__).resolve().parents[1] / "train" / "twinrouterbench_pool.json").read_text())["tier_to_model"]
    digest = lambda row: render(build(trb.router_view(row)["messages"]))  # noqa: E731
    files = {
        "f1_cal.json": {r["id"]: digest(r) for r in f1_rows_and_trajectories()},
        "f1_held.json": {r["id"]: digest(r) for r in trb.load() if trb.split(r) == "held"},
        "f3_held.json": {f"{it['dataset']}/{it['native_id']}": render(build([{"role": "user", "content": it["query"]}])) for it in f3.items() if it["split"] == "held"},
        "f1_cells.json": [pool[t] for t in ("low", "mid", "mid_high", "high")],
        "f3_cells.json": list(f3.MODELS),
    }
    for name, value in files.items():
        (out / name).write_text(json.dumps(value), encoding="utf-8")
    print(json.dumps({k: len(v) for k, v in files.items()}))


if __name__ == "__main__":
    main()
