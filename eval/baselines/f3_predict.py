"""F3 baselines: fit on tune, and write every configuration's HELD choices, without scoring them.

For each router and each configuration it writes a model choice per held item to
<cache>/f3-predictions/<embedder>/<router>.json. Nothing here reads a held score: the A6 read
scores these files together with g1's (PREREG-g1 Amendments 3-4). The τ / α grid is the
19-point 0.05…0.95.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval import llmrouterbench_f3 as f3
from eval.baselines import avengers_pro, embedllm
from suites.base import CACHE

GRID = [round(0.05 * i, 2) for i in range(1, 20)]


def load_vectors(path: Path) -> dict[str, np.ndarray]:
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["id"]] = np.asarray(rec["e"], dtype=np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--embeddings", required=True, help="JSON lines {id: prompt_hash, e: vector}")
    ap.add_argument("--embedder", required=True, help="a name for the output directory, e.g. qwen3-embedding-0.6b")
    ap.add_argument("--routers", default="avengers_pro,embedllm")
    args = ap.parse_args()
    items = f3.items()
    by_hash = load_vectors(Path(args.embeddings))
    missing = {it["prompt_hash"] for it in items} - set(by_hash)
    if missing:
        raise SystemExit(f"{len(missing)} F3 prompts have no embedding")
    emb = {it["query"]: by_hash[it["prompt_hash"]] for it in items}
    tune = [it for it in items if it["split"] == "tune"]
    held = [it for it in items if it["split"] == "held"]
    out_dir = CACHE.parent / "f3-predictions" / args.embedder
    out_dir.mkdir(parents=True, exist_ok=True)
    held_keys = [f"{it['dataset']}/{it['native_id']}" for it in held]

    if "avengers_pro" in args.routers:
        configs = {}
        for alpha in GRID:
            choice = avengers_pro.route(tune, held, emb, alpha=alpha)
            configs[f"alpha={alpha}"] = [choice[(it["dataset"], it["native_id"])] for it in held]
        (out_dir / "avengers_pro.json").write_text(json.dumps({"items": held_keys, "configs": configs}))
        print("avengers_pro", len(configs), "configurations", flush=True)

    if "embedllm" in args.routers:
        epochs, curve = embedllm.choose_epochs(tune, emb)
        model = embedllm.fit(tune, emb, epochs=epochs)
        cost = embedllm.expected_costs(tune)
        probs = model.predict([emb[it["query"]] for it in held])
        configs = {f"tau={tau}": [embedllm.pick(p, cost, tau) for p in probs] for tau in GRID}
        (out_dir / "embedllm.json").write_text(json.dumps({"items": held_keys, "configs": configs, "epochs": epochs, "tune_val_curve": curve}))
        print("embedllm epochs", epochs, "tune-val curve", curve, flush=True)


if __name__ == "__main__":
    main()
