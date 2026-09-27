"""P(solve) for texts × cells from a g1 checkpoint: calibration and the A6 read both use this.

    python -m train.predict --ckpt best.pt --base <laya snapshot> --texts texts.json --cells cells.json \\
        --public public.json --measured measured.json --out probs.json

texts.json is {id: digest text}; cells.json is a list of cell names. A cell absent from measured.json
has an empty measured block, which is the zero-shot case the training dropout rehearsed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train.g1 import G1, cell_vectors, load_encoder


def probabilities(net, tok, texts: list[str], desc: torch.Tensor, device, batch: int = 64, max_len: int = 512) -> list[list[float]]:
    out = []
    net.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
            with ctx:
                logits = net(enc["input_ids"], enc["attention_mask"], desc)
            out += torch.sigmoid(logits.float()).tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for a in ("--ckpt", "--base", "--texts", "--cells", "--public", "--measured", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--mem-frac", type=float, default=0.08)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cells = json.loads(Path(args.cells).read_text())
    public = json.loads(Path(args.public).read_text())
    measured = json.loads(Path(args.measured).read_text())
    desc = cell_vectors(cells, public, measured).to(device)
    if desc.shape[1] != ck["meta"]["desc_width"]:
        raise SystemExit(f"descriptor width {desc.shape[1]} is not the checkpoint's {ck['meta']['desc_width']}")
    net = G1(load_encoder(Path(args.base)), desc_width=desc.shape[1]).to(device)
    net.load_state_dict(ck["model"])
    tok = AutoTokenizer.from_pretrained(str(Path(args.base) / "tokenizer"))
    texts = json.loads(Path(args.texts).read_text())
    ids = sorted(texts)
    probs = probabilities(net, tok, [texts[i] for i in ids], desc, device)
    Path(args.out).write_text(json.dumps({"ckpt_step": ck["step"], "cells": cells, "p": {i: p for i, p in zip(ids, probs, strict=True)}}))
    print(json.dumps({"texts": len(ids), "cells": len(cells), "ckpt_step": ck["step"]}))


if __name__ == "__main__":
    main()
