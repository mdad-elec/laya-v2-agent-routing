"""Laya-g1: P(cell solves | session) = σ(θ(digest) · a(cell) − b(cell)) (PREREG-g1 Amendments 1, 5).

- θ: the Laya encoder (convaiinnovations/laya @55cf4c4e, its ModernBERT only) over the digest,
  mean-pooled, projected to R^64.
- (a, b): an MLP over the cell descriptor, which is the public block (atlas/descriptors.py) plus
  the measured block (per-domain tune accuracy, a value and a mask per domain). A cell is
  described, never an id.
- Training: binary cross-entropy on [0, 1] targets, each source weighted to its pre-registered
  share. The measured block is dropped with p = 0.5 per cell per step. The bottom 8 encoder layers
  are frozen. Early stopping reads calibration BCE only.

Run on the Spark (it resumes from `last.pt`; the GB10's SM fault can kill a run):
    python -m train.g1 --data <train dir> --base <laya snapshot> --out <ckpt dir> [--probe]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from train.data import DOMAINS, SOURCE_WEIGHT

D = 64
MEASURED_WIDTH = 2 * len(DOMAINS)
FROZEN_LAYERS = 8
P_DROP_MEASURED = 0.5


class G1(nn.Module):
    def __init__(self, encoder: nn.Module, desc_width: int, d: int = D):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Linear(encoder.config.hidden_size, d)
        self.cell = nn.Sequential(nn.Linear(desc_width, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, d + 1))

    def task(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.proj((h * m).sum(1) / m.sum(1).clamp(min=1.0))

    def cells(self, desc):
        out = self.cell(desc)
        return out[:, :-1], out[:, -1]

    def forward(self, input_ids, attention_mask, desc):
        theta = self.task(input_ids, attention_mask)
        a, b = self.cells(desc)
        return theta @ a.T - b


def drop_measured(desc: torch.Tensor, public_width: int, p: float = P_DROP_MEASURED) -> torch.Tensor:
    keep = (torch.rand(desc.shape[0], 1, device=desc.device) >= p).to(desc.dtype)
    out = desc.clone()
    out[:, public_width:] = out[:, public_width:] * keep
    return out


def loss_on(logits: torch.Tensor, pairs: list[tuple[int, int, float, float]]) -> torch.Tensor:
    t, c, y, w = zip(*pairs, strict=True)
    z = logits[torch.tensor(t), torch.tensor(c)]
    y = torch.tensor(y, dtype=z.dtype, device=z.device)
    w = torch.tensor(w, dtype=z.dtype, device=z.device)
    return (nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none") * w).sum() / w.sum()


def cell_vectors(models: list[str], public: dict[str, list[float]], measured: dict[str, dict[str, float]]) -> torch.Tensor:
    rows = []
    for m in models:
        block = measured.get(m, {})
        rows.append(list(public[m]) + [block.get(d, 0.0) for d in DOMAINS] + [float(d in block) for d in DOMAINS])
    return torch.tensor(rows, dtype=torch.float32)


def load_encoder(base: Path):
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(base / "encoder")
    encoder = AutoModel.from_config(config)
    state = {k.removeprefix("encoder."): v for k, v in load_file(str(base / "model.safetensors")).items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"the Laya encoder did not load cleanly: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return encoder


def freeze_bottom(encoder: nn.Module, n: int = FROZEN_LAYERS) -> None:
    for p in encoder.embeddings.parameters():
        p.requires_grad = False
    for layer in encoder.layers[:n]:
        for p in layer.parameters():
            p.requires_grad = False


def _hash_pct(prefix: str, key: str, mod: int) -> bool:
    return int(hashlib.sha256(f"{prefix}{key}".encode()).hexdigest()[:8], 16) % mod == 0


def is_calibration(example: dict, trajectory_of: dict[str, str]) -> bool:
    """Amendment 5: 5% of LLMRouterBench/BFCL tune prompts, 20% of TwinRouterBench tune trajectories."""
    if example["source"] == "twinrouterbench":
        return _hash_pct("g1-cal/", trajectory_of[example["native_id"]], 5)
    return _hash_pct("g1-cal/", example["prompt_hash"], 20)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True, help="the convaiinnovations/laya snapshot directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--public", required=True, help="JSON {model: public descriptor vector}")
    ap.add_argument("--trajectories", required=True, help="JSON {TwinRouterBench row id: trajectory key}")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--mem-frac", type=float, default=0.12)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (a smoke run); 0 = the full recipe")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    from labels.splits import probe_trainable

    torch.manual_seed(42)
    random.seed(42)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    data = Path(args.data)
    examples = [json.loads(line) for line in (data / "examples.jsonl").open(encoding="utf-8")]
    if args.probe:
        examples = [e for e in examples if probe_trainable(e["model"])]
    digests = json.loads((data / "digests.json").read_text(encoding="utf-8"))
    measured = json.loads((data / "measured.json").read_text(encoding="utf-8"))
    public = json.loads(Path(args.public).read_text(encoding="utf-8"))
    trajectory_of = json.loads(Path(args.trajectories).read_text(encoding="utf-8"))
    models = sorted({e["model"] for e in examples})
    cell_index = {m: i for i, m in enumerate(models)}
    desc = cell_vectors(models, public, measured).to(device)
    public_width = len(next(iter(public.values())))

    count = defaultdict(int)
    for e in examples:
        count[e["source"]] += 1
    weight = {s: SOURCE_WEIGHT[s] * len(examples) / count[s] for s in count}
    by_text: dict[str, list] = defaultdict(list)
    cal_text: dict[str, list] = defaultdict(list)
    for e in examples:
        pair = (cell_index[e["model"]], e["target"], weight[e["source"]])
        (cal_text if is_calibration(e, trajectory_of) else by_text)[e["text"]].append(pair)

    tok = AutoTokenizer.from_pretrained(str(Path(args.base) / "tokenizer"))
    net = G1(load_encoder(Path(args.base)), desc_width=desc.shape[1]).to(device)
    freeze_bottom(net.encoder)
    heads = list(net.proj.parameters()) + list(net.cell.parameters())
    enc = [p for p in net.encoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc, "lr": 2e-5}, {"params": heads, "lr": 1e-3}], weight_decay=0.01)

    def batches(texts, keys):
        for i in range(0, len(keys), args.batch):
            chunk = keys[i:i + args.batch]
            enc_in = tok([digests[k] for k in chunk], padding=True, truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
            pairs = [(j, c, y, w) for j, k in enumerate(chunk) for c, y, w in texts[k]]
            yield enc_in, pairs

    def calibration_bce():
        net.eval()
        total, n = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for enc_in, pairs in batches(cal_text, sorted(cal_text)):
                loss = loss_on(net(enc_in["input_ids"], enc_in["attention_mask"], desc).float(), pairs)
                total += loss.item() * len(pairs)
                n += len(pairs)
        net.train()
        return total / max(n, 1)

    step, epoch, pos, best, order = 0, 0, 0, math.inf, None
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, epoch, pos, best, order = ck["step"], ck["epoch"], ck["pos"], ck["best"], ck["order"]
        print(f"resumed at step {step}, epoch {epoch}, batch {pos}, best cal BCE {best:.4f}", flush=True)
    log = (out / "train.log").open("a", encoding="utf-8")
    meta = {"probe": args.probe, "cells": models, "public_width": public_width, "desc_width": desc.shape[1], "texts": len(by_text),
            "calibration_texts": len(cal_text), "examples": dict(count)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    net.train()
    while epoch < args.epochs:
        if order is None:
            order = sorted(by_text)
            random.Random(f"g1-epoch-{epoch}").shuffle(order)
            pos = 0
        t0 = time.time()
        for enc_in, pairs in batches(by_text, order[pos * args.batch:]):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(enc_in["input_ids"], enc_in["attention_mask"], drop_measured(desc, public_width))
            loss = loss_on(logits.float(), pairs)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
            pos += 1
            if args.max_steps and step >= args.max_steps:
                break
            if step % 50 == 0:
                print(json.dumps({"step": step, "epoch": epoch, "loss": round(loss.item(), 4), "s": round(time.time() - t0, 1)}), file=log, flush=True)
            if step % args.eval_every == 0:
                cal = calibration_bce()
                print(json.dumps({"step": step, "epoch": epoch, "cal_bce": round(cal, 5)}), file=log, flush=True)
                if cal < best:
                    best = cal
                    torch.save({"model": net.state_dict(), "step": step, "cal_bce": cal, "meta": meta}, out / "best.pt")
                torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": epoch, "pos": pos, "best": best, "order": order}, last)
        if args.max_steps and step >= args.max_steps:
            break
        epoch += 1
        order = None
    cal = calibration_bce()
    if cal < best:
        best = cal
        torch.save({"model": net.state_dict(), "step": step, "cal_bce": cal, "meta": meta}, out / "best.pt")
    print(json.dumps({"done": True, "step": step, "best_cal_bce": round(best, 5)}), file=log, flush=True)
    print("DONE", best, flush=True)


if __name__ == "__main__":
    main()
