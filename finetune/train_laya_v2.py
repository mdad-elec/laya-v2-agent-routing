#!/usr/bin/env python3
"""Rung 4: fine-tune Laya on this estate's routing decisions — upstream's RLCD recipe
(`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`, train_ddp.py), ported to ONE GPU
beside a production model, resumable, and never shown a held-out ask.

    ~/venvs/laya/bin/python scripts/laya/train_laya_v2.py --model-dir <base checkpoint dir> \\
        --data <build_dataset.py out dir> --out ~/laya-ckpts/laya-v2-policy [--device cuda] [--resume]

What changed from upstream, and why:
  * one process, no DDP (the Spark has one GPU and a 27B server on it);
  * the base checkpoint's OWN tokenisation (max_len / head_max_len from its config — 512 / 192 for
    the root), so training and serving build the same sequence; upstream widens to 1024 / 256;
  * bf16 autocast on an Ampere-or-newer GPU (no GradScaler), fp32 on CPU;
  * a memory cap (set_per_process_memory_fraction) and a free-memory floor checked BEFORE loading,
    so the production model is never evicted;
  * a checkpoint per epoch with a DONE marker written last, `--resume` from the highest complete
    epoch, and a per-epoch shuffle seeded by the epoch alone — a crash (the GB10's SM fault kills
    runs mid-way) replays the same epoch, not a different one;
  * the epoch is SELECTED and the temperature FITTED on the val slice (build_dataset.py), per
    option-count bucket, and written to `temperature_by_options` — the bucket the SDK actually
    applies; upstream fits per type on a slice of the training items and leaves the old buckets on.
Hyperparameters are upstream's: 4 epochs, GROUP_SIZE 4, sigma 0.4→0.1, LR 2.5e-5 encoder /
1e-4 head, weight decay 0.01, cosine, clip 1.0, proper_reward(w_sph=0.75, w_rps=1.0), loss =
policy gradient + 1.0 × soft cross-entropy; effective batch 64 (MICRO_BATCH 8 × GRAD_ACCUM 8).
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

EPOCHS, MICRO_BATCH, GRAD_ACCUM, GROUP_SIZE = 4, 8, 8, 4
LR_ENCODER, LR_HEAD, SIGMA_START, SIGMA_END = 2.5e-5, 1.0e-4, 0.4, 0.1


# ------------------------------------------------------------------ data

def training_item(tok, cfg, state, q, gold_q):
    """One sequence for one question, the notebook's build_training_item."""
    from laya.common import QTYPES, build_sequence, render_options
    t, crit = q["type"], q.get("criteria", {})
    if t == "choice":
        target = [gold_q["probabilities"].get(k, 0.0) for k in crit]
    elif t == "noul":
        target = [gold_q["probabilities"].get("false", 0.5), gold_q["probabilities"].get("true", 0.5)]
    else:
        n = len(crit) if isinstance(crit, list) else 4
        target = [gold_q["probabilities"].get(str(i), 0.0) for i in range(n)]
    s = sum(target)
    target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
    k = len(render_options({"t": t, "crit": crit}))
    seq, markers = build_sequence(tok, state, {"t": t, "ins": q["instructions"], "crit": crit}, cfg["max_len"], cfg["head_max_len"])
    if len(markers) != k:
        return None
    return {"ids": seq, "markers": markers, "qtype": QTYPES[t], "target": target, "label": target.index(max(target))}


def items_from(path, tok, cfg):
    items, skipped = [], 0
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        state, questions, gold = json.loads(row["state"]), json.loads(row["questions"]), json.loads(row["gold"])
        for qid, q in questions.items():
            if qid in gold:
                it = training_item(tok, cfg, state, q, gold[qid])
                if it is None:
                    skipped += 1
                else:
                    items.append(it)
    return items, skipped


def collate(items, pad_id):
    import torch
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "target": target,
            "qtype": torch.tensor([it["qtype"] for it in items]), "label": torch.tensor([it["label"] for it in items])}


# ------------------------------------------------------------------ checkpoints

def resume_point(out):
    """The highest epoch whose checkpoint finished (DONE written last), or 0."""
    best = 0
    if os.path.isdir(out):
        for name in os.listdir(out):
            if name.startswith("checkpoint_epoch") and os.path.exists(os.path.join(out, name, "DONE")):
                best = max(best, int(name[len("checkpoint_epoch"):]))
    return best


def save_epoch(out, epoch, model, optimizer, scheduler, meta, tok=None, encoder_config=None):
    import torch
    from safetensors.torch import save_file
    d = os.path.join(out, f"checkpoint_epoch{epoch}")
    if os.path.isdir(d):
        shutil.rmtree(d)                        # a half-written epoch from a crash is replaced whole
    os.makedirs(d)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}, os.path.join(d, "model.safetensors"))
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, os.path.join(d, "optim.pt"))
    if encoder_config is not None:
        encoder_config.save_pretrained(os.path.join(d, "encoder"))
    if tok is not None:
        tok.save_pretrained(os.path.join(d, "tokenizer"))
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    open(os.path.join(d, "DONE"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    return d


# ------------------------------------------------------------------ the loop

def amp_policy(device_type, bf16_ok):
    """(autocast dtype, whether a GradScaler is needed). bf16 where the GPU has it; fp16 plus a scaler
    on older cards (a Turing RTX 2080 Ti has no bf16); full precision on CPU."""
    import torch
    if device_type != "cuda":
        return torch.float32, False
    return (torch.bfloat16, False) if bf16_ok else (torch.float16, True)


def evaluate(model, items, pad_id, device, dtype, batch=16):
    """Argmax accuracy, soft accuracy and the raw logits (for the temperature fit), per item."""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            b = collate(chunk, pad_id)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                                  b["marker_mask"].to(device), b["qtype"].to(device))
            logits = logits.float().cpu()
            for r, it in enumerate(chunk):
                out.append((it, logits[r, :len(it["markers"])].tolist()))
    model.train()
    acc = sum(max(range(len(z)), key=z.__getitem__) == it["label"] for it, z in out) / max(1, len(out))
    soft = 0.0
    for it, z in out:
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        soft += sum(t * v / s for t, v in zip(it["target"], e))
    return {"acc": acc, "soft_acc": soft / max(1, len(out)), "n": len(out)}, out


def train(model, tok_pad_id, train_items, val_items, out, device, epochs=EPOCHS, micro_batch=MICRO_BATCH,
          grad_accum=GRAD_ACCUM, resume=False, on_epoch=None, max_steps=None):
    import torch
    from laya.common import proper_reward
    dtype, use_scaler = amp_policy(device.type, device.type == "cuda" and torch.cuda.is_bf16_supported())
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    enc = [p for n, p in model.named_parameters() if "encoder." in n]
    head = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW([{"params": enc, "lr": LR_ENCODER}, {"params": head, "lr": LR_HEAD}], weight_decay=0.01)
    total_updates = max(1, (len(train_items) // (micro_batch * grad_accum)) * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)
    start = resume_point(out) if resume else 0
    if start:
        from safetensors.torch import load_file
        d = os.path.join(out, f"checkpoint_epoch{start}")
        model.load_state_dict({k: v.float() for k, v in load_file(os.path.join(d, "model.safetensors")).items()}, strict=True)
        st = torch.load(os.path.join(d, "optim.pt"), weights_only=False)
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        print(f"[train] resumed after epoch {start}", flush=True)
    model.to(device)
    model.train()
    history = []
    steps = 0
    for epoch in range(start, epochs):
        order = list(train_items)
        random.Random(42 + epoch).shuffle(order)          # the epoch alone seeds it: a replay is the same epoch
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, epochs - 1))
        optimizer.zero_grad(set_to_none=True)
        t0, total, n = time.time(), 0.0, 0
        for b_idx in range(0, len(order), micro_batch):
            chunk = order[b_idx:b_idx + micro_batch]
            b = collate(chunk, tok_pad_id)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                logits, act = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                                    b["marker_mask"].to(device), b["qtype"].to(device))
            logits = logits.float()
            mask = b["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = b["target"].to(device)
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), b["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = (r - r.mean(0, keepdim=True)) / ((r - r.mean(0, keepdim=True)).std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / grad_accum + 0.0 * act.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1} batch {b_idx // micro_batch}: {loss.item()}")
            scaler.scale(loss).backward()
            n += 1
            total += loss.item() * grad_accum
            if n % grad_accum == 0 or b_idx + micro_batch >= len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            steps += 1
            if max_steps and steps >= max_steps:
                break
        val, _ = evaluate(model, val_items, tok_pad_id, device, dtype)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        meta = {"epoch": epoch + 1, "epochs": epochs, "avg_loss": total / max(1, n), "val": val,
                "seconds": round(time.time() - t0, 1), "train_items": len(train_items), "val_items": len(val_items)}
        history.append(meta)
        print(f"[train] epoch {epoch + 1}/{epochs} loss {meta['avg_loss']:.4f} val acc {val['acc']:.3f} soft {val['soft_acc']:.3f} ({meta['seconds']}s)", flush=True)
        (on_epoch or (lambda *a: None))(epoch + 1, model, optimizer, scheduler, meta)
        if max_steps and steps >= max_steps:
            break
    return history


def fit_bucket_temperatures(val_out):
    """One temperature per (question type, option count) bucket on the val slice, by NLL."""
    from laya.common import temp_bucket
    import fit_temperature
    by = {}
    for it, z in val_out:
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        probs = {str(i): v / s for i, v in enumerate(e)}
        gold = {str(i): t for i, t in enumerate(it["target"])}
        by.setdefault(temp_bucket(it["qtype"], len(z)), []).append({"probabilities": probs, "gold": gold})
    return {b: fit_temperature.fit(rows, t0=1.0)[0] for b, rows in by.items() if len(rows) >= 10}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--mem-fraction", type=float, default=0.12)
    ap.add_argument("--min-free-gb", type=float, default=25.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--micro-batch", type=int, default=MICRO_BATCH)
    ap.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    ap.add_argument("--checkpointing", action="store_true", help="gradient checkpointing (upstream uses it on 16 GB T4s)")
    ap.add_argument("--name", default="laya-v2-routing")
    args = ap.parse_args(argv)
    import torch
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model
    device = torch.device(args.device)
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if free < args.min_free_gb * 1e9:
            sys.exit(f"only {free / 1e9:.1f} GB free on the GPU (need {args.min_free_gb}); the production model must not be evicted")
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    _fix_tokenizer_config(args.model_dir)
    cfg = json.load(open(os.path.join(args.model_dir, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))
    train_items, s1 = items_from(os.path.join(args.data, "train.jsonl"), tok, cfg)
    val_items, s2 = items_from(os.path.join(args.data, "val.jsonl"), tok, cfg)
    print(f"[train] {len(train_items)} train / {len(val_items)} val sequences (skipped {s1 + s2} whose markers did not fit "
          f"{cfg['max_len']}/{cfg['head_max_len']})", flush=True)
    model = build_model(cfg, encoder_dir=os.path.join(args.model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(args.model_dir, "model.safetensors")), strict=True)
    try:
        model.encoder.config.reference_compile = False
    except Exception:
        pass
    if args.checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    os.makedirs(args.out, exist_ok=True)
    started = time.time()

    def on_epoch(epoch, m, opt, sch, meta):
        save_epoch(args.out, epoch, m, opt, sch, meta, tok=tok, encoder_config=m.encoder.config)

    history = train(model, tok.pad_token_id, train_items, val_items, args.out, device, epochs=args.epochs,
                    micro_batch=args.micro_batch, grad_accum=args.grad_accum, resume=args.resume, on_epoch=on_epoch)
    metas = [json.load(open(os.path.join(args.out, f"checkpoint_epoch{e}", "meta.json"))) for e in range(1, resume_point(args.out) + 1)]
    best = max(metas, key=lambda m: (m["val"]["acc"], m["epoch"]))           # ties → the later epoch
    chosen = os.path.join(args.out, f"checkpoint_epoch{best['epoch']}")
    model.load_state_dict({k: v.float() for k, v in load_file(os.path.join(chosen, "model.safetensors")).items()}, strict=True)
    model.to(device)
    _, val_out = evaluate(model, val_items, tok.pad_token_id, device,
                          amp_policy(device.type, device.type == "cuda" and torch.cuda.is_bf16_supported())[0])
    temps = fit_bucket_temperatures(val_out)
    final = os.path.join(args.out, "final")
    if os.path.isdir(final):
        shutil.rmtree(final)
    shutil.copytree(chosen, final, ignore=shutil.ignore_patterns("optim.pt", "DONE", "meta.json"))
    cfg["temperature_by_options"] = {**cfg.get("temperature_by_options", {}), **temps}
    cfg["fine_tuned"], cfg["model_name"] = True, args.name
    cfg["training"] = {"recipe": "RLCD (upstream train_ddp.py, single-GPU port: scripts/laya/train_laya_v2.py)",
                       "base": os.path.realpath(args.model_dir), "data": os.path.realpath(args.data),
                       "epochs": args.epochs, "selected_epoch": best["epoch"], "history": [
                           {"epoch": m["epoch"], "avg_loss": m["avg_loss"], "val": m["val"]} for m in metas],
                       "fitted_temperatures": temps, "hours": round((time.time() - started) / 3600, 3)}
    json.dump(cfg, open(os.path.join(final, "rl_agent_config.json"), "w"), indent=2)
    print(f"[train] selected epoch {best['epoch']} (val acc {best['val']['acc']:.3f}); temperatures {temps}; wrote {final}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
