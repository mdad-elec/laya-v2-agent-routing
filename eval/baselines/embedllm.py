"""EmbedLLM (F3 baseline): the `TextMF` matrix factorisation from LLMRouterBench's
`baselines/EmbedLLM/algorithm/mf.py` @ c77cb050, with its hyperparameters (model embedding dim 1024,
question-noise α 0.05, Adam lr 1e-4, weight decay 1e-5, batch 2048, cross-entropy on 2 classes).

What differs from its script (PREREG-g1 Amendment 4):
- The epoch count is chosen on 10% of TUNE prompts, never on the test set as its script does,
  and the model is then refit on all of tune.
- The label is 1 if score ≥ 0.5.
- EmbedLLM has no cost knob, so it routes by g1's policy: the cheapest model whose P̂ ≥ τ,
  otherwise the argmax. "Cheapest" is by EXPECTED cost (a model's mean cost per tune item),
  because a router cannot see an item's cost before running it.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from dataclasses import dataclass

import numpy as np

from eval.baselines.lrb_code import checkout

DIM, NOISE, LR, WD, BATCH = 1024, 0.05, 1e-4, 1e-5, 2048
SOLVED = 0.5


def _mf_module():
    root = checkout()
    sys.modules.setdefault("wandb", types.ModuleType("wandb"))  # imported at mf.py's top, never called here
    spec = importlib.util.spec_from_file_location("lrb_embedllm_mf", root / "baselines" / "EmbedLLM" / "algorithm" / "mf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class Fit:
    models: list[str]
    net: object

    def predict(self, vectors) -> list[dict[str, float]]:
        import torch

        with torch.no_grad():
            q = self.net.text_proj(torch.tensor(np.asarray(vectors), dtype=torch.float32))
            out = []
            for row in q:
                p = self.net.P.weight  # (models, dim): TextMF.forward in test mode, for every model at once
                probs = torch.softmax(self.net.classifier(p * row), dim=1)[:, 1]
                out.append(dict(zip(self.models, probs.tolist(), strict=True)))
        return out


def _pairs(items, prompt_index, models):
    rows = [(models.index(m), prompt_index[it["query"]], int(it["scores"][m] >= SOLVED)) for it in items for m in models]
    return np.array(rows, dtype=np.int64)


def _train(items, embeddings, epochs, seed=42, on_epoch=None) -> Fit:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    models = sorted(items[0]["scores"])
    queries = sorted({it["query"] for it in items})
    prompt_index = {q: i for i, q in enumerate(queries)}
    table = torch.tensor(np.stack([embeddings[q] for q in queries]), dtype=torch.float32)
    net = _mf_module().TextMF(question_embeddings=table, model_embedding_dim=DIM, alpha=NOISE, num_models=len(models), num_prompts=len(queries), text_dim=table.shape[1])
    pairs = torch.tensor(_pairs(items, prompt_index, models))
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    fit = Fit(models=models, net=net)
    for epoch in range(1, epochs + 1):
        net.train()
        for idx in torch.randperm(len(pairs), generator=gen).split(BATCH):
            b = pairs[idx]
            opt.zero_grad()
            loss_fn(net(b[:, 0], b[:, 1]), b[:, 2]).backward()
            opt.step()
        net.eval()
        if on_epoch:
            on_epoch(epoch, fit)
    return fit


def fit(items: list[dict], embeddings: dict, epochs: int) -> Fit:
    return _train(items, embeddings, epochs)


def _is_val(prompt_hash: str) -> bool:
    return int(hashlib.sha256(f"embedllm-val/{prompt_hash}".encode()).hexdigest()[:8], 16) % 10 == 0


def choose_epochs(items: list[dict], embeddings: dict, candidates=(8, 16, 32, 64, 128, 256, 512)) -> tuple[int, dict[int, float]]:
    """The epoch count with the best correctness accuracy on 10% of tune prompts (fewest epochs on a tie)."""
    val = [it for it in items if _is_val(it["prompt_hash"])]
    train = [it for it in items if not _is_val(it["prompt_hash"])]
    curve: dict[int, float] = {}

    def score(epoch, f):
        if epoch in candidates:
            preds = f.predict([embeddings[it["query"]] for it in val])
            hits = [int((p[m] >= 0.5) == (it["scores"][m] >= SOLVED)) for it, p in zip(val, preds, strict=True) for m in f.models]
            curve[epoch] = sum(hits) / len(hits)

    _train(train, embeddings, max(candidates), on_epoch=score)
    best = max(sorted(curve), key=lambda e: curve[e])
    return best, curve


def expected_costs(items: list[dict]) -> dict[str, float]:
    models = sorted(items[0]["costs"])
    return {m: sum(it["costs"][m] for it in items) / len(items) for m in models}


def pick(p: dict[str, float], expected_cost: dict[str, float], tau: float) -> str:
    ok = [m for m in p if p[m] >= tau]
    return min(ok, key=lambda m: (expected_cost[m], m)) if ok else max(p, key=lambda m: (p[m], m))
