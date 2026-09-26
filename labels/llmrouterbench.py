"""LLMRouterBench (NPULH/LLMRouterBench) as public item-level outcomes: 42 models x 27 datasets.

Pinned: HF revision 0e5af1b8, bench-release.tar.gz sha256 b79f8cde... (1.28 GB). Each result file
is one model on one dataset, with a `records` list carrying per item its index, original query,
score, cost and tokens. The archive is streamed, never extracted, and the output keeps text and
labels apart:
- outcome rows (source, dataset, native id, model, score, cost, tokens, prompt hash): no text;
- a prompt table (prompt hash -> original query): cached locally, never committed. It is Laya's
  training input, and it is also how a label item is matched against our suites' held items.

Licence: the project README shows an MIT badge, but the repo has no LICENSE file and the HF card
states none (checked 2026-09-26). Internal evaluation now; the authors confirm before a published
checkpoint is trained on it (gate O3). Only scores become labels, never a model's output text.

GPQA is excluded: its authors ask that items not be revealed, and a published checkpoint trained on
the text could carry them.
"""
from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path

REVISION = "0e5af1b84bf73437a01a1849c0f1d2468baa93fc"
ARCHIVE_SHA256 = "b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1"
SOURCE = f"llmrouterbench@{REVISION[:8]}"
EXCLUDED_DATASETS = {"gpqa": "its authors ask that items not be revealed; a published checkpoint trained on them could"}


def norm_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def prompt_hash(text: str) -> str:
    return hashlib.sha256(norm_prompt(text).encode()).hexdigest()[:16]


def _num(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_archive(path: Path) -> tuple[list[dict], dict[str, str], dict]:
    rows: list[dict] = []
    prompts: dict[str, str] = {}
    report = {"files": 0, "excluded": {}, "unreadable_scores": 0, "unreadable_files": []}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith(".json")):
                continue
            parts = member.name.split("/")
            dataset = parts[1] if len(parts) > 2 else ""
            if dataset in EXCLUDED_DATASETS:
                report["excluded"][dataset] = EXCLUDED_DATASETS[dataset]
                continue
            try:
                payload = json.load(tar.extractfile(member))
            except (json.JSONDecodeError, UnicodeDecodeError):
                report["unreadable_files"].append(member.name)
                continue
            report["files"] += 1
            model = payload.get("model_name") or parts[2]
            for record in payload.get("records") or []:
                score = _num(record.get("score"))
                if score is None:
                    report["unreadable_scores"] += 1
                    continue
                query = record.get("origin_query") or record.get("prompt") or ""
                h = prompt_hash(query)
                prompts.setdefault(h, query)
                rows.append({"source": SOURCE, "dataset": payload.get("dataset_name") or dataset, "native_id": str(record.get("index")),
                             "model": model, "score": score, "cost": _num(record.get("cost")),
                             "input_tokens": _num(record.get("prompt_tokens")), "output_tokens": _num(record.get("completion_tokens")),
                             "prompt_hash": h})
    return rows, prompts, report
