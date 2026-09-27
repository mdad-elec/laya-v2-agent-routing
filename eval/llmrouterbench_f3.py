"""F3: LLMRouterBench's performance-cost setting on our split (PREREG-g1 Amendment 3).

The setting is its `config/baseline_config_performance_cost.yaml` @ c77cb050: 13 flagship models and
10 datasets, minus GPQA. An item is (dataset, native_id): on tau2, 278 tasks share 167 prompt texts,
so the prompt is not the item. The split still follows the prompt hash, so an identical prompt
never straddles tune and held. A model with no result on an item scores 0 at cost 0, which is the
benchmark's own `fill_missing_models_scores`, and the fill is recorded on the item.

The formulas are the paper's §6 (its repository ships no code for them):
- AvgAcc is macro over datasets.
- PerfGain = AvgAcc(θ*)/AvgAcc(ℬ) − 1.
- CostSave = 1 − Cost(θ†)/Cost(ℬ).
- ParetoDist is the mean L1 distance to the frontier of the union of all configurations, in
  min-max-normalised (AvgAcc, log Cost).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

from suites.base import CACHE, split_of

MODELS = ("claude-sonnet-4", "deepseek-v3-0324", "deepseek-v3.1-terminus", "deepseek-r1-0528", "gemini-2.5-flash", "gemini-2.5-pro", "gpt-5-chat", "gpt-5",
          "qwen3-235b-a22b-2507", "qwen3-235b-a22b-thinking-2507", "glm-4.6", "kimi-k2-0905", "intern-s1")
DATASETS = ("aime", "livemathbench", "hle", "livecodebench", "mmlupro", "swe-bench", "simpleqa", "tau2", "arenahard")
SOLVED = 0.5
SLICE_MAX_CORRECT = 3


def items() -> list[dict]:
    base = CACHE.parent / "public" / "llmrouterbench"
    prompts = json.loads((base / "prompts.json").read_text(encoding="utf-8"))
    by_item: dict[tuple, dict] = defaultdict(lambda: {"scores": {}, "costs": {}})
    with (base / "outcomes.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["model"] not in MODELS or r["dataset"] not in DATASETS:
                continue
            it = by_item[(r["dataset"], r["native_id"])]
            it["prompt_hash"] = r["prompt_hash"]
            it["scores"][r["model"]] = float(r["score"])
            it["costs"][r["model"]] = float(r["cost"] or 0.0)
    out = []
    for (dataset, native_id), it in sorted(by_item.items()):
        filled = [m for m in MODELS if m not in it["scores"]]
        for m in filled:
            it["scores"][m], it["costs"][m] = 0.0, 0.0
        out.append({"dataset": dataset, "native_id": native_id, "prompt_hash": it["prompt_hash"], "split": split_of(it["prompt_hash"]),
                    "query": prompts[it["prompt_hash"]], "scores": it["scores"], "costs": it["costs"], "filled": filled})
    return out


def _key(it: dict) -> tuple:
    return (it["dataset"], it["native_id"])


def avg_acc(items_: list[dict], choice: dict[tuple, str]) -> float:
    per: dict[str, list[float]] = defaultdict(list)
    for it in items_:
        per[it["dataset"]].append(it["scores"][choice[_key(it)]])
    return sum(sum(v) / len(v) for v in per.values()) / len(per)


def total_cost(items_: list[dict], choice: dict[tuple, str]) -> float:
    return sum(it["costs"][choice[_key(it)]] for it in items_)


def perf_gain(configs: dict[str, tuple[float, float]], router: list[str], best: str) -> float:
    return max(configs[c][0] for c in router) / configs[best][0] - 1


def cost_save(configs: dict[str, tuple[float, float]], router: list[str], best: str) -> float | None:
    reach = [configs[c][1] for c in router if configs[c][0] >= configs[best][0]]
    return 1 - min(reach) / configs[best][1] if reach else None


def pareto_dist(configs: dict[str, tuple[float, float]], router: list[str]) -> float:
    pts = {c: (acc, math.log(max(cost, 1e-12))) for c, (acc, cost) in configs.items()}
    accs, costs = [p[0] for p in pts.values()], [p[1] for p in pts.values()]

    def norm(p):
        a = (p[0] - min(accs)) / ((max(accs) - min(accs)) or 1)
        c = (p[1] - min(costs)) / ((max(costs) - min(costs)) or 1)
        return a, c

    frontier = [p for c, p in pts.items() if not any((q[0] >= p[0] and q[1] <= p[1]) and (q[0] > p[0] or q[1] < p[1]) for q in pts.values())]
    nf = [norm(p) for p in frontier]
    dists = [min(abs(n[0] - f[0]) + abs(n[1] - f[1]) for f in nf) for n in (norm(pts[c]) for c in router)]
    return sum(dists) / len(dists)


def recall_slice(items_: list[dict]) -> list[dict]:
    return [it for it in items_ if sum(s >= SOLVED for s in it["scores"].values()) <= SLICE_MAX_CORRECT]
