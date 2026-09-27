"""Laya-g1's training examples (PREREG-g1 Amendments 1-3): public sources, tune split only.

One example is (session digest, cell, target in [0, 1]):
- **LLMRouterBench** @0e5af1b8: single-turn items × its 39 models (`openrouter` is a reference router,
  never a cell). The target is the item's score.
- **BFCL-Result** @cadd76a8: multi-turn tool episodes × 109 models. The digest is the first user turn
  plus the tools offered; the target is the episode's pass.
- **TwinRouterBench** @7cbb0dea: agent-step prefixes × its 4 pool models. A row labelled "cheapest
  sufficient tier t" means the pool model at tier ≥ t handles the step and the ones below do not.

Nothing held trains: split_of(prompt hash) for single prompts and BFCL episodes, and the trajectory
split for TwinRouterBench. No prompt containing an evaluation question trains
(labels.leakage.build_eval_index). The measured block of a cell's descriptor (its per-domain tune
accuracy) is computed from tune rows only.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from labels.splits import probe_trainable
from router.digest import build, render
from suites.base import CACHE, split_of

REFERENCE_ROUTERS = {"openrouter"}
TIERS = ("low", "mid", "mid_high", "high")
DOMAIN_OF = {
    **dict.fromkeys(("aime", "livemathbench", "math500", "mathbench"), "math"),
    **dict.fromkeys(("humaneval", "mbpp", "livecodebench", "swe-bench"), "code"),
    **dict.fromkeys(("mmlupro", "simpleqa", "medqa", "hle", "arcc", "finqa"), "knowledge"),
    **dict.fromkeys(("bbh", "korbench", "kandk", "winogrande", "arc-agi"), "reasoning"),
    **dict.fromkeys(("arenahard", "arenahard_coding", "arenahard_creative_writing", "arenahard_math"), "chat"),
    **dict.fromkeys(("meld", "emorynlp"), "emotion"),
    **dict.fromkeys(("tau2", "bfcl_multi_turn_base"), "tools_multiturn"),
    "twinrouterbench": "agentic",
}
DOMAINS = ("math", "code", "knowledge", "reasoning", "chat", "emotion", "tools_multiturn", "agentic")
SOURCE_WEIGHT = {"llmrouterbench": 0.5, "bfcl": 0.25, "twinrouterbench": 0.25}  # each source's share of the loss


def single_turn_examples(rows: list[dict], prompts: dict[str, str], index, digests: dict | None = None) -> list[dict]:
    clean: dict[str, bool] = {}
    out = []
    for r in rows:
        h = r["prompt_hash"]
        if r["model"] in REFERENCE_ROUTERS or split_of(h) != "tune":
            continue
        if h not in clean:
            clean[h] = not index.hits(prompts[h])
        if not clean[h]:
            continue
        if digests is not None and h not in digests:
            digests[h] = render(build([{"role": "user", "content": prompts[h]}]))
        out.append({"source": "llmrouterbench", "dataset": r["dataset"], "native_id": r["native_id"], "prompt_hash": h, "text": h,
                    "model": r["model"], "target": min(1.0, max(0.0, float(r["score"])))})
    return out


def tier_outcomes(row: dict, tier_map: dict[str, str]) -> list[dict]:
    return [{"source": "twinrouterbench", "dataset": "twinrouterbench", "native_id": row["id"], "text": row["id"], "model": tier_map[t],
             "target": 1.0 if i >= row["target_tier_id"] else 0.0} for i, t in enumerate(TIERS)]


def for_checkpoint(examples: list[dict], probe: bool) -> list[dict]:
    return [e for e in examples if probe_trainable(e["model"])] if probe else list(examples)


def measured_blocks(rows: list[dict], is_tune) -> dict[str, dict[str, float]]:
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        domain = DOMAIN_OF.get(r["dataset"])
        if domain and is_tune(r) and r["model"] not in REFERENCE_ROUTERS:
            acc[r["model"]][domain].append(min(1.0, max(0.0, float(r.get("score", r.get("target", 0.0))))))
    return {m: {d: sum(v) / len(v) for d, v in by.items()} for m, by in acc.items()}


def _bfcl_examples(index, digests: dict) -> tuple[list[dict], list[dict]]:
    from labels.bfcl_result import load_all
    from labels.llmrouterbench import prompt_hash
    from suites.bfcl.suite import BFCL

    items = {i.native_id: i for i in BFCL().load()}
    rows, _ = load_all()
    out, measured_rows = [], []
    for r in rows:
        item = items[r["native_id"]]
        text = item.meta["source_text"]
        h = prompt_hash(text)
        target = 1.0 if r["pass"] else 0.0
        if split_of(h) != "tune" or index.hits(text):
            continue
        if h not in digests:
            tools = [{"type": "function", "function": {"name": f["name"]}} for f in item.gold["functions"]]
            digests[h] = render(build([{"role": "user", "content": text}], tools=tools))
        out.append({"source": "bfcl", "dataset": "bfcl_multi_turn_base", "native_id": r["native_id"], "prompt_hash": h, "text": h, "model": r["model"], "target": target})
        measured_rows.append({"dataset": "bfcl_multi_turn_base", "model": r["model"], "score": target, "prompt_hash": h})
    return out, measured_rows


def _trb_examples(digests: dict) -> tuple[list[dict], list[dict]]:
    from labels import twinrouterbench as trb

    tier_map = json.loads((Path(__file__).resolve().parents[1] / "train" / "twinrouterbench_pool.json").read_text())["tier_to_model"]
    out, measured_rows = [], []
    for row in trb.load():
        if trb.split(row) != "tune":
            continue
        digests[row["id"]] = render(build(trb.router_view(row)["messages"]))
        ex = tier_outcomes(row, tier_map)
        out += ex
        measured_rows += [{"dataset": "twinrouterbench", "model": e["model"], "score": e["target"]} for e in ex]
    return out, measured_rows


def build_all(out_dir: Path) -> dict:
    from labels import routerarena as ra
    from labels import twinrouterbench as trb
    from labels.gate_s3 import prereg_sha
    from labels.leakage import build_eval_index
    from labels.leakage_report import suite_items

    base = CACHE.parent / "public" / "llmrouterbench"
    prompts = json.loads((base / "prompts.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (base / "outcomes.jsonl").open(encoding="utf-8")]
    index = build_eval_index(ra.questions(), trb.load(), suite_items())
    digests: dict[str, str] = {}
    examples = single_turn_examples(rows, prompts, index, digests)
    bfcl, bfcl_measured = _bfcl_examples(index, digests)
    trb_ex, trb_measured = _trb_examples(digests)
    examples += bfcl + trb_ex
    measured = measured_blocks(rows + bfcl_measured + trb_measured, is_tune=lambda r: r["dataset"] == "twinrouterbench" or split_of(r["prompt_hash"]) == "tune")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "examples.jsonl").open("w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(e) + "\n")
    (out_dir / "digests.json").write_text(json.dumps(digests), encoding="utf-8")
    (out_dir / "measured.json").write_text(json.dumps(measured, indent=1), encoding="utf-8")
    counts = defaultdict(int)
    for e in examples:
        counts[e["source"]] += 1
    manifest = {"prereg_sha": prereg_sha(), "examples": dict(counts), "texts": len(digests), "models": len({e["model"] for e in examples}),
                "measured_models": len(measured)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(CACHE.parent / "train" / "g1-v1"))
    print(json.dumps(build_all(Path(ap.parse_args().out)), indent=1))


if __name__ == "__main__":
    main()
