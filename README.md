# Laya-v2: routing an AI agent's turn with a System-One model

Every turn of an agent has to go to some model at some reasoning effort. The usual router asks a large LLM for a
JSON verdict before each turn. This repo tests whether an open **System-One** decision model can make that call
better. The model is [Laya](https://huggingface.co/convaiinnovations/laya), a 421M ModernBERT-large encoder that
answers a typed question in one forward pass and generates no tokens.

It holds the technical report, the recorded side-by-side video, every evaluation behind the numbers, the public
dataset, a runnable benchmark package, and the fine-tuned checkpoint with the code and data that trained it.

## Findings

| held-out measure | LLM judge (27B) | Laya-v2 (fine-tuned) |
|---|---|---|
| band accuracy, 151 requests | 0.662 | **0.801** (exact McNemar p = 0.0027) |
| of which: 91 borrowed tier-labelled requests | 0.637 | 0.857 |
| of which: 60 of our own checkable asks | 0.700 | 0.717 (not shown apart) |
| answer quality over 23 measured model × effort cells, 60 asks | 0.887 | **0.944**, 90% CI of the gain [+0.016, +0.100] |
| median cost per turn | 220 µUSD | 218 µUSD |
| decision tokens generated | 57 | 0 |

- **Zero-shot Laya lost to the judge.** Three defects in our own harness hid part of its ability: the wrong
  checkpoint was served, the brain saw a wrapped state instead of the bare request, and the options reached the
  wire in alphabetical order. Fixing them lifted it from 0.563 to 0.629, still below the judge.
- **A short RLCD fine-tune passes the judge.** It trained on 119 policy-labelled requests plus 708 filtered
  paraphrases, at about 80 s an epoch on one RTX 2080 Ti.
- **The band-accuracy margin comes from the borrowed requests.** On our own asks the two tie. Laya-v2 still wins on
  quality there, because the judge picks too cheap a tier more often: 15 of 60 against 9.
- **Reasoning effort barely mattered for correctness.** The lowest effort was within 0.05 of a model's best on
  112 of 120 asks, so the effort question was dropped.
- **It does not transfer to RouterBench.** Neither the zero-shot nor the fine-tuned Laya reaches the single-model
  hull on any of six tasks. Laya-v2 learned a spending policy, and RouterBench labels which model solved the prompt.
- **Against vLLM Semantic Router (pre-registered, amendment 4),** which ranks 6th of 32 on the
  [RouterArena](https://github.com/RouteWorks/RouterArena) leaderboard and is the highest-ranked entry with open
  code and weights, Laya-v2 wins the band decision: 0.801 against 0.530 (p = 6.5e-9). It also wins on our own asks
  alone: 0.717 against 0.450 (p = 0.0008). Answer quality on the cell menu is 0.944 against 0.888, CI
  [+0.017, +0.106], at 2.6× lower cost, and the models-only ladder is a tie. Its released intent classifier was run
  the way RouterArena's adapter uses it. The leaderboard entry's unpublished extra signals were not reproduced
  (`results/vllmsr.md`).
- **Against RouteLLM's released BERT router (pre-registered, amendment 3),** a 2024 baseline that RouterArena ranks
  31st of 32, Laya-v2 wins the decision this repo
  is about. Band accuracy is 0.801 against 0.430 (p = 2e-10). Answer quality on the cell menu is 0.944 against
  0.842, CI [+0.053, +0.157], and the models-only ladder is a tie at 2.6× lower cost. On RouteLLM's own
  RouterBench pair, GPT-4 against Mixtral, RouteLLM is ahead by 0.006 AIQ on average, and only MMLU is clearly
  apart. Each router wins the question it was trained on. RouteLLM's other three routers were not run: two need
  OpenAI embeddings and one needs an 8B model on a GPU (`results/routellm.md`).
- **In one long conversation, a session-hold rule undid Laya's choice.** The recording caught Laya naming `powerful`
  while the gateway kept a cheap cell it had reached earlier.

Every hypothesis was pre-registered before measurement (`results/PREREG.md`). Each held-out split was read once per
rung (`results/heldout-ledger.md`). The verdict is `results/VERDICT.md`.

## What is here

| path | contents |
|---|---|
| `report/report.md`, `report/index.html` | the technical report; the HTML is one self-contained page with the figures inlined |
| `report/figures/` | four SVG figures, recomputed from the held-out result files |
| `video/laya-v2-vs-judge-workbench.mp4` | the side-by-side take: today's judge router against Laya-v2 over effort cells, real seats, one conversation per arm |
| `results/` | pre-registration, verdict, per-rung reports, factorial, RouterBench leg, training records |
| `dataset/` | the public dataset: cases, per-cell outcomes, router decisions, frozen splits, provenance, `manifest.sha256` |
| `benchmark/` | a runnable benchmark package (117 cases, `run.py`, `audit.py`, frozen `results/v1` for Laya root, Laya-v2 and the judge) |
| `finetune/` | the training, dataset-building, paraphrase, temperature-fit and serving code, plus the 708 kept paraphrases and the 6 dropped by the held-out filters |
| `model/` | the fine-tuned checkpoint's card, config, tokenizer and manifest; the weights are a release asset |
| `eval/` | the measurement instruments: statistics, brain bench, menu replay, factorial, judge adapter, RouterBench leg |

The report cites source paths from the private platform repo, such as `scripts/ab/results/laya-v2/`. In this repo
those files live under `results/`, and the scripts under `eval/` and `finetune/`.

## The fine-tuned checkpoint

Download `model.safetensors` from this repo's **v1 release** into `model/`. Its sha256 is
`6098ad7664ef42ddcbdbd3d3b52da500f724ad8d7038762d6c2b5439fac17bb5`. Then serve or benchmark it:

```bash
sha256sum model/model.safetensors
PYTHONPATH=<laya repo> python benchmark/run.py --backend laya --checkpoint model --device cpu --output /tmp/run
python benchmark/audit.py --run-dir /tmp/run
```

Read Laya-v2's benchmark numbers on the **held** split only. It was trained on the tune split.

## About the video

The take ran 12 held-out asks. Two of them (m07 and h07) name the company's own ERP, and their rows are private in
the corpus, so they were cut from the film. After the cut, the conversation history on screen still showed one of
their answers. That history area is therefore masked from the cut to the end, and a label on screen says so. The
routing captions come from each turn's ledger row and are unchanged. On-screen seat names such as
`spark-27b/qwen3.8-27b` are the platform's internal names for the local Qwen 27B, which the dataset calls `local-27b`.

## Limits

- The policy labels come from one annotator. A blind pass puts the ceiling near 0.82 on the borrowed rows.
- The answer-grading panel has not yet been calibrated against human grades.
- The held sets have 151 and 60 rows, so each number carries roughly ±0.06 to ±0.08.
- Everything was measured on one estate's model menu, in English, at list prices.
- Two held pilot asks are private. A reproduction from `dataset/` therefore has 149 H1 rows and 58 H2 asks.

## Credits and license

180 of the tier-labelled requests come from [glukicov/laya_router](https://github.com/glukicov/laya_router)
(Apache-2.0); see `ATTRIBUTION.md`. RouterBench data belongs to its authors and is not redistributed here; only
derived results are. Everything else is Apache-2.0 (`LICENSE`).

Related work: RouterBench ([arXiv 2403.12031](https://arxiv.org/abs/2403.12031)), RouteLLM
([arXiv 2406.18665](https://arxiv.org/abs/2406.18665)), FrugalGPT ([arXiv 2305.05176](https://arxiv.org/abs/2305.05176)),
Hybrid LLM ([arXiv 2404.14618](https://arxiv.org/abs/2404.14618)).
