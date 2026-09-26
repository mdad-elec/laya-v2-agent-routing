# Laya routing

A general LLM router: pick the cheapest model *and reasoning effort* that will solve the turn, from a
measured understanding of every model, and the whole session, not just the last message.

This repository holds two things:

| | What | Status |
|---|---|---|
| [`studies/v2/`](studies/v2/) | **Laya-v2**: a pre-registered study of a fine-tuned Laya System-One router against an LLM judge, RouteLLM and vLLM Semantic Router. Held-out band accuracy **0.801 vs the judge's 0.662** (McNemar p = 0.0027). Frozen at tag [`v2-study-final`](../../tree/v2-study-final). | Done (published) |
| everything else | **Laya-g1**: a general routing checkpoint for all AI work, trained on **public data only**, seeing session history and tools, and choosing cells from a measured **Model Atlas**. | **In progress** |

## Laya-g1, in one paragraph

Laya-v2 won on the asks it was tuned on, and failed RouterBench's general tasks (0.01–0.37 band accuracy).
It also read only the latest user message, so a short follow-up to hard work ("yes, go ahead") looked easy.
g1 fixes both:
- **General labels.** Every catalogue model is measured at every reasoning effort on eight public,
  machine-checked suites (below). The label for an item is "the cheapest cell that solves it", taken
  from those measurements.
- **Session-aware input.** Laya is fine-tuned on a compact session digest: the request, prior turns,
  the tools offered, and what the tools did. The digest never includes model or seat names; including
  them is what collapsed v2's accuracy from 0.711 to 0.342.

Claims are pre-registered, and are read once on held-out data, RouterArena, LLMRouterBench and RouterBench.

## The pieces

| Folder | What it is | Checked by |
|---|---|---|
| [`atlas/`](atlas/) | The Model Atlas: one profile per (model, effort, domain), either measured by us or imported with attribution. Imports so far: [LMArena](https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset) and [Epoch AI](https://epoch.ai/benchmarks) (both CC BY 4.0), covering 46 of 68 cells as priors. | schema tests; every row validated |
| [`suites/`](suites/) | Eight public suites, each fetched at a pinned revision. Only item ids are committed. | a golden test per suite |
| [`harness/`](harness/) | The measurement runner (resumable, one lane per seat, loud on seat limits) and its backends. | unit tests with fake backends |

The suites:

| Domain | Suite | Its golden check |
|---|---|---|
| code | Aider polyglot, pass@1 (Python/Rust/JS), in a network-less Docker sandbox | all 111 references pass; all stubs fail |
| sql | BIRD Mini-Dev, execution accuracy | all 498 gold queries pass |
| fin_table | TAT-QA arithmetic | pre-registered numeric tolerance |
| instruct | IFEval, prompt-level strict, using the benchmark's own checker | GPT-4's shipped responses reproduce the paper's 76.89% |
| knowledge | MMLU-Pro | standard answer extraction |
| long_ctx | RULER-style synthetic tasks at 32k tokens | lengths and answers verified |
| tools_multiturn | BFCL v4 multi-turn, using BFCL's own APIs and checker | all 200 ground-truth episodes replay valid |
| chat | Arena-Hard-Auto v2, pairwise against its baselines, positions swapped | judge prompts byte-identical to Arena-Hard's |

```bash
python3 -m venv ~/venvs/laya-g1 && ~/venvs/laya-g1/bin/pip install -r requirements.txt
~/venvs/laya-g1/bin/python -m unittest discover -s suites/tests -t .
docker build -t laya-g1-polyglot:2 harness/sandbox/polyglot   # the code suite's sandbox
```

Licences are per source, in each suite's docstring and in [`ATTRIBUTION.md`](ATTRIBUTION.md). No
benchmark item text is committed.
