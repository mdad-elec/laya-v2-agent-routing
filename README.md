# Laya routing

A general LLM router: pick the cheapest model *and reasoning effort* that will solve the turn, from a
measured understanding of every model, and the whole session, not just the last message.

This repository holds two things:

| | What | Status |
|---|---|---|
| [`studies/v2/`](studies/v2/) | **Laya-v2**: a pre-registered study of a fine-tuned Laya System-One router against an LLM judge, RouteLLM and vLLM Semantic Router. Held-out band accuracy **0.801 vs the judge's 0.662** (McNemar p = 0.0027). Frozen at tag [`v2-study-final`](../../tree/v2-study-final). | Done (published) |
| everything else | **Laya-g1**: a general router trained on **public data only**, reading the session, choosing from measured outcomes. Held read once: **F1 met** (TwinRouterBench, +9.6 Combined over SR-KNN, CI90 [5.1, 14.1]); **F3 not met** by one check (LLMRouterBench: better than Avengers-Pro and EmbedLLM on accuracy, and on the hard slice vs EmbedLLM, but not significantly vs Avengers-Pro; CI lower bound −0.0006); **F2 (RouterArena) not run**. See [`VERDICT-g1.md`](VERDICT-g1.md). | Read, published as is |

## Laya-g1, in one paragraph

Laya-v2 won on the asks it was tuned on and failed RouterBench's general tasks. g1's first recipe, a
fine-tuned encoder predicting per-item P(solve), lost to a plain per-model × dataset success table
(calibration BCE 0.571 vs 0.530), so it was retired before any held read. Recipe 2 is three layers:
- **A System-One brain answers typed questions over a session digest** (goal, prior turns, tools,
  tool rounds, errors, tests, context). The digest never names a model or seat.
- **A measured table gives P(solve)** per (answers, model) from public outcomes.
- **A policy with continuity rules in code** picks the cheapest model above τ.

On F1 a logistic-regression brain on Qwen3 embeddings already answers the task at 99.8%, so the Laya
fine-tune was not needed there. **What it does not show:** that session features beat knowing the task.
On TwinRouterBench a task-only table scores the same, and the gain over SR-KNN is concentrated on
SWE-bench. Every claim is pre-registered (`PREREG-g1.md`, Amendments 1–10, including the reads and
the one bar restatement), and every held read is logged in `results/g1/heldout-ledger.jsonl`.

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
