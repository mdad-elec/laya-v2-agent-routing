# H7 — Laya-v2 against RouteLLM's BERT router

Pre-registered as amendment 3 of `scripts/laya/PREREG.md`, committed (332fc7f) before any RouteLLM router ran.

**Router.** `routellm/bert_gpt4_augmented` (Ong et al., arXiv 2406.18665), weights sha256 `e2a92240bad1…`, scored
with RouteLLM's own `calculate_strong_win_rate`. RouteLLM's other three released routers were **not run**. Its
matrix-factorisation and similarity-weighted routers need OpenAI's embedding API (no key here), and its causal-LLM
router is an 8B model with no GPU free for it. Nothing here speaks for them.

**Mapping.** Two thresholds on the score, fitted on the tune split for band accuracy: `t_lo` 0.65, `t_hi` 0.69.
Tune accuracy was 0.443 on 149 rows. The score barely separates this corpus's tiers: median 0.39 / 0.36 / 0.37
for small / medium / powerful on the borrowed rows. So the best fit sends 89% of held requests to `small`.

## Band decision (151 held rows, one read)

| rows | RouteLLM BERT | Laya-v2 | judge | Laya-v2 only right / RouteLLM only right | one-sided exact McNemar |
|---|---|---|---|---|---|
| all 151 | 0.430 | **0.801** | 0.662 | 70 / 14 | p = 2e-10 |
| 91 borrowed tier rows | 0.352 | 0.857 | 0.637 | 48 / 2 | p = 1e-12 |
| 60 pilot asks | 0.550 | 0.717 | 0.700 | 22 / 12 | p = 0.061 |

RouteLLM underspends on 77 of 151 and overspends on 9. **H7 on band accuracy: met** (+0.371, p < 0.05). On the
pilot asks alone the margin is +0.167 but not shown apart at n = 60.

## Answer quality (60 held pilot asks, graded cell replay)

| menu | RouteLLM BERT | Laya-v2 | Laya-v2 − RouteLLM, 90% CI | cost p50 per turn |
|---|---|---|---|---|
| arm C's cell menu | 0.842 | **0.944** | [+0.053, +0.157] | 148 vs 218 µUSD |
| models-only ladder | 0.842 | 0.838 | [−0.053, +0.037] | 106 vs 41 µUSD |

On the cell menu Laya-v2's answers are better, and the interval excludes zero. On the ladder the two are a tie on
quality, and Laya-v2 is 2.6× cheaper per median turn.

## RouterBench, RouteLLM's own setting (GPT-4 against Mixtral-8x7B, 2,451 test prompts)

Each router sends the prompts it scores highest to GPT-4 first. AIQ is the area under the hull of its cost-quality
curve, between the two models' costs. The straight line between the two models is the no-information baseline.

| task | n | line | RouteLLM BERT | Laya-v2 | RouteLLM − Laya-v2, 90% CI |
|---|---|---|---|---|---|
| arc | 441 | 0.921 | 0.935 | 0.927 | [−0.001, +0.016] |
| gsm8k | 500 | 0.585 | 0.592 | 0.590 | [−0.002, +0.008] |
| hellaswag | 500 | 0.639 | 0.648 | 0.646 | [−0.014, +0.018] |
| mbpp | 129 | 0.636 | 0.652 | 0.663 | [−0.030, +0.012] |
| mmlu | 500 | 0.719 | **0.746** | 0.719 | [+0.015, +0.038] |
| winogrande | 381 | 0.675 | 0.683 | 0.678 | [−0.010, +0.020] |
| mean over tasks | | | | | +0.006 [+0.0001, +0.012] |

On its home ground RouteLLM is ahead on MMLU, and ties on the other five tasks. Neither router is far above the
line on any task: both recover little of GPT-4's lead here.

## What this says

- **On the decision this programme needs, Laya-v2 beats RouteLLM's BERT router by a wide margin.** That decision is
  which of three price bands this estate's policy wants. RouteLLM's score was trained on Chatbot Arena preferences
  between GPT-4 and Mixtral, and it barely tracks our tiers.
- **On RouteLLM's own question, it is slightly better than Laya-v2.** That question is whether GPT-4 beats Mixtral on
  this prompt. The average gap is 0.006 AIQ, and it comes mostly from MMLU.
- **Each router is best at the question it was trained on.** Neither generalises to the other's.
