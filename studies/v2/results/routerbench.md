# RouterBench leg — Laya on a public routing benchmark

Data: withmartian/routerbench `routerbench_0shot.pkl` (36,497 rows, 11 models; not redistributed — only derived labels
and ids are in `routerbench/`). Six tasks with enough rows; models banded 4/4/3 by measured mean cost; menu per band =
its most accurate model (Mixtral-8x7B / Yi-34B / GPT-4); a prompt's label = the cheapest band whose menu model scores 1.
Test = 30% per task, up to 500 prompts sampled by label (2,451 total). A router's curve sweeps the low-confidence
upgrade over 9 thresholds; AIQ = area under its non-decreasing convex hull over the single models' cost range, zero
below its cheapest point. The Zero router is the hull of the eleven single models on the same test prompts.

| task | n test | zero-router AIQ | laya-root AIQ (band acc) | laya-v2-policy AIQ (band acc) | label-router (cost, quality) | oracle (cost, quality) |
|---|---|---|---|---|---|---|
| mmlu | 500 | 0.732 | 0.410 (0.45) | 0.342 (0.34) | (0.0004, 0.888) | (0.0001, 0.974) |
| hellaswag | 500 | 0.779 | 0.505 (0.35) | 0.585 (0.37) | (0.0005, 0.938) | (0.0002, 0.970) |
| gsm8k | 500 | 0.644 | 0.545 (0.02) | 0.579 (0.01) | (0.0084, 0.661) | (0.0001, 0.750) |
| arc | 441 | 0.927 | 0.319 (0.33) | 0.657 (0.34) | (0.0001, 0.984) | (0.0000, 0.991) |
| winogrande | 381 | 0.715 | 0.488 (0.45) | 0.651 (0.37) | (0.0002, 0.937) | (0.0000, 1.000) |
| mbpp | 129 | 0.701 | 0.502 (0.06) | 0.503 (0.21) | (0.0037, 0.791) | (0.0004, 0.884) |


**Pre-registered claim — "on or above the Zero-router hull on ≥ 3 of 6 tasks": not met, on any task, by either decider.**
Band accuracy against RouterBench's labels is near chance or below (GSM8K 0.01–0.02, where the label is almost always
`powerful`). Fine-tuning helps on four of six tasks (ARC 0.32 → 0.66 AIQ) but does not approach the hull. What this
says: Laya-v2 learned THIS estate's spending policy ("the cheapest tier that answers it well", as the corpus labels
it) — it is not a general "which model will solve this" predictor, and RouterBench's label is exactly that. A router
for RouterBench's task would be trained on its 70% (the paper's own predictive routers do); that is a separate
experiment, not attempted here. Paper numbers are not quoted beside these: our sample, bands and curve rule differ from
the paper's routers, so only the same-sample Zero router is a fair reference.
