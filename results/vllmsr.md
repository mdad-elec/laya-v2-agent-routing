# H8 — Laya-v2 against vLLM Semantic Router

Pre-registered as amendment 4 of `scripts/laya/PREREG.md` (7ecca79), committed before the router ran on these
corpora. vLLM Semantic Router ranks 6th of 32 on the RouterArena leaderboard. It is the highest-ranked entry there
with open code and weights.

**What was run.** RouterArena's adapter for this router asks its intent endpoint for a category, then looks the
category up in a category-to-model table. Here the released classifier behind that endpoint,
`llm-semantic-router/mmbert32k-intent-classifier-merged` (weights sha256 `f53cd591a3d6…`), is run directly. The
category-to-band table is fitted on tune, which favours the router. The leaderboard entry also used prompt-structure
and projection signals whose configuration is not published. Those were **not reproduced**.

**The fitted table.** business, psychology, physics and law go to `powerful`. Computer science goes to `medium`.
Everything else goes to `small`. Tune accuracy is 0.544.

## Band decision (151 held rows, one read)

| rows | vLLM-SR | Laya-v2 | judge | Laya-v2 only right / vLLM-SR only right | one-sided exact McNemar |
|---|---|---|---|---|---|
| all 151 | 0.530 | **0.801** | 0.662 | 48 / 7 | p = 6.5e-9 |
| 91 borrowed tier rows | 0.582 | 0.857 | 0.637 | 28 / 3 | p = 2.3e-6 |
| 60 pilot asks | 0.450 | 0.717 | 0.700 | 20 / 4 | p = 0.0008 |

**H8 on band accuracy: met**, and met on this estate's own asks alone.

## Answer quality (60 held pilot asks, graded cell replay)

| menu | vLLM-SR | Laya-v2 | Laya-v2 − vLLM-SR, 90% CI | cost p50 per turn |
|---|---|---|---|---|
| arm C's cell menu | 0.888 | **0.944** | [+0.017, +0.106] | 576 vs 218 µUSD |
| models-only ladder | 0.855 | 0.838 | [−0.078, +0.041] | 72 vs 41 µUSD |

On the cell menu Laya-v2 is better and 2.6× cheaper. On the ladder the two tie on quality, and Laya-v2 is cheaper.

## What this says

A subject category is weak evidence of how hard an ask is: "What is 17 × 23?" and a proof are both `math`. This
router was built to route by domain, and it does that job. On the job this programme measures, choosing the
cheapest tier that answers well, the fine-tuned Laya decides better than a category table, whichever subject the
ask is in.
