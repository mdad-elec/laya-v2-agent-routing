# Laya-v2 — verdict against the pre-registration

`scripts/laya/PREREG.md` (with amendments 1 and 2). Held-out rows are the frozen held halves: 151 headline rows for
H1 (91 borrowed tier rows + 60 pilot asks; the 30 unaudited DSS rows apart), 60 pilot asks for H2. Two of the 60 pilot
asks (m07, h07) are private, so a reproduction from the public dataset has 149 H1 rows and 58 H2 asks.
Adopted decider: **laya-v2-policy** (root checkpoint, RLCD fine-tune on tune + filtered paraphrases, bare request,
example-led wording, order powerful/medium/small; weights sha256 `6098ad7664ef42dd…`).

| hypothesis | pre-registered bar | measured | verdict |
|---|---|---|---|
| **H1** band decision | Laya-v2 − judge ≥ 0.05 and one-sided exact McNemar p < 0.05 | 0.801 vs 0.662: **+0.139**; 37 vs 16 discordant, **p = 0.0027** | **met** |
| **H2** quality, cells menu (arm C's, `menus/cells.json`) | ≥ judge − 0.02, 90% CI lower bound > −0.05 | 0.944 vs 0.887: **+0.057, CI [+0.016, +0.100]** | **met; superior (CI excludes 0)** |
| **H2** quality, models-only ladder (`menus/ladder.json`) | same | 0.838 vs 0.813: **+0.025, CI [+0.005, +0.046]** | **met; superior** |
| **H3** decision latency | p95 ≤ 250 ms on the Spark, across the tailnet | not measured on the Spark for the fine-tune (the Spark sidecar was not restarted — gate G5). On this box's CPU under contention: p50 585 ms, p95 817 ms. Same architecture as Ship 0's 62 / 192 ms tailnet measurement | **not measured** |
| **H4** cost | 0 generated tokens vs the judge's | Laya: 0 output tokens, ~120 input. Judge: 57 output + 180 input tokens per decision, p50 2.3 s / p95 3.8 s on local-27b | **met** |
| **H5** off-local rate | descriptive | cells menu: v2-policy 33 of 60 turns off the local seat, judge 31, labels 30 | reported |
| **H7** against RouteLLM's BERT router (amendment 3) | H1's bar on the same 151 rows; quality CI on both menus | band accuracy 0.801 vs **0.430**, 70 vs 14 discordant, p = 2e-10. Quality: cell menu 0.944 vs 0.842, CI [+0.053, +0.157]; ladder 0.838 vs 0.842, a tie at 2.6× lower cost. RouterBench GPT-4/Mixtral: RouteLLM ahead by 0.006 AIQ on average, only MMLU apart (`routellm.md`) | **met** on this estate's decision; RouteLLM slightly better on its own |
| **H8** against vLLM Semantic Router, RouterArena #6 (amendment 4) | H1's bar on the same 151 rows; quality CI on both menus | band accuracy 0.801 vs **0.530**, 48 vs 7, p = 6.5e-9 (pilot alone 0.717 vs 0.450, p = 0.0008). Quality: cell menu 0.944 vs 0.888, CI [+0.017, +0.106], at 218 vs 576 µUSD; ladder a tie (`vllmsr.md`) | **met** |
| **H6** effort question | beat the best constant by ≥ 0.10 (amendment 1) | constant `minimal` 0.917 on tune; bar unreachable | **dropped** |

**By rung** (held band accuracy): as shipped 0.563 → rung 1 (checkpoint, bare state, measured order) 0.629 →
rung 2 no improvement → rung 3 no improvement (ECE 0.112 > 0.08) → **rung 4 0.801**. The zero-shot fixes recover a
third of the distance; the fine-tune is what beats the judge — as upstream's own card predicts ("a fast base to
specialise, not a zero-shot decision engine").

**Where H1's margin lives.** Split by corpus, the held read is 0.857 against 0.637 on the 91 borrowed tier rows (26 vs 6 discordant, p = 0.0005), and 0.717 against 0.700 on the 60 pilot asks (11 vs 10, not shown apart). H1 was pre-registered on the pooled 151 and is met there; on this estate's own asks Laya-v2 ties the judge on band accuracy. It still wins on quality there (H2), because the judge's errors are mostly underspends (15 of 60, against Laya-v2's 9; overspends 3 against 8), and an underspend costs the answer.

**What this does not show:** that the labels are right beyond one annotator (ceiling ~0.82 on the borrowed rows) —
v2-policy's 0.801 sits near it; that the panel is human-calibrated (the audit slice is ungraded, gate G2); that the
result holds past this estate's menu, English, or 60/151 rows (±0.06–0.08); H3 on the Spark.
