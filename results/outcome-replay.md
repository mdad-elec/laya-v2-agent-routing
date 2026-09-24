# v2-outcome on answer quality: the replay, and why it loses

Both R4 variants, replayed over the same 60 held pilot asks and the graded cell answers (`factorial.py`).
v2-outcome's replay was already in `factorial.md`; this adds the paired comparison with v2-policy and the cause.

| decider | cell menu quality | vs v2-policy, 90% CI | cost p50 | models-only ladder |
|---|---|---|---|---|
| v2-policy | **0.944** | — | 218 µUSD | 0.838 |
| perfect policy labels | 0.934 | [−0.022, +0.009] | 184 µUSD | 0.855 |
| judge | 0.887 | [−0.100, −0.016] | 220 µUSD | 0.813 |
| v2-outcome | 0.876 | **[−0.121, −0.016]** | 63 µUSD | 0.838 |
| best graded cell per ask | 0.998 | — | 53 µUSD | — |

v2-outcome is significantly worse on quality on the cell menu, and ties on the ladder. It is cheaper because it
sends more asks to the free local seat: 33 of 60 to `small`, against 27.

**Where the quality goes.** Medium asks. v2-outcome sends 8 of the held medium asks to `small`, against v2-policy's 3.
Quality on medium asks falls from 0.993 to 0.793.

**Why.** Its training target blends the policy label 50/50 with a measured target. The measured part puts 0.6 on the
cheapest band whose cell scored ≥ 0.8 on that ask. The local 27B with thinking off passed many medium tune asks, so
the blended targets for medium asks carry 0.25 of their mass on `small`, against 0.05 in v2-policy's. The model
learned that medium-looking asks are often fine on the small seat. But whether the local model passes a given
medium ask is not predictable from the text (it scores 0.67 on medium asks overall), so on held asks the shift costs
answers.

**What it says for combining policy and outcomes.** Each measured target rests on one graded answer per cell, and
"the cheapest band that passed once" is an optimistic statistic. The gap between perfect labels (0.934) and the best
cell per ask (0.998) is real. But closing it needs outcome targets that are repeated, or averaged over similar asks,
before a router can learn from them.
