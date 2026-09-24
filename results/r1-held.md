# Rung 1 — the one held-out read

Candidate named on tune before this read: **`laya-root / bare / example-led`, options `powerful, medium, small`**
(tune 0.711 on 149 rows, `r1-report.md`). The as-shipped baseline (`typed-decisions`, the router's wrapped
state, the alphabetical order the wire carried) and today's judge (`judge-held.jsonl`, `t_small` fitted on tune)
are read on the same rows; neither is a candidate, so their reads select nothing.

| decider | held band accuracy (151 rows, DSS apart) |
|---|---|
| Laya as shipped | 0.563 |
| **Laya, rung 1** | **0.629** |
| today's judge | 0.662 |

- Rung 1 vs as shipped: +0.066, exact McNemar 26 vs 16 discordant, one-sided p = 0.082.
- Rung 1 vs the judge: −0.033, 28 vs 33 discordant, two-sided p = 0.61 — not shown apart, and on the wrong side.
- Tune → held: 0.711 → 0.629, inside the ±0.07 selection optimism of 21 configurations on 149 rows.
- The error that remains is one-directional: powerful asks called medium (20) or small (6) of 46 — the
  "systematically cheap" bias Ship 0 recorded. The judge's is the same shape, milder (21 + 0).

**Adopted (PREREG rule R1):** the harness reproduced Ship 0 exactly (0.744 / 0.738 on the 180) and this is the
best tune configuration. The serving state is the bare request and the checkpoint is the root (amendment 2).
Zero-shot, fixing the three confounds is worth about seven points and does **not** beat the judge; rungs 2-4
are where that has to come from.
