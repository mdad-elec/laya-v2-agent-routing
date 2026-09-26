# Rung 4 — the fine-tune

Upstream's RLCD recipe (`scripts/laya/train_laya_v2.py`, a single-GPU port) on the root checkpoint, trained on an RTX 2080 Ti workstation
(RTX 2080 Ti, fp16 + gradient checkpointing, micro-batch 4 × 16 = upstream's 64) — ~80 s an epoch, 4 epochs, epoch
selected and temperature fitted on the 28-row val slice of tune. Training rows: the 120 tune asks with DSS rows held back
until audited, t04 dropped as a near duplicate of held sml-02, plus 708 label-preserving paraphrases from the local 27B
(6 dropped by the held-out filters) — 827 train, 28 val. Two variants pre-registered: **v2-policy** (smoothed corpus
labels) and **v2-outcome** (0.5 × label + 0.5 × the measured band target, 57 rows carrying one).

| variant | epoch | val acc | fitted T (choice:3-5) | weights sha256 |
|---|---|---|---|---|
| v2-policy | 4 | 0.821 | 1.518 | `6098ad7664ef42dd…` |
| v2-outcome | 3 | 0.821 | 1.225 | `41f0649032d4591f…` |

**The held-out read** (151 rows, DSS apart; one read per pre-declared variant):

| decider | band accuracy | vs judge | exact McNemar, one-sided | under | over | ECE (signed) |
|---|---|---|---|---|---|---|
| today's judge | 0.662 | — | — | | | |
| Laya as shipped | 0.563 | −0.099 | | 0.159 | 0.278 | 0.132 (−0.132) |
| rung 1 (zero-shot, fixed) | 0.629 | −0.033 | p = 0.78 | 0.238 | 0.132 | 0.122 (−0.055) |
| **v2-policy** | **0.801** | **+0.139** | **p = 0.0027** (37 vs 16) | 0.119 | 0.079 | 0.104 (−0.075) |
| v2-outcome | 0.748 | +0.086 | p = 0.065 (38 vs 25) | 0.159 | 0.093 | 0.060 (−0.058) |

DSS held rows (unaudited, never in the headline): v2-policy 0.667, v2-outcome 0.700 (n = 30).

**Adopted: v2-policy.** It meets H1 (margin ≥ 0.05, p < 0.05) and the rung rule (+0.172 over rung 1). It was the plan's
default variant before any number existed (gate G6: trained on policy labels only, no targets derived from paid-seat
answers), so adopting it does not rest on comparing the two variants on held-out. v2-outcome improves too but does not
meet H1 on its own.
