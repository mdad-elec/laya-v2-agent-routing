# H9 — a Laya-verified cascade against pre-routing

Pre-registered as amendment 5 of `scripts/laya/PREREG.md` (b584f0f), committed before any verifier ran. This is
TypeSafe's `jev-verified-cascade` pattern with Laya as the verifier. The cheap tier answers first; the verifier reads
the request and the answer; the turn escalates one tier when P(correct) < τ. Replayed over the graded answers of the
120-ask measurement, on arm C's cell menu: local 27B @ off → codex luna @ high → codex sol @ high.

## Held read (60 asks), each arm once

| arm | τ (fitted on tune) | quality | vs v2-policy, 90% CI | mean cost per ask | vs v2-policy, 90% CI | escalated | verifier AUROC, all held answers |
|---|---|---|---|---|---|---|---|
| v2-policy (pre-routing) | — | 0.944 | — | 5,300 µUSD | — | — | — |
| A: root zero-shot verifier | 0.95 | 0.978 | [+0.009, +0.074] | 19,536 µUSD | [+11,641, +16,802] | 58 of 60 | 0.341 |
| B: fine-tuned verifier | 0.90 | 0.963 | [−0.010, +0.065] | 21,502 µUSD | [+13,717, +18,693] | 60 of 60 | 0.456 |
| B-hybrid (starts at v2-policy's band) | 0.05 | 0.944 | [0, 0] | 5,300 µUSD | [0, 0] | 0 of 60 | 0.456 |
| oracle verifier (ceiling) | — | 0.954 | [−0.014, +0.052] | **1,836 µUSD** | **[−6,358, −971]** | 17 of 60 | — |

**H9: not met by A or B.** Neither verifier can tell a correct answer from a wrong one. The fitted thresholds therefore
escalate nearly everything, which buys quality at 3.7–4× the cost. The hybrid's threshold never escalates, so it
reduces to v2-policy. The oracle meets the claim: at the same quality, a perfect verifier would cut mean cost by about
two thirds.

On the tier that matters most, the local 27B's answers, AUROC is 0.47 for A and 0.55 for B. The local model is right
on 43 of those 60 held asks, but the verifier cannot see which 43.

## Variant B as trained

- **Data.** 1,089 train and 275 validation rows: tune asks only, all 23 cells, validation split by ask. 93% of the
  labels are `correct`.
- **Model.** RLCD fine-tune of the root at a 1,024-token context. The embeddings and the bottom 20 of 28 encoder layers
  were frozen, leaving 125M parameters trainable, so the run fit beside a rendering service on an RTX 2080 Ti.
- **Epochs.** Epoch 4 hit a non-finite fp16 loss, and the trainer's guard stopped it. Selection ran over epochs 1–3 by
  validation accuracy. That was 0.9745 at every epoch, which is exactly the share of correct labels: the argmax is
  always `correct`. Epoch 3 was chosen, ties going to the later epoch, with temperature 1.115.

## Why it fails, and what it would take

- **The label is hard for an encoder.** Judging whether code runs, arithmetic holds, or a design is sound needs reasoning
  over the answer, not the pattern-matching a 421M encoder does in one pass. TypeSafe's cookbook verifies something
  much easier: whether each fact in a support answer appears in the retrieved excerpts.
- **The data is thin and imbalanced.** There are 1,089 rows at 93% `correct`, with one graded answer per cell. Switchyard's
  prefill router asks for 3,000–5,000 judged samples per model.
- **The fine-tune was partial.** Only the top 8 layers trained, at a context that truncates the longest answers.

The ceiling is real and large. A verifier that could separate the local model's right answers from its wrong ones is
worth about two thirds of the spend. This Laya cannot. Hosted Jev 1.13 (option C), or a generative judge as today's
router uses, remains untested in this role.
