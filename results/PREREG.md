# Laya-v2 — pre-registration

Written 2026-09-23, **before any Laya-v2 measurement**. Every result JSON under
`scripts/ab/results/laya-v2/` carries `prereg_sha` = `git hash-object scripts/laya/PREREG.md` at run
time; a result whose `prereg_sha` is not a committed version of this file is not quotable. Amendments
are appended at the bottom, dated, and never edit a rule above them.

## What is being compared

- **The judge**: today's `dss/auto` classifier — `CLASSIFIER_PROMPT` on `local-27b/qwen3.8-27b @ off`
  (`backend/crates/dss-gateway/src/router.rs`), its verdict mapped to a band by
  `scripts/ab/judge_adapter.py`. The adapter's one free parameter (`t_small`, where a `supported`
  verdict stops being `medium`) is fitted on the **tune** split, so the judge gets the same tuning
  courtesy Laya gets.
- **Laya-shipped**: the brain as it ran on 2026-09-23 (checkpoint `typed-decisions`, wrapped state,
  alphabetical option order) — the baseline the live A/B measured.
- **Laya-v2**: the configuration the ladder below adopts. Two fine-tune variants are pre-declared:
  **v2-policy** (trained on the corpus labels, label-smoothed) and **v2-outcome** (trained on 0.5 × the
  smoothed label + 0.5 × the measured band solve-rate from the graded cells).
- **Oracles** (ceilings, not contenders): `oracle-label` (the corpus band) and `oracle-cell` (the best
  graded cell per ask).

## Splits

Frozen files, committed before this document's first measurement:
`scripts/ab/corpus/splits/cells-pilot.split.json` (the pilot asks; the 24 held-out ids of the
2026-09-23 three-arm run stay held-out forever) and `scripts/ab/corpus/splits/tier-240.split.json`
(the 240 tier-labelled rows, stratified by corpus and tier, seed 20260923). No held-out ask, no
paraphrase of one, and no n-gram or embedding near-duplicate of one appears in any tuning,
calibration, fine-tuning or selection step. Every guard has a test.

## Hypotheses

**H1 — band decision (primary).** On `H_band` = the held half of the 240 (the 90 borrowed rows are the
headline; the 30 DSS rows are reported separately and excluded from the headline until
`audited: true`) plus the held pilot asks mapped through `checks.BAND_OF`: Laya-v2 "beats the judge"
iff (a) Laya-v2 band accuracy − judge band accuracy ≥ 0.05, **and** (b) a one-sided exact McNemar
test on the discordant pairs (Laya-v2 right where the judge is wrong, more often than the reverse)
gives p < 0.05. Power, stated in advance: ~150 paired decisions with ~35% discordance detect a
10-point gap at ~80%; a smaller true gap will most likely read "not shown apart", and it is reported
in those words.

**H2 — answer quality.** On the held pilot asks, the quality of a `decider × menu` pair is read from
the graded answers of the 23 measured cells (the cell the menu resolves the decider's band to; no new
model calls). Non-inferiority: Laya-v2 × cells ≥ judge × cells − 0.02 in mean score, with the lower
bound of a 90% paired bootstrap interval (2,000 resamples over asks, seed 20260923) above −0.05; the
same on the models-only ladder. Superiority is reported if the interval excludes 0; it is not
required. **"Laya-v2 beats the judge" is only ever claimed on the same menu.** An unscored answer is
counted as unscored, never as 0.

**H3 — latency.** Decision latency p95 ≤ 250 ms for Laya-v2 served on the Spark and called across
the tailnet. The judge's decision latency p50 is recorded beside it.

**H4 — cost.** Laya-v2 generates 0 tokens per decision; the judge's input and output tokens per
decision are recorded.

**H5 — off-local rate.** Reported per decider × menu. Descriptive, not gated: it is the owner's policy
weight (`docs/dss/threads/router.md` §5b).

**H6 — effort.** The effort question is kept only if its accuracy against the measured effort label
beats the constant baseline (always `standard`) by ≥ 0.10 in cross-validation on tune. Effort labels
use three canonical levels mapped from the measured efforts: `none|off|low → minimal`,
`high|on → standard`, `max → maximal`; an ask's label is the lowest level whose graded quality is
within δ = 0.05 of the model's best, histogrammed over the models measured at ≥ 2 efforts.

## The ladder and its adoption rule

| Rung | Change | Adopted only if (on held-out, read once) |
|---|---|---|
| R1 form | checkpoint × state shape × wording, decided on tune | reproduces 0.744 ± 0.01 on `root / bare / example-led / powerful,medium,small` over all 180 borrowed rows (harness check); the adopted configuration is the best on tune |
| R2 prompt | staged wording search on tune; v2 schema (band + effort + auxiliaries); combiner | band accuracy ≥ +0.03 over R1's adopted configuration and underspend up by no more than 0.02 |
| R3 calibration | per-question temperature in the served directory | held ECE ≤ 0.08; argmax unchanged |
| R4 fine-tune | v2-policy and v2-outcome, RLCD, tune asks + tune-only paraphrases | band accuracy ≥ +0.03 over R3's adopted configuration |

A rung that does not meet its rule is recorded as **"no improvement"** with its tune and held numbers,
the previous configuration stays adopted, and the ladder continues. Ties on accuracy go to lower
underspend.

**One read per rung.** Held-out is read once per rung, by one candidate named on tune before the
read, and every read is a line in `scripts/ab/results/laya-v2/heldout-ledger.md`
(`date | rung | candidate | file`). There is no quick look.

## Declared bias

The shipped wording and order were selected on all 180 borrowed rows in Ship 0, so their number on
the held half is optimistic by up to the selection optimism of picking 1 of 18 configurations.
Laya-v2 is selected on tune only. H1 is therefore conservative against Laya-v2.

## Panel calibration

The rubric half is graded by a blind cross-family panel. It is called "human-calibrated" only if
Spearman ρ between panel and owner scores is ≥ 0.5 in every band with n ≥ 20. Until then every
rubric-derived number carries "panel not human-calibrated", and the machine-checkable half is
printed beside it as the number that cannot be argued with.

## Amendments

(none yet)

### Amendment 1 — 2026-09-23, before any effort question was measured

H6's bar was "beats the constant baseline (always `standard`) by ≥ 0.10". The effort labels, derived
as written above from the 1,100 graded pilot answers (`scripts/ab/effort_labels.py`,
`scripts/ab/corpus/derived/effort-gold.jsonl`), are 44 of 48 `minimal`: on tune the constant
`minimal` is right 0.875 of the time, `standard` 0.125, `maximal` 0.000. A bar set against
`standard` would be met by a question that always answers `minimal` and knows nothing. The bar is
therefore the **best constant on tune** (the majority class, here `minimal` at 0.875): the effort
question is kept only if it beats that by ≥ 0.10 in cross-validation on tune — which on this corpus
means ≥ 0.975. This is stricter than the original rule, and it is set before the question is asked.
Recorded as a finding in its own right: on these asks, for a given model, effort changes time and
cost far more often than it changes whether the answer is right.

### Amendment 2 — 2026-09-23, after rung 1's held-out read, before rung 2

Rung 1 adopted `laya-root / bare / example-led / powerful,medium,small` (`scripts/ab/results/laya-v2/r1-held.md`).
From here every rung measures, tunes, calibrates and trains on that serving form: the **root checkpoint**
(`convaiinnovations/laya`, sha256 of model.safetensors `891102d372688fc2…` as served) shown the **bare
request** `{"request": …}`. The router sends that state (Task 10). Held-out band accuracy at adoption:
0.629 against the judge's 0.662 — the bar the remaining rungs must clear.

### Amendment 3 — 2026-09-24, before any RouteLLM router was run

**H7: Laya-v2 against RouteLLM** (Ong et al., arXiv 2406.18665), the router the programme set out to beat.
Of RouteLLM's four released routers, only the BERT classifier (`routellm/bert_gpt4_augmented`) can run
here: the matrix-factorisation and similarity-weighted routers embed every prompt with OpenAI's
embedding API (no key on this box), and the causal-LLM router is an 8B model with no GPU free for it.
Those three are **not measured**, and nothing below speaks for them.

- **Score.** RouteLLM's own `calculate_strong_win_rate`, verbatim: softmax over the three logits,
  score = 1 − P(tie) − P(weak wins).
- **Bands.** RouteLLM routes two ways. Our decision has three bands, so the score maps through two
  thresholds `t_lo ≤ t_hi` (below `t_lo` small, below `t_hi` medium, else powerful), fitted on the tune
  split by grid (step 0.01) for band accuracy, ties to the lowest `t_lo` then the lowest `t_hi`. This is
  the same treatment the judge received (a one-parameter adapter fitted on tune), with one more
  parameter in RouteLLM's favour.
- **Read once on held**, the same 151 H1 rows and 60 H2 asks, DSS rows apart.
- **Claim.** "Laya-v2 beats RouteLLM's BERT router" is made only if band accuracy Laya-v2 − RouteLLM ≥ 0.05
  **and** one-sided exact McNemar p < 0.05 (H1's bar). On quality, both menus are reported with the paired
  90% bootstrap CI; "better quality" is claimed only where the CI excludes 0. A tie is reported as a tie.
- **RouterBench, RouteLLM's native setting.** The strong/weak pair the RouteLLM paper uses is GPT-4 against
  Mixtral-8x7B, which are exactly our RouterBench `powerful` and `small` menu models. RouteLLM's router is
  swept over its threshold on that pair; Laya-v2 is swept the same way over P(powerful). Both are scored
  by AIQ against the same-sample hull of the two models, on the same 2,451 test prompts. No refit on
  RouterBench for either router.
