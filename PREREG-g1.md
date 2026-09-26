# Pre-registration: Laya-g1, a general session-aware router

Committed before any g1 label is used for training and before any held item is read. Every g1
result file carries this file's git blob sha, and a result from before an amendment names the
version it was run under. Amendments are appended, dated, never edited in place.

## Data

- **Label sources** (public, item-level, one row per (item, model, score)):
  - LLMRouterBench @ `0e5af1b8`: 26 datasets, 42 models. GPQA is excluded, at its authors' request.
  - BFCL-Result @ `cadd76a8`, snapshot 2025-12-16: multi-turn base, 109 models.
- **Splits.** `tune` / `held` = `suites.base.split_of(prompt_hash)` with salt `laya-g1/2026-09-26`.
  The same prompt appearing in several datasets always falls in one split.
- **Leakage.** A label item whose prompt hash equals any HELD item of this repository's suites is
  dropped from tune. The estate corpus (studies/v2 and our live answers) never enters tune.
  Each of these rules is a test.

## Label (labels/tiers.py)

- **Ability.** A model's ability on a dataset is its mean score over that dataset's tune items.
- **Tiers.** A dataset's models are split into thirds by ability: small, medium, powerful.
- **The label.** An item's label is the lowest tier whose models' mean score reaches `TAU = 0.5`.
  An item no tier reaches is `powerful` and is flagged `unsolved`.
- **The soft label.** The per-tier mean scores.

## Gate S3: is the band learnable from text at all? (tune only, read before any training)

- **The router under test.** TF-IDF on word 1-2 grams, then k-nearest-neighbours with k = 15 on
  cosine similarity. It is scored by 5-fold cross-validation grouped by prompt hash, one classifier
  per dataset.
- **The baseline.** The dataset's majority band.
- **PASS** requires both:
  - kNN − majority ≥ 0.05 on at least 3 datasets;
  - pooled kNN − pooled majority ≥ 0.03.
- **Datasets that don't discriminate.** A dataset with more than 85% of its labels in one band is
  reported and left out of training (it cannot teach the band).
- **FAIL** means pivot to the calibrated combiner (P(cell solves) from features), and the finding is
  published as such.

## Hypotheses (read once, on held, after training)

| # | Claim | Test |
|---|---|---|
| H1 | Laya-g1 band accuracy on the public held split beats Laya-v2 by ≥ 0.05 | one-sided exact McNemar, p < 0.05 |
| H2 | Laya-g1 is non-inferior to Laya-v2 on v2's own held 151 rows, within 0.03 | paired bootstrap 90% CI lower bound > −0.03 (**owner to confirm the margin, gate O1**) |
| H3 | g1 with the session digest beats g1 on the bare request, on BFCL held multi-turn episodes | one-sided exact McNemar, p < 0.05; and single-turn non-inferior within 0.02 |
| H4 | Routing by g1 over the Atlas reaches the quality of the best single model at lower cost on held items | AIQ above the zero-router hull; cost at 95% of best quality below the best single model's |
| H5 | Decision latency | p95 ≤ 250 ms on the Spark GB10 |

A hypothesis that is not met is reported as not met.
