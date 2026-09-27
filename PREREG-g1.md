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

## Amendment 1 (2026-09-27): after the S3 FAIL, the frontier model and its bars

Written before any g1 model is trained and before any held item of F1–F3 is read. Results from
before this amendment (`results/g1/s3-gate.json`) name the original version, blob `627ad263`.

### What happened, and the pivot the rule requires

- Gate S3 read FAIL on tune: pooled kNN − majority = +0.020 (< 0.03); 8 datasets ≥ 0.05. By this
  file's own rule we pivot to **the calibrated combiner: P(cell solves) from features**. We do
  not relax the S3 threshold, and the three-band label is no longer trained directly.
- The literature agrees with the finding. On LLMRouterBench, routers are nearly indistinguishable
  and the embedding backbone does not matter (arXiv 2601.07206). Prompt-only routers recover
  7.5–14.4% of the oracle gap (arXiv 2608.08265).

### The model (Laya-g1)

- **Digest tower.** Laya (ModernBERT, 421M, 512 tokens) encodes the session digest to θ ∈ R^d.
  The digest carries:
  - the request;
  - prior user turns;
  - the tools offered;
  - the tool rounds;
  - the context bucket and turn index;
  - **`goal`**: the standing objective, from the system/task instruction or the first user turn, ≤ 60 tokens;
  - **`progress`**: step markers seen so far.

  It never carries model, rung or seat names.
- **Catalogue tower.** A small MLP maps a cell's **descriptor** (Atlas per-domain priors, effort,
  price in/out, context window, family, open/closed) to (a_c ∈ R^d, b_c). Cells are described,
  never given a learned id, so an unseen model is routable with no retraining.
- **Heads.**
  - P(solve | session, c) = σ(a_c · θ − b_c), trained with binary cross-entropy on per-(item, model) outcomes.
  - Expected output tokens per cell (regression), so cost = tokens × price.
- **Policy.** Pick the cheapest candidate with P ≥ τ. τ is fixed on tune data only, per benchmark,
  and never on a held or eval-only item. Continuity stays as rules in code (a tool loop stays on the
  model that started it; never downgrade mid-session except settled/tests-passed; critical error →
  top tier). Those rules are ablated, not learned.

### Data and splits

- **Items.** `split_of` as above. For TwinRouterBench the key is the trajectory,
  `twinrouterbench/<benchmark>/<instance_id>`, so every step of one trajectory falls in one split.
  That is about 50/50, the repository's single split rule.
- **Held-out models.** A model whose `split_of("model/" + name)` is `held` never contributes
  training outcomes. They measure zero-shot catalogue awareness.
- **Leakage.** No prompt hash from an F1-held row, from any RouterArena split, or from a suite's held
  items trains or tunes anything. Each has a test.

### The frontier bars (all three required before anything is pushed)

- **F1: TwinRouterBench static** @ `7cbb0deac8f697b5faa8489c309560e53d2ef088` (Apache-2.0),
  970 rows / 520 trajectories.
  - **Metric:** its own `main/eval/section11` `combined_score_percent` on the **held** rows.
  - **Tiers.** g1 chooses among descriptors of the four pool models in `data/dynamic/tier_to_model.json`,
    and the chosen model is reported as its tier.
  - **Bar, part 1:** g1 − SR-KNN > 0 (paired bootstrap over trajectories, 90% CI lower bound > 0).
    SR-KNN is re-fitted on the **tune** rows only, with its published method and embedder, and scored
    on the same held rows.
  - **Bar, part 2:** g1's held point estimate > 77.89. That is SR-KNN's published number, which is
    **in-sample** (its kNN index contains the scored rows; leaderboard note "In-sample KNN reference
    over question-bank embeddings"). So this part is the stronger claim.
- **F2: RouterArena** @ `cff9659dd6a3a07815f4842fcdadea5a597a1df1` (Apache-2.0, **evaluation-only**:
  no component, τ included, is fitted or tuned on its data).
  - **Metric:** Arena score (β = 0.1) from its own `router_evaluation/compute_scores.py` on the `full`
    split (8,400).
  - **Bar:** > 77.63 (Paix2, rank 1 in its README at that commit).
  - **Pool.** Chosen from `model_cost.json` names that have Atlas descriptors, and committed before
    the run. The robustness split is run and reported too.
  - This bar needs one inference run of the chosen pool, a spend gate for the owner.
- **F3: LLMRouterBench** @ `0e5af1b8`, our held split, 41 models (GPQA excluded).
  - **Metrics:** the benchmark's own performance-cost metric, and accuracy on the slice where ≤ 3 models
    are correct (its "model recall" failure).
  - **Bar:** g1 beats both Avengers-Pro and EmbedLLM, each re-fitted on tune with the benchmark's own
    framework (paired bootstrap, 90% CI lower bound > 0), on both metrics.

### Reported, not gating

- v2's held 151 rows: non-inferior to v2 within 0.03 (O1; the owner confirms the margin).
- The estate transfer check.
- p95 decision latency ≤ 250 ms on the Spark GB10.
- Ablations on F1 held: −goal, −progress, −tools, −history, descriptors → learned ids, rules off.
- Zero-shot on the held-out models.

### The original hypotheses

- H1 (band accuracy) is withdrawn, because no band is trained.
- H2 is kept, as "reported, not gating".
- H3 is re-stated as the −history/−tools ablation on F1 held.
- H4 is subsumed by F3.
- H5 is kept.

### Reads

- Every held read of F1 or F3, and every F2 run, is appended to `results/g1/heldout-ledger.jsonl`
  with this file's blob sha and the model checkpoint's sha.
- After a miss, another attempt needs a new dated amendment before the next read, and the verdict
  reports how many reads were made.
- A bar not met is reported as not met.
