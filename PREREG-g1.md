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

## Amendment 2 (2026-09-27): held-out models are a probe, not a cut from the scored checkpoint

Written before any g1 model is trained and before any held item is read. Amendment 1's "a
held-out model never contributes training outcomes" would have removed about half of
LLMRouterBench's 41 models from the checkpoint scored on F1–F3.

**The corrected rule:**
- **The scored checkpoint** (F1–F3, and everything "reported") trains on the tune items of every model.
- **A separate probe checkpoint** trains identically, but without the outcomes of any model whose
  `split_of("model/" + name)` is `held`. It is scored only for the zero-shot report: its P(solve)
  on the held models' held items, against the scored checkpoint on the same rows.
- Neither checkpoint's training can see the other's evaluation rows.

## Amendment 3 (2026-09-27): F3's exact setting and formulas; the model list

Written before any F3 number is computed, and before any g1 training.

### The model list

- LLMRouterBench @ `0e5af1b8` lists `openrouter`, a commercial router, among its models. It is a
  *reference*, never a trainable cell, so the pool is 39 models. Amendment 1 said 41; 39 is correct.
- BFCL-Result's `-FC` names (native function calling) and its prompted names are two cells of one
  model, described with the same model descriptor and a `native_tools` flag.

### F3, exactly

- **Authority.** LLMRouterBench's code @ `c77cb0506949d8f959e97967d2fefca0e8ff1b05`
  (`config/baseline_config_performance_cost.yaml`) and its paper's §6 formulas, since the repo ships
  no code for these metrics.
- **Setting.** Its performance-cost setting.
  - **Models:** the 13 flagship models named in that config (all 13 are in our data).
  - **Datasets:** its 10 datasets, minus GPQA (excluded, as above): aime, livemathbench, hle,
    livecodebench, mmlupro, swe-bench, simpleqa, tau2 and arenahard.
  - **Split:** routers are fitted on our **tune** items of these and scored on our **held** items.
    It's one fixed split, not their five seeds; stated as a difference.
- **Per-item cost** is the record's own `cost` for the chosen model. Accuracy is `score`, macro over
  datasets (AvgAcc).
- **Configurations.**
  - g1 sweeps τ over {0.05, 0.10, …, 0.95}.
  - Avengers-Pro and EmbedLLM sweep their own trade-off parameter as their adaptors expose it,
    otherwise over the same 19-point grid.
  - Each single model is one configuration.
- **Best Single (ℬ)** is the model with the highest tune AvgAcc.
- **The formulas:**
  - PerfGain = AvgAcc(θ*)/AvgAcc(ℬ) − 1, where θ* is the configuration with the highest AvgAcc on held.
  - CostSave = 1 − Cost(θ†)/Cost(ℬ), where θ† is the cheapest configuration with AvgAcc(θ) ≥ AvgAcc(ℬ).
    If there is none, CostSave is reported as "none reaches ℬ".
  - ParetoDist: the mean L1 distance of a router's configurations to the frontier of the union of
    all routers' and single models' configurations, in min-max-normalised (AvgAcc, log Cost).
- **The recall slice** is the held items where ≤ 3 of the 13 models score ≥ 0.5. It is scored as
  θ*'s accuracy on those items.
- **The bar.** g1 beats Avengers-Pro and EmbedLLM each:
  - on PerfGain and on slice accuracy, with a paired bootstrap over held items stratified by
    dataset, 10,000 resamples, 90% CI lower bound of the difference > 0;
  - and on CostSave (point estimate).
  - ParetoDist is reported.
- **A θ read.** Choosing θ* and θ† on held is part of the metric's definition, not tuning: every
  configuration is fixed from tune before the read.

### F1's baseline, exactly

- SR-KNN re-fitted on tune uses k = 1 (the paper's "1-nearest-neighbor"). A k = 1 vote needs no tie rule.
- The in-sample reproduction runs over all 970 rows. Only aggregate scores are computed from it, and
  it is logged in `results/g1/heldout-ledger.jsonl`.

## Amendment 4 (2026-09-27): how F3's two baselines are run

Written before any F3 prediction is scored and before any g1 training.

- **Embedder (both baselines, and SR-KNN in F1).** Qwen/Qwen3-Embedding-0.6B, run on the Spark.
  - Avengers-Pro's config names gte-Qwen2-7B-instruct behind a remote API. We don't run it: the
    only GPU box with room would put the Spark's resident 27B at risk.
  - LLMRouterBench's own ablation found no significant difference between gte-Qwen2-7B,
    nli-bert-base and all-MiniLM-L6-v2 for Avengers, EmbedLLM and GraphRouter.
  - The same F3 read therefore also reports both baselines with all-MiniLM-L6-v2 (CPU), as a
    sensitivity check.
- **Avengers-Pro.** `baselines/AvengersPro/balance_cluster_router.py` @ `c77cb050`, run unmodified.
  - Its `balance_config.json` values: 25 clusters, top_k 3, β 9, seed 42, max_router 1.
  - Its query-embedding cache is prefilled with our vectors, and its network embedder raises on any miss.
  - Trade-off: performance_weight = α, cost_sensitivity = 1 − α, for α on the 19-point grid.
- **EmbedLLM.** `baselines/EmbedLLM/algorithm/mf.py`'s `TextMF` @ `c77cb050` (embedding dim 1024,
  α-noise 0.05, Adam 1e-4, weight decay 1e-5, batch 2048).
  - Label: 1 if score ≥ 0.5.
  - Its training script selects the epoch on the TEST set. We don't: the epoch count is chosen on
    10% of tune prompts held aside (by prompt hash), and the model is then refit on all of tune.
  - EmbedLLM has no cost knob. It gets g1's policy: the cheapest model with P̂(correct) ≥ τ
    (argmax P̂ if none), over the same τ grid.
- **Every configuration** of every router is fixed from tune before the F3 read. The held
  predictions are computed first and scored once, together with g1's.

## Amendment 5 (2026-09-27): g1's recipe, calibration splits and τ rules, fixed before training

Written before any g1 model is trained.

- **Start.** The public root `convaiinnovations/laya` @ `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`
  (its ModernBERT-large encoder only). Not the v2 checkpoint, which saw estate asks.
- **Architecture.**
  - The digest (router/digest.py, ≤ 512 tokens) is mean-pooled from the encoder and projected to θ ∈ R^64.
  - The cell descriptor is the public block of atlas/descriptors.py (63 values) plus a measured block:
    8 domains, each a value and a mask. It goes through an MLP (256, 256, GELU) to (a ∈ R^64, b).
  - The logit is θ·a − b. The loss is binary cross-entropy on targets in [0, 1].
  - Each source's share of the loss: LLMRouterBench 0.5, BFCL 0.25, TwinRouterBench 0.25.
  - During training each cell's measured block is dropped (value and mask zeroed) with probability
    0.5, so a cell with public data only is a trained case.
- **Calibration splits** (inside tune; never train):
  - 5% of LLMRouterBench and BFCL tune prompts;
  - 20% of TwinRouterBench tune trajectories;
  - both by salted hash (`g1-cal/`).
  - TwinRouterBench calibration rows are still tune rows: they never touch F1's held rows.
- **Optimisation.** AdamW with weight decay 0.01.
  - The encoder uses lr 2e-5 with its bottom 8 layers frozen; heads use lr 1e-3.
  - bf16, 3 epochs over the training texts, a batch of 32 texts with all their cells.
  - Early stopping on calibration BCE, which is the only thing calibration is read for, besides τ below.
- **Policy and τ.** Pick the cheapest candidate with P ≥ τ, else the argmax.
  - **F1:** candidates are the 4 pool tiers; cost is the tier order; τ_F1 maximises the
    benchmark's Combined score on the TwinRouterBench calibration rows.
  - **F3:** τ sweeps the 19-point grid (Amendment 3). Expected cost is each model's mean cost per tune item.
  - **F2:** τ_F2 maximises RouterArena's arena-score formula (β = 0.1, its own `compute_arena_score`),
    computed on the LLMRouterBench calibration items with their recorded costs, over the F2 pool's
    models that LLMRouterBench has. It is fixed before the F2 run and never touched by RouterArena data.
- **The probe checkpoint** trains identically, without the outcomes of held-out models (Amendment 2).

## Amendment 6 (2026-09-27): the read's mechanics

Written before any read of F1 or F3, and before g1's training finished.

- **The checkpoint** scored is `best.pt`, the lowest calibration BCE. τ_F1 comes from
  `train.calibrate` on that same checkpoint. The read refuses mismatched steps.
  - Gradient checkpointing is on in the encoder: the same computation in less memory. The first run
    hit its own 0.12 memory cap, and the Spark's other services were untouched.
- **F1 bootstrap.**
  - 2,000 paired resamples of held trajectories, seed 20260927.
  - A trajectory drawn twice is scored as two trajectories (its rows are renamed), so repeats never merge.
  - The statistic is g1's Combined minus the tune-fitted SR-KNN's, both from the benchmark's own
    scorer. Its token counts are memoised, which a test shows changes no number.
- **F3 bootstrap.**
  - 10,000 paired resamples, stratified by dataset, seed 20260927.
  - θ* is chosen once on the full held set and kept fixed inside the resamples.
  - PerfGain's comparison is the macro AvgAcc difference of the two θ* configurations: same sign,
    and ℬ is common to both.
- **F2** is not part of this read. It needs a RouterArena inference run, which is an owner spend gate.
  Until F2 is met, nothing is pushed.
- **The results.** `results/g1/verdict.json`, and a line in `results/g1/heldout-ledger.jsonl`.

## Amendment 7 (2026-09-29): recipe 2, a System-One brain over a measured table, climbed by the blog's protocol

Written before any recipe-2 result exists and before any held item of F1 or F3 is read. The
held-read ledger holds two lines, both aggregate in-sample SR-KNN reproductions with
`held_rows_scored_individually: 0`. **No g1 model has been read on held.**

### What happened, and what this replaces

- Recipe 1 (Amendment 5: Laya digest tower × descriptor-MLP catalogue tower, IRT logit) reached a
  best calibration BCE of 0.5712, at step 500 of 900, and overfit from epoch 2. On the same
  calibration pairs the text-blind yardsticks read global 0.710, per cell 0.566, per cell × dataset
  0.530. By Amendment 5's own rule it does not go to the held read, so it is retired and its held
  read count is 0.
- **Disclosed design inputs, read before this amendment.** Diagnostics on the tune pairs, run while
  diagnosing recipe 1 and named here so nobody has to trust that they did not shape this design:
  descriptor MLP alone 0.516; a learned cell id 0.479; kNN over per-model outcomes 0.469;
  cluster-IRT with K = 32 0.472; per cell × dataset 0.459. They chose the family below. They are
  not gates and no bar is stated on them.
- The rest of Amendment 5 stands: the calibration splits (`g1-cal/`) and its τ rules, the encoder
  root (`convaiinnovations/laya` @ `55cf4c4e`), and the three sources' shares.

### Recipe 2

- **Layer 1, a System-One brain.** Laya answers typed questions over the session digest
  (`router/digest.py`, unchanged) in one pass, served on the Jev wire
  (`POST /v1/systemone {state, questions}`, as `studies/v2/finetune/systemone_server.py`), so a
  provider that speaks the wire can replace it. Question set v0:
  - **Asked of the brain:** `task` ∈ {code_fix, code_write, math, knowledge_qa, reasoning,
    writing, structured_extraction, tool_agentic, chat}; `domain` ∈ {software, math,
    science_medicine, business_finance, general, language_creative, other}; `level` ∈ {routine,
    demanding, frontier}; `continues_goal` (bool).
  - **Computed from the digest, never asked:** `stage` ∈ {first_step, exploring, editing, testing,
    wrapping_up}; `tool_loop_open`; `error_state` ∈ {none, some, critical}; `tests_passed`;
    `context_bucket`.
  - **Training labels** are deterministic maps from a source's own fields (dataset, benchmark,
    scenario, step position), stated in `train/questions.py` and committed with the first run.
    `level` is labelled from outcomes only on the eight datasets where Gate S3 found the ability
    band learnable (bbh, humaneval, korbench, livecodebench, livemathbench, math500, mbpp, tau2);
    elsewhere the table ignores it.
- **Layer 2, a measured table.** `P(solve | group, cell) = (k + m·prior_cell) / (n + m)`, where
  the group is the tuple of a subset of the questions above (which subset, and m, are hill-climbed
  below) and `prior_cell` is the cell's overall rate, or its Atlas descriptor prior when the cell
  has no outcomes, so an unmeasured model routes zero-shot. Expected cost is mean tokens × price
  per (group, cell). It is fitted on HC-train only, never on a held, RouterArena or contaminated
  prompt (`results/g1/leakage.json` and its tests).
  - **Lone-correct-model fallback.** When a group's row is flat, meaning no cell reaches τ or the
    best two are within the noise floor, the pick falls back to kNN over per-item outcomes with the
    cached Qwen3 embeddings. One round decides whether it stays.
- **Layer 3, policy and rules.** Cheapest cell with P ≥ τ, else the argmax (`train/policy.py`);
  τ from Amendment 5's calibration split, unchanged. One round tries KT-ModelRouter's Pareto
  operating point instead. Continuity stays as rules in code and is ablated, never learned:
  a tool loop stays on the model that started it; never downgrade mid-session except on
  settled/tests-passed; a critical error goes to the top tier; an explicit request wins; a brain
  failure keeps the current model.

### The hill-climb protocol

Adopted from Anthropic's "Automating eval design and hillclimbing": a split the climber reads and
a split it never opens, one change per round, keep only what improves both, revert what improves
only the one it reads, and diagnose by cause when a round stalls.

- **Splits, all inside tune, all by salted hash, none touching held.**
  - **F3 / LLMRouterBench and BFCL:** HC-test is the 15% of tune-minus-calibration prompts with
    `sha256("g1-hc/" + key) mod 100 < 15`; HC-train is the rest. τ still comes from the
    calibration split, so acceptance and τ never share data.
  - **F1 / TwinRouterBench:** tune is only 500 rows in 257 trajectories, so one split would be
    noise. Five trajectory-grouped folds, `fold = sha256("g1-hc/" + trajectory_key) mod 5`. Folds
    0–2 are HC-train (failures readable), folds 3–4 are HC-test. Every score is out of fold: the
    table that scores a fold is fitted without it.
  - **HC-test failures are never opened by the climber**, only its numbers. A test refuses a round
    whose diagnostics read them.
- **Noise floor first.** A paired bootstrap on HC-test (trajectory-grouped for F1, stratified by
  dataset for F3) gives the CI half-width. A change is worth keeping only when it beats
  max(that half-width, 0.5 Combined points, 0.005 BCE).
- **Allowed changes, one per round:** a question added, removed or rephrased; the subset of
  questions that keys the table; the cluster count; m or the prior source; τ against the Pareto
  point; a continuity rule on or off; the kNN fallback. **Never:** an item, transcript or failure
  pasted into anything trained or prompted; a feature that exists only to fix HC-train items.
- **Acceptance.** HC-train and HC-test both improve past the floor: keep. Only HC-train improves:
  revert, as overfitting. Either regresses: revert.
- **Stall.** After two rounds without a kept change, every HC-train failure is sorted by cause
  (wrong task answer / right group, flat row / policy too greedy / continuity rule fired /
  excluded as contaminated) before the next change.
- **Ledger.** `results/g1/hillclimb.jsonl`, one line per round: the one-line change, HC-train and
  HC-test before and after, the noise floor, the decision, and this file's blob sha. Every kept
  round is its own commit.

### Phase gates

- **Phase 1, oracle-group ceilings (CPU, cached files).** With the true groups from each source's
  own fields, fit the table on HC-train and score HC-test: F1 Combined against the SR-KNN
  re-fitted on the same folds, F3 performance-cost and the ≤ 3-correct slice against Avengers-Pro
  and EmbedLLM re-fitted on the same split. **Pass** means the ceiling beats each baseline with a
  paired 90% CI lower bound above 0, on both benchmarks. If it does not, no classifier can reach
  the bar through this design; nothing is trained and this file gets a new amendment.
- **Phase 2, a text-classifier proxy on cached embeddings, then the climb.** **Phase 3, fine-tune
  Laya on the surviving question set** on a GPU window the owner grants (`studies/v2/finetune`'s
  freeze-bottom, resume and memory cap; never the Spark's 27B on :8003). **Phase 4, the read.**
  Nothing in Phases 1–2 needs the GPU.

### F1 part 2, restated apples-to-apples

Amendment 1 set part 2 at "g1's held point estimate > 77.89". That number is SR-KNN's
**in-sample** score (its kNN index contains the scored rows), and it did not reproduce under any
reading of the paper (measured in `results/g1/a3-sr-knn.json`: leave-one-out 1-NN 57.94, 5-NN
including the row 74.03, 1-NN including the row 90.74, tune-only 5-fold 65.06). No held row has
been read, so this is restated before it can matter:

- **Part 1 stands and is the gate:** g1 − SR-KNN > 0 on the held rows, paired bootstrap over
  trajectories, 90% CI lower bound > 0, SR-KNN re-fitted on tune only.
- **Part 2 becomes:** g1's **in-sample** Combined, with the table fitted on the scored rows as
  SR-KNN's index contains them, must exceed the published 77.89. This is labelled in-sample
  everywhere it appears, is never used to choose anything, and is reported **beside** g1's honest
  held (out-of-sample) Combined and SR-KNN's own readings above. It uses the held rows' labels, and
  only for this one comparison, because that is the published protocol.
- The headline claim is part 1 plus the held out-of-sample number. Part 2 exists so the comparison
  the leaderboard actually makes is made once, on its own terms.

### What is unchanged

F2 and F3 as written, the reported-not-gating list, the held-out-model probe (Amendment 2), the
read mechanics (Amendment 6), and the reads rule: every held read of F1 or F3 and every F2 run is
appended to `results/g1/heldout-ledger.jsonl`; after a miss, another attempt needs a new dated
amendment first; the verdict reports how many reads were made; a bar not met is reported as not
met. **Nothing is pushed until F1, F2 and F3 are all met.**

## Amendment 8 (2026-09-30): F3's Layer 2 gains per-item evidence, after its Phase 1 ceiling failed

Written after the F3 ceiling's first read (`results/g1/ceiling-f3.json`, NOT MET) and before its
second. No held item has been read. F1's ceiling passed and is unaffected by this amendment.

### What happened

With the task answered perfectly (the item's dataset), the dataset-level table reached AvgAcc 0.563
on HC-test, against Avengers-Pro 0.551 (difference CI90 [−0.006, +0.029]) and EmbedLLM 0.591
([−0.051, −0.005]). On the recall slice it beat Avengers-Pro ([+0.003, +0.053]) and tied EmbedLLM.
A perfect task answer is not enough on single-turn items: which model solves an item varies inside
a dataset, and EmbedLLM's per-item estimate sees it. So the change is to Layer 2, not the brain.

### The change: task-conditioned kNN over per-model outcomes, shrunk toward the task row

For an item q with task t (the oracle dataset in this ceiling; the brain's answer later):

    P(model solves | q) = (Σ_{j ∈ N_k(q, t)} score_j(model) + m · row_t(model)) / (k' + m)

where N_k(q, t) is the k nearest HC-train items **of the same task** by cosine over the same Qwen3
embeddings the baselines use, k' = |N_k| (fewer when the task has fewer items), and row_t is the
Amendment 7 task row. Expected cost is shrunk the same way. The policy (cheapest above τ, 19-point
grid) and F3's formulas are unchanged.

- **Menu, fixed now:** k ∈ {8, 16, 32, 64}, m ∈ {1, 4, 16}, neighbours weighted uniformly.
  Chosen on HC-train alone, by 5-fold out-of-fold best AvgAcc (the `g1-hc-fold/` folds already in
  `eval/ceiling_f3.py`).
- **Reported beside it, never chosen from:** the same estimator with neighbours drawn across all
  tasks (no task answer at all).
- **The gate is unchanged:** it beats each of Avengers-Pro and EmbedLLM on AvgAcc(θ*) and on
  recall-slice accuracy, paired bootstrap stratified by dataset, 10,000 resamples, 90% CI lower
  bound > 0.
- **Reads.** This is HC-test's **second** F3 read. Both reads are reported. If this one also fails,
  a third needs another dated amendment first. HC-test is a tune split: its items still never touch
  held, and its failures are still never opened.

## Amendment 9 (2026-09-30): F3's Layer 2 becomes a learned per-item estimator, on a stratified HC-test

Written after F3's second ceiling read (`results/g1/ceiling-f3-knn.json`, NOT MET) and before its
third. No held item has been read. F1 is unaffected: Phase 2 passed (`results/g1/phase2-f1.json`).

### What happened

Task-conditioned kNN over per-model outcomes (Amendment 8; k 16, m 16, chosen on HC-train) reached
AvgAcc 0.558 against EmbedLLM's 0.591 (CI90 of the difference [−0.057, −0.008]); the dataset-level
table had reached 0.563. Neither non-parametric estimator matches EmbedLLM's learned per-item model
on single-turn items. A **flaw in Amendment 7's split** is recorded too: HC-test held 1 AIME and 5
LiveMathBench items, while AvgAcc weights each dataset equally, so one item was a ninth of the
score.

### The split, fixed

HC-test is re-cut **stratified by dataset**, with a fresh salt. Within each dataset, usable tune
items are ordered by `sha256("g1-hc9/" + prompt_hash)`, and the first
`max(⌈0.15·n⌉, min(⌈n/2⌉, 20))` go to HC-test; the rest are HC-train. With today's counts that is
AIME 8/16, LiveMathBench 20/57 and tau2 20/118, about 727 items in all. It overlaps the old
HC-test partly. Only its numbers were ever read, never its items, and this is disclosed; every
earlier read stays reported.

### The estimator

Our router keeps the policy (cheapest above τ, 19-point grid) and the task row's expected cost.
Layer 2's P(solve) is a learned per-item model, using EmbedLLM's own recipe and code
(`eval/baselines/embedllm.py`, its epoch rule on 10% of the fit items) in one of two ways:

- **`mf+task`:** EmbedLLM's model, with the item's task answer (the oracle dataset here, the brain's
  answer later) appended to its input vector as a one-hot.
- **`blend(w)`:** `w · P_embedllm + (1 − w) · row_task`, with w ∈ {0.25, 0.5, 0.75}.

**Menu, fixed now:** {mf+task, blend(0.25), blend(0.5), blend(0.75)}, chosen on HC-train alone by
5-fold out-of-fold best AvgAcc (the `g1-hc-fold/` folds).

**The gate is unchanged:** it beats each of Avengers-Pro and EmbedLLM (EmbedLLM exactly as before,
its input without the task) on AvgAcc(θ*) and on recall-slice accuracy, paired bootstrap stratified
by dataset, 10,000 resamples, 90% CI lower bound > 0.

**Reads.** This is the first read of the stratified HC-test and the third F3 ceiling read in all.
If it fails, F3 is reported as not reachable by this programme's estimators so far, and the next
step is the owner's call. The threshold is not moved.

## Amendment 10 (2026-09-30): F3 restated, Phase 3 skipped for F1, and what the held read scores

Written after Phases 1–2 and the F1 climb, and **before any held item of F1 or F3 is read**. The
owner approved each change below on 2026-09-30. This amendment changes a pre-registered bar,
**after three HC-test reads of F3** (`results/g1/ceiling-f3*.json`), and says so: those reads
are the reason for the change, and every one of them stays reported.

### F3, restated

Amendment 3 required g1 to beat Avengers-Pro and EmbedLLM each on PerfGain and slice accuracy
(CI90 lower bound > 0) and on CostSave (point estimate). On the stratified HC-test the three routers
are within about a point of each other on AvgAcc, which is the benchmark paper's own finding, while
our estimator led both on the recall slice. The bar now asks for what the design claims:

- **Non-inferiority on AvgAcc(θ*):** g1 − baseline, 90% CI lower bound > **−0.02**, against each
  of Avengers-Pro and EmbedLLM.
- **Superiority on recall-slice accuracy** (held items where ≤ 3 of the 13 models score ≥ 0.5):
  g1 − baseline, 90% CI lower bound > 0, against each.
- **CostSave and ParetoDist are reported and no longer gating.** This is part of the restatement
  and is named here so it is not missed.
- Mechanics are unchanged: the paired bootstrap over held items stratified by dataset, 10,000
  resamples, θ* fixed first.
- **g1's F3 estimator** is Amendment 9's, chosen on HC-train: `blend(0.5)` (EmbedLLM's model and the
  task row, half each) with the task row's expected cost and the cheapest-above-τ policy. For the
  read it is fitted on all usable tune items (tune minus calibration, minus contaminated prompts),
  as are both baselines.
- **What HC-test already suggests, stated before the read:** the restated bar would NOT have been met
  there. AvgAcc against Avengers-Pro had a lower bound of −0.033, below −0.02, and the slice lower
  bounds were −0.017 and −0.003. Held has about 8× the items, so the intervals will be narrower,
  and the result can go either way.

### Phase 3 is skipped for F1

The Phase 2 proxy brain (logistic regression on Qwen3-Embedding-0.6B vectors, `eval/phase2_f1.py`)
answers v0's `task`/`domain` at 99.5% on HC-test and routes identically to the v0 vocabulary
ceiling. A fine-tuned Laya brain has nothing left to recover on F1. **The router scored at F1's held
read is Phase 2's:** that proxy, the task + stage + errors table, m 0, τ 0.76, fitted on all tune
rows. Fine-tuning Laya as the System-One brain is deferred to the production wiring (S8), where its
latency and the Jev wire matter. It is not used for any bar.

### Reported beside it, never selected

- **R4** (stage from tool rounds, `results/g1/hillclimb.jsonl`) is reported at the held read as a
  secondary router, with its own score, because its HC-test gain came with an HC-train regression
  that the protocol rejects. It is not the scored router and cannot replace it after the read.
- The ablations listed in Amendment 1 stay reported.

### Blocked, not done

Growing the labelled agentic data was approved but cannot be done from public data:
- BFCL-Result has 48 multi-turn tune entries and lacks the pool's mid-tier model.
- TwinRouterBench's repository ships no labelled steps beyond its 970-row bank.
It needs a measurement run (owner gate O4).

### Unchanged

F1 (both parts, as Amendment 7 restated part 2), F2, the reads rule, and the rule that nothing is
pushed until F1, F2 and F3 are all met.
