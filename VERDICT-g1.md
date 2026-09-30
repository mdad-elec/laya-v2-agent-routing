# Laya-g1 verdict (2026-09-30)

Pre-registration: `PREREG-g1.md` at blob `f74aa092` (Amendments 1–10). Code: `211b125`. Each bar was
read once on its held split (`results/g1/heldout-ledger.jsonl`). Full numbers:
`results/g1/verdict.json`.

| Bar | Verdict | In the pre-registration's words |
|---|---|---|
| **F1** TwinRouterBench static | **MET** | Part 1: g1 − SR-KNN (tune-fitted) = +9.60 Combined, paired bootstrap over 263 held trajectories, CI90 [+5.12, +14.14] > 0. Part 2: g1's in-sample Combined 84.34 > 77.89 (labelled in-sample; g1's honest held Combined is 84.22). |
| **F2** RouterArena | **NOT RUN** | Needs the owner's spend approval for one inference run. |
| **F3** LLMRouterBench (held, 6,138 items) | **NOT MET** | AvgAcc non-inferiority holds against both, and g1 is better than both (+0.0115, CI90 [+0.0054, +0.0177] vs Avengers-Pro; +0.0132, [+0.0074, +0.0197] vs EmbedLLM). Recall-slice superiority holds against EmbedLLM ([+0.0033, +0.0180]) but not against Avengers-Pro ([−0.0006, +0.0219]). One of four checks fails. Under Amendment 3's original bar the same single check fails. |

Nothing is pushed: the bar requires F1, F2 and F3 all met.

## What the router is

Recipe 2 (Amendments 7–10): classifier → measured table → policy.
- **F1:** a logistic-regression brain on Qwen3-Embedding-0.6B vectors answers question set v0's `task`
  (99.8% correct on held), and a table of P(tier t solves | task, stage, errors) routes to the
  cheapest tier with P ≥ 0.76.
- **F3:** EmbedLLM's per-item model is averaged half-and-half with the task's measured row.

The Laya fine-tune (Phase 3) was skipped for F1 by Amendment 10: the proxy already matched the
ceiling.

## What it does and does not show

- **Knowing the benchmark does most of F1.** On held, a task-only table scores 84.34 and "always
  low" 67.71. The gain over SR-KNN is concentrated on SWE-bench (62.99 vs 33.59), the only benchmark
  whose labels spread across tiers.
- **Session awareness is not yet shown to add value.** Session structure alone scores 71.0,
  above every constant but below SR-KNN. R4 (step position from tool rounds), which the climb
  rejected on HC-train, scores 79.35 on held, below the kept router: the rejection was right.
- **The F3 routers differ by about a point**, as LLMRouterBench's paper reports. g1's lead on AvgAcc
  is significant on held; its slice lead over Avengers-Pro is not, by 0.0006.

## Reads and changes, disclosed

- F3's Phase 1 ceiling was read three times on HC-test (Amendments 7–9), and F3's bar was restated
  (Amendment 10) after those reads and before the held read. F1's climb ran five rounds; none was kept.
- A new F3 attempt needs a new dated amendment and a genuinely new estimator, and would be held read 2.

## Publication decision (owner, 2026-09-30)

Amendment 1 said nothing would be pushed until F1, F2 and F3 were all met. After reading this verdict,
the owner lifted that rule **for publication only**: the repository is published with this verdict
exactly as written. F1 is met, F3 is not met, F2 was not run. No bar, threshold or read was changed
for it. F2 is not run, and F3 is not retried.

## The estate transfer check (reported, not gating; read once, 2026-09-30)

`eval/estate_gate.py` ran recipe 2 on the Laya-v2 study's frozen splits: 148 tune asks and 149 held,
the published pilot and borrowed rows. The band table was fitted on the tune asks, as the owner
allowed for the production table; the task brain was trained on public data only. The
configuration chosen on tune (neighbours, k 9, m 1, τ 0.62) reached **held band accuracy 0.524**
(Wilson 90% [0.456, 0.590]; borrowed 0.549, pilot 0.483). **Laya-v2 reaches 0.801** on the same
held halves. The S8 bar was 0.771, so it is **not met**.

On single asks, difficulty lives in the wording. A router fine-tuned on labelled asks learns it; a
task table with neighbours over 148 asks does not. **Recipe 2 is therefore not wired into the estate's
router, and Laya-v2 remains the better router for this estate.** Recipe 2's result is F1's
agentic-benchmark result, not a replacement for v2 on single-ask traffic.
