# Estimate: Terminal-Bench Lite on our CLI seats

A planning estimate, not a measurement. It anchors on what the 120-ask cell measurement actually consumed, plus one
external figure for the size of an agent task.

## What our graded run measured (2,748 answers, 23 cells)

| seat | calls | tokens (incl. cache) | list-price value | active hours | tokens per active hour |
|---|---|---|---|---|---|
| Claude Code (subscription) | 1,080 | 4.36M | $29 | 5.0 | 0.87M |
| Codex (subscription) | 1,440 | 12.67M | $44 | 6.4 | 1.98M |

Calls ran one at a time. The run spanned about 28 hours with roughly 22 hours of pauses per seat; the log does not
separate rate-limit pauses from session breaks. Opus 5's blended rate in our ledger is $8.15 per million tokens.

## The size of an agent task (external anchor)

On Cognition's FrontierCode, Opus 5 costs about $4.30 per task at list price (NVIDIA's Switchyard post: staged routing
at $3.11 is 28% lower). Terminal-Bench Lite is calibrated easier and faster, so this estimate takes $1–4 per task for an
Opus-class cell, and roughly a third of that for Sonnet- and luna-class cells. This range is the largest uncertainty, a
factor of 4.

## The grid

Eight cells: Claude Opus 5, Fable 5 and Sonnet 5 at `high`, Sonnet 5 at `low`; Codex sol, terra and luna at `high`; the
local 27B through Codex's custom-provider setting ($0). The benchmark's own tests grade every task pass or fail, so
there is no panel.

| scope | agent runs | list-price value on the subscriptions | tokens | wall time |
|---|---|---|---|---|
| 20-task subset | 160 | ≈ $80–330 | ≈ 20–80M (5–18× the 120-ask run) | ≈ 8 h of agent time at 4 in parallel; 1–3 days with subscription windows |
| all 100 tasks | 800 | ≈ $400–1,650 | ≈ 100–400M | about 1–2 weeks with subscription windows |

## What each scope can decide

- **20 tasks** grade pass or fail, so a router's solve rate carries a standard error of about ±0.10. Routers within
  about 0.2 of each other cannot be told apart. This is a go/no-go, not a claim.
- **100 tasks** bring the standard error to about ±0.05, comparable to our 60-ask pilot, so H1/H2-style claims become possible.

## What it would test

Per-task routing: Laya-v2, the judge, vLLM Semantic Router, RouteLLM and the oracle, replayed over the matrix. After
H9 the verifier cascade has no working verifier to test, unless it uses the agent's own test run or hosted Jev.
