# results/v1

Three frozen runs over the 117 cases (3 repeats; Laya is deterministic, so the repeats agree).
`summary.json` is first-repeat accuracy per split.

| run | what | tune (59) | held (58) |
|---|---|---|---|
| `laya-root/` | `convaiinnovations/laya` root, zero-shot, this package's `questions.json` | 0.593 | 0.500 |
| `laya-v2/` | the RLCD fine-tune `laya-v2-agent-routing` (weights sha256 `6098ad7664ef42dd…`) | 0.949 | **0.724** |
| `judge/` | a 27B LLM judge (`CLASSIFIER_PROMPT`, JSON p_solve + boundary) mapped to a band by a one-threshold adapter fitted on tune, replayed from `judge-decisions.jsonl` | 0.661 | 0.690 |

**Read `laya-v2` on `held` only.** It was fine-tuned on the tune split (plus paraphrases of it), so its tune
column is training accuracy. On held, it and the judge are not shown apart at this size (58 cases).
Latency in `raw.jsonl` is a contended laptop CPU; on a GPU the same checkpoint answers in ~125 ms.
