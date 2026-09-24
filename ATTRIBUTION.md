# Where this corpus comes from

## `requests.jsonl` — 180 requests, not ours

Copied **byte for byte** from [`glukicov/laya_router`](https://github.com/glukicov/laya_router)
`data/requests.jsonl` at commit `a2278d969067` (2026-09-20), which is licensed
**Apache License 2.0**. It is reused rather than rewritten for two reasons:

1. It is the set the published Laya-vs-GPT-5-nano numbers were measured on
   (route accuracy 0.600 vs 0.600, ECE 0.093 vs 0.172, p50 184 ms vs 6,415 ms). Reusing it means
   our Ship-0 number is directly comparable with theirs — a different corpus would have made
   "we measured 0.61" uninterpretable.
2. Its labelling policy is **written down** (`LABELLING.md` here restates it). A router can only
   be scored against a policy that exists somewhere; scoring against a notion of correctness that
   lives in one person's head is not a measurement.

The upstream set's own stated limits carry over unchanged and are repeated here rather than
quietly dropped: the requests are **written, not collected**; there is **one annotator** and so no
inter-annotator agreement; a blind adjudication pass puts the **ceiling on `tier` at ~0.82**, not
1.0; and 180 rows means differences smaller than about 7 points are inside the noise.

## We reproduced their published numbers before trusting ours

Reusing a corpus is only worth anything if the harness reading it agrees with the harness that
produced the published result. Run on 2026-09-22 against `convaiinnovations/laya` (the 421M
english root), tier question only, sending the state under the key `message` exactly as upstream's
`laya_backend.py` does:

| wording      | our accuracy | their accuracy | our macro-F1 | their macro-F1 |
|--------------|--------------|----------------|--------------|----------------|
| shipped      | 0.600        | 0.600          | 0.535        | 0.535          |
| example-led  | 0.639        | 0.639          | 0.631        | 0.631          |

Identical to three decimals on both metrics. That is the evidence that `brain_bench.py` scores the
same thing `metrics.py` scores, and it is why the rest of our numbers are quoted without hedging.

One deliberate difference: our own runs send the state under the key `request`, because that is the
name the question text points at (`"Which model tier should answer the user's `request`?"`), while
upstream sends `message`. Matching them moved example-led +2.2 points and shipped −1.1 — both well
inside the ±7 the upstream set's size allows, so this is recorded as a reproduction detail and
**not** claimed as a finding. `brain_bench.py --state-key` exists so the question stays measurable.

## `dss-requests.jsonl` — ours, and NOT YET AUDITED

Written for this A/B because the upstream set contains no request that looks like the traffic
`dss/auto` actually routes: ERPNext/Frappe questions, Omniverse/USD asks, fleet and deploy
operations, dashboard and ledger questions, and the short back-and-forth of the console's copilot
dock. Labelled by the **same rules** as the upstream set (`LABELLING.md`).

> **These labels are a proposal, not a policy.** "The cheapest tier that can answer this well" is a
> statement about what DSI is willing to spend and what it considers a good answer — only the owner
> can state it. Until `audited: true` appears on a row, `score.py` reports it in a separate block and
> it does **not** contribute to any headline accuracy number.

## License

Both files are distributed under the Apache License 2.0, matching the upstream set. See
<https://github.com/glukicov/laya_router/blob/main/LICENSE>.
