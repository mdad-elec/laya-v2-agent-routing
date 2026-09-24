# Agent routing: which model tier should answer this request?

> **NOTICE.** The tier label policy used here (`small` / `medium` / `powerful`, "the cheapest tier
> that can answer the request well") is quoted from
> [glukicov/laya_router](https://github.com/glukicov/laya_router) `data/README.md` at commit
> `a2278d969067`, Apache-2.0. The wording of the tier question follows that project's
> example-led style. This package is licensed Apache-2.0 ([LICENSE](LICENSE)); it does not alter
> Laya's license.

A small, self-contained community benchmark in the shape of [`feishu_zh`](../feishu_zh/):
**117 hand-written English requests, 4 difficulty bands, 3 routing tiers**, a frozen one-question
Laya schema, and a model-free audit. The requests come from an internal agent-platform A/B of a
model router; only the public rows are included, and the same rows are published with the
Hugging Face dataset `agent-routing-cells`.

## What is being decided

An agent platform has several models at different prices. Before a turn is answered, a router
asks: **given this request, what is the cheapest tier that will answer it well?** A Laya
`choice` question answers that in one forward pass, with a probability per option and no
generated tokens.

- **State:** `{"request": <the user's message>}` and nothing else — no session, no model list.
- **Question:** one `choice` named `tier` with three options, loaded byte-for-byte from
  [`data/questions.json`](data/questions.json).
- **Reference label:** each request was written for a difficulty band, folded to a tier:
  `trivial`/`easy` → `small`, `medium` → `medium`, `hard` → `powerful`.

**Why three tiers and not a list of models.** The obvious design is "show Laya every available
model and let it pick". Laya calibrates per option-count bucket, and the English root checkpoint
ships a temperature for the `choice:11+` bucket (about 0.1006) that falls outside the SDK's
accepted range `[0.5, 5]`; on load the SDK replaces it and warns that confidence from that bucket
is uncalibrated. A real platform has well over eleven models, so a pick-the-model question would
sit in exactly that bucket. Three tiers keep the question in a healthy bucket; choosing the
concrete model inside a tier is a deterministic step (price, availability) outside this benchmark.

**Why option order is part of the prompt.** The SDK lays a `choice` question out as
`[MASK] opt0 [MASK] opt1 …` and reads each option off its own mask position, so the same three
options in two orders are two different prompts. The frozen order is
`powerful, medium, small`. The manifest records it, `prompts.py` refuses a schema in any other
order, and `data/questions.json` is hashed as bytes (the request digest sorts keys, so it cannot
see order on its own).

**Why overspend and underspend are reported separately.** Routing a `small` request to `powerful`
is a cost line; routing a `powerful` request to `small` is a bad answer somebody has to notice.
One accuracy number hides which of the two a router is doing.

## Start here: verify offline

From the repository root, using Python 3.10+:

```bash
python research/benchmarks/agent_routing/audit.py
python -m unittest discover -s research/benchmarks/agent_routing/tests -v
```

No model downloads, API keys or third-party packages are needed. The audit checks the
`SOURCE.json` byte hashes, the manifest's case/request/question hashes, the frozen option order,
every case ID, band, reference tier and split, and the manifest's per-band, per-tier and
per-split counts. When runs are archived under `results/v1/`, it also checks request equality,
statuses, duplicate `(id, repeat)` pairs, that every saved prediction re-decodes from its raw
answers, and failure accounting, before recomputing metrics. With no archived runs it reports
"no archived results" and passes on the data and prompts alone.

**Results.** No results are archived yet. Once a run is audited it is archived under
`results/v1/<run>/` (`metadata.json` + `raw.jsonl`), `audit.py --write-summary` writes
`results/v1/summary.json`, and `render_cards.py` prints the results table from that file. Numbers
in this README will only ever be copied from that generated table.

## Run Laya on your machine

Use the repository's normal installation instructions. Run a smoke test first (downloads only the
runtime files of the chosen checkpoint, then runs offline):

```bash
PYTHONPATH=. python research/benchmarks/agent_routing/run.py --backend laya \
  --model convaiinnovations/laya --device cpu --limit 3 --repeats 1 \
  --output /tmp/routing-laya-smoke
python research/benchmarks/agent_routing/audit.py --run-dir /tmp/routing-laya-smoke
```

Use `--checkpoint DIR` instead of `--model` for a local checkpoint directory (no download), and
`--subfolder NAME` for a checkpoint bundled inside a repository (e.g. `typed-decisions`). For the
full protocol omit `--limit` and `--repeats` (defaults: 117 cases, 3 repeats). `--split tune|held`
restricts the run to one half; `--device cuda|mps` when available; `--threads N` pins torch
intra-op threads (inter-op is then set to 1, since each call is a single forward pass). Every run
needs a new `--output` directory.

The runner calls `agent.system_one(state, questions)` once per request, after one excluded warmup,
in a seeded shuffled order. Each line of `raw.jsonl` records `id`, `band`, `expected`, `split`,
`repeat`, `request_sha256`, `status` (`ok`/`error`), the raw `answers`, `predicted`, `p_chosen` and
`latency_ms`. `p_chosen` is the probability of the chosen option, not the SDK's `confidence` field
(an entropy-based score on a different scale); ECE is computed on `p_chosen`. `metadata.json`
records the Laya version and source commit, torch/transformers versions, device, thread count,
checkpoint config, the sha256 of `model.safetensors`, and this benchmark's source commit.
Exceptions are recorded by type only; failed requests are never retried and stay in the
denominator.

## Optional judge comparison

Any other router — an LLM judge, a rules engine, a person — can be scored on the same footing by
replaying its decisions:

```bash
python research/benchmarks/agent_routing/run.py --backend judge-replay \
  --model my-judge --decisions decisions.jsonl --output /tmp/routing-judge
python research/benchmarks/agent_routing/audit.py --run-dir /tmp/routing-judge
```

`decisions.jsonl` has one line per request: `{"id": "t01", "band": "trivial"}` (a band or a tier
name; an optional integer `repeat` for several passes). A case with no decision is written as a
failed request, not dropped. Replayed rows have the same shape as Laya rows with `latency_ms` and
`p_chosen` null, so ECE and latency are reported as not applicable. The runner makes no network
calls; how the decisions were produced (and what that cost) is outside this benchmark.

## Files and design choices

| Path | Purpose |
|---|---|
| `data/cases.jsonl` | 117 public requests: `id`, `band`, `expected` tier, `prompt`, `check_kind`, `split` |
| `data/questions.json` | Byte copy of the frozen tier question, key order preserved |
| `data/manifest.json` | Counts, hashes, label policy, band→tier map, option order, repetitions, latency policy |
| `SOURCE.json` | Source repository and commit, and byte hashes of the frozen data files |
| `prompts.py` | Label policy, question loading, request construction and answer decoding |
| `run.py` | Opt-in Laya runner and judge replay; fresh output directories only |
| `audit.py`, `metrics.py`, `tests/` | Model-free validation, scoring, corruption and failure tests |
| `render_cards.py` | Prints the markdown results table from `results/v1/summary.json` (stdlib) |

Quality metrics use the preselected first repeat (`repeat == 0`), not the best repeat: accuracy,
macro-F1, overspend and underspend counts, ECE of `p_chosen` in 10 equal-width bins, and the
confusion matrix, overall and per band and per split. Repeat consistency counts cases whose tier
is identical across all repeats. Latency p50/p95 cover every successful timed request across
repeats, excluding the warmup.

`split` is a fixed seeded half-split (`tune` 59 / `held` 58, stratified by band). If you tune a
wording, option order, threshold or combiner, tune on `tune` and report `held`; a score tuned on
all 117 rows is not a held-out result. `check_kind` (`exact`, `rubric`, `python`, `sql`) records
how the original A/B graded the answers to each request; answer grading is not part of this
benchmark, and no gold answers are included.

## Limitations and attribution

117 English requests written by one team for one platform's A/B, with bands assigned by their
author and not independently multi-annotated. Band is a proxy for tier: a `hard` request may in
some settings be answerable by a mid-sized model, and the fold `trivial`/`easy` → `small` makes the
label distribution uneven (60 / 29 / 28). The state is the bare request; routers that see session
history or tool results are answering a different question. Results depend on checkpoint, option
order, question wording and calibration, and a 117-row set cannot separate small wording or order
effects from noise. This is a regression diagnostic, not a general model ranking.

Label policy: glukicov/laya_router (Apache-2.0), see NOTICE above. Requests, split and runner:
contributed from an internal agent-platform A/B (source commit in `SOURCE.json`), with Claude Code
assistance, structured after the `feishu_zh` contribution. No credentials, private requests, gold
answers, host names or model weights are included.
