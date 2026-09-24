---
license: apache-2.0
language:
- en
pretty_name: Agent Routing Cells
task_categories:
- text-classification
tags:
- routing
- llm-routing
- reasoning-effort
- laya
- system-one
- benchmark
size_categories:
- 1K<n<10K
configs:
- config_name: cases
  data_files: data/cases.jsonl
- config_name: outcomes
  data_files: data/outcomes.jsonl
- config_name: decisions
  data_files: data/decisions.jsonl
- config_name: human_audit
  data_files: data/human_audit.jsonl
- config_name: requests_tier
  data_files: data/requests_tier.jsonl
---

# Agent Routing Cells

Requests an AI agent platform had to route, each answered by every **cell** — a (model × reasoning
effort) pair — and graded, so a router can be scored against measured outcomes rather than against
opinions. Built for the Laya-v2 study (routing an agent's turn to a band and an effort with a
System-One decision model); the method, pre-registration and results are in the accompanying report.

## What is in it

| file | one row is |
|---|---|
| `data/cases.jsonl` | an ask: its band (`trivial / easy / medium / hard`, folded to `small / medium / powerful`), the prompt, its check (`exact`, `python`, `sql` against a bundled fixture, or a `rubric`) and, for checkable asks, a gold answer |
| `data/outcomes.jsonl` | one cell's answer to one ask: status, latency, token usage, list-price cost, the effort the seat applied, the machine check or the blind panel's scores — and the answer as `answer_sha256`, `answer_chars` and its first 200 characters |
| `data/decisions.jsonl` | where each router arm sent each ask, what it cost, how it scored, and the routing decision the platform recorded (`routed`, when the run kept it) |
| `data/human_audit.jsonl` | a blind human score beside the panel's, for the audited slice |
| `data/requests_tier.jsonl` | the 180 borrowed tier-labelled requests, byte-identical |
| `data/splits.json` | the frozen tune / held-out splits every published number uses |

Counts are in `provenance.json`; `manifest.sha256` lists every file's sha256.

## How it was made

- **Asks.** The first 48 were written for the pilot; 72 more were drafted from the corpus's written
  band policy by fresh drafting sessions, validated in code (every gold passes its own check; every
  python/sql check carries a plausible wrong answer that must fail; no near duplicates; no private
  terms) and read by the owner. Asks naming the author's own systems were kept private and are not here.
- **Outcomes.** Every cell answered every ask once through the platform's gateway. Checkable asks are
  scored by the check; rubric asks by a blind cross-family panel (a grader never scores its own
  family, and never sees which model or effort wrote the answer). A failed answer is recorded as
  failed; a grade the panel could not give is recorded as unscored, never as zero.
- **The human slice.** A seeded tenth of the rubric answers was graded blind by a person; the
  agreement (Spearman ρ per band, with an interval) is in the report.

## Outputs

The answers came from commercial model seats whose terms assign outputs to the user; this dataset
still publishes grades, hashes and a short head rather than full text. Full answers are available on
request for research.

## Licence and attribution

Apache-2.0. `data/requests_tier.jsonl` is copied unchanged from
[glukicov/laya_router](https://github.com/glukicov/laya_router) at `a2278d969067` (Apache-2.0).

## Limitations

- One drafting process and one owner read for the asks; no inter-annotator agreement on their bands.
- The rubric half is panel-graded; it is human-calibrated only to the ρ the report measures.
- One platform's menu of models and efforts, English only, list prices on subscription seats.
- Runs whose ledger join predates keeping the routing decision carry `routed: null`
  (`provenance.json` names them).
