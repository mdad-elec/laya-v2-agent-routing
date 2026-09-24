---
license: apache-2.0
base_model: convaiinnovations/laya
base_model_relation: finetune
library_name: laya
pipeline_tag: text-classification
tags: [laya, system-one, routing, llm-routing, rlcd]
---

# laya-v2-agent-routing

A fine-tune of [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (the 421M English root) that
answers one question about an AI agent's incoming request: **which price tier should answer it — `small`, `medium` or
`powerful` — the cheapest that answers it well.** One forward pass, no generated tokens.

## Results (held-out, 151 requests)

| decider | band accuracy |
|---|---|
| **this model** | **0.801** |
| the base checkpoint, same wording and state | 0.629 |
| the platform's LLM judge (27B, JSON verdict) | 0.662 |

It beats the platform's LLM judge on the same held-out rows by +0.139 (exact McNemar, one-sided p = 0.0027). Held-out labels follow one annotator's policy whose blind-adjudication
ceiling is ~0.82. On a public benchmark whose label is "the cheapest model that solved it" (RouterBench) it does not
reach the single-model hull: it learned a spending policy, not a solvability predictor.

## How to ask it

```python
import laya
agent = laya.load("<this repo>")
agent.system_one({"request": "Design a rollout plan with a rollback"}, QUESTIONS)   # QUESTIONS: questions.json in this repo
```
The state is the bare request (a wrapped session state costs the base checkpoint ~37 points); the option order
`powerful, medium, small` is part of the prompt (`questions.json`, byte-identical to what it was trained on).

## Training

Upstream's RLCD recipe (policy gradient against proper scoring rules + soft cross-entropy), single GPU, fp16:
4 epochs, selected epoch 4, 0.096 h. Data: the tune half
of 240 tier-labelled requests (180 from glukicov/laya_router, Apache-2.0) and 60 pilot asks, plus label-preserving
paraphrases from an open 27B model filtered against every held-out request. Temperature refitted on a validation slice:
{'choice:3-5': 1.5184241704075447}. Weights sha256 `6098ad7664ef42ddcbdbd3d3b52da500f724ad8d7038762d6c2b5439fac17bb5`.

## Limitations

English; one platform's notion of "cheapest tier that answers well"; ~150 held-out requests (±0.07); the policy labels
are one annotator's. Not a model-selection oracle for arbitrary model menus.
