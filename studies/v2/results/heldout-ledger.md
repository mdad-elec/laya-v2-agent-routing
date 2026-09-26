# Held-out reads — Laya-v2

One line per read, written in the same commit as the result it names (`scripts/laya/PREREG.md`, "One read per rung").

| date | rung | candidate | file |
|---|---|---|---|
| 2026-09-23 | R1 | laya-root / bare / example-led / powerful,medium,small (+ baselines: as-shipped, judge) | r1-held-candidate-root-bare-example-led-{tier-240,cells-pilot}.json, r1-held-baseline-asshipped-*.json, r1-held.md |
| 2026-09-24 | R3 | laya-root-cal (T 1.097 on choice:3-5) | r3-held-root-cal-{tier-240,cells-pilot}.json, r3-report.md |
| 2026-09-24 | R4 | laya-v2-policy (pre-declared default) | r4-held-v2-policy-{tier-240,cells-pilot}.json, r4-report.md |
| 2026-09-24 | R4 | laya-v2-outcome (pre-declared second variant) | r4-held-v2-outcome-{tier-240,cells-pilot}.json, r4-report.md |
| 2026-09-24 | H7 | routellm/bert_gpt4_augmented, t_lo 0.65 / t_hi 0.69 fitted on tune (amendment 3) | routellm-tier-240.jsonl, routellm-cells-pilot.jsonl, routellm.json, routellm.md |
| 2026-09-24 | H8 | vLLM-SR intent classifier (mmbert32k-intent-classifier-merged), category table fitted on tune (amendment 4) | vllmsr-tier-240.jsonl, vllmsr-cells-pilot.jsonl, vllmsr.json, vllmsr.md |
| 2026-09-24 | H9-A | Laya root zero-shot verifier cascade (amendment 5), τ 0.95 fitted on tune; oracle-verifier ceiling | cascade/verify-A.jsonl, cascade.json |
| 2026-09-24 | H9-B | fine-tuned Laya verifier (epoch 3 of 3 completed; epoch 4 NaN), cascade τ 0.90 and hybrid τ 0.05 fitted on tune | cascade/verify-B.jsonl, cascade.json, cascade.md |
