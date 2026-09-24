# Rung 2 — prompt, schema and combiner (tune only)

Search: 49 configurations on 149 headline tune rows. Winner `C:example-led/pms/i0/request` — tune 0.711, selection-adjusted 0.697. It is rung 1's configuration.

| stage | best three (tune accuracy) |
|---|---|
| A | `example-led` 0.711; `c06` 0.631; `shipped` 0.597 |
| B | `example-led/pms` 0.711; `example-led/msp` 0.678; `example-led/mps` 0.658 |
| C | `example-led/pms/i0/request` 0.711; `example-led/pms/i1/request` 0.698; `example-led/pms/i0/task` 0.685 |

- None of the eight new criteria sets beat upstream's example-led wording (best new: c06, 0.631).
- **Effort question: dropped by PREREG H6.** Measured effort labels over the 120 asks are 112 `minimal`; the constant `minimal` is 0.917 on tune, so the bar (+0.10) cannot be met.
- **v2 schema (band + effort + needs_tools / needs_code / long_output in one call):** the band answer is bit-for-bit the one-question answer (0.787 / 0.600 on the two tune corpora), latency ~2.3× on CPU.
- **Combiner: not adopted.** 5-fold CV on tune 0.665 against 0.711 for the band question's argmax.

**Rung 2: no improvement.** The adopted configuration is unchanged, so no held-out read is spent (it would re-read rung 1's configuration).
