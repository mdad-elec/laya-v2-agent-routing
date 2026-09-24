# Rung 3 — calibration

Fitted on the 149 headline tune rows of the adopted configuration (root, bare, example-led): the tier
question's `choice:3-5` temperature 1.760 → **1.097** (inside the SDK's [0.5, 5] clamp); tune ECE 0.150 → 0.071,
signed gap −0.150 → −0.050 (`r3-temperature-fit.json`). Served from `~/laya-ckpts/laya-root-cal`
(MANIFEST names the base, the weights' sha256 `891102d3…`, the temperature and the schema sha;
`check_questions.py --served` confirmed the schema).

**The one held-out read (151 rows):** accuracy 0.629 → 0.629 (argmax identical on all 151, as a
temperature must leave it); ECE 0.122 → 0.112; signed gap −0.055 → +0.042.

**Not adopted:** PREREG's rule is held ECE ≤ 0.08. Calibration moves the brain from modestly
under-confident to modestly over-confident without reaching the bar, so the uncalibrated root
stays and `DSS_ROUTER_MIN_CONFIDENCE` stays 0.35. Recorded as "no improvement" for rung 3.
