# Rung 1 — checkpoint × state shape × wording (tune split)

Pooled over the headline tune rows of tier-240 and the pilot; DSS rows apart. Selection optimism for 21 configurations on 149 rows: ±0.071.

| checkpoint | state | wording / order | n | acc | macro-F1 | under | over | mean p | p<0.45 | p<0.35 | signed gap | in tok | p50 ms | DSS acc (n) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| laya-root | bare | example-led | 149 | 0.711 | 0.702 | 0.128 | 0.161 | 0.561 | 0.13 | 0.01 | -0.150 | 126 | 627 | 0.400 (30) |
| laya-root | wrapped-noenv | example-led | 149 | 0.691 | 0.691 | 0.081 | 0.228 | 0.508 | 0.37 | 0.00 | -0.183 | 157 | 998 | 0.500 (30) |
| laya-typed-decisions | bare | example-led | 149 | 0.685 | 0.680 | 0.168 | 0.148 | 0.514 | 0.20 | 0.00 | -0.170 | 126 | 916 | 0.500 (30) |
| laya-typed-decisions | bare | shipped | 149 | 0.658 | 0.624 | 0.221 | 0.121 | 0.483 | 0.38 | 0.00 | -0.175 | 138 | 862 | 0.667 (30) |
| laya-typed-decisions | wrapped-noenv | shipped | 149 | 0.658 | 0.627 | 0.282 | 0.060 | 0.462 | 0.50 | 0.02 | -0.196 | 169 | 885 | 0.567 (30) |
| laya-typed-decisions | wrapped | example-led | 149 | 0.611 | 0.613 | 0.081 | 0.309 | 0.432 | 0.64 | 0.01 | -0.178 | 340 | 1250 | 0.567 (30) |
| laya-typed-decisions | wrapped | shipped | 149 | 0.604 | 0.601 | 0.148 | 0.248 | 0.414 | 0.80 | 0.03 | -0.190 | 352 | 1345 | 0.500 (30) |
| laya-root | bare | shipped | 149 | 0.597 | 0.537 | 0.215 | 0.188 | 0.541 | 0.26 | 0.00 | -0.056 | 138 | 632 | 0.567 (30) |
| laya-root | wrapped-noenv | shipped | 149 | 0.597 | 0.520 | 0.315 | 0.087 | 0.520 | 0.28 | 0.01 | -0.077 | 169 | 956 | 0.567 (30) |
| laya-typed-decisions | wrapped | example-led-asshipped | 149 | 0.557 | 0.565 | 0.107 | 0.336 | 0.431 | 0.72 | 0.01 | -0.126 | 340 | 1167 | 0.500 (30) |
| laya-typed-decisions | wrapped-noenv | example-led | 149 | 0.537 | 0.545 | 0.174 | 0.289 | 0.478 | 0.40 | 0.00 | -0.059 | 157 | 813 | 0.533 (30) |
| laya-typed-decisions | bare | dss | 149 | 0.530 | 0.518 | 0.154 | 0.315 | 0.471 | 0.53 | 0.02 | -0.059 | 136 | 821 | 0.567 (30) |
| laya-typed-decisions | wrapped | shipped-asshipped | 149 | 0.523 | 0.531 | 0.148 | 0.329 | 0.414 | 0.89 | 0.02 | -0.110 | 352 | 1151 | 0.333 (30) |
| laya-root | bare | dss | 149 | 0.483 | 0.444 | 0.141 | 0.376 | 0.499 | 0.40 | 0.02 | +0.015 | 136 | 673 | 0.533 (30) |
| laya-root | wrapped-noenv | dss | 149 | 0.416 | 0.353 | 0.054 | 0.530 | 0.493 | 0.36 | 0.00 | +0.077 | 167 | 858 | 0.400 (30) |
| laya-typed-decisions | wrapped-noenv | dss | 149 | 0.409 | 0.388 | 0.101 | 0.490 | 0.423 | 0.77 | 0.07 | +0.013 | 167 | 790 | 0.533 (30) |
| laya-typed-decisions | wrapped | dss | 149 | 0.403 | 0.338 | 0.020 | 0.577 | 0.405 | 0.85 | 0.07 | +0.002 | 350 | 1165 | 0.433 (30) |
| laya-root | wrapped | shipped | 149 | 0.376 | 0.290 | 0.027 | 0.597 | 0.505 | 0.34 | 0.00 | +0.129 | 352 | 1196 | 0.300 (30) |
| laya-typed-decisions | wrapped | dss-asshipped | 149 | 0.362 | 0.278 | 0.013 | 0.624 | 0.417 | 0.78 | 0.01 | +0.055 | 350 | 1289 | 0.433 (30) |
| laya-root | wrapped | example-led | 149 | 0.342 | 0.231 | 0.000 | 0.658 | 0.518 | 0.26 | 0.01 | +0.176 | 340 | 1186 | 0.333 (30) |
| laya-root | wrapped | dss | 149 | 0.302 | 0.156 | 0.013 | 0.685 | 0.519 | 0.15 | 0.00 | +0.217 | 350 | 1259 | 0.367 (30) |

**Selected for the one held-out read:** `laya-root / bare / example-led` (tune accuracy 0.711, underspend 0.128).
