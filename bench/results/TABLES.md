# Current standard-pipeline results

R reference: R 4.6.0, DESeq2 1.52.0, apeglm 1.34.0. GPU measurements use one A100-SXM4-40GB.

## Matched sum of stage medians (ms)

R and cuDESeq2 use the same five stage boundaries. These totals are sums of independently measured stage medians, not direct end-to-end observations.

| dataset | P | n | R, 1 worker | eager | graph | Triton | best cuDESeq2 vs. R |
|---|--:|--:|--:|--:|--:|--:|--:|
| airway | 5 | 8 | 32472 | 2578 | 1707 | 1531 | 21.2× |
| airway_cell | 4 | 8 | 30326 | 4260 | 3665 | 3629 | 8.4× |
| airway_dex | 2 | 8 | 26320 | 1330 | 1010 | 944 | 27.9× |
| gtex_blood_muscle | 2 | 300 | 279604 | 8526 | 8284 | 4744 | 58.9× |
| pasilla | 2 | 7 | 9485 | 1022 | 673 | 751 | 14.1× |
| pasilla_2fac | 3 | 7 | 9999 | 1536 | 960 | 876 | 11.4× |

## Direct R end-to-end wall time: one vs. 12 workers (ms)

Each cell is the median of direct `DESeq() + results() + lfcShrink()` observations. This separately collected table is not combined with the stage-summed GPU measurements above.

| dataset | n | 1 worker | 12 workers |
|---|--:|--:|--:|
| airway | 8 | 32130 | 12099 |
| airway_cell | 8 | 29341 | 10421 |
| airway_dex | 8 | 25968 | 11856 |
| gtex_blood_muscle | 300 | 276016 | 63445 |
| pasilla | 7 | 9447 | 7336 |
| pasilla_2fac | 7 | 10200 | 7371 |

## Per-stage wall time (ms)

| dataset | substep | R | eager | graph | Triton |
|---|---|--:|--:|--:|--:|
| airway | normalization | 189 | 1 | 1 | 1 |
| airway | dispersion | 7463 | 1586 | 719 | 583 |
| airway | glm_fit | 4438 | 62 | 61 | 61 |
| airway | significance | 895 | 118 | 119 | 120 |
| airway | lfc_shrink | 19488 | 811 | 808 | 767 |
| airway_cell | normalization | 190 | 1 | 1 | 1 |
| airway_cell | dispersion | 5668 | 753 | 185 | 60 |
| airway_cell | glm_fit | 3850 | 38 | 36 | 37 |
| airway_cell | significance | 863 | 115 | 115 | 119 |
| airway_cell | lfc_shrink | 19755 | 3353 | 3327 | 3412 |
| airway_dex | normalization | 182 | 1 | 1 | 1 |
| airway_dex | dispersion | 4732 | 427 | 110 | 52 |
| airway_dex | glm_fit | 5205 | 57 | 56 | 56 |
| airway_dex | significance | 1391 | 145 | 126 | 124 |
| airway_dex | lfc_shrink | 14810 | 699 | 716 | 711 |
| gtex_blood_muscle | normalization | 3262 | 4 | 4 | 4 |
| gtex_blood_muscle | dispersion | 184599 | 3507 | 3625 | 200 |
| gtex_blood_muscle | glm_fit | 51134 | 3148 | 2794 | 2694 |
| gtex_blood_muscle | significance | 492 | 613 | 610 | 609 |
| gtex_blood_muscle | lfc_shrink | 40117 | 1254 | 1252 | 1237 |
| pasilla | normalization | 153 | 1 | 1 | 1 |
| pasilla | dispersion | 1760 | 417 | 71 | 20 |
| pasilla | glm_fit | 2035 | 39 | 39 | 39 |
| pasilla | significance | 223 | 62 | 61 | 62 |
| pasilla | lfc_shrink | 5314 | 504 | 501 | 630 |
| pasilla_2fac | normalization | 153 | 1 | 1 | 1 |
| pasilla_2fac | dispersion | 2134 | 730 | 163 | 83 |
| pasilla_2fac | glm_fit | 1465 | 334 | 338 | 335 |
| pasilla_2fac | significance | 204 | 75 | 60 | 59 |
| pasilla_2fac | lfc_shrink | 6043 | 396 | 399 | 398 |

## Output parity against DESeq2 1.52.0

The retained reference-parity artifact scores eager mode; graph and Triton differences are retained separately in `parity.json`.

| dataset | substep | metric | eager vs. R | tolerance | verdict |
|---|---|---|--:|--:|:--:|
| airway | normalization | max rel | 4.14162e-15 | ≤1e-06 | PASS |
| airway | dispersion | p95 rel | 0.000448875 | ≤0.1 | PASS |
| airway | glm_fit | p95 |Δ| | 3.03784e-05 | ≤0.01 | PASS |
| airway | significance | Jaccard@.05 | 1 | ≥0.95 | PASS |
| airway | lfc_shrink | Pearson r | 0.999998 | ≥0.9 | PASS |
| airway_cell | normalization | max rel | 4.14162e-15 | ≤1e-06 | PASS |
| airway_cell | dispersion | p95 rel | 0.00283559 | ≤0.1 | PASS |
| airway_cell | glm_fit | p95 |Δ| | 8.0882e-05 | ≤0.01 | PASS |
| airway_cell | significance | Jaccard@.05 | 1 | ≥0.95 | PASS |
| airway_cell | lfc_shrink | Pearson r | 0.997224 | ≥0.9 | PASS |
| airway_dex | normalization | max rel | 4.14162e-15 | ≤1e-06 | PASS |
| airway_dex | dispersion | p95 rel | 0.00104057 | ≤0.1 | PASS |
| airway_dex | glm_fit | p95 |Δ| | 2.50371e-05 | ≤0.01 | PASS |
| airway_dex | significance | Jaccard@.05 | 0.99963 | ≥0.95 | PASS |
| airway_dex | lfc_shrink | Pearson r | 0.998593 | ≥0.9 | PASS |
| gtex_blood_muscle | normalization | max rel | 5.69019e-15 | ≤1e-06 | PASS |
| gtex_blood_muscle | dispersion | p95 rel | 0.00439594 | ≤0.1 | PASS |
| gtex_blood_muscle | glm_fit | p95 |Δ| | 1.79702e-05 | ≤0.01 | PASS |
| gtex_blood_muscle | significance | Jaccard@.05 | 0.999971 | ≥0.95 | PASS |
| gtex_blood_muscle | lfc_shrink | Pearson r | 0.999998 | ≥0.9 | PASS |
| pasilla | normalization | max rel | 2.9261e-15 | ≤1e-06 | PASS |
| pasilla | dispersion | p95 rel | 0.000667198 | ≤0.1 | PASS |
| pasilla | glm_fit | p95 |Δ| | 2.34212e-05 | ≤0.01 | PASS |
| pasilla | significance | Jaccard@.05 | 1 | ≥0.95 | PASS |
| pasilla | lfc_shrink | Pearson r | 0.999997 | ≥0.9 | PASS |
| pasilla_2fac | normalization | max rel | 2.9261e-15 | ≤1e-06 | PASS |
| pasilla_2fac | dispersion | p95 rel | 0.000235398 | ≤0.1 | PASS |
| pasilla_2fac | glm_fit | p95 |Δ| | 1.45539e-05 | ≤0.01 | PASS |
| pasilla_2fac | significance | Jaccard@.05 | 1 | ≥0.95 | PASS |
| pasilla_2fac | lfc_shrink | Pearson r | 0.999991 | ≥0.9 | PASS |
