# Current standard-pipeline results

R reference: R 4.6.0, DESeq2 1.52.0, apeglm 1.34.0. GPU measurements use one A100-SXM4-40GB.

## Matched sum of stage medians (ms)

R and cuDESeq2 use the same five stage boundaries. These totals are sums of independently measured stage medians, not direct end-to-end observations.

| dataset | P | n | R, 1 worker | eager | graph | Triton | best cuDESeq2 vs. R |
|---|--:|--:|--:|--:|--:|--:|--:|
| airway | 5 | 8 | 32961 | 2470 | 1545 | 1386 | 23.8× |
| airway_cell | 4 | 8 | 30122 | 3593 | 2982 | 2852 | 10.6× |
| airway_dex | 2 | 8 | 26070 | 1296 | 888 | 773 | 33.7× |
| gtex_blood_muscle | 2 | 300 | 287354 | 5081 | 4749 | 1678 | 171.2× |
| pasilla | 2 | 7 | 9573 | 966 | 583 | 524 | 18.3× |
| pasilla_2fac | 3 | 7 | 10104 | 1528 | 925 | 837 | 12.1× |

## Direct end-to-end wall time (ms)

Each cell is the median of five complete workflow observations, including dataset construction and host-to-device transfer. These values are kept separate from the matched sums of stage medians above.

| dataset | n | R, 1 worker | eager | graph | Triton | best cuDESeq2 vs. R |
|---|--:|--:|--:|--:|--:|--:|
| airway | 8 | 33063 | 2479 | 1552 | 1401 | 23.6× |
| airway_cell | 8 | 30564 | 3596 | 3003 | 2870 | 10.6× |
| airway_dex | 8 | 26678 | 1332 | 858 | 790 | 33.8× |
| gtex_blood_muscle | 300 | 289337 | 5284 | 4910 | 1807 | 160.2× |
| pasilla | 7 | 9945 | 972 | 588 | 525 | 19.0× |
| pasilla_2fac | 7 | 10462 | 1541 | 932 | 848 | 12.3× |

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
| airway | normalization | 201 | 1 | 1 | 1 |
| airway | dispersion | 7863 | 1660 | 723 | 585 |
| airway | glm_fit | 4398 | 62 | 62 | 64 |
| airway | significance | 310 | 74 | 74 | 73 |
| airway | lfc_shrink | 20189 | 673 | 685 | 663 |
| airway_cell | normalization | 188 | 1 | 1 | 1 |
| airway_cell | dispersion | 5456 | 801 | 198 | 62 |
| airway_cell | glm_fit | 3875 | 39 | 38 | 39 |
| airway_cell | significance | 332 | 78 | 78 | 75 |
| airway_cell | lfc_shrink | 20271 | 2675 | 2667 | 2674 |
| airway_dex | normalization | 188 | 1 | 1 | 1 |
| airway_dex | dispersion | 4481 | 581 | 141 | 56 |
| airway_dex | glm_fit | 5171 | 58 | 59 | 57 |
| airway_dex | significance | 1387 | 82 | 109 | 82 |
| airway_dex | lfc_shrink | 14843 | 573 | 578 | 576 |
| gtex_blood_muscle | normalization | 3437 | 4 | 4 | 4 |
| gtex_blood_muscle | dispersion | 187567 | 3574 | 3612 | 217 |
| gtex_blood_muscle | glm_fit | 53421 | 849 | 491 | 432 |
| gtex_blood_muscle | significance | 456 | 85 | 73 | 95 |
| gtex_blood_muscle | lfc_shrink | 42474 | 570 | 569 | 929 |
| pasilla | normalization | 159 | 1 | 1 | 1 |
| pasilla | dispersion | 1782 | 463 | 78 | 21 |
| pasilla | glm_fit | 2079 | 42 | 42 | 41 |
| pasilla | significance | 209 | 40 | 40 | 40 |
| pasilla | lfc_shrink | 5343 | 420 | 422 | 421 |
| pasilla_2fac | normalization | 163 | 1 | 1 | 1 |
| pasilla_2fac | dispersion | 2179 | 777 | 174 | 87 |
| pasilla_2fac | glm_fit | 1495 | 354 | 354 | 352 |
| pasilla_2fac | significance | 184 | 50 | 50 | 49 |
| pasilla_2fac | lfc_shrink | 6082 | 347 | 346 | 347 |

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
