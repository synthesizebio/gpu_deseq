# Current standard-pipeline results

R reference: R 4.6.0, DESeq2 1.52.0, apeglm 1.34.0. GPU measurements use one A100-SXM4-40GB.

## Practical GPU acceleration versus 12-worker R (ms)

Each cell is the median of five direct, complete workflow observations. The CPU baseline uses `BiocParallel::MulticoreParam(12)` with BLAS and OpenMP pinned to one thread per worker.

| dataset | P | n | R, 12 workers | eager | graph | Triton | best GPU vs. R |
|---|--:|--:|--:|--:|--:|--:|--:|
| airway | 5 | 8 | 12099 | 2479 | 1552 | 1401 | 8.6× |
| airway_cell | 4 | 8 | 10421 | 3596 | 3003 | 2870 | 3.6× |
| airway_dex | 2 | 8 | 11856 | 1332 | 858 | 790 | 15.0× |
| gtex_blood_muscle | 2 | 300 | 63445 | 5284 | 4910 | 1807 | 35.1× |
| pasilla | 2 | 7 | 7336 | 972 | 588 | 525 | 14.0× |
| pasilla_2fac | 3 | 7 | 7371 | 1541 | 932 | 848 | 8.7× |

## Controlled serial stage diagnostic (ms)

This table preserves matched stage boundaries for attributing where time is spent. Its R column deliberately uses one worker to isolate algorithmic work; it is not the practical CPU baseline and no headline acceleration is computed from it. Totals are sums of stage medians, not direct observations.

| dataset | P | n | R, 1 worker (diagnostic) | eager | graph | Triton |
|---|--:|--:|--:|--:|--:|--:|
| airway | 5 | 8 | 32961 | 2470 | 1545 | 1386 |
| airway_cell | 4 | 8 | 30122 | 3593 | 2982 | 2852 |
| airway_dex | 2 | 8 | 26070 | 1296 | 888 | 773 |
| gtex_blood_muscle | 2 | 300 | 287354 | 5081 | 4749 | 1678 |
| pasilla | 2 | 7 | 9573 | 966 | 583 | 524 |
| pasilla_2fac | 3 | 7 | 10104 | 1528 | 925 | 837 |

## Controlled per-stage diagnostic (ms)

The R values below are the same one-worker diagnostic observations, not the practical baseline used in the acceleration table above.

| dataset | substep | R, 1 worker (diagnostic) | eager | graph | Triton |
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
