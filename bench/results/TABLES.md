# Current standard-pipeline results

R reference: R 4.6.0, DESeq2 1.52.0, apeglm 1.34.0. GPU measurements use one A100-SXM4-40GB.

## Practical GPU acceleration versus 12-worker R (ms)

Each cell is the median of five direct, complete workflow observations. The CPU baseline uses `BiocParallel::MulticoreParam(12)` with BLAS and OpenMP pinned to one thread per worker.

| dataset | P | n | R, 12 workers | eager | graph | Triton | best GPU vs. R |
|---|--:|--:|--:|--:|--:|--:|--:|
| airway | 5 | 8 | 12099 | 2479 | 1552 | **1401** | 8.6× |
| airway_cell | 4 | 8 | 10421 | 3596 | 3003 | **2870** | 3.6× |
| airway_dex | 2 | 8 | 11856 | 1332 | 858 | **790** | 15.0× |
| gtex_blood_muscle | 2 | 300 | 63445 | 5284 | 4910 | **1807** | 35.1× |
| pasilla | 2 | 7 | 7336 | 972 | 588 | **525** | 14.0× |
| pasilla_2fac | 3 | 7 | 7371 | 1541 | 932 | **848** | 8.7× |

## GPU pipeline-stage measurements (ms)

Bold values identify the fastest GPU mode for each measured stage. Component totals sum independently measured stage medians and are not direct end-to-end observations.

| dataset | substep | eager | graph | Triton |
|---|---|--:|--:|--:|
| airway | normalization | **1** | 1 | 1 |
| airway | dispersion | 1660 | 723 | **585** |
| airway | glm_fit | 62 | **62** | 64 |
| airway | significance | 74 | 74 | **73** |
| airway | lfc_shrink | 673 | 685 | **663** |
| airway | *component total* | 2470 | 1545 | 1386 |
| airway_cell | normalization | 1 | 1 | **1** |
| airway_cell | dispersion | 801 | 198 | **62** |
| airway_cell | glm_fit | 39 | **38** | 39 |
| airway_cell | significance | 78 | 78 | **75** |
| airway_cell | lfc_shrink | 2675 | **2667** | 2674 |
| airway_cell | *component total* | 3593 | 2982 | 2852 |
| airway_dex | normalization | 1 | 1 | **1** |
| airway_dex | dispersion | 581 | 141 | **56** |
| airway_dex | glm_fit | 58 | 59 | **57** |
| airway_dex | significance | **82** | 109 | 82 |
| airway_dex | lfc_shrink | **573** | 578 | 576 |
| airway_dex | *component total* | 1296 | 888 | 773 |
| gtex_blood_muscle | normalization | 4 | 4 | **4** |
| gtex_blood_muscle | dispersion | 3574 | 3612 | **217** |
| gtex_blood_muscle | glm_fit | 849 | 491 | **432** |
| gtex_blood_muscle | significance | 85 | **73** | 95 |
| gtex_blood_muscle | lfc_shrink | 570 | **569** | 929 |
| gtex_blood_muscle | *component total* | 5081 | 4749 | 1678 |
| pasilla | normalization | **1** | 1 | 1 |
| pasilla | dispersion | 463 | 78 | **21** |
| pasilla | glm_fit | 42 | 42 | **41** |
| pasilla | significance | 40 | **40** | 40 |
| pasilla | lfc_shrink | **420** | 422 | 421 |
| pasilla | *component total* | 966 | 583 | 524 |
| pasilla_2fac | normalization | 1 | 1 | **1** |
| pasilla_2fac | dispersion | 777 | 174 | **87** |
| pasilla_2fac | glm_fit | 354 | 354 | **352** |
| pasilla_2fac | significance | 50 | 50 | **49** |
| pasilla_2fac | lfc_shrink | 347 | **346** | 347 |
| pasilla_2fac | *component total* | 1528 | 925 | 837 |

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
