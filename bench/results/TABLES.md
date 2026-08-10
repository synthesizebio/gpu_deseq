# Current standard-pipeline results

R reference: R 4.6.0, DESeq2 1.52.0, apeglm 1.34.0. GPU timings are from one A100-SXM4-40GB standard-pipeline run.
End-to-end R timings use the same standard call path with one and 12 BiocParallel workers; the speedup column uses the faster R result.

## End-to-end wall time (ms)

| dataset | P | n | R, 1 worker | R, 12 workers | eager | graph | Triton | best cuDESeq2 vs. best R |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| airway | 5 | 8 | 34739 | 12044 | 2559 | 1681 | 1504 | 8.0× |
| airway_cell | 4 | 8 | 29417 | 10887 | 4169 | 3579 | 3495 | 3.1× |
| airway_dex | 2 | 8 | 25948 | 12412 | 1307 | 968 | 898 | 13.8× |
| gtex_blood_muscle | 2 | 300 | 277686 | 63471 | 8510 | 8293 | 4754 | 13.4× |
| pasilla | 2 | 7 | 9493 | 7172 | 982 | 650 | 717 | 11.0× |
| pasilla_2fac | 3 | 7 | 10120 | 7423 | 1481 | 922 | 841 | 8.8× |
## Per-substep wall time (ms)
These substep timings are one-worker measurements. DESeq2 parallelizes some stages together, so they cannot be partitioned into comparable 12-worker substeps.
| dataset | substep | R | eager | graph | Triton |
|---|---|--:|--:|--:|--:|
| airway | normalization | 223 | 1 | 1 | 1 |
| airway | dispersion | 8077 | 1586 | 713 | 581 |
| airway | glm_fit | 12686 | 62 | 60 | 60 |
| airway | significance | 872 | 117 | 117 | 117 |
| airway | lfc_shrink | 19712 | 793 | 790 | 746 |
| airway_cell | normalization | 183 | 1 | 1 | 1 |
| airway_cell | dispersion | 5794 | 748 | 186 | 60 |
| airway_cell | glm_fit | 9584 | 37 | 36 | 36 |
| airway_cell | significance | 362 | 114 | 115 | 115 |
| airway_cell | lfc_shrink | 20157 | 3268 | 3240 | 3283 |
| airway_dex | normalization | 189 | 1 | 1 | 1 |
| airway_dex | dispersion | 4707 | 429 | 109 | 51 |
| airway_dex | glm_fit | 9924 | 55 | 53 | 53 |
| airway_dex | significance | 1397 | 145 | 124 | 123 |
| airway_dex | lfc_shrink | 14715 | 676 | 681 | 670 |
| gtex_blood_muscle | normalization | 3115 | 4 | 4 | 4 |
| gtex_blood_muscle | dispersion | 188957 | 3500 | 3623 | 200 |
| gtex_blood_muscle | glm_fit | 239691 | 3147 | 2810 | 2709 |
| gtex_blood_muscle | significance | 496 | 623 | 623 | 621 |
| gtex_blood_muscle | lfc_shrink | 39330 | 1235 | 1233 | 1220 |
| pasilla | normalization | 143 | 1 | 1 | 1 |
| pasilla | dispersion | 1722 | 402 | 71 | 20 |
| pasilla | glm_fit | 3878 | 38 | 37 | 37 |
| pasilla | significance | 206 | 63 | 63 | 63 |
| pasilla | lfc_shrink | 5143 | 478 | 478 | 596 |
| pasilla_2fac | normalization | 145 | 1 | 1 | 1 |
| pasilla_2fac | dispersion | 2140 | 708 | 161 | 80 |
| pasilla_2fac | glm_fit | 3755 | 320 | 321 | 322 |
| pasilla_2fac | significance | 185 | 74 | 60 | 60 |
| pasilla_2fac | lfc_shrink | 5912 | 378 | 379 | 378 |

## Output parity against DESeq2 1.52.0

| dataset | substep | metric | value | tolerance | verdict |
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
