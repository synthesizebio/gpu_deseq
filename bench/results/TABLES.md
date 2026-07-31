# Current standard-pipeline results

R reference: R 4.6.0, DESeq2 1.52.0, apeglm 1.34.0. The GTEx GPU stages affected by count replacement/refitting are pending a new A100 run.

## End-to-end wall time (ms)

| dataset | P | n | R | eager | graph | Triton | best measured speedup |
|---|--:|--:|--:|--:|--:|--:|--:|
| airway | 5 | 8 | 41570 | 2521 | 1650 | 1501 | 27.7× |
| airway_cell | 4 | 8 | 36080 | 4178 | 3621 | 3469 | 10.4× |
| airway_dex | 2 | 8 | 30932 | 1733 | 1414 | 1361 | 22.7× |
| gtex_blood_muscle | 2 | 300 | 471588 | pending | pending | pending | pending |
| pasilla | 2 | 7 | 11091 | 1149 | 801 | 702 | 15.8× |
| pasilla_2fac | 3 | 7 | 12137 | 1511 | 946 | 866 | 14.0× |

## Per-substep wall time (ms)

| dataset | substep | R | eager | graph | Triton |
|---|---|--:|--:|--:|--:|
| airway | normalization | 223 | 1 | 1 | 1 |
| airway | dispersion | 8077 | 1590 | 715 | 583 |
| airway | glm_fit | 12686 | 57 | 57 | 57 |
| airway | significance | 872 | 119 | 119 | 119 |
| airway | lfc_shrink | 19712 | 754 | 758 | 741 |
| airway_cell | normalization | 183 | 1 | 1 | 1 |
| airway_cell | dispersion | 5794 | 750 | 182 | 60 |
| airway_cell | glm_fit | 9584 | 33 | 33 | 33 |
| airway_cell | significance | 362 | 117 | 116 | 116 |
| airway_cell | lfc_shrink | 20157 | 3277 | 3290 | 3259 |
| airway_dex | normalization | 189 | 1 | 1 | 1 |
| airway_dex | dispersion | 4707 | 429 | 109 | 52 |
| airway_dex | glm_fit | 9924 | 53 | 53 | 53 |
| airway_dex | significance | 1397 | 146 | 144 | 145 |
| airway_dex | lfc_shrink | 14715 | 1104 | 1107 | 1111 |
| gtex_blood_muscle | normalization | 3115 | 4 | 4 | 4 |
| gtex_blood_muscle | dispersion | 188957 | 3501 | 3620 | 200 |
| gtex_blood_muscle | glm_fit | 239691 | pending | pending | pending |
| gtex_blood_muscle | significance | 496 | pending | pending | pending |
| gtex_blood_muscle | lfc_shrink | 39330 | pending | pending | pending |
| pasilla | normalization | 143 | 1 | 1 | 1 |
| pasilla | dispersion | 1722 | 422 | 78 | 21 |
| pasilla | glm_fit | 3878 | 36 | 36 | 37 |
| pasilla | significance | 206 | 62 | 61 | 62 |
| pasilla | lfc_shrink | 5143 | 628 | 625 | 581 |
| pasilla_2fac | normalization | 145 | 1 | 1 | 1 |
| pasilla_2fac | dispersion | 2140 | 730 | 162 | 83 |
| pasilla_2fac | glm_fit | 3755 | 327 | 329 | 329 |
| pasilla_2fac | significance | 185 | 60 | 62 | 59 |
| pasilla_2fac | lfc_shrink | 5912 | 393 | 392 | 395 |

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
| gtex_blood_muscle | normalization | max rel | 4.97848e-15 | ≤1e-06 | PASS |
| gtex_blood_muscle | dispersion | p95 rel | 0.00426616 | ≤0.1 | PASS |
| gtex_blood_muscle | glm_fit | p95 |Δ| | 1.78371e-05 | ≤0.01 | PASS |
| gtex_blood_muscle | significance | Jaccard@.05 | 0.999971 | ≥0.95 | PASS |
| gtex_blood_muscle | lfc_shrink | Pearson r | 0.999998 | ≥0.9 | PASS |
| pasilla | normalization | max rel | 2.53595e-15 | ≤1e-06 | PASS |
| pasilla | dispersion | p95 rel | 0.000667198 | ≤0.1 | PASS |
| pasilla | glm_fit | p95 |Δ| | 2.34211e-05 | ≤0.01 | PASS |
| pasilla | significance | Jaccard@.05 | 1 | ≥0.95 | PASS |
| pasilla | lfc_shrink | Pearson r | 0.999997 | ≥0.9 | PASS |
| pasilla_2fac | normalization | max rel | 2.53595e-15 | ≤1e-06 | PASS |
| pasilla_2fac | dispersion | p95 rel | 0.00023554 | ≤0.1 | PASS |
| pasilla_2fac | glm_fit | p95 |Δ| | 1.45539e-05 | ≤0.01 | PASS |
| pasilla_2fac | significance | Jaccard@.05 | 1 | ≥0.95 | PASS |
| pasilla_2fac | lfc_shrink | Pearson r | 0.999991 | ≥0.9 | PASS |
