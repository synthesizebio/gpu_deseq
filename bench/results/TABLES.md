## Table 1 — total pipeline wall time (ms) and speedup vs R

| dataset | P | n | R | pydeseq2 | eager | graph | triton | best cuDESeq2 vs R | pydeseq2 vs R |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| airway | 5 | 8 | 33456 | 34695 | 2059 | 1341 | 2078 | 25.0× (graph) | 1.0× |
| airway_cell | 4 | 8 | 27899 | 36532 | 4284 | 3601 | 3469 | 8.0× (triton) | 0.8× |
| airway_dex | 2 | 8 | 25664 | 30289 | 1705 | 1390 | 1339 | 19.2× (triton) | 0.8× |
| gtex_blood_muscle | 2 | 300 | 350435 | 78203 | 8951 | 8560 | 4685 | 74.8× (triton) | 4.5× |
| pasilla | 2 | 7 | 9307 | 11911 | 1119 | 798 | 691 | 13.5× (triton) | 0.8× |
| pasilla_2fac | 3 | 7 | 10346 | 11845 | 1497 | 945 | 1499 | 10.9× (graph) | 0.9× |

## Table 2 — per-substep wall time (ms)

| dataset | substep | R | pydeseq2 | eager | graph | triton |
|---|---|--:|--:|--:|--:|--:|
| airway | normalization | 213 | 13 | 1 | 1 | 1 |
| airway | dispersion | 10423 | 15309 | 1124 | 401 | 1138 |
| airway | glm_fit | 4738 | 6528 | 55 | 55 | 55 |
| airway | significance | 365 | 2512 | 117 | 118 | 118 |
| airway | lfc_shrink | 17717 | 10332 | 762 | 766 | 765 |
| airway | **total** | 33456 | 34695 | 2059 | 1341 | 2078 |
| airway_cell | normalization | 201 | 13 | 1 | 1 | 1 |
| airway_cell | dispersion | 5724 | 15368 | 845 | 178 | 60 |
| airway_cell | glm_fit | 4190 | 5524 | 34 | 33 | 33 |
| airway_cell | significance | 362 | 2594 | 118 | 117 | 117 |
| airway_cell | lfc_shrink | 17423 | 13033 | 3287 | 3272 | 3257 |
| airway_cell | **total** | 27899 | 36532 | 4284 | 3601 | 3469 |
| airway_dex | normalization | 198 | 12 | 1 | 1 | 1 |
| airway_dex | dispersion | 6212 | 16879 | 425 | 109 | 52 |
| airway_dex | glm_fit | 5666 | 4198 | 52 | 51 | 52 |
| airway_dex | significance | 1299 | 2653 | 126 | 125 | 126 |
| airway_dex | lfc_shrink | 12290 | 6547 | 1102 | 1104 | 1108 |
| airway_dex | **total** | 25664 | 30289 | 1705 | 1390 | 1339 |
| gtex_blood_muscle | normalization | 2954 | 825 | 4 | 4 | 4 |
| gtex_blood_muscle | dispersion | 231276 | 43938 | 3501 | 3623 | 200 |
| gtex_blood_muscle | glm_fit | 42321 | 14517 | 404 | 402 | 424 |
| gtex_blood_muscle | significance | 17865 | 4931 | 2817 | 2801 | 2821 |
| gtex_blood_muscle | lfc_shrink | 56020 | 13993 | 2226 | 1731 | 1236 |
| gtex_blood_muscle | **total** | 350435 | 78203 | 8951 | 8560 | 4685 |
| pasilla | normalization | 175 | 5 | 1 | 1 | 1 |
| pasilla | dispersion | 2335 | 6405 | 407 | 85 | 20 |
| pasilla | glm_fit | 2138 | 1798 | 36 | 36 | 36 |
| pasilla | significance | 277 | 1032 | 61 | 61 | 62 |
| pasilla | lfc_shrink | 4382 | 2671 | 614 | 615 | 572 |
| pasilla | **total** | 9307 | 11911 | 1119 | 798 | 691 |
| pasilla_2fac | normalization | 176 | 5 | 1 | 1 | 1 |
| pasilla_2fac | dispersion | 2791 | 6153 | 712 | 161 | 713 |
| pasilla_2fac | glm_fit | 1502 | 1801 | 323 | 323 | 324 |
| pasilla_2fac | significance | 240 | 993 | 72 | 72 | 72 |
| pasilla_2fac | lfc_shrink | 5637 | 2893 | 389 | 389 | 389 |
| pasilla_2fac | **total** | 10346 | 11845 | 1497 | 945 | 1499 |

## Table 3 — output parity

cuDESeq2 (eager, representative) and PyDESeq2 (competitor) each scored against the same R~DESeq2 ground truth per substep. The PASS/FAIL verdict and its tolerance apply to cuDESeq2 (our bit-exactness claim); the PyDESeq2 column is shown for comparison. `GPU Δ` is the largest disagreement among the three cuDESeq2 GPU modes (0 ⇒ bit-identical).

| dataset | substep | metric | cuDESeq2 vs R | PyDESeq2 vs R | tol | cuDESeq2 verdict | GPU Δ |
|---|---|---|--:|--:|--:|:--:|--:|
| airway | normalization | max rel | 4.14e-15 | 4.14e-15 | ≤1e-06 | ✅ PASS | 0.0e+00 |
| airway | dispersion | p95 rel | 0.0669 | 0.409 | ≤0.1 | ✅ PASS | 8.9e-15 |
| airway | glm_fit | p95 |Δ| | 0.00483 | 0.0359 | ≤0.01 | ✅ PASS | 7.8e-14 |
| airway | significance | Jaccard@.05 | 0.977 | 0.915 | ≥0.95 | ✅ PASS | 2.8e-13 |
| airway | lfc_shrink | Pearson r | 1 (ρ=1.000) | 0.988 | ≥0.9 | ✅ PASS | 8.4e-10 |
| airway_cell | normalization | max rel | 4.14e-15 | 4.14e-15 | ≤1e-06 | ✅ PASS | 0.0e+00 |
| airway_cell | dispersion | p95 rel | 0.00284 | 0.212 | ≤0.1 | ✅ PASS | 8.1e-08 |
| airway_cell | glm_fit | p95 |Δ| | 8.09e-05 | 0.0131 | ≤0.01 | ✅ PASS | 4.3e-09 |
| airway_cell | significance | Jaccard@.05 | 0.971 | 0.901 | ≥0.95 | ✅ PASS | 1.2e-08 |
| airway_cell | lfc_shrink | Pearson r | 0.997 (ρ=0.996) | 0.96 | ≥0.9 | ✅ PASS | 1.7e-04 |
| airway_dex | normalization | max rel | 4.14e-15 | 4.14e-15 | ≤1e-06 | ✅ PASS | 0.0e+00 |
| airway_dex | dispersion | p95 rel | 0.00104 | 0.137 | ≤0.1 | ✅ PASS | 2.3e-07 |
| airway_dex | glm_fit | p95 |Δ| | 2.5e-05 | 0.00611 | ≤0.01 | ✅ PASS | 1.8e-09 |
| airway_dex | significance | Jaccard@.05 | 1 | 0.977 | ≥0.95 | ✅ PASS | 1.5e-07 |
| airway_dex | lfc_shrink | Pearson r | 0.999 (ρ=1.000) | 0.999 | ≥0.9 | ✅ PASS | 8.0e-07 |
| gtex_blood_muscle | normalization | max rel | 5.69e-15 | 5.47e-15 | ≤1e-06 | ✅ PASS | 0.0e+00 |
| gtex_blood_muscle | dispersion | p95 rel | 0.00371 | 0.00604 | ≤0.1 | ✅ PASS | 2.6e-01 |
| gtex_blood_muscle | glm_fit | p95 |Δ| | 1.81e-05 | 4.52e-05 | ≤0.01 | ✅ PASS | 1.2e-04 |
| gtex_blood_muscle | significance | Jaccard@.05 | 0.991 | 0.971 | ≥0.95 | ✅ PASS | 1.8e-03 |
| gtex_blood_muscle | lfc_shrink | Pearson r | 1 (ρ=1.000) | 1 | ≥0.9 | ✅ PASS | 5.7e-03 |
| pasilla | normalization | max rel | 2.93e-15 | 2.93e-15 | ≤1e-06 | ✅ PASS | 0.0e+00 |
| pasilla | dispersion | p95 rel | 0.000667 | 0.121 | ≤0.1 | ✅ PASS | 1.9e-07 |
| pasilla | glm_fit | p95 |Δ| | 2.34e-05 | 0.00821 | ≤0.01 | ✅ PASS | 1.3e-09 |
| pasilla | significance | Jaccard@.05 | 1 | 0.974 | ≥0.95 | ✅ PASS | 1.8e-06 |
| pasilla | lfc_shrink | Pearson r | 1 (ρ=1.000) | 0.995 | ≥0.9 | ✅ PASS | 5.4e-07 |
| pasilla_2fac | normalization | max rel | 2.93e-15 | 2.93e-15 | ≤1e-06 | ✅ PASS | 0.0e+00 |
| pasilla_2fac | dispersion | p95 rel | 0.000235 | 0.233 | ≤0.1 | ✅ PASS | 1.8e-08 |
| pasilla_2fac | glm_fit | p95 |Δ| | 1.46e-05 | 0.0216 | ≤0.01 | ✅ PASS | 3.1e-09 |
| pasilla_2fac | significance | Jaccard@.05 | 1 | 0.939 | ≥0.95 | ✅ PASS | 1.4e-06 |
| pasilla_2fac | lfc_shrink | Pearson r | 1 (ρ=1.000) | 0.995 | ≥0.9 | ✅ PASS | 4.5e-08 |
