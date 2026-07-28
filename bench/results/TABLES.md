## Table 1 — total pipeline wall time (ms) and speedup vs R

| dataset | P | n | R | pydeseq2 | eager | graph | triton | best cuDESeq2 vs R | pydeseq2 vs R |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| airway | 5 | 8 | 33456 | 34685 | 2521 | 1650 | 1501 | 22.3× (triton) | 1.0× |
| airway_cell | 4 | 8 | 27899 | 37878 | 4178 | 3621 | 3469 | 8.0× (triton) | 0.7× |
| airway_dex | 2 | 8 | 25664 | 29601 | 1733 | 1414 | 1361 | 18.9× (triton) | 0.9× |
| gtex_blood_muscle | 2 | 300 | 350435 | 80111 | 8429 | 8386 | 4483 | 78.2× (triton) | 4.4× |
| pasilla | 2 | 7 | 9307 | 11436 | 1149 | 801 | 702 | 13.3× (triton) | 0.8× |
| pasilla_2fac | 3 | 7 | 10346 | 12133 | 1511 | 946 | 866 | 11.9× (triton) | 0.9× |

## Table 2 — per-substep wall time (ms)

| dataset | substep | R | pydeseq2 | eager | graph | triton |
|---|---|--:|--:|--:|--:|--:|
| airway | normalization | 213 | 12 | 1 | 1 | 1 |
| airway | dispersion | 10423 | 15256 | 1590 | 715 | 583 |
| airway | glm_fit | 4738 | 6348 | 57 | 57 | 57 |
| airway | significance | 365 | 2685 | 119 | 119 | 119 |
| airway | lfc_shrink | 17717 | 10384 | 754 | 758 | 741 |
| airway | **total** | 33456 | 34685 | 2521 | 1650 | 1501 |
| airway_cell | normalization | 201 | 13 | 1 | 1 | 1 |
| airway_cell | dispersion | 5724 | 16430 | 750 | 182 | 60 |
| airway_cell | glm_fit | 4190 | 5757 | 33 | 33 | 33 |
| airway_cell | significance | 362 | 2738 | 117 | 116 | 116 |
| airway_cell | lfc_shrink | 17423 | 12941 | 3277 | 3290 | 3259 |
| airway_cell | **total** | 27899 | 37878 | 4178 | 3621 | 3469 |
| airway_dex | normalization | 198 | 13 | 1 | 1 | 1 |
| airway_dex | dispersion | 6212 | 16149 | 429 | 109 | 52 |
| airway_dex | glm_fit | 5666 | 4182 | 53 | 53 | 53 |
| airway_dex | significance | 1299 | 2586 | 146 | 144 | 145 |
| airway_dex | lfc_shrink | 12290 | 6671 | 1104 | 1107 | 1111 |
| airway_dex | **total** | 25664 | 29601 | 1733 | 1414 | 1361 |
| gtex_blood_muscle | normalization | 2954 | 775 | 4 | 4 | 4 |
| gtex_blood_muscle | dispersion | 231276 | 44959 | 3501 | 3620 | 200 |
| gtex_blood_muscle | glm_fit | 42321 | 15131 | 397 | 417 | 417 |
| gtex_blood_muscle | significance | 17865 | 5142 | 2785 | 2609 | 2625 |
| gtex_blood_muscle | lfc_shrink | 56020 | 14104 | 1742 | 1736 | 1237 |
| gtex_blood_muscle | **total** | 350435 | 80111 | 8429 | 8386 | 4483 |
| pasilla | normalization | 175 | 5 | 1 | 1 | 1 |
| pasilla | dispersion | 2335 | 6177 | 422 | 78 | 21 |
| pasilla | glm_fit | 2138 | 1683 | 36 | 36 | 37 |
| pasilla | significance | 277 | 991 | 62 | 61 | 62 |
| pasilla | lfc_shrink | 4382 | 2579 | 628 | 625 | 581 |
| pasilla | **total** | 9307 | 11436 | 1149 | 801 | 702 |
| pasilla_2fac | normalization | 176 | 5 | 1 | 1 | 1 |
| pasilla_2fac | dispersion | 2791 | 6310 | 730 | 162 | 83 |
| pasilla_2fac | glm_fit | 1502 | 1803 | 327 | 329 | 329 |
| pasilla_2fac | significance | 240 | 1087 | 60 | 62 | 59 |
| pasilla_2fac | lfc_shrink | 5637 | 2927 | 393 | 392 | 395 |
| pasilla_2fac | **total** | 10346 | 12133 | 1511 | 946 | 866 |

## Table 3 — output parity

cuDESeq2 (eager, representative) and PyDESeq2 (competitor) each scored against the same R~DESeq2 ground truth per substep. The PASS/FAIL verdict and its tolerance apply to cuDESeq2 (our bit-exactness claim); the PyDESeq2 column is shown for comparison. `GPU Δ` is the largest disagreement among the three cuDESeq2 GPU modes (0 ⇒ bit-identical).

| dataset | substep | metric | cuDESeq2 vs R | PyDESeq2 vs R | tol | cuDESeq2 verdict | GPU Δ |
|---|---|---|--:|--:|--:|:--:|--:|
| airway | normalization | max rel | 4.14e-15 | 4.14e-15 | ≤1e-06 | ✅ PASS | 0.0e+00 |
| airway | dispersion | p95 rel | 0.000449 | 0.409 | ≤0.1 | ✅ PASS | 3.1e-07 |
| airway | glm_fit | p95 |Δ| | 3.04e-05 | 0.0359 | ≤0.01 | ✅ PASS | 9.7e-09 |
| airway | significance | Jaccard@.05 | 1 | 0.915 | ≥0.95 | ✅ PASS | 1.5e-06 |
| airway | lfc_shrink | Pearson r | 1 (ρ=1.000) | 0.988 | ≥0.9 | ✅ PASS | 7.9e-05 |
| airway_cell | normalization | max rel | 4.14e-15 | 4.14e-15 | ≤1e-06 | ✅ PASS | 0.0e+00 |
| airway_cell | dispersion | p95 rel | 0.00284 | 0.212 | ≤0.1 | ✅ PASS | 6.1e-08 |
| airway_cell | glm_fit | p95 |Δ| | 8.09e-05 | 0.0131 | ≤0.01 | ✅ PASS | 2.9e-09 |
| airway_cell | significance | Jaccard@.05 | 1 | 0.901 | ≥0.95 | ✅ PASS | 1.3e-08 |
| airway_cell | lfc_shrink | Pearson r | 0.997 (ρ=0.996) | 0.96 | ≥0.9 | ✅ PASS | 6.4e-05 |
| airway_dex | normalization | max rel | 4.14e-15 | 4.14e-15 | ≤1e-06 | ✅ PASS | 0.0e+00 |
| airway_dex | dispersion | p95 rel | 0.00104 | 0.137 | ≤0.1 | ✅ PASS | 2.3e-07 |
| airway_dex | glm_fit | p95 |Δ| | 2.5e-05 | 0.00611 | ≤0.01 | ✅ PASS | 1.8e-09 |
| airway_dex | significance | Jaccard@.05 | 1 | 0.977 | ≥0.95 | ✅ PASS | 1.5e-07 |
| airway_dex | lfc_shrink | Pearson r | 0.999 (ρ=1.000) | 0.999 | ≥0.9 | ✅ PASS | 9.1e-06 |
| gtex_blood_muscle | normalization | max rel | 5.69e-15 | 5.47e-15 | ≤1e-06 | ✅ PASS | 0.0e+00 |
| gtex_blood_muscle | dispersion | p95 rel | 0.00371 | 0.00604 | ≤0.1 | ✅ PASS | 2.6e-01 |
| gtex_blood_muscle | glm_fit | p95 |Δ| | 1.81e-05 | 4.52e-05 | ≤0.01 | ✅ PASS | 1.2e-04 |
| gtex_blood_muscle | significance | Jaccard@.05 | 1 | 0.971 | ≥0.95 | ✅ PASS | 2.0e-03 |
| gtex_blood_muscle | lfc_shrink | Pearson r | 1 (ρ=1.000) | 1 | ≥0.9 | ✅ PASS | 5.7e-03 |
| pasilla | normalization | max rel | 2.93e-15 | 2.93e-15 | ≤1e-06 | ✅ PASS | 0.0e+00 |
| pasilla | dispersion | p95 rel | 0.000667 | 0.121 | ≤0.1 | ✅ PASS | 1.9e-07 |
| pasilla | glm_fit | p95 |Δ| | 2.34e-05 | 0.00821 | ≤0.01 | ✅ PASS | 1.3e-09 |
| pasilla | significance | Jaccard@.05 | 1 | 0.974 | ≥0.95 | ✅ PASS | 1.8e-06 |
| pasilla | lfc_shrink | Pearson r | 1 (ρ=1.000) | 0.995 | ≥0.9 | ✅ PASS | 5.4e-07 |
| pasilla_2fac | normalization | max rel | 2.93e-15 | 2.93e-15 | ≤1e-06 | ✅ PASS | 0.0e+00 |
| pasilla_2fac | dispersion | p95 rel | 0.000235 | 0.233 | ≤0.1 | ✅ PASS | 1.1e-07 |
| pasilla_2fac | glm_fit | p95 |Δ| | 1.46e-05 | 0.0216 | ≤0.01 | ✅ PASS | 1.4e-08 |
| pasilla_2fac | significance | Jaccard@.05 | 1 | 0.939 | ≥0.95 | ✅ PASS | 1.4e-06 |
| pasilla_2fac | lfc_shrink | Pearson r | 1 (ρ=1.000) | 0.995 | ≥0.9 | ✅ PASS | 1.5e-07 |
