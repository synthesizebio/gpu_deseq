## Table 1 — total pipeline wall time (ms) and speedup vs R

| dataset | P | n | R | eager | graph | triton | best vs R |
|---|--:|--:|--:|--:|--:|--:|--:|
| airway | 5 | 8 | 33456 | 1445 | 704 | 1442 | 47.5× (graph) |
| airway_cell | 4 | 8 | 27899 | 1164 | 455 | 359 | 77.6× (triton) |
| airway_dex | 2 | 8 | 25664 | 719 | 394 | 340 | 75.5× (triton) |
| gtex_blood_muscle | 2 | 300 | 350435 | 7064 | 7013 | 3570 | 98.2× (triton) |
| pasilla | 2 | 7 | 9307 | 610 | 270 | 219 | 42.6× (triton) |
| pasilla_2fac | 3 | 7 | 10346 | 1219 | 653 | 1227 | 15.8× (graph) |

## Table 2 — per-substep wall time (ms)

| dataset | substep | R | eager | graph | triton |
|---|---|--:|--:|--:|--:|
| airway | normalization | 213 | 1 | 1 | 1 |
| airway | dispersion | 10423 | 1138 | 396 | 1133 |
| airway | glm_fit | 4738 | 55 | 54 | 56 |
| airway | significance | 365 | 115 | 118 | 117 |
| airway | lfc_shrink | 17717 | 136 | 135 | 136 |
| airway | **total** | 33456 | 1445 | 704 | 1442 |
| airway_cell | normalization | 201 | 1 | 1 | 1 |
| airway_cell | dispersion | 5724 | 845 | 158 | 54 |
| airway_cell | glm_fit | 4190 | 33 | 33 | 33 |
| airway_cell | significance | 362 | 114 | 113 | 114 |
| airway_cell | lfc_shrink | 17423 | 170 | 150 | 157 |
| airway_cell | **total** | 27899 | 1164 | 455 | 359 |
| airway_dex | normalization | 198 | 1 | 1 | 1 |
| airway_dex | dispersion | 6212 | 429 | 109 | 52 |
| airway_dex | glm_fit | 5666 | 52 | 52 | 52 |
| airway_dex | significance | 1299 | 124 | 121 | 123 |
| airway_dex | lfc_shrink | 12290 | 113 | 111 | 112 |
| airway_dex | **total** | 25664 | 719 | 394 | 340 |
| gtex_blood_muscle | normalization | 2954 | 4 | 4 | 4 |
| gtex_blood_muscle | dispersion | 231276 | 3508 | 3623 | 200 |
| gtex_blood_muscle | glm_fit | 42321 | 398 | 415 | 397 |
| gtex_blood_muscle | significance | 17865 | 2755 | 2574 | 2571 |
| gtex_blood_muscle | lfc_shrink | 56020 | 400 | 397 | 397 |
| gtex_blood_muscle | **total** | 350435 | 7064 | 7013 | 3570 |
| pasilla | normalization | 175 | 1 | 1 | 1 |
| pasilla | dispersion | 2335 | 410 | 71 | 19 |
| pasilla | glm_fit | 2138 | 36 | 36 | 36 |
| pasilla | significance | 277 | 60 | 60 | 61 |
| pasilla | lfc_shrink | 4382 | 103 | 103 | 102 |
| pasilla | **total** | 9307 | 610 | 270 | 219 |
| pasilla_2fac | normalization | 176 | 1 | 1 | 1 |
| pasilla_2fac | dispersion | 2791 | 723 | 162 | 730 |
| pasilla_2fac | glm_fit | 1502 | 326 | 326 | 327 |
| pasilla_2fac | significance | 240 | 58 | 58 | 58 |
| pasilla_2fac | lfc_shrink | 5637 | 111 | 105 | 112 |
| pasilla_2fac | **total** | 10346 | 1219 | 653 | 1227 |

## Table 3 — output parity

cuDESeq2 (eager, representative) vs R per substep, with PASS/FAIL against an explicit tolerance; `GPU Δ` is the largest disagreement among the three GPU modes (0 ⇒ eager/graph/triton bit-identical for that substep).

| dataset | substep | metric | value | tol | verdict | GPU Δ |
|---|---|---|--:|--:|:--:|--:|
| airway | normalization | max rel | 4.14e-15 | ≤1e-06 | ✅ PASS | 0.0e+00 |
| airway | dispersion | p95 rel | 0.0669 | ≤0.1 | ✅ PASS | 8.9e-15 |
| airway | glm_fit | p95 |Δ| | 0.00483 | ≤0.01 | ✅ PASS | 7.8e-14 |
| airway | significance | Jaccard@.05 | 0.977 | ≥0.95 | ✅ PASS | 2.8e-13 |
| airway | lfc_shrink | Pearson r | 0.999 (ρ=1.000) | ≥0.9 | ✅ PASS | 1.0e-07 |
| airway_cell | normalization | max rel | 4.14e-15 | ≤1e-06 | ✅ PASS | 0.0e+00 |
| airway_cell | dispersion | p95 rel | 0.00284 | ≤0.1 | ✅ PASS | 8.1e-08 |
| airway_cell | glm_fit | p95 |Δ| | 8.09e-05 | ≤0.01 | ✅ PASS | 4.3e-09 |
| airway_cell | significance | Jaccard@.05 | 0.971 | ≥0.95 | ✅ PASS | 1.2e-08 |
| airway_cell | lfc_shrink | Pearson r | 0.866 (ρ=0.996) | ≥0.9 | ❌ FAIL | 5.0e+00 |
| airway_dex | normalization | max rel | 4.14e-15 | ≤1e-06 | ✅ PASS | 0.0e+00 |
| airway_dex | dispersion | p95 rel | 0.00104 | ≤0.1 | ✅ PASS | 2.3e-07 |
| airway_dex | glm_fit | p95 |Δ| | 2.5e-05 | ≤0.01 | ✅ PASS | 1.8e-09 |
| airway_dex | significance | Jaccard@.05 | 1 | ≥0.95 | ✅ PASS | 1.5e-07 |
| airway_dex | lfc_shrink | Pearson r | 0.977 (ρ=1.000) | ≥0.9 | ✅ PASS | 2.0e-06 |
| gtex_blood_muscle | normalization | max rel | 5.69e-15 | ≤1e-06 | ✅ PASS | 0.0e+00 |
| gtex_blood_muscle | dispersion | p95 rel | 0.00371 | ≤0.1 | ✅ PASS | 2.6e-01 |
| gtex_blood_muscle | glm_fit | p95 |Δ| | 1.81e-05 | ≤0.01 | ✅ PASS | 1.2e-04 |
| gtex_blood_muscle | significance | Jaccard@.05 | 0.991 | ≥0.95 | ✅ PASS | 1.8e-03 |
| gtex_blood_muscle | lfc_shrink | Pearson r | 1 (ρ=1.000) | ≥0.9 | ✅ PASS | 5.7e-03 |
| pasilla | normalization | max rel | 2.93e-15 | ≤1e-06 | ✅ PASS | 0.0e+00 |
| pasilla | dispersion | p95 rel | 0.000667 | ≤0.1 | ✅ PASS | 1.9e-07 |
| pasilla | glm_fit | p95 |Δ| | 2.34e-05 | ≤0.01 | ✅ PASS | 1.3e-09 |
| pasilla | significance | Jaccard@.05 | 1 | ≥0.95 | ✅ PASS | 1.8e-06 |
| pasilla | lfc_shrink | Pearson r | 0.982 (ρ=0.949) | ≥0.9 | ✅ PASS | 1.4e-06 |
| pasilla_2fac | normalization | max rel | 2.93e-15 | ≤1e-06 | ✅ PASS | 0.0e+00 |
| pasilla_2fac | dispersion | p95 rel | 0.000235 | ≤0.1 | ✅ PASS | 1.8e-08 |
| pasilla_2fac | glm_fit | p95 |Δ| | 1.46e-05 | ≤0.01 | ✅ PASS | 3.1e-09 |
| pasilla_2fac | significance | Jaccard@.05 | 1 | ≥0.95 | ✅ PASS | 1.4e-06 |
| pasilla_2fac | lfc_shrink | Pearson r | 0.929 (ρ=0.694) | ≥0.9 | ✅ PASS | 5.2e-07 |
