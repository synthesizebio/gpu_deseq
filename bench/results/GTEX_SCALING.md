# Real-GTEx scaling and A100 memory boundary

The gene axis is fixed at 54,922. Direct GPU times are medians of five complete observations after one warm-up; stage sums add independently measured stage medians, and peak allocation is the maximum over the five staged repetitions. R endpoint rows are single cold one-worker staged observations and are not pooled with the main five-repetition headline benchmark.

## Full-workflow GPU timing

| case | samples | P | direct (s) | stage sum (s) | peak allocated (GiB) |
|---|---:|---:|---:|---:|---:|
| p2_300 | 300 | 2 | 1.753 | 1.686 | 1.98 |
| p2_600 | 600 | 2 | 2.740 | 2.501 | 3.95 |
| p2_912 | 912 | 2 | 5.430 | 5.139 | 5.99 |
| p3_fixed300 | 900 | 3 | 6.005 | 5.677 | 5.92 |
| p4_fixed300 | 1200 | 4 | 8.990 | 8.514 | 8.38 |
| p5_fixed300 | 1500 | 5 | 9.898 | 10.511 | 11.09 |
| p6_fixed300 | 1800 | 6 | 14.323 | 13.745 | 14.77 |
| p6_all | 2451 | 6 | 22.936 | 23.415 | 20.10 |

## One-worker R endpoint observations

| case | samples | P | R stage sum (s) | GPU stage sum (s) | observed speedup | parity |
|---|---:|---:|---:|---:|---:|:---:|
| p2_912 | 912 | 2 | 952.305 | 5.139 | 185.3× | PASS |
| p6_all | 2451 | 6 | 3211.540 | 23.415 | 137.2× | PASS |

## Fresh-process A100 feasibility boundary

| case | samples | P | path | status | peak allocated (GiB) | failed stage |
|---|---:|---:|---|:---:|---:|---|
| p8_all | 3147 | 8 | eager_fallback | PASS | 30.98 | — |
| p9_all | 3478 | 9 | eager_fallback | PASS | 37.09 | — |
| p10_all | 3784 | 10 | eager_fallback | OOM | 27.88 | dispersion |
| p12_all | 4338 | 12 | eager_fallback | OOM | 35.51 | dispersion |
