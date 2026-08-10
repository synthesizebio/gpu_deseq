# A100-host R DESeq2 multi-worker benchmark (archived)

This file preserves the same-host CPU baseline that was superseded when the
benchmark suite moved to the six real-data standard-pipeline designs.

- **Date:** 2026-07-14
- **GPU:** NVIDIA A100-SXM4-40GB
- **R baseline:** R DESeq2 with 1 worker and 12 workers
- **GPU modes:** eager, CUDA graph, and Triton
- **Workloads:** four synthetic count-matrix designs
- **Raw measurements:** `results_per_step_a100_12core_2026-07-14.json`
- **Benchmark code revision:** `f2f13a8`

The 12-worker R result is a full DESeq2 analysis using
`DESeq(parallel=TRUE, BPPARAM=MulticoreParam(12))`; DESeq2 combines
dispersion fitting and GLM fitting in that call. The corresponding serial and
12-worker totals (ms), with the best GPU mode, were:

| design | R, 1 worker | R, 12 workers | Triton | Triton vs. best R |
|---|---:|---:|---:|---:|
| 6 samples × 2,000 genes | 3,126 | 5,280 | 162 | 19.3× |
| 60 samples × 2,000 genes | 4,474 | 5,810 | 155 | 28.9× |
| 60 samples × 20,000 genes | 28,287 | 12,356 | 330 | 37.5× |
| 60 samples × 1,500 genes, P=4 | 4,951 | 6,248 | 186 | 26.6× |

This archive is not the source for the manuscript's current real-data
10.3--99.2× result, which is based on the later six-design A100 rerun with
DESeq2 1.52.0 and a one-worker R baseline. Keeping the records separate
prevents an old synthetic benchmark from being presented as a current
real-data measurement.
