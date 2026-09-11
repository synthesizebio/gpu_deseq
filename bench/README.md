# cuDESeq2 benchmark harness

One reproducible harness that, for every dataset, answers four questions and
renders them as **three tables**:

1. **Total-pipeline speed** of all four versions (Table 1).
2. **Per-substep speed** of all four versions (Table 2).
3. **Output parity** — do the versions produce equivalent results, per substep
   and overall (Table 3, which also confirms the three GPU modes agree with each
   other)?

The current reference is R 4.6.0 with DESeq2 1.52.0 and apeglm 1.34.0. cuDESeq2 runs in
three execution modes (**eager**, **CUDA-graph**, **Triton**), and **PyDESeq2**
as a **competitor baseline** — a third-party CPU/Python DESeq2 reimplementation
that is timed and scored against R exactly as cuDESeq2 is, but is *never* a
parity target for us. PyDESeq2 is a benchmark-only dependency (`pip install
pydeseq2`); it is imported lazily and only in `run_pydeseq2.py` (never by the
`gpu_deseq` package), and if it is not installed the competitor column is simply
skipped. Table 3 shows cuDESeq2-vs-R matches R far more tightly than
PyDESeq2-vs-R on every substep.

The **five substeps** are `normalization` (size factors), `dispersion`
(Cox–Reid gene-est → trend → MAP), `glm_fit` (IRLS NB-GLM, Wald, and
Cook's-outlier replacement/refit where eligible),
`significance` (Cook's / independent filtering + BH), and `lfc_shrink` (apeGLM).

## Run

```bash
# Download checksum-pinned sources and reconstruct validation/data/<case>/.
make container-data
Rscript bench/run_r.R                      # R timings + intermediates -> bench/cache/
PYTHONPATH=src python bench/bench.py       # cuDESeq2 timings + parity -> bench/results/
```

`bench.py` runs `run_r.R` itself by default; pass `--skip-r` to reuse
`bench/cache/`, `--only pasilla,airway_dex` to subset, `--device cpu` to force CPU.

Outputs (committed): `bench/results/TABLES.md` (the three tables),
`timings.json`, `parity.json`, and `reference_parity.json`. Use
`reference_parity.py` when the R reference changes but a new GPU timing run is
not available; it validates outputs without overwriting GPU timings.
The [result manifest](results/README.md) identifies the source record for each
manuscript table and keeps the latest same-host worker comparison separate from
historical synthetic and CPU-only-host experiments.

## What the numbers mean / honest caveats

- **The graph and Triton flags select the dispersion optimizer.** They affect
  the main dispersion stage and any dispersion fit performed during the
  standard outlier-refit branch. The other code paths are shared.

- **The Triton kernel covers P ∈ {2,...,6}, so it runs on all six datasets.** Wider
  designs fall back to eager, and the harness would then report a Triton
  dispersion time equal to eager rather than an error. Earlier tables showed
  exactly that for `airway` (P=5) and `pasilla_2fac` (P=3), which predate the
  kernel covering odd P — worth knowing when comparing against an older run.

- **Absolute timings shift 10–25% between sessions**, and the first dataset timed
  in a run pays CUDA/JIT warmup (`airway`, alphabetically first, is the usual
  victim). All six datasets are timed in one session on an otherwise idle GPU, so
  mode-vs-mode comparisons within a row are sound; do not read cross-run deltas on
  substeps that no mode flag touches (`glm_fit`, `lfc_shrink`) as regressions. Run
  nothing else on the GPU while benchmarking.

- **The graph mode is bit-identical to eager; Triton is deliberately not.** Table 3
  reports cuDESeq2-vs-R once (eager, representative) plus a `GPU Δ` column = the
  largest disagreement among eager/graph/triton for that substep. Graph replay
  contributes exactly 0 to that column on every substep. Triton re-derives the
  same mathematics in registers with a different reduction order and a
  hand-written `digamma`, so it agrees with eager to ~1e-7 in dispersion on the
  five small-*n* sets. On GTEx, the largest per-gene relative difference is
  2.6e-1 at one boundary gene; the called-set Jaccard remains 0.99997.

- **Equivalence vs R is tolerance-based**, not bit-exact: reduction order in
  batched GPU kernels differs from R's sequential per-gene loops. (The small-dof
  prior-variance estimator used to be a second source here; it now replays R's
  RNG stream and smoother exactly — see `_r_rng` — and reproduces R's scalar bit
  for bit.) Each substep has an explicit PASS tolerance:

  | substep | metric | PASS tolerance | why |
  |---|---|---|---|
  | normalization | max relative Δ (size factors) | ≤ 1e-6 | deterministic |
  | dispersion    | p95 relative Δ                | ≤ 0.10 | intermediate; observed worst: 4.4e-3 |
  | glm_fit       | p95 \|Δ\| (raw LFC)           | ≤ 1e-2 | drives significance; near-exact |
  | significance  | Jaccard of {padj<0.05}        | ≥ 0.95 | borderline-gene flicker at the threshold |
  | lfc_shrink    | Pearson r (Spearman in-cell)  | ≥ 0.90 | soft ranking quantity, see below |

  Current results: all 30 substep checks pass; the worst dispersion p95 relative
  error is 4.4e-3 (`gtex_blood_muscle`). `airway` — the only case with
  residual dof ≤ 3, hence the only one entering R's Monte-Carlo prior-variance
  branch — reproduces R's prior variance bit for bit, lands at 4.5e-4, and calls
  an identical significant-gene set (3993/3993). The least favorable shrunk-LFC
  correlation is 0.997 (`airway_cell`).

- **`lfc_shrink` is scored by Pearson r (Spearman shown in-cell).** The
  apeGLM-shrunk LFC is a MAP estimate under a heavy-tailed prior used for
  *ranking / visualization*, not for calling significance (that's raw LFC + padj,
  which match R near-exactly). Pearson weights the high-effect genes that matter;
  global Spearman is unfairly harsh here because thousands of genes shrunk to ≈0
  reorder as pure noise (which is why e.g. pasilla_2fac reads ρ=0.69 while raw LFC
  is exact). Both are shown so nothing is hidden.

- **Timing method.** cuDESeq2 substeps are timed with `cuda.synchronize()` around
  each, with five measured repetitions after an untimed warm-up. R substeps call
  `estimateSizeFactors → estimateDispersions → nbinomWaldTest` plus eligible
  outlier refitting `→ results → lfcShrink`. R also uses five measured
  repetitions and one worker for the matched stage comparison. Current harness
  output keeps `stage_total` (the sum of independently measured stage medians)
  separate from `total` (the median of direct observations including dataset
  construction and, for CUDA, host-to-device transfer). The separate R worker
  comparison uses direct `DESeq() + results() + lfcShrink()` observations at one and 12
  `MulticoreParam` workers, with standard count-outlier replacement/refitting;
  its record is `bench/results/r_parallel_a100_12worker.json`.

## Files

- `run_r.R` — R side: per-substep timing + intermediate export → `bench/cache/`.
- `run_cu.py` — cuDESeq2 side: per-substep timing + intermediate capture
  (importable; also runnable standalone for one case/mode).
- `bench.py` — orchestrator: runs both sides, verifies parity, renders the three
  tables to `bench/results/`.

## Real-GTEx sample, design-width, and memory scaling

The main six-case suite deliberately fixes GTEx at 300 samples. The separate
scaling harness uses all 9,662 samples in the same checksum-pinned recount2
source while holding the gene axis fixed at the paper's 54,922 genes. It tests
nested two-tissue cohorts, balanced P=3--6 designs, all-sample designs, and the
A100 OOM boundary.

Materialize the source matrix once (the 2.1 GB output remains under ignored
`bench/cache/`):

```bash
Rscript bench/extract_gtex_scaling.R \
  validation/sources/SRP012682_rse_gene.Rdata \
  validation/data/gtex_blood_muscle/counts.csv \
  bench/cache/gtex_scaling/matrix

PYTHONPATH=src python bench/gtex_scaling.py --list
```

Run one case per clean process. Paper-grade timing uses one warm-up, five
stage-timed repetitions, and five independent direct repetitions. Feasibility
probes use no warm-up, one staged repetition, and no separate direct run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
  python bench/gtex_scaling.py --case p6_all --mode triton \
  --warmups 1 --reps 5 --direct-reps 5 --verify-matrix \
  --capture /tmp/p6_all_gpu.npz --output /tmp/p6_all_gpu.json

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
  python bench/gtex_scaling.py --case p10_all --mode triton \
  --warmups 0 --reps 1 --direct-reps 0 --output /tmp/p10_all_probe.json
```

The JSON records the exact one-based source columns, their sample-ID hash,
matrix hashes, software/commit provenance, every raw timing, and peak CUDA
allocated/reserved memory. Designs wider than P=6 explicitly report the eager
fallback. Use the exact GPU case JSON for the one-worker R reference:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
Rscript bench/run_gtex_scaling_r.R \
  validation/sources/SRP012682_rse_gene.Rdata \
  bench/cache/gtex_scaling/matrix /tmp/p6_all_gpu.json /tmp/p6_all_r
```
