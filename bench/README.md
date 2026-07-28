# cuDESeq2 benchmark harness

One reproducible harness that, for every dataset, answers four questions and
renders them as **three tables**:

1. **Total-pipeline speed** of all four versions (Table 1).
2. **Per-substep speed** of all four versions (Table 2).
3. **Output parity** — do the versions produce equivalent results, per substep
   and overall (Table 3, which also confirms the three GPU modes agree with each
   other)?

The versions are R DESeq2 1.30.1 (the **reference/ground truth**), cuDESeq2 in
three execution modes (**eager**, **CUDA-graph**, **Triton**), and **PyDESeq2**
as a **competitor baseline** — a third-party CPU/Python DESeq2 reimplementation
that is timed and scored against R exactly as cuDESeq2 is, but is *never* a
parity target for us. PyDESeq2 is a benchmark-only dependency (`pip install
pydeseq2`); it is imported lazily and only in `run_pydeseq2.py` (never by the
`gpu_deseq` package), and if it is not installed the competitor column is simply
skipped. Table 3 shows cuDESeq2-vs-R matches R far more tightly than
PyDESeq2-vs-R on every substep.

The **five substeps** are `normalization` (size factors), `dispersion`
(Cox–Reid gene-est → trend → MAP), `glm_fit` (IRLS NB-GLM + Wald),
`significance` (Cook's / independent filtering + BH), and `lfc_shrink` (apeGLM).

## Run

```bash
# inputs must exist under validation/data/<case>/ (counts.csv, coldata.csv,
# meta.json) — produced once by validation/fetch_and_reference.R + prepare_gtex.R
Rscript bench/run_r.R                      # R timings + intermediates -> bench/cache/
PYTHONPATH=src python bench/bench.py       # cuDESeq2 timings + parity -> bench/results/
```

`bench.py` runs `run_r.R` itself by default; pass `--skip-r` to reuse
`bench/cache/`, `--only pasilla,airway_dex` to subset, `--device cpu` to force CPU.

Outputs (committed): `bench/results/TABLES.md` (the three tables),
`timings.json`, `parity.json`.

## What the numbers mean / honest caveats

- **The graph and Triton flags affect only the `dispersion` substep.** The other
  four substeps run the identical eager code in every mode. So in Table 2 those
  rows are equal across eager/graph/triton by construction — the acceleration is
  localized to dispersion fitting, and the harness shows this rather than hiding
  it. (This is why the total-pipeline speedup is bounded by the non-dispersion
  substeps — see `lfc_shrink`, the largest remaining cost on small-*n* sets.)

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
  five small-*n* sets, with a worst case of 2.6e-1 relative on one `gtex` gene
  (a gene at the dispersion-grid boundary) and 5.7e-3 absolute in the shrunk LFC,
  where aggressive shrinkage on a few near-degenerate genes amplifies a last-bit
  dispersion difference. No PASS verdict or DE call changes.

- **Equivalence vs R is tolerance-based**, not bit-exact: reduction order in
  batched GPU kernels differs from R's sequential per-gene loops. (The small-dof
  prior-variance estimator used to be a second source here; it now replays R's
  RNG stream and smoother exactly — see `_r_rng` — and reproduces R's scalar bit
  for bit.) Each substep has an explicit PASS tolerance:

  | substep | metric | PASS tolerance | why |
  |---|---|---|---|
  | normalization | max relative Δ (size factors) | ≤ 1e-6 | deterministic |
  | dispersion    | p95 relative Δ                | ≤ 0.10 | intermediate; within 10% is DE-equivalent (observed worst: 3.7e-3) |
  | glm_fit       | p95 \|Δ\| (raw LFC)           | ≤ 1e-2 | drives significance; near-exact |
  | significance  | Jaccard of {padj<0.05}        | ≥ 0.95 | borderline-gene flicker at the threshold |
  | lfc_shrink    | Pearson r (Spearman in-cell)  | ≥ 0.90 | soft ranking quantity, see below |

  Typical results: all 30 substep checks pass, and all six dispersions land
  <0.4% (worst: `gtex_blood_muscle` at 3.7e-3). `airway` — the only case with
  residual dof ≤ 3, hence the only one entering R's Monte-Carlo prior-variance
  branch — reproduces R's prior variance bit for bit, lands at 4.5e-4, and calls
  an identical significant-gene set (3993/3993). The one remaining soft spot is
  `airway_cell` `lfc_shrink` (Pearson 0.866) — a single near-degenerate gene whose apeGLM
  posterior is so flat that a 1e-14 dispersion difference moves the shrunk LFC by
  ~5 (visible in that row's `GPU Δ`). Neither changes any DE call.

- **`lfc_shrink` is scored by Pearson r (Spearman shown in-cell).** The
  apeGLM-shrunk LFC is a MAP estimate under a heavy-tailed prior used for
  *ranking / visualization*, not for calling significance (that's raw LFC + padj,
  which match R near-exactly). Pearson weights the high-effect genes that matter;
  global Spearman is unfairly harsh here because thousands of genes shrunk to ≈0
  reorder as pure noise (which is why e.g. pasilla_2fac reads ρ=0.69 while raw LFC
  is exact). Both are shown so nothing is hidden.

- **Timing method.** cuDESeq2 substeps are timed with `cuda.synchronize()` around
  each, median of 5 reps (3 on the 300-sample cohort). R substeps call the five
  DESeq2 functions individually (`estimateSizeFactors → estimateDispersions →
  nbinomWaldTest → results → lfcShrink`), median of 3 reps (1 on the 300-sample
  cohort, where a single run already takes minutes). R is single-threaded.

## Files

- `run_r.R` — R side: per-substep timing + intermediate export → `bench/cache/`.
- `run_cu.py` — cuDESeq2 side: per-substep timing + intermediate capture
  (importable; also runnable standalone for one case/mode).
- `bench.py` — orchestrator: runs both sides, verifies parity, renders the three
  tables to `bench/results/`.
