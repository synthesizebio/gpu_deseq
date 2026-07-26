# cuDESeq2 benchmark harness

One reproducible harness that, for every dataset, answers four questions and
renders them as **three tables**:

1. **Total-pipeline speed** of all four versions (Table 1).
2. **Per-substep speed** of all four versions (Table 2).
3. **Output parity** — do the versions produce equivalent results, per substep
   and overall (Table 3, which also confirms the three GPU modes agree with each
   other)?

The **four versions** are R DESeq2 1.30.1 (reference) and cuDESeq2 in three
execution modes: **eager**, **CUDA-graph**, **Triton**.

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

- **The three GPU modes are bit-identical to each other**, so Table 3 reports
  cuDESeq2-vs-R once (eager, representative) plus a `GPU Δ` column = the largest
  disagreement among eager/graph/triton for that substep. `GPU Δ ≈ 0` confirms
  the modes agree; the only non-trivial entry is `lfc_shrink`, where a ~1e-14
  dispersion-kernel difference can be amplified by aggressive shrinkage on a few
  near-degenerate genes.

- **Equivalence vs R is tolerance-based**, not bit-exact: floating-point order and
  R's Mersenne-Twister RNG (used in the small-dof prior-variance estimator) can't
  be reproduced exactly. Each substep has an explicit PASS tolerance:

  | substep | metric | PASS tolerance | why |
  |---|---|---|---|
  | normalization | max relative Δ (size factors) | ≤ 1e-6 | deterministic |
  | dispersion    | p95 relative Δ                | ≤ 0.10 | intermediate; within 10% is DE-equivalent; RNG-limited at small dof (below) |
  | glm_fit       | p95 \|Δ\| (raw LFC)           | ≤ 1e-2 | drives significance; near-exact |
  | significance  | Jaccard of {padj<0.05}        | ≥ 0.95 | borderline-gene flicker at the threshold |
  | lfc_shrink    | Pearson r (Spearman in-cell)  | ≥ 0.90 | soft ranking quantity, see below |

  Typical results: 29/30 substep checks pass tightly (5/6 dispersions land
  <0.4%). The known exceptions: `airway` dispersion 6.7% (RNG-limited, n=8/P=5 ⇒
  dof=3 — R's `set.seed(2)` Mersenne-Twister in the prior-variance estimator is
  not reproducible in NumPy, and 6.7% is DE-equivalent), and `airway_cell`
  `lfc_shrink` (Pearson 0.866) — a single near-degenerate gene whose apeGLM
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
