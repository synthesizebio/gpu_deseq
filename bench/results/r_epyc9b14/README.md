# R DESeq2 baseline — AMD EPYC 9B14 (second host)

An **additional R-only measurement**, added as new columns. Nothing that was
already committed changed: `bench/results/TABLES.md`, `timings.json` and
`parity.json` are untouched, and `scripts/audit_paper_numbers.py` still passes
99/99 against them.

## Read this first

This host runs **DESeq2 1.46.0**, not the reference **1.30.1**, so hardware and
library version moved **together**. A ratio between this column and the published
one cannot be attributed to either cause alone. Docker was unavailable for the
pinned Bioconductor 3.12 image on this box (apt is broken by an unrelated
dependency), which is why `bench/run_r_scaling.sh --allow-any-deseq2` was used;
that flag exists precisely so this substitution is explicit and recorded, never
silent. `summary.json` carries `comparable_to_reference: false` for this reason.

The column is still worth having: it is *what DESeq2 costs today*, on a current
CPU, which is the baseline a reader will compare against. It is simply not a
controlled A/B against the 1.30.1 measurement.

## What is here

| file | what it is |
|---|---|
| `summary.json` | **Machine-readable.** Per dataset: R serial substeps for both hosts, the multi-core sweep, and every speedup with its denominator named. Start here for tooling. |
| `r_scaling.jsonl` | Raw harness output, one JSON record per line (`meta`, `serial_fine` ×6, `sweep` ×30). The authoritative source; `summary.json` is derived from it. |
| `R_SCALING.md` | Tables A/B/C as emitted by the harness, with a full provenance header. |
| `blas_probe.txt` | BLAS sensitivity measurement (see below). |
| `../R_HOSTS.md` | Both hosts side by side, plus the recomputed speedups. The human-readable view. |

## Provenance

- Host: AMD EPYC 9B14, 15 physical / 30 logical cores, 117 GB RAM, **no GPU**.
- R 4.5.0, DESeq2 1.46.0, apeglm 1.32.0, reference BLAS, all thread counts pinned to 1.
- Six datasets regenerated from Bioconductor on this host; dimensions match the
  documented reference **exactly** (pasilla 7×12 359, airway 8×33 469,
  gtex_blood_muscle 300×54 922), so the inputs are the same data.
- Measured 2026-07-28, on an otherwise idle machine.

## Findings

1. **~2× faster per core than the original baseline host**, on every dataset.
   GTEx total 350 s → 147 s; its dominant dispersion step 231 s → 81 s.
2. **This box parallelises small-*n* data** instead of regressing on it, unlike
   the original 12-core host where `MulticoreParam` made small inputs *slower*.
   Best-case 3.6× on GTEx; the two pasilla cases still peak at w=4 and get worse
   by w=15, so the fork-overhead effect is real, just smaller here.
3. **`results()` does not parallelise at all** — flat ~11 s on GTEx across
   w=1…15, where `deseq_fit` scaled 7.9×. That serial remainder is what caps the
   curve, exactly the plateau the sweep exists to expose.
4. **Speedups shrink but do not invert.** Against the already-published A100
   timings, best cuDESeq2 stays ahead on all six datasets:

   | baseline | range |
   |---|---|
   | published (A100-host R, 1 thread, DESeq2 1.30.1) | 8–78× |
   | this host's R, 1 thread | 4.3–32.8× |
   | this host's R, best multi-core (≤15 workers) | 1.3–8.8× |

   The GPU side of that arithmetic is the **existing** A100 measurement — this
   host has no GPU — so those are recomputations, not new measurements, and they
   compare two different machines.

5. **The thin case is `airway_cell`: 1.3×** (3 469 ms GPU vs 4 572 ms best R), a
   1.1 s absolute gap that sits inside the 10–25% cross-session drift the bench
   README already warns about. Its GPU total is **94% `lfc_shrink`** (3 259 of
   3 469 ms); dispersion — what the CUDA-graph/Triton work accelerates — is 60 ms.
   So that row measures apeGLM shrinkage, not the dispersion port, and R's
   `lfcShrink` parallelises well enough to nearly catch it. It is also the row
   most likely to invert on a box with more than 15 cores.
   The durable win is the dispersion-bound large-cohort end (GTEx, 8.8–32.8×).

6. **BLAS is not a confound.** It differed from the reference host, so it was
   measured rather than assumed (`bench/blas_sensitivity.R`, output in
   `blas_probe.txt`): system BLAS vs OpenBLAS-pthread differ by less than the
   run-to-run spread of the *same* configuration, and not in a consistent
   direction. DESeq2's per-gene IRLS is C++ scalar work that does not dispatch
   through BLAS — matching `benchmarks/RESULTS.md`, which found serial DESeq2
   time unchanged at 12 vs 1 BLAS threads.

## If these numbers go into the paper

`scripts/audit_paper_numbers.py` loads only `parity.json`, `timings.json`,
`benchmarks/results_per_step.json` and `validation/results/*.json`. Any `\fact{}`
citing this host will be flagged UNBACKED until `summary.json` (or
`r_scaling.jsonl`) is added as a source there and the claims registered. That is
deliberate — the audit is what stops an unsourced number from shipping.

## Reproduce / add a third host

```bash
# R-only; no GPU needed. Datasets must exist under validation/data/<case>/.
R_DESEQ2_LIB=/path/to/rlib bash bench/run_r_scaling.sh \
    --allow-any-deseq2 --out bench/results/<hostname>

# optional: is this host's BLAS a confound?
R_DESEQ2_LIB=/path/to/rlib Rscript bench/blas_sensitivity.R
LD_PRELOAD=/path/to/libopenblas.so.0 R_DESEQ2_LIB=/path/to/rlib \
    Rscript bench/blas_sensitivity.R

# re-render the comparison (repeat --new/--label per host)
python bench/render_r_hosts.py \
    --new bench/results/<hostname>/r_scaling.jsonl --label "<hostname>" \
    --blas-probe bench/results/<hostname>/blas_probe.txt \
    --json-out bench/results/<hostname>/summary.json
```

Drop `--allow-any-deseq2` if the host can supply DESeq2 1.30.1 (native or via
the `bioconductor/bioconductor_docker:RELEASE_3_12` fallback) — then the column
*is* a controlled comparison and `comparable_to_reference` becomes true.
