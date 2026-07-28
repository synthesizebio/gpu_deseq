#!/usr/bin/env Rscript
# Does the BLAS implementation change DESeq2's serial cost on this host?
#
# run_r_scaling.sh records the BLAS in its output header and warns that absolute
# serial times are only comparable across machines when the BLAS matches. When a
# new host's BLAS differs from the reference host's (Debian reference BLAS here
# vs OpenBLAS-pthread on the A100 host), that warning either invalidates the
# cross-host comparison or it does not -- which is a measurable question, not a
# judgement call. This script measures it.
#
# Times the three expensive substeps once per BLAS, single-threaded. Select the
# BLAS with LD_PRELOAD so no system state is touched:
#
#   # as configured
#   R_DESEQ2_LIB=... Rscript bench/blas_sensitivity.R
#   # against another BLAS
#   LD_PRELOAD=/path/to/libopenblas.so.0 R_DESEQ2_LIB=... Rscript bench/blas_sensitivity.R
#
# If the two agree within run-to-run noise, DESeq2's per-gene C++ work does not
# go through BLAS (consistent with benchmarks/RESULTS.md finding serial DESeq2
# time unchanged at 12 vs 1 BLAS threads), and the BLAS mismatch therefore does
# not confound a cross-host R comparison.
#
# Env: BENCH_CASE (default airway), BENCH_REPS (default 3), R_DESEQ2_LIB.
.libPaths(c(Sys.getenv("R_DESEQ2_LIB", unset = "~/R/library"), .libPaths()))
suppressMessages({library(DESeq2); library(apeglm)})

CASE <- Sys.getenv("BENCH_CASE", unset = "airway")
REPS <- as.integer(Sys.getenv("BENCH_REPS", unset = "3"))
D <- file.path("validation/data", CASE)

# Same minimal flat-JSON reader as bench/run_r.R, so metadata is read identically.
read_meta <- function(p) {
  txt <- paste(readLines(p), collapse = " ")
  get <- function(k) { m <- regmatches(txt, regexpr(sprintf('"%s"\\s*:\\s*"?([^",}]+)', k), txt)); sub(sprintf('"%s"\\s*:\\s*"?', k), "", m) }
  list(design = get("design"), factor = get("factor"), ref = get("ref"), coef = get("coef"))
}
meta <- read_meta(file.path(D, "meta.json"))
cts <- as.matrix(read.csv(file.path(D, "counts.csv"), row.names = 1)); storage.mode(cts) <- "integer"
cd  <- read.csv(file.path(D, "coldata.csv"), row.names = 1)
cd[[meta$factor]] <- relevel(factor(cd[[meta$factor]]), ref = meta$ref)

secs <- function(t0) as.numeric(Sys.time() - t0, units = "secs")
tt <- list(dispersion = c(), glm_fit = c(), lfc_shrink = c())
for (i in seq_len(REPS)) {
  dds <- DESeqDataSetFromMatrix(cts, cd, as.formula(meta$design))
  dds <- estimateSizeFactors(dds)
  t0 <- Sys.time(); dds <- estimateDispersions(dds, quiet = TRUE); tt$dispersion <- c(tt$dispersion, secs(t0))
  t0 <- Sys.time(); dds <- nbinomWaldTest(dds);                    tt$glm_fit    <- c(tt$glm_fit,    secs(t0))
  invisible(results(dds, name = meta$coef))
  t0 <- Sys.time(); invisible(lfcShrink(dds, coef = meta$coef, type = "apeglm", quiet = TRUE))
                                                                   tt$lfc_shrink <- c(tt$lfc_shrink, secs(t0))
}
cat(sprintf("case=%s reps=%d DESeq2=%s\nBLAS=%s\ndispersion=%.0f glm_fit=%.0f lfc_shrink=%.0f ms\n",
            CASE, REPS, packageVersion("DESeq2"), sessionInfo()$BLAS,
            median(tt$dispersion) * 1000, median(tt$glm_fit) * 1000, median(tt$lfc_shrink) * 1000))
