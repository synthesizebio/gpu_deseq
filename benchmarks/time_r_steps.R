#!/usr/bin/env Rscript
# Per-step R DESeq2 timing on the matrices written by bench_per_step.py, split
# into the same 5 steps gpu_deseq exposes:
#   normalization  estimateSizeFactors
#   dispersion     estimateDispersions
#   glm_fit        nbinomWaldTest
#   significance   results
#   lfc_shrink     lfcShrink(type="apeglm")
# Single-threaded; median of n_reps. Writes <casedir>/r_step_timings.csv.
#
# Usage: Rscript time_r_steps.R <casedir> [n_reps]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: time_r_steps.R <casedir> [n_reps] [ncores]")
casedir <- args[[1]]
n_reps <- if (length(args) >= 2) as.integer(args[[2]]) else 3L
ncores <- if (length(args) >= 3) as.integer(args[[3]]) else parallel::detectCores()

.libPaths(c(Sys.getenv("R_DESEQ2_LIB", unset = "~/R/library"), .libPaths()))
suppressMessages(library(DESeq2))
suppressMessages(library(apeglm))
suppressMessages(library(BiocParallel))
cat("DESeq2", as.character(packageVersion("DESeq2")),
    "apeglm", as.character(packageVersion("apeglm")), "| ncores", ncores, "\n")
bp <- MulticoreParam(ncores)

manifest <- read.csv(file.path(casedir, "manifest.csv"), stringsAsFactors = FALSE)
# per-step columns are serial (DESeq2 exposes parallelism only at the DESeq()
# level, chunking genes across the whole gene-wise flow — not per step). The two
# full-pipeline columns capture serial vs multi-core end-to-end.
cols <- c("tag", "r_norm_ms", "r_disp_ms", "r_glm_ms", "r_sig_ms", "r_shrink_ms",
          "r_full_ms", "r_full_mc_ms")
out <- setNames(data.frame(matrix(ncol = length(cols), nrow = 0)), cols)

med_ms <- function(fn) {
  fn()
  ts <- numeric(n_reps)
  for (r in seq_len(n_reps)) { t0 <- Sys.time(); fn(); ts[r] <- as.numeric(Sys.time() - t0, units = "secs") }
  median(ts) * 1000
}

for (i in seq_len(nrow(manifest))) {
  tag <- manifest$tag[i]
  design <- as.formula(manifest$design[i])
  counts <- as.matrix(read.csv(file.path(casedir, paste0(tag, "_counts.csv")), row.names = 1))
  coldata <- read.csv(file.path(casedir, paste0(tag, "_coldata.csv")), row.names = 1)
  for (cn in colnames(coldata)) coldata[[cn]] <- factor(coldata[[cn]])
  mk <- function() DESeqDataSetFromMatrix(countData = counts, colData = coldata, design = design)

  norm_ms <- med_ms(function() invisible(estimateSizeFactors(mk())))
  sf <- estimateSizeFactors(mk())
  disp_ms <- med_ms(function() invisible(estimateDispersions(sf, quiet = TRUE)))
  dp <- estimateDispersions(sf, quiet = TRUE)
  glm_ms <- med_ms(function() invisible(nbinomWaldTest(dp)))
  wd <- nbinomWaldTest(dp)
  sig_ms <- med_ms(function() invisible(results(wd)))
  coef <- grep("condition", resultsNames(wd), value = TRUE)[1]
  shrink_ms <- med_ms(function() invisible(lfcShrink(wd, coef = coef, type = "apeglm", quiet = TRUE)))

  # Full pipeline: serial vs multi-core (the charitable "optimal R"). DESeq2
  # fans the per-gene dispersion + GLM work across cores via parallel=TRUE.
  full_ms <- med_ms(function() invisible(results(DESeq(mk(), quiet = TRUE))))
  full_mc_ms <- med_ms(function() invisible(
    results(DESeq(mk(), parallel = TRUE, BPPARAM = bp, quiet = TRUE))))

  cat(sprintf("  %-22s norm=%.1f disp=%.1f glm=%.1f sig=%.1f shrink=%.1f | full 1c=%.0f %dc=%.0f ms\n",
              tag, norm_ms, disp_ms, glm_ms, sig_ms, shrink_ms, full_ms, ncores, full_mc_ms))
  out[nrow(out) + 1, ] <- list(tag, norm_ms, disp_ms, glm_ms, sig_ms, shrink_ms, full_ms, full_mc_ms)
}
write.csv(out, file.path(casedir, "r_step_timings.csv"), row.names = FALSE)
