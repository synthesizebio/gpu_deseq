#!/usr/bin/env Rscript
# Time R DESeq2 on the matrices written by bench_vs_r.py, for both:
#   - dispersion stage : estimateDispersions (gene-est + trend + MAP + outlier)
#   - full pipeline     : estimateSizeFactors + estimateDispersions +
#                         nbinomWaldTest + results  (i.e. DESeq() + results())
# Single-threaded; median of n_reps runs. Writes <casedir>/r_timings.csv.
#
# Usage: Rscript time_r_disp.R <casedir> [n_reps]
# DESeq2 1.30.1 must be importable; set R_DESEQ2_LIB if it is not on the default
# .libPaths (defaults to ~/R/library).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: time_r_disp.R <casedir> [n_reps]")
casedir <- args[[1]]
n_reps <- if (length(args) >= 2) as.integer(args[[2]]) else 5L

.libPaths(c(Sys.getenv("R_DESEQ2_LIB", unset = "~/R/library"), .libPaths()))
suppressMessages(library(DESeq2))
cat("DESeq2", as.character(packageVersion("DESeq2")), "\n")

manifest <- read.csv(file.path(casedir, "manifest.csv"), stringsAsFactors = FALSE)
out <- data.frame(tag = character(), r_disp_ms = numeric(), r_full_ms = numeric(),
                  stringsAsFactors = FALSE)

med_ms <- function(fn) {
  fn()  # warm
  ts <- numeric(n_reps)
  for (r in seq_len(n_reps)) {
    t0 <- Sys.time(); fn(); ts[r] <- as.numeric(Sys.time() - t0, units = "secs")
  }
  median(ts) * 1000
}

for (i in seq_len(nrow(manifest))) {
  tag <- manifest$tag[i]
  design <- as.formula(manifest$design[i])
  counts <- as.matrix(read.csv(file.path(casedir, paste0(tag, "_counts.csv")), row.names = 1))
  coldata <- read.csv(file.path(casedir, paste0(tag, "_coldata.csv")), row.names = 1)
  for (cn in colnames(coldata)) coldata[[cn]] <- factor(coldata[[cn]])
  mk <- function() DESeqDataSetFromMatrix(countData = counts, colData = coldata, design = design)

  sf_dds <- estimateSizeFactors(mk())
  disp_ms <- med_ms(function() invisible(estimateDispersions(sf_dds, quiet = TRUE)))
  full_ms <- med_ms(function() invisible(results(DESeq(mk(), quiet = TRUE))))

  cat(sprintf("  %-22s disp=%.1f ms  full=%.1f ms\n", tag, disp_ms, full_ms))
  out <- rbind(out, data.frame(tag = tag, r_disp_ms = disp_ms, r_full_ms = full_ms))
}
write.csv(out, file.path(casedir, "r_timings.csv"), row.names = FALSE)
