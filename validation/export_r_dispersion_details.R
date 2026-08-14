#!/usr/bin/env Rscript
# Real-data validation, step 1b (R side): export the *components* of DESeq2's
# dispersion estimate for each prepared case, which fetch_and_reference.R does
# not save (it writes only the final `dispersions(dds)`).
#
# The canonical DESeq2 diagnostic, plotDispEsts(), draws three series against the
# mean of normalized counts: the gene-wise MLE, the fitted trend, and the final
# (MAP, or MLE-if-outlier) estimate. Reproducing that plot for cuDESeq2 next to R
# requires all three from R, so write them per case to:
#   validation/data/<name>/r_disp_details.csv
#     gene, baseMean, dispGeneEst, dispFit, dispersion, dispOutlier
#
# Counts/coldata are re-read from the already-exported CSVs and the model is
# rebuilt exactly as in fetch_and_reference.R (same relevel, same design), so the
# fit is the same one the reference results came from.
#
# Usage: Rscript validation/export_r_dispersion_details.R [case ...]
custom_lib <- Sys.getenv("R_DESEQ2_LIB", unset = "")
if (nzchar(custom_lib)) .libPaths(c(custom_lib, .libPaths()))
suppressMessages(library(DESeq2))

read_meta <- function(path) {
  text <- paste(readLines(path), collapse = " ")
  value <- function(key) {
    match <- regmatches(
      text, regexpr(sprintf('"%s"\\s*:\\s*"?([^",}]+)', key), text)
    )
    sub(sprintf('"%s"\\s*:\\s*"?', key), "", match)
  }
  list(
    design = value("design"),
    factor = value("factor"),
    ref = value("ref"),
    n_samples = as.integer(value("n_samples"))
  )
}

OUT <- "validation/data"
cases <- commandArgs(trailingOnly = TRUE)
if (!length(cases)) cases <- sort(basename(dirname(Sys.glob(file.path(OUT, "*", "meta.json")))))

for (name in cases) {
  d <- file.path(OUT, name)
  meta <- read_meta(file.path(d, "meta.json"))
  counts <- as.matrix(read.csv(file.path(d, "counts.csv"), row.names = 1, check.names = FALSE))
  storage.mode(counts) <- "integer"
  coldata <- read.csv(file.path(d, "coldata.csv"), row.names = 1, check.names = FALSE)
  coldata[[meta$factor]] <- relevel(factor(coldata[[meta$factor]]), ref = meta$ref)

  t0 <- Sys.time()
  dds <- DESeq(DESeqDataSetFromMatrix(counts, coldata, as.formula(meta$design)), quiet = TRUE)
  m <- mcols(dds)
  out <- data.frame(
    gene        = rownames(dds),
    baseMean    = as.numeric(m$baseMean),
    dispGeneEst = as.numeric(m$dispGeneEst),
    dispFit     = as.numeric(m$dispFit),
    dispersion  = as.numeric(m$dispersion),
    dispOutlier = as.logical(m$dispOutlier)
  )
  write.csv(out, file.path(d, "r_disp_details.csv"), row.names = FALSE)
  cat(sprintf("  %-18s %5d genes  %6.1f s -> %s\n", name, nrow(out),
              as.numeric(Sys.time() - t0, units = "secs"),
              file.path(d, "r_disp_details.csv")))
}
