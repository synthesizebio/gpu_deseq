#!/usr/bin/env Rscript
# Benchmark harness, R side. For every prepared case in validation/data/<case>/
# (counts.csv, coldata.csv, meta.json) this:
#   (1) runs the installed DESeq2 standard Wald pipeline substep-by-substep,
#       including DESeq()'s Cook's-outlier replacement/refit in glm_fit, and
#       times the five substeps (normalization, dispersion, glm_fit,
#       significance, lfc_shrink), median over BENCH_R_REPS reps;
#   (2) exports every per-substep intermediate so the Python side can verify
#       equivalence: size factors, dispersions, results (LFC/stat/padj/baseMean),
#       apeGLM-shrunk LFC.
# Writes bench/cache/<case>/{r_timings.json, r_sizefactors.csv, r_dispersions.csv,
# r_results.csv, r_shrink.csv}.
#
# Usage: Rscript bench/run_r.R [case ...]   (default: all cases)
custom_lib <- Sys.getenv("R_DESEQ2_LIB", unset = "")
if (nzchar(custom_lib)) .libPaths(c(custom_lib, .libPaths()))
suppressMessages({library(DESeq2); library(apeglm)})

DATA <- "validation/data"; CACHE <- "bench/cache"
args <- commandArgs(trailingOnly = TRUE)
cases <- if (length(args)) args else list.dirs(DATA, recursive = FALSE, full.names = FALSE)
cases <- Filter(function(c) file.exists(file.path(DATA, c, "meta.json")), cases)

# minimal JSON reader for the flat meta.json we write
read_meta <- function(p) {
  txt <- paste(readLines(p), collapse = " ")
  get <- function(k) { m <- regmatches(txt, regexpr(sprintf('"%s"\\s*:\\s*"?([^",}]+)', k), txt)); sub(sprintf('"%s"\\s*:\\s*"?', k), "", m) }
  list(design = get("design"), factor = get("factor"), ref = get("ref"), nonref = get("nonref"),
       coef = get("coef"), n_samples = as.integer(get("n_samples")))
}

med_ms <- function(f, reps) { ts <- numeric(reps); for (i in seq_len(reps)) { t0 <- Sys.time(); f(); ts[i] <- as.numeric(Sys.time() - t0, units = "secs") }; median(ts) * 1000 }

for (case in cases) {
  d <- file.path(DATA, case); out <- file.path(CACHE, case); dir.create(out, recursive = TRUE, showWarnings = FALSE)
  meta <- read_meta(file.path(d, "meta.json"))
  cts <- as.matrix(read.csv(file.path(d, "counts.csv"), row.names = 1)); storage.mode(cts) <- "integer"
  cd  <- read.csv(file.path(d, "coldata.csv"), row.names = 1)
  cd[[meta$factor]] <- relevel(factor(cd[[meta$factor]]), ref = meta$ref)
  reps <- as.integer(Sys.getenv("BENCH_R_REPS", unset = if (meta$n_samples > 100) "1" else "3"))
  # Use the FULL design (e.g. ~ cell + dex), not just the primary factor, so R
  # fits the same model as cuDESeq2 (which reads meta$design).
  mk <- function() DESeqDataSetFromMatrix(cts, cd, as.formula(meta$design))

  # Per-substep timing: each rep runs the full chain, timing each stage.
  tt <- list(normalization = c(), dispersion = c(), glm_fit = c(), significance = c(), lfc_shrink = c())
  for (i in seq_len(reps)) {
    dds <- mk()
    t0 <- Sys.time(); dds <- estimateSizeFactors(dds);                 tt$normalization <- c(tt$normalization, as.numeric(Sys.time()-t0, units="secs"))
    t0 <- Sys.time(); dds <- estimateDispersions(dds, quiet = TRUE);   tt$dispersion    <- c(tt$dispersion,    as.numeric(Sys.time()-t0, units="secs"))
    # DESeq() recognizes the existing size factors and dispersions, so this
    # call times the Wald fit and the standard count-outlier replacement/refit.
    t0 <- Sys.time(); dds <- DESeq(dds, test = "Wald", quiet = TRUE); tt$glm_fit       <- c(tt$glm_fit,       as.numeric(Sys.time()-t0, units="secs"))
    t0 <- Sys.time(); res <- results(dds, name = meta$coef);          tt$significance  <- c(tt$significance,  as.numeric(Sys.time()-t0, units="secs"))
    t0 <- Sys.time(); sh  <- lfcShrink(dds, coef = meta$coef, type = "apeglm", quiet = TRUE); tt$lfc_shrink <- c(tt$lfc_shrink, as.numeric(Sys.time()-t0, units="secs"))
  }
  timings <- lapply(tt, function(x) median(x) * 1000)
  timings$total <- Reduce(`+`, timings)

  # Intermediates (deterministic — from the last rep's dds/res/sh).
  write.csv(data.frame(sample = colnames(dds), sizeFactor = sizeFactors(dds)), file.path(out, "r_sizefactors.csv"), row.names = FALSE)
  write.csv(data.frame(gene = rownames(dds), dispersion = dispersions(dds)),   file.path(out, "r_dispersions.csv"), row.names = FALSE)
  rdf <- as.data.frame(res); rdf$gene <- rownames(res)
  write.csv(rdf[, c("gene","baseMean","log2FoldChange","lfcSE","stat","pvalue","padj")], file.path(out, "r_results.csv"), row.names = FALSE)
  sdf <- as.data.frame(sh); sdf$gene <- rownames(sh)
  write.csv(sdf[, c("gene","log2FoldChange","lfcSE")], file.path(out, "r_shrink.csv"), row.names = FALSE)

  writeLines(sprintf('{"case":"%s","reps":%d,"r_version":"%s","deseq2_version":"%s","apeglm_version":"%s","normalization":%.3f,"dispersion":%.3f,"glm_fit":%.3f,"significance":%.3f,"lfc_shrink":%.3f,"total":%.3f}',
    case, reps, as.character(getRversion()), as.character(packageVersion("DESeq2")),
    as.character(packageVersion("apeglm")), timings$normalization, timings$dispersion,
    timings$glm_fit, timings$significance, timings$lfc_shrink, timings$total),
    file.path(out, "r_timings.json"))
  cat(sprintf("  %-18s reps=%d  norm=%.0f disp=%.0f glm=%.0f sig=%.0f shrink=%.0f  total=%.0f ms\n",
    case, reps, timings$normalization, timings$dispersion, timings$glm_fit, timings$significance, timings$lfc_shrink, timings$total))
}
cat("R side done ->", CACHE, "\n")
