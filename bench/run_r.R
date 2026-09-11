#!/usr/bin/env Rscript
# Benchmark harness, R side. For every prepared case in validation/data/<case>/
# (counts.csv, coldata.csv, meta.json) this:
#   (1) runs the installed DESeq2 standard Wald pipeline substep-by-substep,
#       including DESeq2's Cook's-outlier replacement/refit in glm_fit, and
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

DATA <- Sys.getenv("BENCH_DATA", unset = "validation/data")
CACHE <- Sys.getenv("BENCH_CACHE", unset = "bench/cache")
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

# The staged equivalent of the part of DESeq(test="Wald") that follows
# estimateDispersions(). Calling DESeq() here would be wrong: DESeq() always
# calls estimateDispersions() itself, even when dispersion estimates exist, and
# would charge a second complete dispersion fit to the glm_fit stage. Keep this
# aligned with cuDESeq2's glm_fit boundary: the initial Wald fit plus eligible
# Cook's-outlier replacement and refitting.
wald_and_outlier_refit <- function(dds, quiet = TRUE) {
  dds <- nbinomWaldTest(dds, betaPrior = FALSE, quiet = quiet)
  model_matrix <- attr(dds, "modelMatrix")
  sufficient_reps <- any(DESeq2:::nOrMoreInCell(model_matrix, 7))
  if (sufficient_reps) {
    dds <- DESeq2:::refitWithoutOutliers(
      dds,
      test = "Wald",
      betaPrior = FALSE,
      full = design(dds),
      quiet = quiet,
      minReplicatesForReplace = 7,
      modelMatrix = NULL
    )
  }
  dds
}

assert_same_numeric <- function(label, observed, expected, tolerance = 0) {
  if (!isTRUE(all.equal(observed, expected, tolerance = tolerance,
                        check.attributes = FALSE))) {
    finite <- is.finite(observed) & is.finite(expected)
    max_abs <- if (any(finite)) max(abs(observed[finite] - expected[finite])) else NA_real_
    stop(sprintf("staged-vs-DESeq output mismatch: %s (max abs diff %.6g)",
                 label, max_abs))
  }
}

for (case in cases) {
  d <- file.path(DATA, case); out <- file.path(CACHE, case); dir.create(out, recursive = TRUE, showWarnings = FALSE)
  meta <- read_meta(file.path(d, "meta.json"))
  cts <- as.matrix(read.csv(file.path(d, "counts.csv"), row.names = 1)); storage.mode(cts) <- "integer"
  cd  <- read.csv(file.path(d, "coldata.csv"), row.names = 1)
  cd[[meta$factor]] <- relevel(factor(cd[[meta$factor]]), ref = meta$ref)
  reps <- as.integer(Sys.getenv("BENCH_R_REPS", unset = "5"))
  # Use the FULL design (e.g. ~ cell + dex), not just the primary factor, so R
  # fits the same model as cuDESeq2 (which reads meta$design).
  mk <- function() DESeqDataSetFromMatrix(cts, cd, as.formula(meta$design))

  # Untimed warm-up, matching the warm timing policy on the GPU side.
  warm <- mk()
  warm <- estimateSizeFactors(warm)
  warm <- estimateDispersions(warm, quiet = TRUE)
  warm <- wald_and_outlier_refit(warm)
  warm_res <- results(warm, name = meta$coef)
  invisible(lfcShrink(warm, coef = meta$coef, type = "apeglm", quiet = TRUE))
  rm(warm, warm_res)

  # Per-substep timing: each rep runs the full chain, timing each stage.
  tt <- list(normalization = c(), dispersion = c(), glm_fit = c(),
             significance = c(), lfc_shrink = c())
  for (i in seq_len(reps)) {
    dds <- mk()
    t0 <- Sys.time(); dds <- estimateSizeFactors(dds);                 tt$normalization <- c(tt$normalization, as.numeric(Sys.time()-t0, units="secs"))
    t0 <- Sys.time(); dds <- estimateDispersions(dds, quiet = TRUE);   tt$dispersion    <- c(tt$dispersion,    as.numeric(Sys.time()-t0, units="secs"))
    t0 <- Sys.time(); dds <- wald_and_outlier_refit(dds);            tt$glm_fit       <- c(tt$glm_fit,       as.numeric(Sys.time()-t0, units="secs"))
    t0 <- Sys.time(); res <- results(dds, name = meta$coef);          tt$significance  <- c(tt$significance,  as.numeric(Sys.time()-t0, units="secs"))
    t0 <- Sys.time(); sh  <- lfcShrink(dds, coef = meta$coef, type = "apeglm", quiet = TRUE); tt$lfc_shrink <- c(tt$lfc_shrink, as.numeric(Sys.time()-t0, units="secs"))
  }
  timings <- lapply(tt, function(x) median(x) * 1000)
  timings$stage_total <- Reduce(
    `+`, timings[c("normalization", "dispersion", "glm_fit", "significance", "lfc_shrink")]
  )
  direct_pipeline <- function() {
    direct_dds <- mk()
    direct_dds <- estimateSizeFactors(direct_dds)
    direct_dds <- estimateDispersions(direct_dds, quiet = TRUE)
    direct_dds <- wald_and_outlier_refit(direct_dds)
    invisible(results(direct_dds, name = meta$coef))
    invisible(lfcShrink(direct_dds, coef = meta$coef, type = "apeglm", quiet = TRUE))
  }
  direct_values <- numeric(reps)
  for (i in seq_len(reps)) {
    direct_start <- proc.time()[["elapsed"]]
    direct_pipeline()
    direct_values[[i]] <- proc.time()[["elapsed"]] - direct_start
  }
  timings$total <- median(direct_values) * 1000

  # Untimed correctness gate: explicit staging must reproduce an ordinary
  # DESeq() call exactly before this case's measurements are accepted. For the
  # large GTEx case, compare against the existing version-matched DESeq() CSVs
  # instead of retaining a second 54,922-gene DESeqDataSet in memory and running
  # a third full pipeline. The cache is read before anything below overwrites it.
  cached_paths <- file.path(out, c("r_sizefactors.csv", "r_dispersions.csv",
                                   "r_results.csv", "r_shrink.csv"))
  force_fresh_gate <- identical(Sys.getenv("BENCH_FORCE_FRESH_GATE", unset = "0"), "1")
  if (meta$n_samples > 100 && !force_fresh_gate && all(file.exists(cached_paths))) {
    cached_sf <- read.csv(cached_paths[1])
    cached_disp <- read.csv(cached_paths[2])
    cached_res <- read.csv(cached_paths[3])
    cached_sh <- read.csv(cached_paths[4])
    csv_tolerance <- 1e-12
    assert_same_numeric("size factors", sizeFactors(dds), cached_sf$sizeFactor,
                        csv_tolerance)
    assert_same_numeric("dispersions", dispersions(dds), cached_disp$dispersion,
                        csv_tolerance)
    for (column in c("baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj")) {
      assert_same_numeric(column, as.data.frame(res)[[column]], cached_res[[column]],
                          csv_tolerance)
    }
    assert_same_numeric("apeglm log2FoldChange", as.data.frame(sh)$log2FoldChange,
                        cached_sh$log2FoldChange, csv_tolerance)
    assert_same_numeric("apeglm lfcSE", as.data.frame(sh)$lfcSE, cached_sh$lfcSE,
                        csv_tolerance)
    validation_source <- "cached version-matched DESeq() reference (CSV tolerance 1e-12)"
  } else {
    standard <- DESeq(mk(), test = "Wald", quiet = TRUE)
    standard_res <- results(standard, name = meta$coef)
    standard_sh <- lfcShrink(standard, coef = meta$coef, type = "apeglm", quiet = TRUE)
    assert_same_numeric("size factors", sizeFactors(dds), sizeFactors(standard))
    assert_same_numeric("dispersions", dispersions(dds), dispersions(standard))
    for (column in c("baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj")) {
      assert_same_numeric(column, as.data.frame(res)[[column]],
                          as.data.frame(standard_res)[[column]])
    }
    assert_same_numeric("apeglm log2FoldChange", as.data.frame(sh)$log2FoldChange,
                        as.data.frame(standard_sh)$log2FoldChange)
    assert_same_numeric("apeglm lfcSE", as.data.frame(sh)$lfcSE,
                        as.data.frame(standard_sh)$lfcSE)
    validation_source <- "fresh DESeq() run"
  }
  cat(sprintf("  %-18s staged output exactly matches %s\n", case, validation_source))

  # Intermediates (deterministic — from the last rep's dds/res/sh).
  write.csv(data.frame(sample = colnames(dds), sizeFactor = sizeFactors(dds)), file.path(out, "r_sizefactors.csv"), row.names = FALSE)
  write.csv(data.frame(gene = rownames(dds), dispersion = dispersions(dds)),   file.path(out, "r_dispersions.csv"), row.names = FALSE)
  rdf <- as.data.frame(res); rdf$gene <- rownames(res)
  write.csv(rdf[, c("gene","baseMean","log2FoldChange","lfcSE","stat","pvalue","padj")], file.path(out, "r_results.csv"), row.names = FALSE)
  sdf <- as.data.frame(sh); sdf$gene <- rownames(sh)
  write.csv(sdf[, c("gene","log2FoldChange","lfcSE")], file.path(out, "r_shrink.csv"), row.names = FALSE)

  total_values_json <- paste(sprintf("%.3f", direct_values * 1000), collapse = ",")
  writeLines(sprintf('{"case":"%s","reps":%d,"pipeline":"staged_standard_wald_no_duplicate_dispersion","glm_fit_scope":"nbinomWaldTest_plus_outlier_replacement_refit","output_equivalent_to_DESeq":true,"total_definition":"median direct wall time including DESeqDataSet construction","stage_total_definition":"sum of independently measured stage medians","r_version":"%s","deseq2_version":"%s","apeglm_version":"%s","normalization":%.3f,"dispersion":%.3f,"glm_fit":%.3f,"significance":%.3f,"lfc_shrink":%.3f,"stage_total":%.3f,"total":%.3f,"total_values_ms":[%s]}',
    case, reps, as.character(getRversion()), as.character(packageVersion("DESeq2")),
    as.character(packageVersion("apeglm")), timings$normalization, timings$dispersion,
    timings$glm_fit, timings$significance, timings$lfc_shrink,
    timings$stage_total, timings$total, total_values_json),
    file.path(out, "r_timings.json"))
  cat(sprintf("  %-18s reps=%d  norm=%.0f disp=%.0f glm=%.0f sig=%.0f shrink=%.0f  direct=%.0f ms\n",
    case, reps, timings$normalization, timings$dispersion, timings$glm_fit,
    timings$significance, timings$lfc_shrink, timings$total))
}
cat("R side done ->", CACHE, "\n")
