#!/usr/bin/env Rscript

# One-worker R DESeq2 reference timing for a cohort emitted by
# bench/gtex_scaling.py. The GPU JSON supplies the exact one-based source
# columns; the pinned recount2 RData supplies counts and tissue annotations.

custom_lib <- Sys.getenv("R_DESEQ2_LIB", unset = "")
if (nzchar(custom_lib)) .libPaths(c(custom_lib, .libPaths()))
suppressMessages({
  library(DESeq2)
  library(apeglm)
  library(BiocParallel)
})
register(SerialParam())

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop(
    "usage: run_gtex_scaling_r.R SOURCE_RDATA MATRIX_DIR ",
    "GPU_CASE_JSON OUTPUT_DIR"
  )
}
source_rdata <- normalizePath(args[[1]])
matrix_dir <- normalizePath(args[[2]])
gpu_json_path <- normalizePath(args[[3]])
output_dir <- args[[4]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
git_commit <- system2("git", c("rev-parse", "HEAD"), stdout = TRUE)
git_dirty <- length(system2("git", c("status", "--porcelain"), stdout = TRUE)) > 0

extract_numeric_array <- function(text, key) {
  pattern <- sprintf('"%s"[[:space:]]*:[[:space:]]*\\[([^]]*)\\]', key)
  matched <- regmatches(text, regexpr(pattern, text, perl = TRUE))
  if (!nzchar(matched)) stop("missing array in GPU JSON: ", key)
  body <- sub('^[^[]*\\[', "", matched)
  body <- sub('\\]$', "", body)
  as.integer(trimws(strsplit(body, ",", fixed = TRUE)[[1]]))
}

gpu_json <- paste(readLines(gpu_json_path, warn = FALSE), collapse = "\n")
source_columns <- extract_numeric_array(gpu_json, "source_columns_one_based")
if (anyNA(source_columns) || any(source_columns < 1)) {
  stop("invalid source columns in GPU JSON")
}

environment <- new.env(parent = emptyenv())
load(source_rdata, envir = environment)
rse <- environment$rse_gene
gene_ids <- readLines(file.path(matrix_dir, "genes.txt"), warn = FALSE)
gene_index <- match(gene_ids, rownames(rse))
stopifnot(!anyNA(gene_index), length(gene_ids) == 54922)

sample_table <- read.csv(
  file.path(matrix_dir, "samples.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
sample_rows <- match(source_columns, sample_table$source_column)
stopifnot(!anyNA(sample_rows))
selected_samples <- sample_table[sample_rows, , drop = FALSE]
stopifnot(identical(selected_samples$source_column, source_columns))

selected_rse <- rse[gene_index, source_columns]
read_length <- as.numeric(colData(selected_rse)$avg_read_length)
read_length[is.na(read_length) | read_length <= 0] <- 100
counts <- round(sweep(assay(selected_rse), 2, read_length, "/"))
storage.mode(counts) <- "integer"
stopifnot(identical(rownames(counts), gene_ids))

tissue_levels <- unique(selected_samples$tissue)
coldata <- data.frame(
  tissue = factor(selected_samples$tissue, levels = tissue_levels),
  row.names = colnames(counts)
)
mk <- function() DESeqDataSetFromMatrix(counts, coldata, ~ tissue)

wald_and_outlier_refit <- function(dds) {
  dds <- nbinomWaldTest(dds, betaPrior = FALSE, quiet = TRUE)
  model_matrix <- attr(dds, "modelMatrix")
  if (any(DESeq2:::nOrMoreInCell(model_matrix, 7))) {
    dds <- DESeq2:::refitWithoutOutliers(
      dds,
      test = "Wald",
      betaPrior = FALSE,
      full = design(dds),
      quiet = TRUE,
      minReplicatesForReplace = 7,
      modelMatrix = NULL
    )
  }
  dds
}

if (identical(Sys.getenv("GTEX_R_DISPERSION_ONLY", unset = "0"), "1")) {
  stage_start <- proc.time()[["elapsed"]]
  dds <- mk()
  normalization_start <- proc.time()[["elapsed"]]
  dds <- estimateSizeFactors(dds)
  normalization_ms <- (proc.time()[["elapsed"]] - normalization_start) * 1000
  dispersion_start <- proc.time()[["elapsed"]]
  dds <- estimateDispersions(dds, quiet = TRUE)
  dispersion_ms <- (proc.time()[["elapsed"]] - dispersion_start) * 1000
  write.csv(
    data.frame(gene = rownames(dds), dispersion = dispersions(dds)),
    file.path(output_dir, "r_dispersions.csv"),
    row.names = FALSE
  )
  writeLines(
    c(
      "{",
      '  "status": "pass",',
      '  "scope": "normalization_and_dispersion_only",',
      sprintf('  "git_commit": "%s",', git_commit),
      sprintf(
        '  "working_tree_dirty_at_start": %s,',
        ifelse(git_dirty, "true", "false")
      ),
      sprintf('  "r_version": "%s",', as.character(getRversion())),
      sprintf(
        '  "deseq2_version": "%s",',
        as.character(packageVersion("DESeq2"))
      ),
      sprintf('  "normalization_ms": %.6f,', normalization_ms),
      sprintf('  "dispersion_ms": %.6f,', dispersion_ms),
      sprintf(
        '  "elapsed_s": %.6f',
        proc.time()[["elapsed"]] - stage_start
      ),
      "}"
    ),
    file.path(output_dir, "r_dispersion_stage.json")
  )
  cat("dispersion-only reference complete\n")
  quit(save = "no", status = 0)
}

run_workflow <- function(timed = FALSE) {
  values <- list()
  measure <- function(name, expression) {
    start <- proc.time()[["elapsed"]]
    value <- force(expression)
    if (timed) values[[name]] <<- (proc.time()[["elapsed"]] - start) * 1000
    value
  }
  dds <- mk()
  dds <- measure("normalization", estimateSizeFactors(dds))
  dds <- measure("dispersion", estimateDispersions(dds, quiet = TRUE))
  stage_dispersion <- dispersions(dds)
  dds <- measure("glm_fit", wald_and_outlier_refit(dds))
  coefficient <- resultsNames(dds)[[2]]
  result <- measure("significance", results(dds, name = coefficient))
  shrunk <- measure(
    "lfc_shrink",
    lfcShrink(dds, coef = coefficient, type = "apeglm", quiet = TRUE)
  )
  list(
    timings = values,
    stage_dispersion = stage_dispersion,
    dds = dds,
    result = result,
    shrunk = shrunk,
    coefficient = coefficient
  )
}

warmups <- as.integer(Sys.getenv("GTEX_R_WARMUPS", unset = "1"))
reps <- as.integer(Sys.getenv("GTEX_R_REPS", unset = "5"))
direct_reps <- as.integer(Sys.getenv("GTEX_R_DIRECT_REPS", unset = "5"))
if (warmups < 0 || reps < 1 || direct_reps < 0) {
  stop("require warmups >= 0, reps >= 1, and direct_reps >= 0")
}
benchmark_start <- proc.time()[["elapsed"]]
for (i in seq_len(warmups)) {
  invisible(run_workflow(FALSE))
  gc()
}

stage_values <- setNames(
  lapply(c("normalization", "dispersion", "glm_fit", "significance", "lfc_shrink"), function(x) numeric()),
  c("normalization", "dispersion", "glm_fit", "significance", "lfc_shrink")
)
last <- NULL
for (i in seq_len(reps)) {
  last <- run_workflow(TRUE)
  for (stage in names(stage_values)) {
    stage_values[[stage]] <- c(stage_values[[stage]], last$timings[[stage]])
  }
  gc()
}

direct_values <- numeric(direct_reps)
for (i in seq_len(direct_reps)) {
  start <- proc.time()[["elapsed"]]
  invisible(run_workflow(FALSE))
  direct_values[[i]] <- (proc.time()[["elapsed"]] - start) * 1000
  gc()
}

# Persist one complete reference result for parity without retaining another
# full DESeqDataSet in memory.
write.csv(
  data.frame(sample = colnames(last$dds), sizeFactor = sizeFactors(last$dds)),
  file.path(output_dir, "r_sizefactors.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(gene = rownames(last$dds), dispersion = last$stage_dispersion),
  file.path(output_dir, "r_dispersions.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(gene = rownames(last$dds), dispersion = dispersions(last$dds)),
  file.path(output_dir, "r_refit_dispersions.csv"),
  row.names = FALSE
)
result_frame <- as.data.frame(last$result)
result_frame$gene <- rownames(last$result)
write.csv(
  result_frame[, c("gene", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj")],
  file.path(output_dir, "r_results.csv"),
  row.names = FALSE
)
shrink_frame <- as.data.frame(last$shrunk)
shrink_frame$gene <- rownames(last$shrunk)
write.csv(
  shrink_frame[, c("gene", "log2FoldChange", "lfcSE")],
  file.path(output_dir, "r_shrink.csv"),
  row.names = FALSE
)

sha256 <- function(path) {
  strsplit(system2("sha256sum", path, stdout = TRUE), " ")[[1]][[1]]
}
medians <- vapply(stage_values, median, numeric(1))
json_numbers <- function(values) paste(sprintf("%.6f", values), collapse = ",")
json_stage_values <- paste(
  sprintf('"%s":[%s]', names(stage_values), vapply(stage_values, json_numbers, character(1))),
  collapse = ","
)
json_stage_medians <- paste(
  sprintf('"%s":%.6f', names(medians), medians),
  collapse = ","
)
direct_median_json <- if (length(direct_values)) {
  sprintf("%.6f", median(direct_values))
} else {
  "null"
}
output <- c(
  "{",
  sprintf('  "status": "pass",'),
  sprintf('  "n_samples": %d,', ncol(counts)),
  sprintf('  "P": %d,', length(tissue_levels)),
  sprintf('  "coefficient": "%s",', last$coefficient),
  sprintf('  "warmups": %d,', warmups),
  sprintf('  "reps": %d,', reps),
  sprintf('  "direct_reps": %d,', direct_reps),
  sprintf('  "git_commit": "%s",', git_commit),
  sprintf('  "working_tree_dirty_at_start": %s,', ifelse(git_dirty, "true", "false")),
  sprintf('  "r_version": "%s",', as.character(getRversion())),
  sprintf('  "deseq2_version": "%s",', as.character(packageVersion("DESeq2"))),
  sprintf('  "apeglm_version": "%s",', as.character(packageVersion("apeglm"))),
  sprintf('  "gpu_case_sha256": "%s",', sha256(gpu_json_path)),
  sprintf('  "stage_values_ms": {%s},', json_stage_values),
  sprintf('  "stage_medians_ms": {%s},', json_stage_medians),
  sprintf('  "stage_total_ms": %.6f,', sum(medians)),
  sprintf('  "direct_values_ms": [%s],', json_numbers(direct_values)),
  sprintf('  "direct_median_ms": %s,', direct_median_json),
  sprintf('  "elapsed_s": %.6f', proc.time()[["elapsed"]] - benchmark_start),
  "}"
)
writeLines(output, file.path(output_dir, "r_timings.json"))
cat(paste(output, collapse = "\n"), "\n")
