#!/usr/bin/env Rscript

# Direct standard-DESeq2 timing for a real-GTEx cohort emitted by
# bench/gtex_scaling.py. Run each worker setting in a fresh process so the
# one-worker and MulticoreParam measurements use the same public call path and
# do not inherit package, allocator, or worker state from one another.

custom_lib <- Sys.getenv("R_DESEQ2_LIB", unset = "")
if (nzchar(custom_lib)) .libPaths(c(custom_lib, .libPaths()))
suppressMessages({
  library(DESeq2)
  library(apeglm)
  library(BiocParallel)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop(
    "usage: run_gtex_scaling_r_direct.R SOURCE_RDATA MATRIX_DIR ",
    "GPU_CASE_JSON OUTPUT_DIR"
  )
}
source_rdata <- normalizePath(args[[1]])
matrix_dir <- normalizePath(args[[2]])
gpu_json_path <- normalizePath(args[[3]])
output_dir <- args[[4]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

workers <- as.integer(Sys.getenv("GTEX_R_WORKERS", unset = "1"))
warmups <- as.integer(Sys.getenv("GTEX_R_WARMUPS", unset = "0"))
reps <- as.integer(Sys.getenv("GTEX_R_REPS", unset = "1"))
if (is.na(workers) || workers < 1) stop("GTEX_R_WORKERS must be at least 1")
if (is.na(warmups) || warmups < 0) stop("GTEX_R_WARMUPS must be non-negative")
if (is.na(reps) || reps < 1) stop("GTEX_R_REPS must be at least 1")

parallel_run <- workers > 1
bp <- if (parallel_run) {
  MulticoreParam(workers = workers, progressbar = FALSE)
} else {
  SerialParam()
}
register(bp)
on.exit(register(SerialParam()), add = TRUE)

command_output <- function(command, args) {
  tryCatch(
    suppressWarnings(system2(command, args, stdout = TRUE, stderr = FALSE)),
    error = function(error) character()
  )
}
sha256 <- function(path) {
  output <- command_output("sha256sum", path)
  if (!length(output)) return(NA_character_)
  strsplit(output[[1]], " ", fixed = TRUE)[[1]][[1]]
}
extract_numeric_array <- function(text, key) {
  pattern <- sprintf('"%s"[[:space:]]*:[[:space:]]*\\[([^]]*)\\]', key)
  matched <- regmatches(text, regexpr(pattern, text, perl = TRUE))
  if (!nzchar(matched)) stop("missing array in GPU JSON: ", key)
  body <- sub('^[^[]*\\[', "", matched)
  body <- sub('\\]$', "", body)
  as.integer(trimws(strsplit(body, ",", fixed = TRUE)[[1]]))
}

git_commit <- command_output("git", c("rev-parse", "HEAD"))
git_status <- command_output("git", c("status", "--porcelain"))
cpu_lines <- if (file.exists("/proc/cpuinfo")) readLines("/proc/cpuinfo") else character()
cpu_match <- grep("^model name", cpu_lines, value = TRUE)
cpu_model <- if (length(cpu_match)) {
  trimws(sub("^[^:]+:", "", cpu_match[[1]]))
} else {
  NA_character_
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

run_workflow <- function() {
  dds <- mk()
  dds <- DESeq(
    dds,
    test = "Wald",
    quiet = TRUE,
    parallel = parallel_run,
    BPPARAM = bp
  )
  coefficient <- resultsNames(dds)[[2]]
  result <- results(
    dds,
    name = coefficient,
    parallel = parallel_run,
    BPPARAM = bp
  )
  shrunk <- lfcShrink(
    dds,
    coef = coefficient,
    type = "apeglm",
    quiet = TRUE,
    parallel = parallel_run,
    BPPARAM = bp
  )
  list(dds = dds, result = result, shrunk = shrunk, coefficient = coefficient)
}

for (i in seq_len(warmups)) {
  invisible(run_workflow())
  gc()
}

measured_utc_start <- format(Sys.time(), tz = "UTC", usetz = TRUE)
direct_values_ms <- numeric(reps)
last <- NULL
for (i in seq_len(reps)) {
  start <- proc.time()[["elapsed"]]
  last <- run_workflow()
  direct_values_ms[[i]] <- (proc.time()[["elapsed"]] - start) * 1000
  if (i < reps) gc()
}
measured_utc_end <- format(Sys.time(), tz = "UTC", usetz = TRUE)

write.csv(
  data.frame(sample = colnames(last$dds), sizeFactor = sizeFactors(last$dds)),
  file.path(output_dir, "r_sizefactors.csv"),
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

json_string <- function(value) {
  if (is.na(value)) return("null")
  escaped <- gsub("\\\\", "\\\\\\\\", value)
  escaped <- gsub('"', '\\\\"', escaped, fixed = TRUE)
  paste0('"', escaped, '"')
}
json_numbers <- function(values) paste(sprintf("%.6f", values), collapse = ",")
result_files <- c(
  "r_sizefactors.csv",
  "r_refit_dispersions.csv",
  "r_results.csv",
  "r_shrink.csv"
)
result_hashes <- paste(
  sprintf(
    '"%s":"%s"',
    result_files,
    vapply(file.path(output_dir, result_files), sha256, character(1))
  ),
  collapse = ","
)
backend <- if (parallel_run) "BiocParallel::MulticoreParam" else "BiocParallel::SerialParam"
output <- c(
  "{",
  '  "status": "pass",',
  sprintf('  "n_samples": %d,', ncol(counts)),
  sprintf('  "P": %d,', length(tissue_levels)),
  sprintf('  "coefficient": %s,', json_string(last$coefficient)),
  sprintf('  "workers": %d,', workers),
  sprintf('  "backend": %s,', json_string(backend)),
  sprintf('  "parallel": %s,', ifelse(parallel_run, "true", "false")),
  sprintf('  "warmups": %d,', warmups),
  sprintf('  "reps": %d,', reps),
  sprintf('  "direct_values_ms": [%s],', json_numbers(direct_values_ms)),
  sprintf('  "direct_median_ms": %.6f,', median(direct_values_ms)),
  '  "total_definition": "direct DESeqDataSet construction + DESeq() + results() + lfcShrink(type=apeglm)",',
  sprintf('  "git_commit": %s,', json_string(if (length(git_commit)) git_commit[[1]] else NA_character_)),
  sprintf('  "working_tree_dirty_at_start": %s,', ifelse(length(git_status) > 0, "true", "false")),
  sprintf('  "r_version": %s,', json_string(as.character(getRversion()))),
  sprintf('  "deseq2_version": %s,', json_string(as.character(packageVersion("DESeq2")))),
  sprintf('  "apeglm_version": %s,', json_string(as.character(packageVersion("apeglm")))),
  sprintf('  "blas_threads": %s,', json_string(Sys.getenv("OPENBLAS_NUM_THREADS", unset = "unset"))),
  sprintf('  "omp_threads": %s,', json_string(Sys.getenv("OMP_NUM_THREADS", unset = "unset"))),
  sprintf('  "cpu_model": %s,', json_string(cpu_model)),
  sprintf('  "logical_cpu_count": %d,', parallel::detectCores(logical = TRUE)),
  sprintf('  "source_rdata_sha256": "%s",', sha256(source_rdata)),
  sprintf('  "matrix_metadata_sha256": "%s",', sha256(file.path(matrix_dir, "matrix_metadata.json"))),
  sprintf('  "gpu_case_sha256": "%s",', sha256(gpu_json_path)),
  sprintf('  "result_file_sha256": {%s},', result_hashes),
  sprintf('  "measured_utc_start": %s,', json_string(measured_utc_start)),
  sprintf('  "measured_utc_end": %s', json_string(measured_utc_end)),
  "}"
)
writeLines(output, file.path(output_dir, "r_direct_timings.json"))
cat(paste(output, collapse = "\n"), "\n")
