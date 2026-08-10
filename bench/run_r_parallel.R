#!/usr/bin/env Rscript
# Measure the complete standard DESeq2 pipeline at one and N workers without
# touching the reference cache or GPU timing records. The two R configurations
# use the same call path: DESeq() followed by results() and apeGLM shrinkage.

custom_lib <- Sys.getenv("R_DESEQ2_LIB", unset = "")
if (nzchar(custom_lib)) .libPaths(c(custom_lib, .libPaths()))
suppressMessages({library(DESeq2); library(apeglm); library(BiocParallel)})

DATA <- Sys.getenv("BENCH_DATA", unset = "validation/data")
OUT <- Sys.getenv("BENCH_PARALLEL_OUTPUT",
                  unset = "bench/results/r_parallel_a100_12worker.json")
WORKERS <- as.integer(Sys.getenv("BENCH_R_WORKERS", unset = "12"))
if (is.na(WORKERS) || WORKERS < 2) stop("BENCH_R_WORKERS must be at least 2")

args <- commandArgs(trailingOnly = TRUE)
cases <- if (length(args)) args else list.dirs(DATA, recursive = FALSE, full.names = FALSE)
cases <- sort(Filter(function(case) file.exists(file.path(DATA, case, "meta.json")), cases))
if (!length(cases)) stop("no benchmark cases found")

read_meta <- function(path) {
  txt <- paste(readLines(path), collapse = " ")
  get <- function(key) {
    hit <- regmatches(txt, regexpr(sprintf('"%s"\\s*:\\s*"?([^",}]+)', key), txt))
    sub(sprintf('"%s"\\s*:\\s*"?', key), "", hit)
  }
  list(design = get("design"), factor = get("factor"), ref = get("ref"),
       coef = get("coef"), n_samples = as.integer(get("n_samples")))
}

timed <- function(f, reps) {
  invisible(f()) # warm-up package loading and method dispatch for this mode
  elapsed <- numeric(reps)
  for (i in seq_len(reps)) {
    start <- proc.time()[["elapsed"]]
    invisible(f())
    elapsed[i] <- proc.time()[["elapsed"]] - start
  }
  list(median_ms = median(elapsed) * 1000, values_ms = elapsed * 1000)
}

run_pipeline <- function(make_dds, coef, parallel, bpparam) {
  dds <- make_dds()
  dds <- DESeq(dds, test = "Wald", quiet = TRUE,
               parallel = parallel, BPPARAM = bpparam)
  invisible(results(dds, name = coef, parallel = parallel, BPPARAM = bpparam))
  invisible(lfcShrink(dds, coef = coef, type = "apeglm", quiet = TRUE,
                      parallel = parallel, BPPARAM = bpparam))
}

serial_bp <- SerialParam()
parallel_bp <- MulticoreParam(workers = WORKERS, progressbar = FALSE)
on.exit(bpparam() <- SerialParam(), add = TRUE)

records <- list()
for (case in cases) {
  path <- file.path(DATA, case)
  meta <- read_meta(file.path(path, "meta.json"))
  counts <- as.matrix(read.csv(file.path(path, "counts.csv"), row.names = 1))
  storage.mode(counts) <- "integer"
  coldata <- read.csv(file.path(path, "coldata.csv"), row.names = 1)
  coldata[[meta$factor]] <- relevel(factor(coldata[[meta$factor]]), ref = meta$ref)
  make_dds <- function() DESeqDataSetFromMatrix(counts, coldata, as.formula(meta$design))
  reps <- if (meta$n_samples > 100) 1L else 3L
  serial <- timed(function() run_pipeline(make_dds, meta$coef, FALSE, serial_bp), reps)
  parallel <- timed(function() run_pipeline(make_dds, meta$coef, TRUE, parallel_bp), reps)
  records[[case]] <- list(n_samples = meta$n_samples, design = meta$design, reps = reps,
                          serial = serial, parallel = parallel)
  cat(sprintf("  %-20s serial=%8.0f ms  %2d workers=%8.0f ms\n",
              case, serial$median_ms, WORKERS, parallel$median_ms))
}

output <- list(
  provenance = list(
    pipeline = "DESeq() + results() + lfcShrink(type=apeglm), standard outlier replacement/refit enabled",
    r_version = R.version.string,
    deseq2_version = as.character(packageVersion("DESeq2")),
    apeglm_version = as.character(packageVersion("apeglm")),
    workers = WORKERS,
    backend = "BiocParallel::MulticoreParam",
    blas_threads = Sys.getenv("OPENBLAS_NUM_THREADS", unset = "unset"),
    measured_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
  ),
  cases = records
)
dir.create(dirname(OUT), recursive = TRUE, showWarnings = FALSE)
json_string <- function(x) {
  escaped <- gsub("\\\\", "\\\\\\\\", x)
  escaped <- gsub('"', '\\\\"', escaped, fixed = TRUE)
  paste0('"', escaped, '"')
}
json_array <- function(x) paste0("[", paste(sprintf("%.3f", x), collapse = ", "), "]")
case_json <- vapply(names(records), function(case) {
  record <- records[[case]]
  paste0("    ", json_string(case), ": {\n",
         "      \"n_samples\": ", record$n_samples, ",\n",
         "      \"design\": ", json_string(record$design), ",\n",
         "      \"reps\": ", record$reps, ",\n",
         "      \"serial\": {\"median_ms\": ", sprintf("%.3f", record$serial$median_ms),
         ", \"values_ms\": ", json_array(record$serial$values_ms), "},\n",
         "      \"parallel\": {\"median_ms\": ", sprintf("%.3f", record$parallel$median_ms),
         ", \"values_ms\": ", json_array(record$parallel$values_ms), "}\n",
         "    }")
}, character(1))
json <- c(
  "{",
  "  \"provenance\": {",
  paste0("    \"pipeline\": ", json_string(output$provenance$pipeline), ","),
  paste0("    \"r_version\": ", json_string(output$provenance$r_version), ","),
  paste0("    \"deseq2_version\": ", json_string(output$provenance$deseq2_version), ","),
  paste0("    \"apeglm_version\": ", json_string(output$provenance$apeglm_version), ","),
  paste0("    \"workers\": ", output$provenance$workers, ","),
  paste0("    \"backend\": ", json_string(output$provenance$backend), ","),
  paste0("    \"blas_threads\": ", json_string(output$provenance$blas_threads), ","),
  paste0("    \"measured_utc\": ", json_string(output$provenance$measured_utc)),
  "  },",
  "  \"cases\": {",
  paste(case_json, collapse = ",\n"),
  "  }",
  "}"
)
writeLines(json, OUT)
cat("wrote", OUT, "\n")
