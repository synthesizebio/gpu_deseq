#!/usr/bin/env Rscript

# Materialize the checksum-pinned recount2 GTEx source as a reusable
# column-major int32 matrix. Cohort selection is performed by
# bench/gtex_scaling.py so every timed run uses the same source matrix.

suppressPackageStartupMessages(library(SummarizedExperiment))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop(
    "usage: extract_gtex_scaling.R SOURCE_RDATA ",
    "REFERENCE_COUNTS_CSV OUTPUT_DIR"
  )
}
source_rdata <- normalizePath(args[[1]])
reference_counts <- normalizePath(args[[2]])
output_dir <- args[[3]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

environment <- new.env(parent = emptyenv())
load(source_rdata, envir = environment)
rse <- environment$rse_gene

# Hold the gene axis fixed at the 54,922 genes in the paper's original GTEx
# case. This makes the experiment isolate sample count and design width.
reference <- read.csv(reference_counts, row.names = 1, check.names = FALSE)
gene_ids <- rownames(reference)
gene_index <- match(gene_ids, rownames(rse))
stopifnot(!anyNA(gene_index), length(gene_index) == 54922)
rm(reference)
gc()

sample_data <- as.data.frame(colData(rse))
tissue <- as.character(sample_data$smtsd)
read_length <- as.numeric(sample_data$avg_read_length)
read_length[is.na(read_length) | read_length <= 0] <- 100

samples <- data.frame(
  source_column = seq_len(ncol(rse)),
  sample_id = colnames(rse),
  tissue = tissue,
  read_length = read_length,
  stringsAsFactors = FALSE
)
write.csv(samples, file.path(output_dir, "samples.csv"), row.names = FALSE)
writeLines(gene_ids, file.path(output_dir, "genes.txt"))

tissue_counts <- sort(table(tissue), decreasing = TRUE)
inventory <- data.frame(
  rank = seq_along(tissue_counts),
  tissue = names(tissue_counts),
  n_samples = as.integer(tissue_counts),
  stringsAsFactors = FALSE
)
write.csv(
  inventory,
  file.path(output_dir, "tissue_inventory.csv"),
  row.names = FALSE
)

binary_path <- file.path(output_dir, "counts_int32_f.bin")
connection <- file(binary_path, open = "wb")
on.exit(close(connection), add = TRUE)
block_size <- 64L
for (start in seq.int(1L, ncol(rse), by = block_size)) {
  stop_at <- min(ncol(rse), start + block_size - 1L)
  columns <- start:stop_at
  block <- assay(rse)[gene_index, columns, drop = FALSE]
  block <- round(sweep(block, 2, read_length[columns], "/"))
  if (any(block < 0) || any(block > .Machine$integer.max)) {
    stop("converted count outside int32 range")
  }
  writeBin(as.integer(block), connection, size = 4L, endian = "little")
  if (start == 1L || stop_at == ncol(rse) || (start - 1L) %% 640L == 0L) {
    cat(sprintf("columns %d..%d / %d\n", start, stop_at, ncol(rse)))
  }
}
close(connection)
on.exit(NULL, add = FALSE)

expected_bytes <- length(gene_ids) * ncol(rse) * 4
actual_bytes <- file.info(binary_path)$size
stopifnot(actual_bytes == expected_bytes)

sha256 <- function(path) {
  strsplit(system2("sha256sum", path, stdout = TRUE), " ")[[1]][[1]]
}
metadata <- c(
  "{",
  sprintf('  "source_sha256": "%s",', sha256(source_rdata)),
  sprintf('  "reference_counts_sha256": "%s",', sha256(args[[2]])),
  sprintf('  "n_genes": %d,', length(gene_ids)),
  sprintf('  "n_samples": %d,', ncol(rse)),
  sprintf('  "n_tissues": %d,', length(tissue_counts)),
  '  "dtype": "int32",',
  '  "order": "F",',
  sprintf('  "binary_bytes": %d,', actual_bytes),
  sprintf('  "binary_sha256": "%s"', sha256(binary_path)),
  "}"
)
writeLines(metadata, file.path(output_dir, "matrix_metadata.json"))
cat("wrote", binary_path, "\n")
