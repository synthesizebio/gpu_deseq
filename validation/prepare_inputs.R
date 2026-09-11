#!/usr/bin/env Rscript
# Deterministically derive all six benchmark inputs from data_sources.json.
# Source archives must first be downloaded and checksum-verified with
# scripts/fetch_validation_data.py.

suppressMessages({
  library(SummarizedExperiment)
})

SOURCE_DIR <- Sys.getenv("VALIDATION_SOURCE_DIR", unset = "validation/sources")
OUT <- Sys.getenv("VALIDATION_DATA_DIR", unset = "validation/data")
GTEX_N_PER_GROUP <- as.integer(Sys.getenv("GTEX_N_PER_GROUP", unset = "150"))
GTEX_SAMPLE_SEED <- as.integer(Sys.getenv("GTEX_SAMPLE_SEED", unset = "1"))

required <- c(
  airway = file.path(SOURCE_DIR, "airway_1.32.0.tar.gz"),
  pasilla = file.path(SOURCE_DIR, "pasilla_1.40.0.tar.gz"),
  gtex = file.path(SOURCE_DIR, "SRP012682_rse_gene.Rdata")
)
missing <- required[!file.exists(required)]
if (length(missing)) {
  stop(
    "missing checksum-pinned source file(s): ",
    paste(missing, collapse = ", "),
    "; run python3 scripts/fetch_validation_data.py first"
  )
}
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

write_json_string <- function(value) {
  value <- gsub("\\\\", "\\\\\\\\", value)
  value <- gsub('"', '\\\\"', value)
  sprintf('"%s"', value)
}

write_case <- function(name, counts, coldata, design, factor_name, ref, nonref,
                       source_id, extra_json = NULL) {
  counts <- round(as.matrix(counts))
  storage.mode(counts) <- "integer"
  counts <- counts[rowSums(counts) > 0, , drop = FALSE]
  coldata <- as.data.frame(coldata)
  coldata[[factor_name]] <- relevel(factor(coldata[[factor_name]]), ref = ref)
  stopifnot(identical(colnames(counts), rownames(coldata)))

  path <- file.path(OUT, name)
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  write.csv(counts, file.path(path, "counts.csv"))
  write.csv(coldata, file.path(path, "coldata.csv"))

  coefficient <- sprintf("%s_%s_vs_%s", factor_name, nonref, ref)
  P <- ncol(model.matrix(as.formula(design), coldata))
  fields <- c(
    sprintf('"name":%s', write_json_string(name)),
    sprintf('"design":%s', write_json_string(design)),
    sprintf('"factor":%s', write_json_string(factor_name)),
    sprintf('"ref":%s', write_json_string(ref)),
    sprintf('"nonref":%s', write_json_string(nonref)),
    sprintf('"coef":%s', write_json_string(coefficient)),
    sprintf('"P":%d', P),
    sprintf('"n_samples":%d', ncol(counts)),
    sprintf('"n_genes":%d', nrow(counts)),
    sprintf('"source_id":%s', write_json_string(source_id)),
    sprintf('"preparation_r_version":%s', write_json_string(as.character(getRversion())))
  )
  if (!is.null(extra_json)) fields <- c(fields, extra_json)
  writeLines(sprintf("{%s}", paste(fields, collapse = ",")), file.path(path, "meta.json"))
  cat(sprintf("  %-20s %d samples x %d genes; P=%d\n", name, ncol(counts), nrow(counts), P))
}

extract_root <- tempfile("gpu_deseq_sources_")
dir.create(extract_root)
on.exit(unlink(extract_root, recursive = TRUE), add = TRUE)

cat("Preparing pasilla inputs\n")
pasilla_root <- file.path(extract_root, "pasilla")
dir.create(pasilla_root)
untar(required[["pasilla"]], exdir = pasilla_root)
pasilla_ext <- file.path(pasilla_root, "pasilla", "inst", "extdata")
pasilla_counts <- as.matrix(read.csv(
  file.path(pasilla_ext, "pasilla_gene_counts.tsv"),
  sep = "\t", row.names = "gene_id"
))
pasilla_annotation <- read.csv(
  file.path(pasilla_ext, "pasilla_sample_annotation.csv"), row.names = 1
)
rownames(pasilla_annotation) <- sub("fb$", "", rownames(pasilla_annotation))
pasilla_annotation <- pasilla_annotation[colnames(pasilla_counts), , drop = FALSE]
pasilla_coldata <- data.frame(
  condition = pasilla_annotation$condition,
  type = pasilla_annotation$type,
  row.names = colnames(pasilla_counts)
)
write_case("pasilla", pasilla_counts, pasilla_coldata, "~ condition",
           "condition", "untreated", "treated", "pasilla")
write_case("pasilla_2fac", pasilla_counts, pasilla_coldata, "~ type + condition",
           "condition", "untreated", "treated", "pasilla")

cat("Preparing airway inputs\n")
airway_root <- file.path(extract_root, "airway")
dir.create(airway_root)
untar(required[["airway"]], exdir = airway_root)
airway_environment <- new.env(parent = emptyenv())
load(file.path(airway_root, "airway", "data", "airway.RData"),
     envir = airway_environment)
airway <- airway_environment$airway
airway_counts <- assay(airway)
airway_coldata <- data.frame(
  cell = colData(airway)$cell,
  dex = colData(airway)$dex,
  row.names = colnames(airway_counts)
)
write_case("airway_dex", airway_counts, airway_coldata, "~ dex",
           "dex", "untrt", "trt", "airway")
write_case("airway_cell", airway_counts, airway_coldata, "~ cell",
           "cell", "N052611", "N061011", "airway")
write_case("airway", airway_counts, airway_coldata, "~ cell + dex",
           "dex", "untrt", "trt", "airway")

cat("Preparing GTEx/recount2 inputs\n")
gtex_environment <- new.env(parent = emptyenv())
load(required[["gtex"]], envir = gtex_environment)
rse_gene <- gtex_environment$rse_gene
sample_data <- colData(rse_gene)
tissue_detail <- as.character(sample_data$smtsd)
blood <- which(tissue_detail == "Whole Blood")
muscle <- which(tissue_detail == "Muscle - Skeletal")
set.seed(GTEX_SAMPLE_SEED)
blood <- sort(sample(blood, min(GTEX_N_PER_GROUP, length(blood))))
muscle <- sort(sample(muscle, min(GTEX_N_PER_GROUP, length(muscle))))
selected <- c(blood, muscle)
rse <- rse_gene[, selected]
read_length <- colData(rse)$avg_read_length
read_length[is.na(read_length) | read_length <= 0] <- 100
gtex_counts <- round(sweep(assay(rse), 2, read_length, "/"))
storage.mode(gtex_counts) <- "integer"
gtex_tissue <- factor(
  ifelse(as.character(colData(rse)$smtsd) == "Whole Blood", "blood", "muscle"),
  levels = c("blood", "muscle")
)
gtex_sample_ids <- paste0("s", seq_len(ncol(gtex_counts)))
colnames(gtex_counts) <- gtex_sample_ids
gtex_coldata <- data.frame(tissue = gtex_tissue, row.names = gtex_sample_ids)
gtex_extra <- c(
  sprintf('"sample_seed":%d', GTEX_SAMPLE_SEED),
  sprintf('"samples_per_tissue":%d', GTEX_N_PER_GROUP),
  sprintf('"selected_source_columns":[%s]', paste(selected, collapse = ",")),
  sprintf('"selected_source_samples":[%s]', paste(
    vapply(colnames(rse), write_json_string, character(1)), collapse = ","
  ))
)
write_case("gtex_blood_muscle", gtex_counts, gtex_coldata, "~ tissue",
           "tissue", "blood", "muscle", "gtex_recount2_srp012682", gtex_extra)

expected_shapes <- list(
  pasilla = c(7, 12359), pasilla_2fac = c(7, 12359),
  airway_dex = c(8, 33469), airway_cell = c(8, 33469), airway = c(8, 33469),
  gtex_blood_muscle = c(300, 54922)
)
for (name in names(expected_shapes)) {
  meta <- readLines(file.path(OUT, name, "meta.json"), warn = FALSE)
  shape <- expected_shapes[[name]]
  stopifnot(grepl(sprintf('"n_samples":%d', shape[[1]]), meta, fixed = TRUE))
  stopifnot(grepl(sprintf('"n_genes":%d', shape[[2]]), meta, fixed = TRUE))
}
cat("All six prepared inputs match the expected dimensions in validation/data\n")
