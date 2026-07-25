#!/usr/bin/env Rscript
# Real-data validation, step 1 (R side): export standard Bioconductor RNA-seq
# datasets and compute the R DESeq2 1.30.1 REFERENCE output that cuDESeq2 is
# validated against. Writes, per dataset, to validation/data/<name>/:
#   counts.csv, coldata.csv            (inputs, shared with the Python side)
#   r_results.csv                      (baseMean, log2FoldChange, lfcSE, stat, pvalue, padj)
#   r_dispersions.csv                  (final per-gene dispersion)
#   r_shrink.csv                       (apeGLM-shrunk log2FoldChange + lfcSE)
#   meta.json                          (design, coefficient, reference levels)
#
# Reference levels are set EXPLICITLY so the Python side can reproduce the exact
# same contrast (untreated/untrt as the base level), avoiding sign flips.
#
# Usage: Rscript validation/fetch_and_reference.R
.libPaths(c(Sys.getenv("R_DESEQ2_LIB", unset = "~/R/library"), .libPaths()))
suppressMessages({library(DESeq2); library(apeglm)})

OUT <- "validation/data"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

write_case <- function(name, counts, coldata, design, factor, ref, coef) {
  d <- file.path(OUT, name); dir.create(d, showWarnings = FALSE)
  coldata[[factor]] <- relevel(factor(coldata[[factor]]), ref = ref)
  # keep only genes with nonzero total count (DESeq2 does this internally anyway)
  counts <- counts[rowSums(counts) > 0, , drop = FALSE]
  write.csv(counts, file.path(d, "counts.csv"))
  write.csv(coldata, file.path(d, "coldata.csv"))

  dds <- DESeqDataSetFromMatrix(counts, coldata, as.formula(design))
  dds <- DESeq(dds, quiet = TRUE)
  res <- as.data.frame(results(dds, name = coef))
  res$gene <- rownames(res)
  write.csv(res[, c("gene","baseMean","log2FoldChange","lfcSE","stat","pvalue","padj")],
            file.path(d, "r_results.csv"), row.names = FALSE)
  disp <- data.frame(gene = rownames(dds), dispersion = dispersions(dds))
  write.csv(disp, file.path(d, "r_dispersions.csv"), row.names = FALSE)
  sh <- as.data.frame(lfcShrink(dds, coef = coef, type = "apeglm", quiet = TRUE))
  sh$gene <- rownames(sh)
  write.csv(sh[, c("gene","log2FoldChange","lfcSE")],
            file.path(d, "r_shrink.csv"), row.names = FALSE)

  meta <- sprintf('{"name":"%s","design":"%s","factor":"%s","ref":"%s","coef":"%s","n_samples":%d,"n_genes":%d}',
                  name, design, factor, ref, coef, ncol(counts), nrow(counts))
  writeLines(meta, file.path(d, "meta.json"))
  cat(sprintf("  %-10s %d samples x %d genes  design=%s  coef=%s\n",
              name, ncol(counts), nrow(counts), design, coef))
}

cat("Preparing datasets + R DESeq2 reference:\n")

## --- pasilla: single 2-level factor (condition), P=2 ---
if (requireNamespace("pasilla", quietly = TRUE)) {
  pf <- system.file("extdata", "pasilla_gene_counts.tsv", package = "pasilla")
  af <- system.file("extdata", "pasilla_sample_annotation.csv", package = "pasilla")
  cts <- as.matrix(read.csv(pf, sep = "\t", row.names = "gene_id"))
  anno <- read.csv(af, row.names = 1)
  rownames(anno) <- sub("fb$", "", rownames(anno))          # match count colnames
  anno <- anno[colnames(cts), , drop = FALSE]
  cd <- data.frame(condition = anno$condition, type = anno$type, row.names = colnames(cts))
  write_case("pasilla", cts, cd, "~ condition", "condition", "untreated",
             "condition_treated_vs_untreated")
} else cat("  pasilla NOT INSTALLED, skipping\n")

## --- airway: multi-factor (cell + dex), P=5 ---
if (requireNamespace("airway", quietly = TRUE)) {
  data("airway", package = "airway")
  cts <- assay(airway)
  cd <- data.frame(cell = airway$cell, dex = airway$dex, row.names = colnames(cts))
  write_case("airway", cts, cd, "~ cell + dex", "dex", "untrt", "dex_trt_vs_untrt")
} else cat("  airway NOT INSTALLED, skipping\n")

cat("done -> ", OUT, "\n")
