#!/usr/bin/env Rscript
# Real-data validation, step 1 (R side): export standard Bioconductor RNA-seq
# datasets across a range of designs (P=2..5, single- and multi-factor, several
# sample sizes) and compute the R DESeq2 1.30.1 REFERENCE output that cuDESeq2 is
# validated against. Writes, per case, to validation/data/<name>/:
#   counts.csv, coldata.csv, meta.json,
#   r_results.csv, r_dispersions.csv, r_shrink.csv
#
# Reference and contrast levels are set EXPLICITLY (ref + nonref) so the Python
# side reproduces the exact same coefficient (no sign flip, unambiguous for
# multi-level factors).
#
# Usage: Rscript validation/fetch_and_reference.R
.libPaths(c(Sys.getenv("R_DESEQ2_LIB", unset = "~/R/library"), .libPaths()))
suppressMessages({library(DESeq2); library(apeglm)})

OUT <- "validation/data"; dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

write_case <- function(name, counts, coldata, design, factor, ref, nonref) {
  d <- file.path(OUT, name); dir.create(d, showWarnings = FALSE)
  counts <- round(as.matrix(counts)); storage.mode(counts) <- "integer"
  counts <- counts[rowSums(counts) > 0, , drop = FALSE]
  coldata[[factor]] <- relevel(factor(coldata[[factor]]), ref = ref)
  coef <- sprintf("%s_%s_vs_%s", factor, nonref, ref)
  dds <- DESeqDataSetFromMatrix(counts, coldata, as.formula(design))
  dds <- DESeq(dds, quiet = TRUE)
  stopifnot(coef %in% resultsNames(dds))
  res <- as.data.frame(results(dds, name = coef)); res$gene <- rownames(res)
  write.csv(counts, file.path(d, "counts.csv"))
  write.csv(coldata, file.path(d, "coldata.csv"))
  write.csv(res[, c("gene","baseMean","log2FoldChange","lfcSE","stat","pvalue","padj")],
            file.path(d, "r_results.csv"), row.names = FALSE)
  write.csv(data.frame(gene = rownames(dds), dispersion = dispersions(dds)),
            file.path(d, "r_dispersions.csv"), row.names = FALSE)
  sh <- as.data.frame(lfcShrink(dds, coef = coef, type = "apeglm", quiet = TRUE)); sh$gene <- rownames(sh)
  write.csv(sh[, c("gene","log2FoldChange","lfcSE")], file.path(d, "r_shrink.csv"), row.names = FALSE)
  # Time R's full pipeline (size factors -> dispersions -> Wald -> results ->
  # apeGLM shrink), single-threaded, median of 3.
  run_full <- function() {
    dd <- DESeq(DESeqDataSetFromMatrix(counts, coldata, as.formula(design)), quiet = TRUE)
    invisible(results(dd, name = coef)); invisible(lfcShrink(dd, coef = coef, type = "apeglm", quiet = TRUE))
  }
  run_full()
  ts <- numeric(3); for (i in 1:3) { t0 <- Sys.time(); run_full(); ts[i] <- as.numeric(Sys.time() - t0, units = "secs") }
  r_full_ms <- median(ts) * 1000
  P <- ncol(model.matrix(as.formula(design), coldata))
  writeLines(sprintf('{"name":"%s","design":"%s","factor":"%s","ref":"%s","nonref":"%s","coef":"%s","P":%d,"n_samples":%d,"n_genes":%d,"r_full_ms":%.1f}',
    name, design, factor, ref, nonref, coef, P, ncol(counts), nrow(counts), r_full_ms), file.path(d, "meta.json"))
  cat(sprintf("  %-16s %2d samp x %5d genes  P=%d  design=%-18s R full=%.0f ms\n",
      name, ncol(counts), nrow(counts), P, design, r_full_ms))
}

cat("Preparing datasets + R DESeq2 reference:\n")

## --- pasilla (Drosophila): P=2 and P=3 designs ---
if (requireNamespace("pasilla", quietly = TRUE)) {
  pf <- system.file("extdata","pasilla_gene_counts.tsv", package="pasilla")
  af <- system.file("extdata","pasilla_sample_annotation.csv", package="pasilla")
  cts <- as.matrix(read.csv(pf, sep="\t", row.names="gene_id"))
  anno <- read.csv(af, row.names=1); rownames(anno) <- sub("fb$","",rownames(anno))
  anno <- anno[colnames(cts),,drop=FALSE]
  cd <- data.frame(condition=anno$condition, type=anno$type, row.names=colnames(cts))
  write_case("pasilla",      cts, cd, "~ condition",        "condition","untreated","treated")
  write_case("pasilla_2fac", cts, cd, "~ type + condition", "condition","untreated","treated")
} else cat("  pasilla NOT INSTALLED\n")

## --- airway (human): P=2 (~dex), P=4 (~cell), P=5 (~cell+dex) ---
if (requireNamespace("airway", quietly = TRUE)) {
  suppressMessages(library(airway)); data("airway", package="airway")
  cts <- assay(airway); cd <- data.frame(cell=airway$cell, dex=airway$dex, row.names=colnames(cts))
  write_case("airway_dex",  cts, cd, "~ dex",        "dex", "untrt","trt")
  write_case("airway_cell", cts, cd, "~ cell",       "cell","N052611","N061011")
  write_case("airway",      cts, cd, "~ cell + dex", "dex", "untrt","trt")
} else cat("  airway NOT INSTALLED\n")

## --- macrophage (human, ~24 samples, condition 4 levels): P=4 ---
if (requireNamespace("macrophage", quietly = TRUE)) {
  suppressMessages(library(macrophage)); data("gse", package="macrophage")
  cd <- as.data.frame(colData(gse))
  fac <- if ("condition_name" %in% colnames(cd)) "condition_name" else "condition"
  cd[[fac]] <- factor(cd[[fac]])
  lv <- levels(cd[[fac]])
  ref <- if ("naive" %in% lv) "naive" else lv[1]
  nonref <- setdiff(lv, ref)[1]
  write_case("macrophage", assay(gse), data.frame(cond=cd[[fac]], row.names=rownames(cd)),
             "~ cond", "cond", ref, nonref)
} else cat("  macrophage NOT INSTALLED (Bioc 3.12 data mirror 504)\n")

cat("done ->", OUT, "\n")
