#!/usr/bin/env Rscript
# Large-cohort real-data case: GTEx (recount2), Whole Blood vs Skeletal Muscle.
# Exports counts + coldata + R DESeq2 reference to
# validation/data/gtex_blood_muscle/ (same format as the other cases, so
# validate.py picks it up). ~tissue is P=2, so it exercises the Triton kernel.
#
# Requires the recount2 GTEx RSE downloaded to $GTEX_RSE (default below):
#   curl -sL -o /var/tmp/geuvadis/rse_gene.Rdata \
#     https://recount-opendata.s3.amazonaws.com/recount2/v2/SRP012682/rse_gene.Rdata
#
# recount2 stores base-coverage counts; we convert to read counts via
# coverage / avg_read_length (DESeq2's median-of-ratios then handles library
# size). Both engines get identical integer counts, so parity is unaffected.
custom_lib <- Sys.getenv("R_DESEQ2_LIB", unset = "")
if (nzchar(custom_lib)) .libPaths(c(custom_lib, .libPaths()))
suppressMessages({library(SummarizedExperiment); library(DESeq2); library(apeglm)})

RSE <- Sys.getenv("GTEX_RSE", unset = "/var/tmp/geuvadis/rse_gene.Rdata")
N_PER_GROUP <- as.integer(Sys.getenv("GTEX_N_PER_GROUP", unset = "150"))
OUT <- "validation/data/gtex_blood_muscle"; dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

load(RSE)  # -> rse_gene
cd <- colData(rse_gene)
tis <- as.character(cd$smtsd)
sel_blood  <- which(tis == "Whole Blood")
sel_muscle <- which(tis == "Muscle - Skeletal")
set.seed(1)
sel_blood  <- sort(sample(sel_blood,  min(N_PER_GROUP, length(sel_blood))))
sel_muscle <- sort(sample(sel_muscle, min(N_PER_GROUP, length(sel_muscle))))
sel <- c(sel_blood, sel_muscle)
rse <- rse_gene[, sel]

rl <- colData(rse)$avg_read_length; rl[is.na(rl) | rl <= 0] <- 100
counts <- round(sweep(assay(rse), 2, rl, "/")); storage.mode(counts) <- "integer"
counts <- counts[rowSums(counts) > 0, , drop = FALSE]
tissue <- factor(ifelse(as.character(colData(rse)$smtsd) == "Whole Blood", "blood", "muscle"),
                 levels = c("blood", "muscle"))
sid <- paste0("s", seq_len(ncol(counts)))
colnames(counts) <- sid
coldata <- data.frame(tissue = tissue, row.names = sid)
coef <- "tissue_muscle_vs_blood"

cat(sprintf("GTEx blood-vs-muscle: %d samples x %d genes\n", ncol(counts), nrow(counts)))
dds <- DESeqDataSetFromMatrix(counts, coldata, ~ tissue)
t0 <- Sys.time()
dds <- DESeq(dds, quiet = TRUE)
res <- results(dds, name = coef)
sh  <- lfcShrink(dds, coef = coef, type = "apeglm", quiet = TRUE)
r_full_ms <- as.numeric(Sys.time() - t0, units = "secs") * 1000
cat(sprintf("R full pipeline: %.0f ms\n", r_full_ms))

resdf <- as.data.frame(res); resdf$gene <- rownames(res)
shdf  <- as.data.frame(sh);  shdf$gene  <- rownames(sh)
write.csv(counts, file.path(OUT, "counts.csv"))
write.csv(coldata, file.path(OUT, "coldata.csv"))
write.csv(resdf[, c("gene","baseMean","log2FoldChange","lfcSE","stat","pvalue","padj")],
          file.path(OUT, "r_results.csv"), row.names = FALSE)
write.csv(data.frame(gene = rownames(dds), dispersion = dispersions(dds)),
          file.path(OUT, "r_dispersions.csv"), row.names = FALSE)
write.csv(shdf[, c("gene","log2FoldChange","lfcSE")], file.path(OUT, "r_shrink.csv"), row.names = FALSE)
writeLines(sprintf('{"name":"gtex_blood_muscle","design":"~ tissue","factor":"tissue","ref":"blood","nonref":"muscle","coef":"%s","P":2,"n_samples":%d,"n_genes":%d,"r_full_ms":%.1f}',
  coef, ncol(counts), nrow(counts), r_full_ms), file.path(OUT, "meta.json"))
cat("done ->", OUT, "\n")
