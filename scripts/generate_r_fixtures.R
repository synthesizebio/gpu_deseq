#!/usr/bin/env Rscript
# Generate R DESeq2 ground-truth fixtures, dumping every intermediate the
# Python pipeline computes so we can validate per-step parity (not just
# end-to-end).
#
# Layout under fixtures/r_deseq2/<label>/:
#   counts.csv, coldata.csv                 inputs
#   size_factors.csv                        sizeFactors(dds)
#   dispersions.csv                         per-gene dispGeneEst, dispFit, dispMAP, dispersion, dispOutlier
#   dispersion_function.csv                 fit_type + parametric trend (asymptDisp, extraPois)
#   coefficients_log2.csv                   β matrix in log2 (R's storage)
#   coefficient_se_log2.csv                 SE matrix in log2
#   mu.csv, H.csv, cooks.csv                (genes × samples) IRLS outputs
#   results.csv                             results(dds, indepFilter=FALSE, cooksCutoff=FALSE)
#   indep_filter.csv, padj_filtered.csv     independentFiltering=TRUE outputs
#   cooks_filtered.csv                      results(cooksCutoff=TRUE)$padj/pvalue (Cook's filter applied)
#   lrt_results.csv                         test="LRT", reduced=~1
#   local_dispfit.csv, mean_dispfit.csv     dispFit per gene under fitType="local" / "mean"
#   apeglm_priorinfo.csv                    apeglm prior_var (lfcShrink)

.libPaths(c("~/R/library", .libPaths()))
suppressPackageStartupMessages({
  library(DESeq2)
})

# -------------------- simulators --------------------

simulate_counts <- function(n_samples, n_genes, seed) {
  set.seed(seed)
  half <- n_samples %/% 2
  true_lfc <- c(rnorm(n_genes - 100, 0, 0.2), sample(c(-2, -1, 1, 2), 100, replace=TRUE))
  true_lfc <- sample(true_lfc)
  base_mu <- pmin(5000, pmax(5, exp(rnorm(n_genes, 5, 2))))
  sf <- exp(rnorm(n_samples, 0, 0.2))
  disp <- 0.05 + 5.0 / base_mu
  counts <- matrix(0L, nrow=n_genes, ncol=n_samples)
  for (g in seq_len(n_genes)) {
    for (s in seq_len(n_samples)) {
      group_lfc <- if (s > half) true_lfc[g] else 0
      mu_gs <- base_mu[g] * sf[s] * 2^group_lfc
      counts[g, s] <- rnbinom(1, mu=mu_gs, size=1/disp[g])
    }
  }
  rownames(counts) <- paste0("gene_", seq_len(n_genes) - 1)
  colnames(counts) <- paste0("sample_", seq_len(n_samples) - 1)
  coldata <- data.frame(
    condition = factor(c(rep("control", half), rep("treated", n_samples - half)),
                        levels = c("control", "treated")),
    row.names = colnames(counts)
  )
  list(counts=counts, coldata=coldata)
}

simulate_multifactor <- function(n_samples, n_genes, seed) {
  # n_samples must be a multiple of 6 (3 batches × 2 conditions).
  stopifnot(n_samples %% 6 == 0)
  set.seed(seed)
  per_cell <- n_samples %/% 6
  batch <- factor(rep(c("A","B","C"), each = n_samples / 3),
                  levels = c("A","B","C"))
  cond  <- factor(rep(rep(c("control","treated"), each = per_cell), 3),
                  levels = c("control","treated"))
  # batch effects (per gene) + condition effect
  true_lfc_cond <- c(rnorm(n_genes - 100, 0, 0.2), sample(c(-2, -1, 1, 2), 100, replace=TRUE))
  true_lfc_cond <- sample(true_lfc_cond)
  true_lfc_batchB <- rnorm(n_genes, 0, 0.3)
  true_lfc_batchC <- rnorm(n_genes, 0, 0.3)
  base_mu <- pmin(5000, pmax(5, exp(rnorm(n_genes, 5, 2))))
  sf <- exp(rnorm(n_samples, 0, 0.2))
  disp <- 0.05 + 5.0 / base_mu

  counts <- matrix(0L, nrow = n_genes, ncol = n_samples)
  for (g in seq_len(n_genes)) {
    for (s in seq_len(n_samples)) {
      lfc_b <- if (batch[s] == "B") true_lfc_batchB[g]
               else if (batch[s] == "C") true_lfc_batchC[g] else 0
      lfc_c <- if (cond[s] == "treated") true_lfc_cond[g] else 0
      mu_gs <- base_mu[g] * sf[s] * 2^(lfc_b + lfc_c)
      counts[g, s] <- rnbinom(1, mu = mu_gs, size = 1 / disp[g])
    }
  }
  rownames(counts) <- paste0("gene_", seq_len(n_genes) - 1)
  colnames(counts) <- paste0("sample_", seq_len(n_samples) - 1)
  coldata <- data.frame(batch = batch, condition = cond,
                        row.names = colnames(counts))
  list(counts = counts, coldata = coldata)
}

simulate_continuous <- function(n_samples, n_genes, seed) {
  set.seed(seed)
  x <- rnorm(n_samples, 0, 1)              # continuous covariate
  true_slope <- c(rnorm(n_genes - 80, 0, 0.15), sample(c(-1.5, -0.7, 0.7, 1.5), 80, replace=TRUE))
  true_slope <- sample(true_slope)
  base_mu <- pmin(5000, pmax(5, exp(rnorm(n_genes, 5, 2))))
  sf <- exp(rnorm(n_samples, 0, 0.2))
  disp <- 0.05 + 5.0 / base_mu

  counts <- matrix(0L, nrow = n_genes, ncol = n_samples)
  for (g in seq_len(n_genes)) {
    for (s in seq_len(n_samples)) {
      mu_gs <- base_mu[g] * sf[s] * 2^(true_slope[g] * x[s])
      counts[g, s] <- rnbinom(1, mu = mu_gs, size = 1 / disp[g])
    }
  }
  rownames(counts) <- paste0("gene_", seq_len(n_genes) - 1)
  colnames(counts) <- paste0("sample_", seq_len(n_samples) - 1)
  coldata <- data.frame(x = x, row.names = colnames(counts))
  list(counts = counts, coldata = coldata)
}

# -------------------- dump helpers --------------------

write_intermediates <- function(dds, dir) {
  sf_df <- data.frame(sample = names(sizeFactors(dds)),
                      size_factor = unname(sizeFactors(dds)))
  write.csv(sf_df, file.path(dir, "size_factors.csv"), row.names=FALSE)

  mc <- mcols(dds)
  disp_df <- data.frame(
    gene         = rownames(dds),
    baseMean     = mc$baseMean,
    dispGeneEst  = mc$dispGeneEst,
    dispFit      = mc$dispFit,
    dispMAP      = mc$dispMAP,
    dispersion   = mc$dispersion,
    dispOutlier  = mc$dispOutlier
  )
  write.csv(disp_df, file.path(dir, "dispersions.csv"), row.names=FALSE)

  dfun <- dispersionFunction(dds)
  tc <- attr(dfun, "coefficients")
  fit_type <- attr(dfun, "fitType")
  if (is.null(fit_type)) fit_type <- "parametric"
  trend_df <- data.frame(
    fit_type    = fit_type,
    asymptDisp  = if (!is.null(tc)) unname(tc["asymptDisp"]) else NA_real_,
    extraPois   = if (!is.null(tc)) unname(tc["extraPois"])  else NA_real_
  )
  write.csv(trend_df, file.path(dir, "dispersion_function.csv"), row.names=FALSE)

  design_cols <- resultsNames(dds)
  beta_log2 <- as.data.frame(mc[, design_cols, drop=FALSE])
  beta_log2$gene <- rownames(dds)
  beta_log2 <- beta_log2[, c("gene", design_cols)]
  write.csv(beta_log2, file.path(dir, "coefficients_log2.csv"), row.names=FALSE)

  se_cols <- paste0("SE_", design_cols)
  se_log2 <- as.data.frame(mc[, se_cols, drop=FALSE])
  se_log2$gene <- rownames(dds)
  se_log2 <- se_log2[, c("gene", se_cols)]
  write.csv(se_log2, file.path(dir, "coefficient_se_log2.csv"), row.names=FALSE)

  write.csv(as.data.frame(assays(dds)$mu),    file.path(dir, "mu.csv"),    row.names=TRUE)
  write.csv(as.data.frame(assays(dds)$H),     file.path(dir, "H.csv"),     row.names=TRUE)
  write.csv(as.data.frame(assays(dds)$cooks), file.path(dir, "cooks.csv"), row.names=TRUE)

  invisible(NULL)
}

run_results_default <- function(dds, contrast_or_name) {
  # Default Wald results for the comparison (no shrinkage, no filters)
  args <- list(independentFiltering = FALSE, cooksCutoff = FALSE)
  if (is.character(contrast_or_name) && length(contrast_or_name) == 1) {
    args$name <- contrast_or_name
  } else {
    args$contrast <- contrast_or_name
  }
  do.call(results, c(list(dds), args))
}

dump_main_results <- function(dds, dir, contrast_or_name, shrink_coef = NULL) {
  res <- run_results_default(dds, contrast_or_name)
  shrunk <- NULL
  prior_var <- NA_real_
  prior_scale <- NA_real_
  prior_no_shrink_scale <- NA_real_
  if (!is.null(shrink_coef)) {
    shrunk <- tryCatch(
      lfcShrink(dds, coef = shrink_coef, type = "apeglm", quiet = TRUE),
      error = function(e) { message("lfcShrink apeglm failed: ", conditionMessage(e)); NULL }
    )
    if (!is.null(shrunk)) {
      pi <- priorInfo(shrunk)
      pc <- pi$prior.control
      if (!is.null(pc)) {
        if (!is.null(pc$prior.var))             prior_var             <- pc$prior.var
        if (!is.null(pc$prior.scale))           prior_scale           <- pc$prior.scale
        if (!is.null(pc$prior.no.shrink.scale)) prior_no_shrink_scale <- pc$prior.no.shrink.scale
      }
    }
  }
  out <- data.frame(
    gene = rownames(res),
    baseMean       = res$baseMean,
    log2FoldChange = res$log2FoldChange,
    lfcSE          = res$lfcSE,
    stat           = res$stat,
    pvalue         = res$pvalue,
    padj           = res$padj,
    dispersion     = dispersions(dds),
    log2FoldChange_apeglm = if (!is.null(shrunk)) shrunk$log2FoldChange else NA_real_,
    lfcSE_apeglm          = if (!is.null(shrunk)) shrunk$lfcSE          else NA_real_
  )
  write.csv(out, file.path(dir, "results.csv"), row.names = FALSE)

  apeglm_df <- data.frame(
    prior_var             = prior_var,
    prior_scale           = prior_scale,
    prior_no_shrink_scale = prior_no_shrink_scale,
    shrink_coef           = ifelse(is.null(shrink_coef), NA, shrink_coef)
  )
  write.csv(apeglm_df, file.path(dir, "apeglm_priorinfo.csv"), row.names = FALSE)
}

dump_indep_filter <- function(dds, dir, contrast_or_name) {
  args <- list(independentFiltering = TRUE, cooksCutoff = FALSE, alpha = 0.1)
  if (is.character(contrast_or_name) && length(contrast_or_name) == 1) {
    args$name <- contrast_or_name
  } else {
    args$contrast <- contrast_or_name
  }
  res_if <- do.call(results, c(list(dds), args))
  md <- metadata(res_if)
  filt_df <- data.frame(
    theta       = if (!is.null(md$filterTheta))     md$filterTheta     else NA_real_,
    threshold   = if (!is.null(md$filterThreshold)) md$filterThreshold else NA_real_,
    alpha       = 0.1
  )
  write.csv(filt_df, file.path(dir, "indep_filter.csv"), row.names=FALSE)

  padj_df <- data.frame(
    gene      = rownames(res_if),
    baseMean  = res_if$baseMean,
    pvalue    = res_if$pvalue,
    padj_filt = res_if$padj
  )
  write.csv(padj_df, file.path(dir, "padj_filtered.csv"), row.names=FALSE)
}

dump_cooks_filtered <- function(dds, dir, contrast_or_name) {
  # results() with cooksCutoff=TRUE (the DESeq2 default).
  args <- list(independentFiltering = FALSE, cooksCutoff = TRUE)
  if (is.character(contrast_or_name) && length(contrast_or_name) == 1) {
    args$name <- contrast_or_name
  } else {
    args$contrast <- contrast_or_name
  }
  res_cf <- do.call(results, c(list(dds), args))
  out <- data.frame(
    gene     = rownames(res_cf),
    pvalue_cf = res_cf$pvalue,
    padj_cf   = res_cf$padj
  )
  write.csv(out, file.path(dir, "cooks_filtered.csv"), row.names=FALSE)
}

dump_lrt <- function(counts, coldata, design, reduced, dir) {
  dds <- DESeqDataSetFromMatrix(countData=counts, colData=coldata, design=design)
  dds <- DESeq(dds, test="LRT", reduced=reduced, fitType="parametric", quiet=TRUE)
  res <- results(dds, independentFiltering=FALSE, cooksCutoff=FALSE)
  lrt_df <- data.frame(
    gene   = rownames(res),
    stat   = res$stat,
    pvalue = res$pvalue,
    padj   = res$padj,
    dispersion = dispersions(dds)
  )
  write.csv(lrt_df, file.path(dir, "lrt_results.csv"), row.names=FALSE)
}

dump_alt_trends <- function(counts, coldata, design, dir) {
  # Run DESeq with fitType="local" and "mean" and dump per-gene dispFit.
  for (ftype in c("local", "mean")) {
    dds <- DESeqDataSetFromMatrix(countData = counts, colData = coldata, design = design)
    dds <- tryCatch(
      DESeq(dds, fitType = ftype, quiet = TRUE),
      error = function(e) { message(sprintf("DESeq fitType=%s failed: %s", ftype, conditionMessage(e))); NULL }
    )
    if (is.null(dds)) next
    df <- data.frame(
      gene        = rownames(dds),
      dispGeneEst = mcols(dds)$dispGeneEst,
      dispFit     = mcols(dds)$dispFit,
      dispMAP     = mcols(dds)$dispMAP,
      dispersion  = mcols(dds)$dispersion
    )
    write.csv(df, file.path(dir, sprintf("%s_dispfit.csv", ftype)), row.names = FALSE)
  }
}

# -------------------- driver --------------------

run_fixture <- function(cfg, out_root) {
  dir <- file.path(out_root, cfg$label)
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  message(sprintf("[R] %s", cfg$label))
  sim <- cfg$simulate(cfg$n_samples, cfg$n_genes, cfg$seed)
  write.csv(sim$counts,  file.path(dir, "counts.csv"),  row.names = TRUE)
  write.csv(sim$coldata, file.path(dir, "coldata.csv"), row.names = TRUE)

  dds <- DESeqDataSetFromMatrix(countData = sim$counts, colData = sim$coldata,
                                 design = cfg$design)
  dds <- DESeq(dds, fitType = "parametric", quiet = TRUE)

  write_intermediates(dds, dir)
  dump_main_results(dds, dir, cfg$contrast_or_name, shrink_coef = cfg$shrink_coef)
  dump_indep_filter(dds, dir, cfg$contrast_or_name)
  dump_cooks_filtered(dds, dir, cfg$contrast_or_name)
  if (isTRUE(cfg$lrt)) {
    dump_lrt(sim$counts, sim$coldata, cfg$design, cfg$lrt_reduced, dir)
  }
  if (isTRUE(cfg$alt_trends)) {
    dump_alt_trends(sim$counts, sim$coldata, cfg$design, dir)
  }
  message(sprintf("[R]   done %s", cfg$label))
}

main <- function() {
  out_root <- "fixtures/r_deseq2"
  dir.create(out_root, showWarnings=FALSE, recursive=TRUE)
  cases <- list(
    list(label = "small_12x200",  simulate = simulate_counts,
         n_samples = 12,  n_genes = 200,  seed = 0,
         design = ~ condition,
         contrast_or_name = c("condition", "treated", "control"),
         shrink_coef = "condition_treated_vs_control",
         lrt = TRUE, lrt_reduced = ~1,
         alt_trends = TRUE),
    list(label = "medium_30x500", simulate = simulate_counts,
         n_samples = 30,  n_genes = 500,  seed = 0,
         design = ~ condition,
         contrast_or_name = c("condition", "treated", "control"),
         shrink_coef = "condition_treated_vs_control",
         lrt = TRUE, lrt_reduced = ~1,
         alt_trends = TRUE),
    list(label = "large_60x2000", simulate = simulate_counts,
         n_samples = 60,  n_genes = 2000, seed = 1,
         design = ~ condition,
         contrast_or_name = c("condition", "treated", "control"),
         shrink_coef = "condition_treated_vs_control",
         lrt = TRUE, lrt_reduced = ~1,
         alt_trends = TRUE),
    list(label = "multi_60x1500", simulate = simulate_multifactor,
         n_samples = 60, n_genes = 1500, seed = 2,
         design = ~ batch + condition,
         contrast_or_name = "condition_treated_vs_control",
         shrink_coef = "condition_treated_vs_control",
         lrt = TRUE, lrt_reduced = ~ batch,    # tests effect of condition above batch
         alt_trends = FALSE),
    list(label = "continuous_60x1000", simulate = simulate_continuous,
         n_samples = 60, n_genes = 1000, seed = 3,
         design = ~ x,
         contrast_or_name = "x",
         shrink_coef = "x",
         lrt = TRUE, lrt_reduced = ~ 1,
         alt_trends = FALSE)
  )
  for (cfg in cases) run_fixture(cfg, out_root)
}

main()
