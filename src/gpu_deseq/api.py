from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from formulaic import model_matrix
from scipy.stats import chi2, norm

from . import _deseq2_core as _core
from . import _shrink as _shrink_core

# Public aliases, in case callers imported these before
MIN_DISP = _core.MIN_DISP
MAX_DISP = _core.MAX_DISP
MIN_MU = _core.MIN_MU
RIDGE_FACTOR = _core.RIDGE


def _to_tensor(counts: object) -> torch.Tensor:
    if isinstance(counts, torch.Tensor):
        tensor = counts.detach().clone()
    else:
        tensor = torch.as_tensor(np.asarray(counts))
    if tensor.ndim != 2:
        raise ValueError("counts must be a 2D matrix of genes x samples")
    if torch.any(tensor < 0):
        raise ValueError("counts must be non-negative")
    return tensor.to(dtype=torch.float64)


def _as_dataframe(coldata: object, sample_ids: Iterable[str]) -> pd.DataFrame:
    if isinstance(coldata, pd.DataFrame):
        frame = coldata.copy()
    else:
        frame = pd.DataFrame(coldata)
    frame = frame.reset_index(drop=True)
    if len(frame) != len(list(sample_ids)):
        raise ValueError("coldata rows must match the number of samples")
    return frame


def _resolve_device(backend: str) -> torch.device:
    if backend != "torch":
        raise ValueError(f"unsupported backend: {backend}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class DeseqResult:
    test_type: str
    design_columns: list[str]
    coefficients: torch.Tensor              # (G, P)
    mu: torch.Tensor | None                 # (G, S) final μ from IRLS
    dispersions: torch.Tensor | None        # (G,) MAP dispersions
    base_mean: torch.Tensor                 # (G,)
    design_matrix: torch.Tensor | None = None      # (S, P) used for Wald SE
    hat_diagonals: torch.Tensor | None = None      # (G, S) IRLS H diag (Cook's)
    counts: torch.Tensor | None = None             # (G, S) raw counts
    normalized_counts: torch.Tensor | None = None  # (G, S) counts / size_factors
    design_df: pd.DataFrame | None = None          # pandas frame (used for cohort value_counts)
    contrast_vector: torch.Tensor | None = None
    reduced_log_likelihood: torch.Tensor | None = None
    log_likelihood: torch.Tensor | None = None  # for LRT only
    degrees_of_freedom: int | None = None
    gene_ids: list[str] | None = None
    non_zero_mask: torch.Tensor | None = None
    # Populated by lfc_shrink(): shrunk standard errors (natural log scale).
    shrunk_se: torch.Tensor | None = None


class DESeqDataset:
    def __init__(
        self,
        counts: object,
        coldata: object,
        design: str,
        gene_ids: Iterable[str] | None = None,
        sample_ids: Iterable[str] | None = None,
        backend: str = "torch",
    ) -> None:
        counts_tensor = _to_tensor(counts)
        n_genes, n_samples = counts_tensor.shape
        if sample_ids is None:
            sample_ids = [f"sample_{idx}" for idx in range(n_samples)]
        if gene_ids is None:
            gene_ids = [f"gene_{idx}" for idx in range(n_genes)]
        self.sample_ids = list(sample_ids)
        self.gene_ids = list(gene_ids)
        if len(self.sample_ids) != n_samples:
            raise ValueError("sample_ids length must match sample count")
        if len(self.gene_ids) != n_genes:
            raise ValueError("gene_ids length must match gene count")
        self.design = design
        self.coldata = _as_dataframe(coldata, self.sample_ids)
        self._design_cpu = model_matrix(design, self.coldata, output="pandas")
        self.design_columns = list(self._design_cpu.columns)
        self.backend = backend
        self.device = _resolve_device(backend)
        self.counts = counts_tensor.to(self.device)
        self.design_matrix = torch.as_tensor(
            self._design_cpu.to_numpy(dtype=np.float64),
            device=self.device,
            dtype=torch.float64,
        )
        self.size_factors: torch.Tensor | None = None
        self.normalized_counts: torch.Tensor | None = None
        self.non_zero_mask: torch.Tensor | None = None
        self.dispersions_gene_wise: torch.Tensor | None = None
        self.dispersion_trend: torch.Tensor | None = None
        self.dispersions_map: torch.Tensor | None = None
        self.dispersions: torch.Tensor | None = None
        self.prior_disp_var: float | None = None
        self.squared_logres: float | None = None

    @property
    def n_genes(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.counts.shape[1])

    def to(self, device: str | torch.device) -> "DESeqDataset":
        target = torch.device(device)
        self.device = target
        self.counts = self.counts.to(target)
        self.design_matrix = self.design_matrix.to(target)
        for attr in (
            "size_factors",
            "normalized_counts",
            "non_zero_mask",
            "dispersions_gene_wise",
            "dispersion_trend",
            "dispersions_map",
            "dispersions",
        ):
            t = getattr(self, attr, None)
            if t is not None:
                setattr(self, attr, t.to(target))
        return self


def _bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    mask = np.isfinite(pvalues)
    adjusted = np.full_like(pvalues, np.nan, dtype=np.float64)
    if not np.any(mask):
        return adjusted
    pv = pvalues[mask]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    scaled = ranked * n / np.arange(1, n + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    scaled = np.clip(scaled, 0.0, 1.0)
    adjusted_subset = np.empty_like(scaled)
    adjusted_subset[order] = scaled
    adjusted[mask] = adjusted_subset
    return adjusted


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def fit_size_factors(dataset: DESeqDataset, method: str = "median_ratio") -> DESeqDataset:
    """Median-of-ratios size factors (DESeq2 default, with poscounts fallback)."""
    if method != "median_ratio":
        raise ValueError("only median_ratio is supported")
    sf, normed = _core.fit_size_factors(dataset.counts)
    dataset.size_factors = sf.to(dataset.device)
    dataset.normalized_counts = normed.to(dataset.device)
    return dataset


def fit_dispersions(
    dataset: DESeqDataset,
    fit_type: str = "parametric",
    use_cuda_graph: bool = False,
    use_triton: bool = False,
) -> DESeqDataset:
    """Faithful port of DESeq2's gene-wise fit + trend + MAP + outlier rule.

    fit_type matches R DESeq2's fitType argument: "parametric" (default),
    "local" (LOESS on log-dispersion vs log-mean), or "mean" (trimmed-mean
    trend — forces the fallback regardless of whether parametric would work).

    The dispersion Newton-Raphson loops (gene-wise + MAP) can be accelerated on
    CUDA without changing the R-parity result:

    - ``use_cuda_graph``: replay the loops from a captured CUDA graph (removes
      per-iteration launch overhead; bit-identical to eager).
    - ``use_triton``: run each gene's whole loop fused in one Triton kernel with
      per-gene early exit (fastest; matches R to ~1e-14 but not bit-identical to
      eager). Falls back to eager for designs Triton doesn't cover (P > 6)
      or on CPU.

    Either can also be selected via the ``GPU_DESEQ_ACCEL`` env var
    (``graph`` / ``triton``), which is convenient for benchmarking and running
    the parity suite through an accelerated path.
    """
    import os
    _accel = os.environ.get("GPU_DESEQ_ACCEL", "").lower()
    if _accel == "graph":
        use_cuda_graph = True
    elif _accel == "triton":
        use_triton = True
    if fit_type not in ("parametric", "local", "mean"):
        raise ValueError(f"unknown fit_type: {fit_type!r} (choose parametric, local, mean)")
    if dataset.normalized_counts is None or dataset.size_factors is None:
        raise ValueError("size factors must be estimated before dispersions")

    counts = dataset.counts
    normed = dataset.normalized_counts
    design = dataset.design_matrix
    size_factors = dataset.size_factors

    # non-zero genes: those not all-zero across samples (DESeq2 drops these).
    non_zero_mask = ~torch.all(counts == 0, dim=1)
    dataset.non_zero_mask = non_zero_mask

    # Full-length outputs, NaN-padded for all-zero genes.
    n_genes = counts.shape[0]
    genewise_full = torch.full((n_genes,), float("nan"), dtype=torch.float64, device=dataset.device)
    trend_full = torch.full((n_genes,), float("nan"), dtype=torch.float64, device=dataset.device)
    disp_full = torch.full((n_genes,), float("nan"), dtype=torch.float64, device=dataset.device)

    map_full = torch.full((n_genes,), float("nan"), dtype=torch.float64, device=dataset.device)

    if int(non_zero_mask.sum().item()) == 0:
        dataset.dispersions_gene_wise = genewise_full
        dataset.dispersion_trend = trend_full
        dataset.dispersions_map = map_full
        dataset.dispersions = disp_full
        return dataset

    idx = torch.nonzero(non_zero_mask, as_tuple=False).squeeze(-1)
    counts_nz = counts[idx]
    normed_nz = normed[idx]

    # Initial MoM (clipped).
    alpha_init = _core.fit_initial_dispersions(normed_nz, size_factors, design)

    # mu_hat initialization per DESeq2 (lin-reg for saturated designs, else IRLS).
    if _core.is_saturated_design(design):
        mu_hat = _core.lin_reg_mu(counts_nz, size_factors, design)
    else:
        _, mu_hat, _, _ = _core.irls_batched(counts_nz, size_factors, design, alpha_init)
        mu_hat = mu_hat.clamp_min(_core.MIN_MU)

    # Cox-Reid adjusted MLE at fixed mu_hat. Pass alpha_init so the NR loop
    # starts from R's exact rough/MoM init (otherwise NR would derive a
    # different init from μ̂ and we'd lose bit-parity on R's noIncrease check).
    alpha_mle = _core.fit_alpha_mle(counts_nz, mu_hat, design, alpha_init=alpha_init,
                                    use_cuda_graph=use_cuda_graph, use_triton=use_triton)
    genewise_full[idx] = alpha_mle

    # Trend fit on CPU.
    normed_means_nz = normed_nz.mean(dim=1).cpu().numpy()
    if fit_type == "parametric":
        trend_values_nz, trend_type = _core.fit_parametric_trend(
            alpha_mle.cpu().numpy(), normed_means_nz
        )
    elif fit_type == "local":
        trend_values_nz, trend_type = _core.fit_local_trend(
            alpha_mle.cpu().numpy(), normed_means_nz
        )
    else:  # "mean"
        trend_values_nz, trend_type = _core.fit_mean_trend(
            alpha_mle.cpu().numpy()
        )
    trend_nz = torch.as_tensor(trend_values_nz, dtype=torch.float64, device=dataset.device)
    trend_full[idx] = trend_nz

    # Prior variance.
    n_samples = counts.shape[1]
    n_vars = design.shape[1]
    prior_var, sq_logres = _core.compute_prior_disp_var(
        alpha_mle.cpu().numpy(), trend_values_nz, n_samples, n_vars
    )

    # MAP shrinkage at fixed mu_hat with prior centered at trend.
    # R initializes the optimizer at log(dispGeneEst) — pass alpha_mle.
    alpha_map = _core.fit_alpha_map(
        counts_nz, mu_hat, design,
        alpha_hat=trend_nz, prior_disp_var=prior_var,
        alpha_init=alpha_mle,
        use_cuda_graph=use_cuda_graph, use_triton=use_triton,
    )
    # R: maxDisp <- max(10, ncol(object)); clamping to the constant MAX_DISP (10)
    # truncates legitimate high dispersions once n_samples > 10 (surfaces on large
    # cohorts, e.g. GTEx n=300 where R reports dispersions up to ~300).
    max_disp_eff = max(_core.MAX_DISP, n_samples)
    alpha_map = alpha_map.clamp(_core.MIN_DISP, max_disp_eff)
    map_full[idx] = alpha_map

    # Outlier rule: keep MLE for very-high genewise vs trend.
    alpha_final = _core.apply_outlier_keep_mle(alpha_map, alpha_mle, trend_nz, sq_logres)
    disp_full[idx] = alpha_final

    dataset.dispersions_gene_wise = genewise_full
    dataset.dispersion_trend = trend_full
    dataset.dispersions_map = map_full
    dataset.dispersions = disp_full
    dataset.prior_disp_var = float(prior_var)
    dataset.squared_logres = float(sq_logres)
    return dataset


def _fit_glm(dataset: DESeqDataset, design_matrix: torch.Tensor | None = None) -> DeseqResult:
    if dataset.dispersions is None or dataset.size_factors is None:
        raise ValueError("size factors and dispersions must be estimated before GLM fitting")
    x = dataset.design_matrix if design_matrix is None else design_matrix.to(dataset.device)
    non_zero_mask = dataset.non_zero_mask
    assert non_zero_mask is not None
    idx = torch.nonzero(non_zero_mask, as_tuple=False).squeeze(-1)

    n_genes = dataset.n_genes
    n_samples = dataset.n_samples
    p = x.shape[1]

    beta_full = torch.full((n_genes, p), float("nan"), dtype=torch.float64, device=dataset.device)
    mu_full = torch.full((n_genes, n_samples), float("nan"), dtype=torch.float64, device=dataset.device)
    hat_full = torch.full((n_genes, n_samples), float("nan"), dtype=torch.float64, device=dataset.device)

    if idx.numel() > 0:
        counts_nz = dataset.counts[idx]
        dispersions_nz = dataset.dispersions[idx]
        beta_nz, mu_nz, H_nz, _conv = _core.irls_batched(
            counts_nz, dataset.size_factors, x, dispersions_nz
        )
        beta_full[idx] = beta_nz
        mu_full[idx] = mu_nz
        hat_full[idx] = H_nz

    base_mean = torch.mean(dataset.normalized_counts, dim=1)
    design_columns = dataset.design_columns if design_matrix is None else [f"coef_{i}" for i in range(p)]

    return DeseqResult(
        test_type="fit",
        design_columns=design_columns,
        coefficients=beta_full,
        mu=mu_full,
        dispersions=dataset.dispersions,
        base_mean=base_mean,
        design_matrix=x,
        hat_diagonals=hat_full,
        counts=dataset.counts,
        normalized_counts=dataset.normalized_counts,
        design_df=dataset._design_cpu if design_matrix is None else None,
        gene_ids=dataset.gene_ids,
        non_zero_mask=non_zero_mask,
        log_likelihood=None,
    )


def _contrast_vector(result: DeseqResult, contrast: str | Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(contrast, str):
        try:
            idx = result.design_columns.index(contrast)
        except ValueError as exc:
            raise ValueError(f"unknown coefficient: {contrast}") from exc
        vector = torch.zeros(len(result.design_columns), dtype=torch.float64, device=result.coefficients.device)
        vector[idx] = 1.0
        return vector
    if isinstance(contrast, torch.Tensor):
        vector = contrast.to(device=result.coefficients.device, dtype=torch.float64)
    else:
        vector = torch.as_tensor(list(contrast), dtype=torch.float64, device=result.coefficients.device)
    if vector.shape != (len(result.design_columns),):
        raise ValueError("contrast vector has the wrong shape")
    return vector


def lfc_shrink(
    fit: DeseqResult,
    coeff: str,
    method: str = "apeglm",
    adapt: bool = True,
    prior_no_shrink_scale: float = 15.0,
) -> DeseqResult:
    """Shrink LFCs using an apeGLM Cauchy prior (matches DESeq2 lfcShrink).

    Args:
        fit: a DeseqResult from `wald_test()` (i.e. test_type == "wald").
        coeff: the design column to shrink (e.g. "condition[T.treated]").
        method: only "apeglm" is supported for now.
        adapt: if True, estimate prior scale by empirical Bayes from MLE LFCs.
        prior_no_shrink_scale: prior SD for coefficients NOT being shrunk
            (intercept, batch, etc.). DESeq2 default = 15.

    Returns: a new DeseqResult with shrunk coefficients and SE replacing the
    MLE values on `fit`. p-values are left unchanged (per apeGLM convention).
    """
    if method != "apeglm":
        raise ValueError("only method='apeglm' is supported")
    if fit.test_type != "wald":
        raise ValueError("lfc_shrink requires a Wald fit")
    assert fit.mu is not None and fit.dispersions is not None
    assert fit.design_matrix is not None and fit.non_zero_mask is not None
    assert fit.counts is not None

    if coeff not in fit.design_columns:
        raise ValueError(f"unknown coefficient: {coeff}")
    shrink_index = fit.design_columns.index(coeff)

    # Size factors per sample. counts[g,s]/normalized_counts[g,s] == sf[s] for
    # EVERY gene with a nonzero count, so take the median over genes. Using a
    # single gene (e.g. gene 0) is wrong: wherever that gene has a zero count the
    # ratio is 0/0 and reads back as sf=1, corrupting the offset for that sample
    # and mis-shrinking every gene (dataset-dependent, e.g. broke pasilla).
    assert fit.normalized_counts is not None
    ratio = torch.where(fit.normalized_counts > 0,
                        fit.counts / fit.normalized_counts,
                        torch.full_like(fit.counts, float("nan")))
    size_factors = torch.nanmedian(ratio, dim=0).values

    # Estimate prior scale (empirical Bayes) using MLE LFCs at the shrink index.
    nz = torch.nonzero(fit.non_zero_mask, as_tuple=False).squeeze(-1)
    mle_beta_nat = fit.coefficients[nz, shrink_index].detach().cpu().numpy()
    # Per-gene Wald SE at this coefficient: rebuild from the sandwich form.
    _, se_nat = _wald_se_and_stat(
        fit.coefficients[nz], fit.mu[nz], fit.dispersions[nz],
        fit.design_matrix, _contrast_vector(fit, coeff),
    )
    se_nat_np = se_nat.detach().cpu().numpy()
    if adapt:
        prior_var = _shrink_core.fit_prior_var(mle_beta_nat, se_nat_np)
        prior_scale = float(min(np.sqrt(prior_var), 1.0))
    else:
        prior_scale = 1.0

    # Batched Newton MAP.
    counts_nz = fit.counts[nz]
    disp_nz = fit.dispersions[nz]
    beta_shrunk_nz, inv_hess_diag_nz, _conv = _shrink_core.apeglm_shrink_batched(
        counts_nz, disp_nz, size_factors, fit.design_matrix,
        shrink_index=shrink_index,
        prior_scale=prior_scale,
        prior_no_shrink_scale=prior_no_shrink_scale,
    )

    # Write shrunk coefficients back (only the shrink column).
    coefficients_new = fit.coefficients.clone()
    coefficients_new[nz, shrink_index] = beta_shrunk_nz[:, shrink_index]

    # SE from Hessian diagonal: SE = sqrt(|inv_hess[idx, idx]|).
    se_new_full = torch.full((fit.coefficients.shape[0],), float("nan"),
                             dtype=torch.float64, device=fit.coefficients.device)
    se_new_full[nz] = torch.sqrt(inv_hess_diag_nz[:, shrink_index].abs())

    # Contrast vector for the shrunk coefficient (unit vector at shrink_index).
    contrast_vector = torch.zeros(len(fit.design_columns), dtype=torch.float64,
                                  device=fit.coefficients.device)
    contrast_vector[shrink_index] = 1.0

    return DeseqResult(
        test_type="wald_shrunk",
        design_columns=fit.design_columns,
        coefficients=coefficients_new,
        mu=fit.mu,
        dispersions=fit.dispersions,
        base_mean=fit.base_mean,
        design_matrix=fit.design_matrix,
        hat_diagonals=fit.hat_diagonals,
        counts=fit.counts,
        normalized_counts=fit.normalized_counts,
        design_df=fit.design_df,
        contrast_vector=contrast_vector,
        gene_ids=fit.gene_ids,
        non_zero_mask=fit.non_zero_mask,
        shrunk_se=se_new_full,
    )


def wald_test(dataset: DESeqDataset, contrast: str | Iterable[float] | torch.Tensor) -> DeseqResult:
    fitted = _fit_glm(dataset)
    contrast_vector = _contrast_vector(fitted, contrast)
    return DeseqResult(
        test_type="wald",
        design_columns=fitted.design_columns,
        coefficients=fitted.coefficients,
        mu=fitted.mu,
        dispersions=fitted.dispersions,
        base_mean=fitted.base_mean,
        design_matrix=fitted.design_matrix,
        hat_diagonals=fitted.hat_diagonals,
        counts=fitted.counts,
        normalized_counts=fitted.normalized_counts,
        design_df=fitted.design_df,
        contrast_vector=contrast_vector,
        gene_ids=dataset.gene_ids,
        non_zero_mask=fitted.non_zero_mask,
    )


def lrt_test(dataset: DESeqDataset, reduced_design: str) -> DeseqResult:
    full = _fit_glm(dataset)
    reduced_frame = model_matrix(reduced_design, dataset.coldata, output="pandas")
    reduced_design_matrix = torch.as_tensor(
        reduced_frame.to_numpy(dtype=np.float64),
        device=dataset.device,
        dtype=torch.float64,
    )
    reduced = _fit_glm(dataset, reduced_design_matrix)
    if full.coefficients.shape[1] <= reduced.coefficients.shape[1]:
        raise ValueError("reduced design must have fewer columns than the full design")

    # Per-gene log-likelihoods for LRT: sum NB log-prob over samples.
    assert dataset.dispersions is not None and dataset.non_zero_mask is not None
    idx = torch.nonzero(dataset.non_zero_mask, as_tuple=False).squeeze(-1)
    if idx.numel() > 0:
        ll_full_nz = -_core._nb_nll_per_gene(dataset.counts[idx], full.mu[idx], dataset.dispersions[idx])
        ll_red_nz = -_core._nb_nll_per_gene(dataset.counts[idx], reduced.mu[idx], dataset.dispersions[idx])
    ll_full = torch.full((dataset.n_genes,), float("nan"), dtype=torch.float64, device=dataset.device)
    ll_red = torch.full((dataset.n_genes,), float("nan"), dtype=torch.float64, device=dataset.device)
    if idx.numel() > 0:
        ll_full[idx] = ll_full_nz
        ll_red[idx] = ll_red_nz

    return DeseqResult(
        test_type="lrt",
        design_columns=full.design_columns,
        coefficients=full.coefficients,
        mu=full.mu,
        dispersions=dataset.dispersions,
        base_mean=full.base_mean,
        design_matrix=full.design_matrix,
        log_likelihood=ll_full,
        reduced_log_likelihood=ll_red,
        degrees_of_freedom=full.coefficients.shape[1] - reduced.coefficients.shape[1],
        gene_ids=dataset.gene_ids,
        non_zero_mask=full.non_zero_mask,
    )


def _wald_se_and_stat(
    coefficients: torch.Tensor,    # (G, P)
    mu: torch.Tensor,              # (G, S)
    dispersions: torch.Tensor,     # (G,)
    design: torch.Tensor,          # (S, P)
    contrast: torch.Tensor,        # (P,)
    ridge: float = _core.RIDGE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DESeq2's Wald sandwich SE + Wald statistic.

    SE = sqrt( cᵀ (M + ridge)⁻¹ M (M + ridge)⁻¹ c ) where M = Xᵀ W X,
    W = μ / (1 + α μ).
    Stat = cᵀ β / SE.
    Returns (stat, se) each of shape (G,).
    """
    G = coefficients.shape[0]
    P = design.shape[1]
    device = coefficients.device
    dtype = coefficients.dtype
    eye = ridge * torch.eye(P, dtype=dtype, device=device)

    alpha = dispersions.unsqueeze(1)  # (G, 1)
    W = mu / (1.0 + mu * alpha)       # (G, S)
    M = torch.einsum("sp,gs,sq->gpq", design, W, design)  # (G, P, P)
    H = torch.linalg.inv(M + eye)     # (G, P, P)
    Hc = torch.einsum("gpq,q->gp", H, contrast)           # (G, P)
    # SE^2 = Hcᵀ M Hc
    se_sq = torch.einsum("gp,gpq,gq->g", Hc, M, Hc)
    se = torch.sqrt(se_sq.clamp_min(1e-30))
    effects = torch.einsum("gp,p->g", coefficients, contrast)  # natural log
    stat = effects / se
    return stat, se


def results(
    fit: DeseqResult,
    contrast: str | Iterable[float] | torch.Tensor | None = None,
    alpha: float = 0.1,
    p_adjust: str = "bh",
    cooks_filter: bool = True,
    independent_filter: bool = True,
) -> pd.DataFrame:
    """Assemble the results DataFrame.

    For Wald tests, SE uses DESeq2's sandwich form; log2FoldChange and lfcSE
    are in log₂ scale (natural-log divided by ln(2)).
    """
    if p_adjust != "bh":
        raise ValueError("only Benjamini-Hochberg adjustment is supported")
    gene_ids = fit.gene_ids or [f"gene_{idx}" for idx in range(fit.coefficients.shape[0])]
    base_mean = fit.base_mean.detach().cpu().numpy()

    if fit.test_type == "lrt":
        assert fit.reduced_log_likelihood is not None and fit.log_likelihood is not None
        assert fit.degrees_of_freedom is not None
        stat = 2.0 * (fit.log_likelihood - fit.reduced_log_likelihood).detach().cpu().numpy()
        pvalue = chi2.sf(stat, fit.degrees_of_freedom)
        padj = _bh_adjust(pvalue)
        frame = pd.DataFrame(
            {
                "baseMean": base_mean,
                "log2FoldChange": np.nan,
                "lfcSE": np.nan,
                "stat": stat,
                "pvalue": pvalue,
                "padj": padj,
                "significant": padj <= alpha,
            },
            index=gene_ids,
        )
        return frame

    resolved_contrast = fit.contrast_vector
    if contrast is not None:
        resolved_contrast = _contrast_vector(fit, contrast)
    if resolved_contrast is None:
        raise ValueError("a contrast is required for Wald results")

    assert fit.mu is not None and fit.dispersions is not None and fit.non_zero_mask is not None
    assert fit.design_matrix is not None, "Wald result must carry design_matrix"

    # Full-length outputs with NaN for all-zero genes.
    n_genes = fit.coefficients.shape[0]
    lfc = np.full(n_genes, np.nan, dtype=np.float64)
    se = np.full(n_genes, np.nan, dtype=np.float64)
    stat = np.full(n_genes, np.nan, dtype=np.float64)
    pvalue = np.full(n_genes, np.nan, dtype=np.float64)

    nz_idx = torch.nonzero(fit.non_zero_mask, as_tuple=False).squeeze(-1)
    if nz_idx.numel() > 0:
        stat_nz, se_nz = _wald_se_and_stat(
            fit.coefficients[nz_idx],
            fit.mu[nz_idx],
            fit.dispersions[nz_idx],
            fit.design_matrix,
            resolved_contrast,
        )
        effects_nz = torch.einsum("gp,p->g", fit.coefficients[nz_idx], resolved_contrast)
        effects_np = effects_nz.detach().cpu().numpy()
        se_np = se_nz.detach().cpu().numpy()
        stat_np = stat_nz.detach().cpu().numpy()
        pval_np = 2.0 * norm.sf(np.abs(stat_np))

        idx_np = nz_idx.detach().cpu().numpy()
        lfc[idx_np] = effects_np / np.log(2.0)
        se[idx_np] = se_np / np.log(2.0)
        stat[idx_np] = stat_np
        pvalue[idx_np] = pval_np

        # For shrunk Wald results, replace LFC + lfcSE with the shrunk MAP values
        # (natural log in `coefficients`, stored SE in natural log). p-value stays
        # at the MLE Wald p-value per DESeq2 lfcShrink convention.
        if fit.test_type == "wald_shrunk" and fit.shrunk_se is not None:
            shrunk_effects = torch.einsum("gp,p->g",
                                          fit.coefficients[nz_idx], resolved_contrast)
            lfc[idx_np] = shrunk_effects.detach().cpu().numpy() / np.log(2.0)
            se[idx_np] = fit.shrunk_se[nz_idx].detach().cpu().numpy() / np.log(2.0)

        # Cook's distance filter: set pvalue = NaN for flagged genes.
        if cooks_filter and fit.hat_diagonals is not None and fit.counts is not None \
                and fit.normalized_counts is not None and fit.design_df is not None:
            from ._filters import cooks_distance, cooks_outlier_mask
            # the filter helpers expect (samples, genes); our tensors are (genes, samples).
            counts_sg = fit.counts[nz_idx].T.detach().cpu().numpy()
            normed_sg = fit.normalized_counts[nz_idx].T.detach().cpu().numpy()
            mu_sg = fit.mu[nz_idx].T.detach().cpu().numpy()
            hat_sg = fit.hat_diagonals[nz_idx].T.detach().cpu().numpy()
            cooks, _ = cooks_distance(counts_sg, normed_sg, mu_sg, hat_sg, fit.design_df)
            outlier_nz = cooks_outlier_mask(
                cooks, counts_sg, fit.design_df,
                num_vars=fit.coefficients.shape[1],
            )
            outlier_full = np.zeros(n_genes, dtype=bool)
            outlier_full[idx_np] = outlier_nz
            pvalue = np.where(outlier_full, np.nan, pvalue)

    if independent_filter and np.isfinite(pvalue).any():
        from ._filters import independent_filtering
        padj = independent_filtering(pvalue, base_mean, alpha)
    else:
        padj = _bh_adjust(pvalue)

    frame = pd.DataFrame(
        {
            "baseMean": base_mean,
            "log2FoldChange": lfc,
            "lfcSE": se,
            "stat": stat,
            "pvalue": pvalue,
            "padj": padj,
            "significant": padj <= alpha,
        },
        index=gene_ids,
    )
    return frame
