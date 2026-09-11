from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from formulaic import model_matrix
from scipy.stats import chi2, f as f_dist, norm, trim_mean

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
    if tensor.is_complex():
        raise ValueError("counts must be real-valued")
    tensor = tensor.to(dtype=torch.float64)
    if not bool(torch.all(torch.isfinite(tensor)).item()):
        raise ValueError("counts must contain only finite values")
    if torch.any(tensor < 0):
        raise ValueError("counts must be non-negative")
    if not bool(torch.all(tensor == torch.round(tensor)).item()):
        raise ValueError("counts must contain integer values")
    return tensor


def _is_default_range_index(index: pd.Index, length: int) -> bool:
    return (
        isinstance(index, pd.RangeIndex)
        and index.start == 0
        and index.stop == length
        and index.step == 1
    )


def _as_dataframe(coldata: object, sample_ids: Iterable[str]) -> pd.DataFrame:
    expected_index = list(sample_ids)
    if isinstance(coldata, pd.DataFrame):
        frame = coldata.copy()
    else:
        frame = pd.DataFrame(coldata)
    if len(frame) != len(expected_index):
        raise ValueError("coldata rows must match the number of samples")
    if isinstance(coldata, pd.DataFrame) and not _is_default_range_index(frame.index, len(frame)):
        if list(frame.index) != expected_index:
            raise ValueError(
                "coldata index must exactly match sample_ids/count columns in the same order"
            )
    frame.index = pd.Index(expected_index, name=frame.index.name)
    return frame


def _is_binary_factor_design(design_frame: pd.DataFrame, coldata: pd.DataFrame) -> bool:
    """Whether DESeq2's two-group low-count Cook's heuristic applies."""
    model_spec = getattr(design_frame, "model_spec", None)
    formula = getattr(model_spec, "formula", None)
    required = getattr(formula, "required_variables", set())
    if len(required) != 1:
        return False
    variable = next(iter(required))
    if variable not in coldata:
        return False
    values = coldata[variable]
    is_factor = (
        isinstance(values.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(values.dtype)
        or pd.api.types.is_string_dtype(values.dtype)
    )
    return bool(is_factor and values.nunique(dropna=False) == 2)


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
    # Preserve the exact fitted offsets.  Reconstructing them from
    # counts / normalized_counts scans an entire G x S matrix and used to be a
    # material part of lfc_shrink() for large cohorts.
    size_factors: torch.Tensor | None = None        # (S,) fitted sample offsets
    design_df: pd.DataFrame | None = None          # pandas frame (used for cohort value_counts)
    contrast_vector: torch.Tensor | None = None
    # Contrast-specific MLE inference cache.  A standard results -> lfc_shrink
    # workflow otherwise rebuilds the same per-gene Wald sandwich matrices
    # three times.
    wald_statistics: torch.Tensor | None = None    # (G,) MLE Wald statistic
    wald_standard_errors: torch.Tensor | None = None  # (G,) natural-log scale
    wald_contrast: torch.Tensor | None = None       # (P,) cache key
    reduced_log_likelihood: torch.Tensor | None = None
    log_likelihood: torch.Tensor | None = None  # for LRT only
    degrees_of_freedom: int | None = None
    gene_ids: list[str] | None = None
    non_zero_mask: torch.Tensor | None = None
    # Populated by lfc_shrink(): shrunk standard errors (natural log scale).
    shrunk_se: torch.Tensor | None = None
    # Populated by deseq(): DESeq()'s count-outlier replacement/refit metadata.
    replaced_genes: torch.Tensor | None = None
    replaceable_samples: torch.Tensor | None = None
    replacement_counts: torch.Tensor | None = None
    # Original pre-refit Cook's distances, retained because DESeq2 records
    # maxCooks from the first Wald fit rather than recomputing them after refit.
    cooks: torch.Tensor | None = None
    # Preserve the MLE coefficients when `coefficients` contains apeGLM MAP
    # estimates. Wald statistics and p-values must continue to use the MLE.
    mle_coefficients: torch.Tensor | None = None
    # DESeq2 applies its low-count Cook's exemption only to one-factor,
    # two-level categorical designs.
    cooks_low_count_heuristic: bool = False


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
        counts_frame = counts if isinstance(counts, pd.DataFrame) else None
        counts_tensor = _to_tensor(counts)
        n_genes, n_samples = counts_tensor.shape
        if sample_ids is None:
            if counts_frame is not None:
                sample_ids = list(counts_frame.columns)
            elif (
                isinstance(coldata, pd.DataFrame)
                and not _is_default_range_index(coldata.index, len(coldata))
            ):
                sample_ids = list(coldata.index)
            else:
                sample_ids = [f"sample_{idx}" for idx in range(n_samples)]
        if gene_ids is None:
            gene_ids = (
                list(counts_frame.index)
                if counts_frame is not None
                else [f"gene_{idx}" for idx in range(n_genes)]
            )
        self.sample_ids = list(sample_ids)
        self.gene_ids = list(gene_ids)
        if len(self.sample_ids) != n_samples:
            raise ValueError("sample_ids length must match sample count")
        if len(self.gene_ids) != n_genes:
            raise ValueError("gene_ids length must match gene count")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique")
        if len(set(self.gene_ids)) != len(self.gene_ids):
            raise ValueError("gene_ids must be unique")
        if counts_frame is not None:
            if list(counts_frame.columns) != self.sample_ids:
                raise ValueError(
                    "sample_ids must exactly match count DataFrame columns in the same order"
                )
            if list(counts_frame.index) != self.gene_ids:
                raise ValueError(
                    "gene_ids must exactly match count DataFrame index in the same order"
                )
        self.design = design
        self.coldata = _as_dataframe(coldata, self.sample_ids)
        self._design_cpu = model_matrix(design, self.coldata, output="pandas")
        if len(self._design_cpu) != n_samples:
            raise ValueError("design variables must not contain missing values")
        self.design_columns = list(self._design_cpu.columns)
        design_np = self._design_cpu.to_numpy(dtype=np.float64)
        if not np.isfinite(design_np).all():
            raise ValueError("design matrix must contain only finite values")
        design_rank = int(np.linalg.matrix_rank(design_np))
        if design_rank < design_np.shape[1]:
            raise ValueError(
                "design matrix is not full rank; remove confounded or redundant terms"
            )
        self.cooks_low_count_heuristic = _is_binary_factor_design(
            self._design_cpu, self.coldata
        )
        self.backend = backend
        self.device = _resolve_device(backend)
        self.counts = counts_tensor.to(self.device)
        self.design_matrix = torch.as_tensor(
            design_np,
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
        self.dispersion_fit_type: str | None = None

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


def fit_size_factors(dataset: DESeqDataset, method: str = "ratio") -> DESeqDataset:
    """Estimate size factors with a DESeq2 normalization method.

    ``method="ratio"`` (or the backward-compatible alias ``"median_ratio"``)
    selects DESeq2's default median-of-ratios estimator. Sparse matrices for
    which every gene contains a zero require the explicit
    ``method="poscounts"`` alternative, matching DESeq2's ``sfType`` choice.
    """
    normalized_method = "ratio" if method == "median_ratio" else method
    if normalized_method not in ("ratio", "poscounts"):
        raise ValueError("method must be 'ratio', 'median_ratio', or 'poscounts'")
    sf, normed = _core.fit_size_factors(
        dataset.counts, method=normalized_method
    )
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
    dataset.dispersion_fit_type = trend_type
    return dataset


def _evaluate_existing_dispersion_trend(
    dataset: DESeqDataset,
    normed_means: np.ndarray,
) -> np.ndarray:
    """Evaluate the fitted dispersion trend at new base means.

    DESeq2's outlier-refit path does not refit the global trend. It evaluates the
    existing dispersion function at the replacement-count base means. The public
    dataset stores that function on the original genes as ``dispersion_trend``;
    this helper reconstructs its parametric/mean representation, or interpolates
    the local fit, and evaluates it at ``normed_means``.
    """
    if dataset.dispersion_trend is None or dataset.normalized_counts is None:
        raise ValueError("the original dispersion trend is required for outlier refitting")
    nz = dataset.non_zero_mask
    if nz is None:
        raise ValueError("the original non-zero mask is required for outlier refitting")
    old_means = dataset.normalized_counts[nz].mean(dim=1).detach().cpu().numpy()
    old_trend = dataset.dispersion_trend[nz].detach().cpu().numpy()
    fit_type = dataset.dispersion_fit_type or "parametric"

    if fit_type == "mean":
        return np.full_like(normed_means, float(np.nanmedian(old_trend)))
    if fit_type == "local":
        order = np.argsort(old_means)
        x = np.log(np.maximum(old_means[order], 1e-300))
        y = np.log(np.maximum(old_trend[order], _core.MIN_DISP))
        x_new = np.log(np.maximum(normed_means, 1e-300))
        return np.exp(np.interp(x_new, x, y, left=y[0], right=y[-1]))

    # Parametric trend: alpha(mu) = a0 + a1 / mu. Reconstructing the two
    # coefficients from all fitted values is stable and exact up to roundoff.
    x = np.column_stack([np.ones_like(old_means), 1.0 / old_means])
    coefs, *_ = np.linalg.lstsq(x, old_trend, rcond=None)
    return coefs[0] + coefs[1] / normed_means


def _refit_replacement_dispersions(
    replacement: DESeqDataset,
    original: DESeqDataset,
    *,
    use_cuda_graph: bool = False,
    use_triton: bool = False,
) -> DESeqDataset:
    """Refit affected genes with DESeq2's existing trend and prior variance."""
    if original.prior_disp_var is None or original.squared_logres is None:
        raise ValueError("the original dispersion prior is required for outlier refitting")
    if replacement.size_factors is None or replacement.normalized_counts is None:
        raise ValueError("replacement counts must carry the original size factors")

    counts = replacement.counts
    normed = replacement.normalized_counts
    design = replacement.design_matrix
    size_factors = replacement.size_factors
    non_zero = ~torch.all(counts == 0, dim=1)
    replacement.non_zero_mask = non_zero
    n_genes = replacement.n_genes
    nan = float("nan")
    genewise = torch.full((n_genes,), nan, dtype=torch.float64, device=replacement.device)
    trend = torch.full_like(genewise, nan)
    mapped = torch.full_like(genewise, nan)
    final = torch.full_like(genewise, nan)
    idx = torch.nonzero(non_zero, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        replacement.dispersions_gene_wise = genewise
        replacement.dispersion_trend = trend
        replacement.dispersions_map = mapped
        replacement.dispersions = final
        return replacement

    counts_nz = counts[idx]
    normed_nz = normed[idx]
    alpha_init = _core.fit_initial_dispersions(normed_nz, size_factors, design)
    if _core.is_saturated_design(design):
        mu_hat = _core.lin_reg_mu(counts_nz, size_factors, design)
    else:
        _, mu_hat, _, _ = _core.irls_batched(
            counts_nz, size_factors, design, alpha_init
        )
        mu_hat = mu_hat.clamp_min(_core.MIN_MU)
    alpha_mle = _core.fit_alpha_mle(
        counts_nz,
        mu_hat,
        design,
        alpha_init=alpha_init,
        use_cuda_graph=use_cuda_graph,
        use_triton=use_triton,
    )
    means = normed_nz.mean(dim=1).detach().cpu().numpy()
    trend_np = _evaluate_existing_dispersion_trend(original, means)
    trend_nz = torch.as_tensor(trend_np, dtype=torch.float64, device=replacement.device)
    alpha_map = _core.fit_alpha_map(
        counts_nz,
        mu_hat,
        design,
        alpha_hat=trend_nz,
        prior_disp_var=original.prior_disp_var,
        alpha_init=alpha_mle,
        use_cuda_graph=use_cuda_graph,
        use_triton=use_triton,
    ).clamp(_core.MIN_DISP, max(_core.MAX_DISP, replacement.n_samples))
    alpha_final = _core.apply_outlier_keep_mle(
        alpha_map, alpha_mle, trend_nz, original.squared_logres
    )

    genewise[idx] = alpha_mle
    trend[idx] = trend_nz
    mapped[idx] = alpha_map
    final[idx] = alpha_final
    replacement.dispersions_gene_wise = genewise
    replacement.dispersion_trend = trend
    replacement.dispersions_map = mapped
    replacement.dispersions = final
    replacement.prior_disp_var = original.prior_disp_var
    replacement.squared_logres = original.squared_logres
    replacement.dispersion_fit_type = original.dispersion_fit_type
    return replacement


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
        size_factors=dataset.size_factors,
        design_df=dataset._design_cpu if design_matrix is None else None,
        gene_ids=dataset.gene_ids,
        non_zero_mask=non_zero_mask,
        log_likelihood=None,
        cooks_low_count_heuristic=dataset.cooks_low_count_heuristic,
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


def _replace_outliers_and_refit_wald(
    dataset: DESeqDataset,
    fit: DeseqResult,
    *,
    min_replicates: int = 7,
    use_cuda_graph: bool = False,
    use_triton: bool = False,
) -> DeseqResult:
    """Apply the count replacement and per-gene refit performed by ``DESeq()``."""
    if min_replicates < 3:
        raise ValueError("min_replicates must be at least 3")
    if (
        fit.mu is None
        or fit.hat_diagonals is None
        or fit.counts is None
        or fit.normalized_counts is None
        or fit.design_df is None
        or fit.non_zero_mask is None
    ):
        raise ValueError("a complete Wald fit is required for outlier replacement")

    from ._filters import (
        _trimmed_mean_tensor,
        cooks_distance,
        cooks_distance_tensor,
        n_or_more_replicates,
    )

    replaceable = n_or_more_replicates(fit.design_df, min_replicates).to_numpy()
    replaceable_t = torch.as_tensor(replaceable, dtype=torch.bool, device=dataset.device)
    fit.replaceable_samples = replaceable_t
    if not replaceable.any():
        fit.replaced_genes = torch.zeros(
            dataset.n_genes, dtype=torch.bool, device=dataset.device
        )
        return fit
    if dataset.n_samples <= fit.design_matrix.shape[1]:
        fit.replaced_genes = torch.zeros(
            dataset.n_genes, dtype=torch.bool, device=dataset.device
        )
        return fit

    nz_idx = torch.nonzero(fit.non_zero_mask, as_tuple=False).squeeze(-1)
    replaced_full = torch.zeros(dataset.n_genes, dtype=torch.bool, device=dataset.device)
    if nz_idx.numel() == 0:
        fit.replaced_genes = replaced_full
        return fit

    cutoff = f_dist.ppf(
        0.99, fit.design_matrix.shape[1], dataset.n_samples - fit.design_matrix.shape[1]
    )
    cooks_full = torch.full(
        (dataset.n_genes, dataset.n_samples), float("nan"),
        dtype=torch.float64, device=dataset.device,
    )

    if fit.counts.is_cuda:
        counts_gs = fit.counts[nz_idx]
        normed_gs = fit.normalized_counts[nz_idx]
        cooks_gs, _ = cooks_distance_tensor(
            counts_gs,
            normed_gs,
            fit.mu[nz_idx],
            fit.hat_diagonals[nz_idx],
            fit.design_df,
        )
        cooks_full[nz_idx] = cooks_gs
        high_gs = cooks_gs > cutoff
        assign = high_gs & replaceable_t.unsqueeze(0)
        # DESeq2 marks a gene for refitting when any sample exceeds the cutoff,
        # even if that sample's cell is too small for replacement.
        replace_nz_t = high_gs.any(dim=1)
        if not bool(replace_nz_t.any().item()):
            fit.cooks = cooks_full
            fit.replaced_genes = replaced_full
            return fit

        # R replaceOutliers(trim=0.2): trimmed normalized-count mean, rescaled
        # by each size factor, converted to integer by truncation.
        trim_base_mean = _trimmed_mean_tensor(normed_gs, 0.2, dim=1)
        candidate = (
            trim_base_mean.unsqueeze(1) * dataset.size_factors.unsqueeze(0)
        ).to(torch.int64).to(fit.counts.dtype)
        replacement_counts_t = fit.counts.clone()
        replacement_nz = replacement_counts_t[nz_idx]
        replacement_nz[assign] = candidate[assign]
        replacement_counts_t[nz_idx] = replacement_nz
    else:
        counts_sg = fit.counts[nz_idx].T.detach().cpu().numpy()
        normed_sg = fit.normalized_counts[nz_idx].T.detach().cpu().numpy()
        mu_sg = fit.mu[nz_idx].T.detach().cpu().numpy()
        hat_sg = fit.hat_diagonals[nz_idx].T.detach().cpu().numpy()
        cooks, _ = cooks_distance(
            counts_sg, normed_sg, mu_sg, hat_sg, fit.design_df
        )
        cooks_full[nz_idx] = torch.as_tensor(
            cooks.T, dtype=torch.float64, device=dataset.device
        )
        high_gs = (cooks > cutoff).T
        assign = high_gs & replaceable[None, :]
        replace_nz = high_gs.any(axis=1)
        if not replace_nz.any():
            fit.cooks = cooks_full
            fit.replaced_genes = replaced_full
            return fit
        replace_nz_t = torch.as_tensor(
            replace_nz, dtype=torch.bool, device=dataset.device
        )
        trim_base_mean = trim_mean(normed_sg, proportiontocut=0.2, axis=0)
        sf = dataset.size_factors.detach().cpu().numpy()
        candidate = (trim_base_mean[:, None] * sf[None, :]).astype(np.int64)
        replacement_counts = fit.counts.detach().cpu().numpy().copy()
        replacement_nz = replacement_counts[nz_idx.detach().cpu().numpy()].copy()
        replacement_nz[assign] = candidate[assign]
        replacement_counts[nz_idx.detach().cpu().numpy()] = replacement_nz
        replacement_counts_t = torch.as_tensor(
            replacement_counts, dtype=torch.float64, device=dataset.device
        )

    fit.cooks = cooks_full
    full_replace_idx = nz_idx[replace_nz_t]
    replaced_full[full_replace_idx] = True
    refit_counts = replacement_counts_t[full_replace_idx]
    refit_gene_ids = [dataset.gene_ids[i] for i in full_replace_idx.detach().cpu().tolist()]
    refit_dataset = DESeqDataset(
        refit_counts,
        dataset.coldata,
        design=dataset.design,
        gene_ids=refit_gene_ids,
        sample_ids=dataset.sample_ids,
        backend=dataset.backend,
    ).to(dataset.device)
    refit_dataset.size_factors = dataset.size_factors.clone()
    refit_dataset.normalized_counts = (
        refit_dataset.counts / refit_dataset.size_factors.unsqueeze(0)
    )
    _refit_replacement_dispersions(
        refit_dataset,
        dataset,
        use_cuda_graph=use_cuda_graph,
        use_triton=use_triton,
    )
    refit = _fit_glm(refit_dataset)

    fit.coefficients = fit.coefficients.clone()
    fit.mu = fit.mu.clone()
    fit.hat_diagonals = fit.hat_diagonals.clone()
    fit.dispersions = fit.dispersions.clone()
    fit.base_mean = fit.base_mean.clone()
    fit.non_zero_mask = fit.non_zero_mask.clone()
    fit.coefficients[full_replace_idx] = refit.coefficients
    fit.mu[full_replace_idx] = refit.mu
    fit.hat_diagonals[full_replace_idx] = refit.hat_diagonals
    fit.dispersions[full_replace_idx] = refit.dispersions
    fit.base_mean[full_replace_idx] = refit.base_mean
    fit.non_zero_mask[full_replace_idx] = refit.non_zero_mask
    fit.replaced_genes = replaced_full
    fit.replacement_counts = replacement_counts_t
    return fit


def deseq(
    dataset: DESeqDataset,
    contrast: str | Iterable[float] | torch.Tensor | None = None,
    *,
    fit_type: str = "parametric",
    sf_type: str = "ratio",
    min_replicates_for_replace: int | None = 7,
    use_cuda_graph: bool = False,
    use_triton: bool = False,
) -> DeseqResult:
    """Run the standard DESeq2 Wald pipeline.

    This high-level entry point mirrors ``DESeq(test="Wald")``: size-factor
    estimation, dispersion estimation, Wald fitting, Cook's-distance count
    replacement for eligible design cells, and per-gene refitting. Pass
    ``min_replicates_for_replace=None`` to disable replacement, corresponding
    to ``minReplicatesForReplace=Inf`` in R. ``sf_type`` accepts DESeq2's
    ``"ratio"`` default or its explicit ``"poscounts"`` alternative.
    """
    if dataset.size_factors is None:
        fit_size_factors(dataset, method=sf_type)
    if dataset.dispersions is None:
        fit_dispersions(
            dataset,
            fit_type=fit_type,
            use_cuda_graph=use_cuda_graph,
            use_triton=use_triton,
        )
    resolved_contrast = dataset.design_columns[-1] if contrast is None else contrast
    fit = wald_test(dataset, contrast=resolved_contrast)
    if min_replicates_for_replace is not None:
        fit = _replace_outliers_and_refit_wald(
            dataset,
            fit,
            min_replicates=min_replicates_for_replace,
            use_cuda_graph=use_cuda_graph,
            use_triton=use_triton,
        )
    return fit


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
    if fit.size_factors is not None:
        size_factors = fit.size_factors
    else:
        # Backward-compatible fallback for callers that manually construct a
        # DeseqResult.  Results produced by this package always carry the exact
        # fitted vector and avoid this full-matrix reconstruction.
        assert fit.normalized_counts is not None
        ratio = torch.where(
            fit.normalized_counts > 0,
            fit.counts / fit.normalized_counts,
            torch.full_like(fit.counts, float("nan")),
        )
        size_factors = torch.nanmedian(ratio, dim=0).values

    # Estimate prior scale (empirical Bayes) using MLE LFCs at the shrink index.
    nz = torch.nonzero(fit.non_zero_mask, as_tuple=False).squeeze(-1)
    mle_beta_nat = fit.coefficients[nz, shrink_index].detach().cpu().numpy()
    shrink_contrast = _contrast_vector(fit, coeff)
    cache_matches = (
        fit.wald_statistics is not None
        and fit.wald_standard_errors is not None
        and fit.wald_contrast is not None
        and torch.equal(fit.wald_contrast, shrink_contrast)
    )
    if cache_matches:
        stat_nat = fit.wald_statistics[nz]
        se_nat = fit.wald_standard_errors[nz]
    else:
        stat_nat, se_nat = _wald_se_and_stat(
            fit.coefficients[nz], fit.mu[nz], fit.dispersions[nz],
            fit.design_matrix, shrink_contrast,
        )
        stat_full = torch.full(
            (fit.coefficients.shape[0],), float("nan"),
            dtype=torch.float64, device=fit.coefficients.device,
        )
        se_full = torch.full_like(stat_full, float("nan"))
        stat_full[nz] = stat_nat
        se_full[nz] = se_nat
        fit.wald_statistics = stat_full
        fit.wald_standard_errors = se_full
        fit.wald_contrast = shrink_contrast.clone()
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
    contrast_vector = shrink_contrast

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
        size_factors=fit.size_factors,
        design_df=fit.design_df,
        contrast_vector=contrast_vector,
        wald_statistics=fit.wald_statistics,
        wald_standard_errors=fit.wald_standard_errors,
        wald_contrast=fit.wald_contrast,
        gene_ids=fit.gene_ids,
        non_zero_mask=fit.non_zero_mask,
        shrunk_se=se_new_full,
        replaced_genes=fit.replaced_genes,
        replaceable_samples=fit.replaceable_samples,
        replacement_counts=fit.replacement_counts,
        cooks=fit.cooks,
        mle_coefficients=fit.coefficients,
        cooks_low_count_heuristic=fit.cooks_low_count_heuristic,
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
        size_factors=fitted.size_factors,
        design_df=fitted.design_df,
        contrast_vector=contrast_vector,
        gene_ids=dataset.gene_ids,
        non_zero_mask=fitted.non_zero_mask,
        cooks_low_count_heuristic=fitted.cooks_low_count_heuristic,
    )


def lrt_test(dataset: DESeqDataset, reduced_design: str) -> DeseqResult:
    full = _fit_glm(dataset)
    reduced_frame = model_matrix(reduced_design, dataset.coldata, output="pandas")
    if len(reduced_frame) != dataset.n_samples:
        raise ValueError("reduced design variables must not contain missing values")
    full_np = dataset.design_matrix.detach().cpu().numpy()
    reduced_np = reduced_frame.to_numpy(dtype=np.float64)
    if not np.isfinite(reduced_np).all():
        raise ValueError("reduced design matrix must contain only finite values")
    reduced_rank = int(np.linalg.matrix_rank(reduced_np))
    if reduced_rank < reduced_np.shape[1]:
        raise ValueError("reduced design matrix is not full rank")
    if full_np.shape[1] <= reduced_np.shape[1]:
        raise ValueError("reduced design must have fewer columns than the full design")
    combined_rank = int(np.linalg.matrix_rank(np.column_stack([full_np, reduced_np])))
    if combined_rank > full_np.shape[1]:
        raise ValueError("reduced design must be nested within the full design")
    reduced_design_matrix = torch.as_tensor(
        reduced_np,
        device=dataset.device,
        dtype=torch.float64,
    )
    reduced = _fit_glm(dataset, reduced_design_matrix)

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

    contrast_vector = torch.zeros(
        full.coefficients.shape[1], dtype=torch.float64, device=dataset.device
    )
    contrast_vector[-1] = 1.0

    return DeseqResult(
        test_type="lrt",
        design_columns=full.design_columns,
        coefficients=full.coefficients,
        mu=full.mu,
        dispersions=dataset.dispersions,
        base_mean=full.base_mean,
        design_matrix=full.design_matrix,
        hat_diagonals=full.hat_diagonals,
        counts=full.counts,
        normalized_counts=full.normalized_counts,
        size_factors=full.size_factors,
        design_df=full.design_df,
        contrast_vector=contrast_vector,
        log_likelihood=ll_full,
        reduced_log_likelihood=ll_red,
        degrees_of_freedom=full.coefficients.shape[1] - reduced.coefficients.shape[1],
        gene_ids=dataset.gene_ids,
        non_zero_mask=full.non_zero_mask,
        cooks_low_count_heuristic=full.cooks_low_count_heuristic,
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
    W = μ̃ / (1 + α μ̃)  and  μ̃ = max(μ, MIN_MU).
    Stat = cᵀ β / SE.
    Returns (stat, se) each of shape (G,).

    The μ floor matters and is not cosmetic: R forms the weights for `sigma` from
    a μ already thresholded at `minmu` (fitBeta.cpp), and `irls_batched` hands
    back the *un*-thresholded μ because Cook's distance needs it raw. Without the
    floor here, any sample with μ < MIN_MU contributes ~0 weight instead of
    ~MIN_MU, XᵀWX loses that sample and the SE comes out too large -- by ~22% on
    genes where one contrasted group is entirely zero, and ~7% on near-zero-count
    genes whose dispersion is pinned at the ceiling. Genes with μ >= MIN_MU
    throughout, i.e. almost all of them, are unaffected either way.
    """
    G = coefficients.shape[0]
    P = design.shape[1]
    device = coefficients.device
    dtype = coefficients.dtype
    eye = ridge * torch.eye(P, dtype=dtype, device=device)

    alpha = dispersions.unsqueeze(1)          # (G, 1)
    mu = mu.clamp_min(_core.MIN_MU)           # R: fitBeta.cpp thresholds at minmu
    W = mu / (1.0 + mu * alpha)               # (G, S)
    M = torch.einsum("sp,gs,sq->gpq", design, W, design)  # (G, P, P)
    H = torch.linalg.inv(M + eye)     # (G, P, P)
    Hc = torch.einsum("gpq,q->gp", H, contrast)           # (G, P)
    # SE^2 = Hcᵀ M Hc
    se_sq = torch.einsum("gp,gpq,gq->g", Hc, M, Hc)
    se = torch.sqrt(se_sq.clamp_min(1e-30))
    effects = torch.einsum("gp,p->g", coefficients, contrast)  # natural log
    stat = effects / se
    return stat, se


def _cooks_outlier_mask_tensor(
    cooks: torch.Tensor,       # (G, S)
    counts: torch.Tensor,      # (G, S)
    design_df: pd.DataFrame,
    num_vars: int,
    *,
    apply_low_count_heuristic: bool = False,
) -> torch.Tensor:
    """GPU/torch equivalent of ``_filters.cooks_outlier_mask``.

    This path is used when ``deseq()`` has already computed and retained Cook's
    distances during its replacement pass.  Re-running the thresholding on the
    retained tensors avoids copying two complete G x S matrices back to the
    host merely to produce one boolean value per gene.
    """
    from ._filters import n_or_more_replicates

    n_samples = cooks.shape[1]
    cutoff = f_dist.ppf(0.99, num_vars, n_samples - num_vars)
    use_for_max = n_or_more_replicates(design_df, 3).to_numpy()
    if not use_for_max.any():
        use_for_max = np.ones(n_samples, dtype=bool)
    use_for_max_t = torch.as_tensor(
        use_for_max, dtype=torch.bool, device=cooks.device
    )

    cooks_exceeds = (cooks[:, use_for_max_t] > cutoff).any(dim=1)
    if not apply_low_count_heuristic or not bool(cooks_exceeds.any().item()):
        return cooks_exceeds

    flagged = torch.nonzero(cooks_exceeds, as_tuple=False).squeeze(-1)
    flagged_cooks = cooks[flagged]
    flagged_counts = counts[flagged]
    max_positions = flagged_cooks.argmax(dim=1)
    threshold_counts = flagged_counts.gather(
        1, max_positions.unsqueeze(1)
    ).squeeze(1)
    n_exceeding = (flagged_counts > threshold_counts.unsqueeze(1)).sum(dim=1)
    result = cooks_exceeds.clone()
    result[flagged] = n_exceeding < 3
    return result


def _apply_cooks_filter(fit: DeseqResult, pvalue: np.ndarray) -> np.ndarray:
    """Apply DESeq2's Cook's-distance p-value filtering to a result vector."""
    if (
        fit.mu is None
        or fit.hat_diagonals is None
        or fit.counts is None
        or fit.normalized_counts is None
        or fit.design_df is None
        or fit.non_zero_mask is None
    ):
        return pvalue

    from ._filters import cooks_distance, cooks_outlier_mask

    nz_idx = torch.nonzero(fit.non_zero_mask, as_tuple=False).squeeze(-1)
    if nz_idx.numel() == 0:
        return pvalue

    if fit.cooks is not None:
        cooks_nz = fit.cooks[nz_idx]
        if (
            fit.replaceable_samples is not None
            and fit.replaced_genes is not None
            and bool(torch.any(fit.replaced_genes).item())
        ):
            cooks_nz = cooks_nz.clone()
            cooks_nz[:, fit.replaceable_samples] = 0.0
        outlier_nz = _cooks_outlier_mask_tensor(
            cooks_nz,
            fit.counts[nz_idx],
            fit.design_df,
            num_vars=fit.coefficients.shape[1],
            apply_low_count_heuristic=fit.cooks_low_count_heuristic,
        )
        outlier_full = torch.zeros(
            fit.coefficients.shape[0], dtype=torch.bool, device=fit.coefficients.device
        )
        outlier_full[nz_idx] = outlier_nz
        return np.where(outlier_full.detach().cpu().numpy(), np.nan, pvalue)

    counts_sg = fit.counts[nz_idx].T.detach().cpu().numpy()
    normed_sg = fit.normalized_counts[nz_idx].T.detach().cpu().numpy()
    mu_sg = fit.mu[nz_idx].T.detach().cpu().numpy()
    hat_sg = fit.hat_diagonals[nz_idx].T.detach().cpu().numpy()
    cooks, _ = cooks_distance(
        counts_sg, normed_sg, mu_sg, hat_sg, fit.design_df
    )
    outlier_nz = cooks_outlier_mask(
        cooks,
        counts_sg,
        fit.design_df,
        num_vars=fit.coefficients.shape[1],
        apply_low_count_heuristic=fit.cooks_low_count_heuristic,
    )
    outlier_full = np.zeros(fit.coefficients.shape[0], dtype=bool)
    outlier_full[nz_idx.detach().cpu().numpy()] = outlier_nz
    return np.where(outlier_full, np.nan, pvalue)


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
        assert fit.mu is not None and fit.dispersions is not None
        assert fit.design_matrix is not None and fit.non_zero_mask is not None
        resolved_contrast = fit.contrast_vector
        if contrast is not None:
            resolved_contrast = _contrast_vector(fit, contrast)
        if resolved_contrast is None:
            raise ValueError("a contrast is required to report the LRT fold change")

        stat = 2.0 * (fit.log_likelihood - fit.reduced_log_likelihood).detach().cpu().numpy()
        pvalue = chi2.sf(stat, fit.degrees_of_freedom)
        n_genes = fit.coefficients.shape[0]
        lfc = np.full(n_genes, np.nan, dtype=np.float64)
        se = np.full(n_genes, np.nan, dtype=np.float64)
        nz_idx = torch.nonzero(fit.non_zero_mask, as_tuple=False).squeeze(-1)
        if nz_idx.numel() > 0:
            _, se_nz = _wald_se_and_stat(
                fit.coefficients[nz_idx],
                fit.mu[nz_idx],
                fit.dispersions[nz_idx],
                fit.design_matrix,
                resolved_contrast,
            )
            effects_nz = torch.einsum(
                "gp,p->g", fit.coefficients[nz_idx], resolved_contrast
            )
            idx_np = nz_idx.detach().cpu().numpy()
            lfc[idx_np] = effects_nz.detach().cpu().numpy() / np.log(2.0)
            se[idx_np] = se_nz.detach().cpu().numpy() / np.log(2.0)
        if cooks_filter:
            pvalue = _apply_cooks_filter(fit, pvalue)
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

    resolved_contrast = fit.contrast_vector
    if contrast is not None:
        resolved_contrast = _contrast_vector(fit, contrast)
    if resolved_contrast is None:
        raise ValueError("a contrast is required for Wald results")
    if (
        fit.test_type == "wald_shrunk"
        and contrast is not None
        and fit.contrast_vector is not None
        and not torch.equal(resolved_contrast, fit.contrast_vector)
    ):
        raise ValueError("a shrunk result can only report the coefficient that was shrunk")

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
        inference_coefficients = (
            fit.mle_coefficients
            if fit.test_type == "wald_shrunk" and fit.mle_coefficients is not None
            else fit.coefficients
        )
        cache_matches = (
            fit.wald_statistics is not None
            and fit.wald_standard_errors is not None
            and fit.wald_contrast is not None
            and torch.equal(fit.wald_contrast, resolved_contrast)
        )
        if cache_matches:
            stat_nz = fit.wald_statistics[nz_idx]
            se_nz = fit.wald_standard_errors[nz_idx]
        else:
            stat_nz, se_nz = _wald_se_and_stat(
                inference_coefficients[nz_idx],
                fit.mu[nz_idx],
                fit.dispersions[nz_idx],
                fit.design_matrix,
                resolved_contrast,
            )
            # Cache only the fit's declared contrast.  Arbitrary reporting
            # contrasts remain one-shot computations and cannot poison the
            # standard results/shrinkage path.
            if (
                fit.contrast_vector is not None
                and torch.equal(fit.contrast_vector, resolved_contrast)
            ):
                stat_full = torch.full(
                    (n_genes,), float("nan"), dtype=torch.float64,
                    device=fit.coefficients.device,
                )
                se_full = torch.full_like(stat_full, float("nan"))
                stat_full[nz_idx] = stat_nz
                se_full[nz_idx] = se_nz
                fit.wald_statistics = stat_full
                fit.wald_standard_errors = se_full
                fit.wald_contrast = resolved_contrast.clone()
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

        # For shrunk Wald results, replace only lfcSE with the posterior SD. LFC
        # already came from the MAP coefficients, while stat/pvalue above came
        # from the preserved MLE coefficients, matching DESeq2 lfcShrink.
        if fit.test_type == "wald_shrunk" and fit.shrunk_se is not None:
            se[idx_np] = fit.shrunk_se[nz_idx].detach().cpu().numpy() / np.log(2.0)

    if cooks_filter:
        pvalue = _apply_cooks_filter(fit, pvalue)

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
