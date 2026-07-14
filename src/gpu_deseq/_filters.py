"""Cook's-distance and independent-filtering, matching R DESeq2 1.30.1.

Both run per DE task (typically ~5k genes), so the filtering is implemented on
CPU via numpy/scipy. The hot work (dispersions, IRLS) stays in the batched GPU
kernels in `_deseq2_core`.

The trimmed method-of-moments dispersion, the ``n_or_more_replicates`` cohort
rule, and the ``lowess`` smoother used by independent filtering are all
implemented here directly (no external DE library) so the package has no
dependency beyond numpy/scipy/pandas/torch.
"""

from __future__ import annotations

from math import ceil, floor

import numpy as np
import pandas as pd
from scipy.stats import f as _f_dist
from scipy.stats import false_discovery_control


# ---------------------------------------------------------------------------
# Trimmed method-of-moments dispersion (for Cook's distance) — DESeq2's
# robustMethodOfMomentsDisp / trimmedCellVariance.
# ---------------------------------------------------------------------------


def _trimmed_mean(x: np.ndarray, trim: float = 0.1, axis: int | None = None) -> np.ndarray | float:
    """Mean after trimming ``floor(n*trim)`` smallest and largest per axis."""
    assert trim <= 0.5
    if axis is not None:
        s = np.sort(x, axis=axis)
        n = x.shape[axis]
        ntrim = floor(n * trim)
        return np.take(s, np.arange(ntrim, n - ntrim), axis).mean(axis)
    n = len(x)
    s = np.sort(x)
    ntrim = floor(n * trim)
    return s[ntrim : n - ntrim].mean()


def _trimmed_variance(x: np.ndarray, trim: float = 0.125, axis: int = 0) -> np.ndarray | float:
    """Trimmed variance with DESeq2's 1.51 bias-correction scale."""
    rm = _trimmed_mean(x, trim=trim, axis=axis)
    sqerror = (x - rm) ** 2
    return 1.51 * _trimmed_mean(sqerror, trim=trim, axis=axis)


def _trimmed_cell_variance(counts: np.ndarray, cells: pd.Series) -> np.ndarray:
    """Per-cohort trimmed variance, maxed across cohorts (DESeq2 trimmedCellVariance).

    Trim fraction and bias scale switch on cohort size: (1/3, 1/4, 1/8) trim
    and (2.04, 1.86, 1.51) scale for n < 3.5, [3.5, 23.5), >= 23.5.
    """
    trimratio = (1 / 3, 1 / 4, 1 / 8)

    def trimfn(x: float) -> int:
        return 2 if x >= 23.5 else 1 if x >= 3.5 else 0

    ns = cells.value_counts()
    sqerror = np.zeros_like(counts)
    for lvl in cells.unique():
        cell_means = _trimmed_mean(
            counts[cells == lvl, :], trim=trimratio[trimfn(ns[lvl])], axis=0
        )
        sqerror[cells == lvl, :] = counts[cells == lvl, :] - cell_means[None, :]
    sqerror **= 2

    var_est = np.zeros((len(ns), counts.shape[1]), dtype=float)
    for i, lvl in enumerate(cells.unique()):
        scale = [2.04, 1.86, 1.51][trimfn(ns[lvl])]
        var_est[i, :] = scale * _trimmed_mean(
            sqerror[cells == lvl, :], trim=trimratio[trimfn(ns[lvl])], axis=0
        )
    return var_est.max(axis=0)


def n_or_more_replicates(design_matrix: pd.DataFrame, min_replicates: int) -> pd.Series:
    """Bool series: does each sample's design cell have >= ``min_replicates`` samples."""
    n_or_more = design_matrix.value_counts() >= min_replicates
    replaceable = n_or_more[pd.MultiIndex.from_frame(design_matrix)]
    replaceable.index = design_matrix.index
    return replaceable


def robust_method_of_moments_disp(
    normed_counts: np.ndarray, design_matrix: pd.DataFrame
) -> np.ndarray:
    """Trimmed method-of-moments dispersion for Cook's-distance outlier detection.

    Groups samples by design cell; uses trimmed cell variance when any cell has
    >= 3 replicates, else a plain trimmed variance. DESeq2 floors this at 0.04
    (not the usual 1e-8) so counts sharing an outlier's cell don't blow up.
    """
    three_or_more = n_or_more_replicates(design_matrix, 3)
    if three_or_more.any():
        filtered_counts = normed_counts[three_or_more.values, :]
        filtered_design = design_matrix.loc[three_or_more, :]
        cell_id = pd.Series(
            filtered_design.groupby(filtered_design.columns.values.tolist()).ngroup(),
            index=filtered_design.index,
        )
        v = _trimmed_cell_variance(filtered_counts, cell_id)
    else:
        v = _trimmed_variance(normed_counts)

    m = normed_counts.mean(0)
    alpha = (v - m) / m**2
    min_disp = 0.04
    np.maximum(alpha, min_disp, out=alpha)
    return alpha


# ---------------------------------------------------------------------------
# Robust locally-weighted regression (lowess) for the independent-filtering
# rejection curve — DESeq2's genefilter::lowess-equivalent smoother.
# ---------------------------------------------------------------------------


def _lowess(
    features: np.ndarray, targets: np.ndarray, frac: float = 2.0 / 3.0, iter: int = 3
) -> np.ndarray:
    """Robust locally-weighted regression; returns smoothed target values."""
    n = len(features)
    r = int(ceil(frac * n))
    h = np.maximum(
        np.array([np.sort(np.abs(features - features[i]))[r] for i in range(n)]), 1e-12
    )
    w = np.clip(
        np.abs(np.nan_to_num((features[:, None] - features[None, :]) / h)), 0.0, 1.0
    )
    w = (1 - w**3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for _ in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array(
                [np.sum(weights * targets), np.sum(weights * targets * features)]
            )
            A = np.array(
                [
                    [np.sum(weights), np.sum(weights * features)],
                    [np.sum(weights * features), np.sum(weights * features * features)],
                ]
            )
            beta = np.linalg.lstsq(A, b, rcond=None)[0]
            yest[i] = beta[0] + beta[1] * features[i]
        residuals = targets - yest
        s = np.median(np.abs(residuals))
        if s == 0:
            delta = (np.abs(residuals) > 0).astype(float)
        else:
            delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta**2) ** 2
    return yest


def cooks_distance(
    counts: np.ndarray,         # (n_samples, n_genes_nz)
    normed_counts: np.ndarray,  # (n_samples, n_genes_nz)
    mu: np.ndarray,             # (n_samples, n_genes_nz)  UN-thresholded
    hat_diag: np.ndarray,       # (n_samples, n_genes_nz)  sqrt(W) * diag(...) * sqrt(W)
    design_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cooks, alpha_robust) matching DESeq2's Cook's-distance computation.

    Args:
        counts: raw counts (sample, gene), non-zero subset.
        normed_counts: counts / size_factors (sample, gene), non-zero subset.
        mu: final μ from IRLS (sample, gene), non-zero subset.
        hat_diag: IRLS H diagonal (sample, gene), non-zero subset.
        design_df: the design DataFrame (pandas; cohort value_counts run on it).
    """
    num_vars = design_df.shape[1]
    alpha_rob = robust_method_of_moments_disp(normed_counts, design_df)  # (n_genes,)
    # Pearson-squared / (tau * p)
    sq_pearson = (counts - mu) ** 2
    V = mu**2 * alpha_rob[None, :] + mu
    sq_pearson = sq_pearson / V
    sq_pearson = sq_pearson / num_vars
    # leverage term
    diag_mul = hat_diag / (1.0 - hat_diag) ** 2
    cooks = sq_pearson * diag_mul  # (n_samples, n_genes)
    return cooks, alpha_rob


def cooks_outlier_mask(
    cooks: np.ndarray,              # (n_samples, n_genes_nz)
    counts: np.ndarray,             # (n_samples, n_genes_nz)
    design_df: pd.DataFrame,
    num_vars: int,
) -> np.ndarray:
    """Return per-gene bool outlier mask, matching DESeq2's Cook's outlier rule.

    A gene is flagged when: (a) max Cooks across samples belonging to cohorts
    with ≥3 replicates exceeds F(0.99, p, n-p), AND (b) fewer than 3 samples
    have more counts than the sample holding that max.
    """
    n_samples = cooks.shape[0]
    cutoff = _f_dist.ppf(0.99, num_vars, n_samples - num_vars)

    use_for_max = n_or_more_replicates(design_df, 3).to_numpy()
    if not use_for_max.any():
        # fall back to the full matrix in that case (no refit path).
        use_for_max = np.ones(n_samples, dtype=bool)

    cooks_exceeds = (cooks[use_for_max, :] > cutoff).any(axis=0)  # (n_genes,)
    if not cooks_exceeds.any():
        return cooks_exceeds

    # For flagged genes: require fewer than 3 other samples exceed the max-Cooks sample.
    pos = cooks[:, cooks_exceeds].argmax(axis=0)  # argmax over samples, per flagged gene
    cooks_flag = cooks_exceeds.copy()
    sub_counts = counts[:, cooks_exceeds]
    threshold_counts = sub_counts[pos, np.arange(len(pos))]
    n_exceeding = (sub_counts > threshold_counts[None, :]).sum(axis=0)
    cooks_flag[cooks_exceeds] = n_exceeding < 3
    return cooks_flag


def independent_filtering(
    pvalues: np.ndarray,    # (n_genes,) may contain NaN
    base_mean: np.ndarray,  # (n_genes,)
    alpha: float,
) -> np.ndarray:
    """Return padj after DESeq2's independent-filtering step.

    For genes with baseMean below the selected cutoff, padj = NaN.
    """
    n_genes = len(pvalues)
    base_mean = np.asarray(base_mean, dtype=np.float64)
    pvalues = np.asarray(pvalues, dtype=np.float64)

    lower_q = float(np.mean(base_mean == 0))
    upper_q = 0.95 if lower_q < 0.95 else 1.0

    theta = np.linspace(lower_q, upper_q, 50)
    cutoffs = np.quantile(base_mean, theta)

    # (n_genes, 50) padj table, NaN by default
    padj_table = np.full((n_genes, len(theta)), np.nan)
    not_nan_p = ~np.isnan(pvalues)
    for i, cutoff in enumerate(cutoffs):
        use = (base_mean >= cutoff) & not_nan_p
        if use.any():
            padj_table[use, i] = false_discovery_control(pvalues[use], method="bh")

    num_rej = (padj_table < alpha).sum(axis=0).astype(int)
    lowess_res = _lowess(theta, num_rej, frac=1 / 5)

    if num_rej.max() <= 10:
        j = 0
    else:
        nonzero = num_rej > 0
        residual = num_rej[nonzero] - lowess_res[nonzero]
        thresh = lowess_res.max() - np.sqrt(np.mean(residual**2))
        above = np.where(num_rej > thresh)[0]
        j = int(above[0]) if len(above) > 0 else 0

    return padj_table[:, j]
