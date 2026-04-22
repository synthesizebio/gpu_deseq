"""Cook's-distance and independent-filtering ports of pydeseq2.

Both run per DE task (typically ~5k genes), so the filtering is implemented on
CPU via numpy/scipy — matching pydeseq2 numerics exactly. The hot work
(dispersions, IRLS) stays in the batched GPU kernels in `_deseq2_core`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydeseq2.utils import (
    lowess as _pydeseq2_lowess,
    robust_method_of_moments_disp,
)
from scipy.stats import f as _f_dist
from scipy.stats import false_discovery_control


def cooks_distance(
    counts: np.ndarray,         # (n_samples, n_genes_nz)
    normed_counts: np.ndarray,  # (n_samples, n_genes_nz)
    mu: np.ndarray,             # (n_samples, n_genes_nz)  UN-thresholded
    hat_diag: np.ndarray,       # (n_samples, n_genes_nz)  sqrt(W) * diag(...) * sqrt(W)
    design_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cooks, alpha_robust) matching pydeseq2 dds.py:988-1042.

    Args:
        counts: raw counts (sample, gene), non-zero subset.
        normed_counts: counts / size_factors (sample, gene), non-zero subset.
        mu: final μ from IRLS (sample, gene), non-zero subset.
        hat_diag: IRLS H diagonal (sample, gene), non-zero subset.
        design_df: the design DataFrame (pandas; pydeseq2 uses value_counts on it).
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
    """Return per-gene bool outlier mask, per pydeseq2 dds.py:1068-1112.

    A gene is flagged when: (a) max Cooks across samples belonging to cohorts
    with ≥3 replicates exceeds F(0.99, p, n-p), AND (b) fewer than 3 samples
    have more counts than the sample holding that max.
    """
    from pydeseq2.utils import n_or_more_replicates

    n_samples = cooks.shape[0]
    cutoff = _f_dist.ppf(0.99, num_vars, n_samples - num_vars)

    use_for_max = n_or_more_replicates(design_df, 3).to_numpy()
    if not use_for_max.any():
        # pydeseq2 still uses the full matrix in that case (no refit path).
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
    """Return padj after pydeseq2's independent-filtering (ds.py:490-531).

    Matches pydeseq2's algorithm exactly. For genes with baseMean below the
    selected cutoff, padj = NaN.
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
    lowess_res = _pydeseq2_lowess(theta, num_rej, frac=1 / 5)

    if num_rej.max() <= 10:
        j = 0
    else:
        nonzero = num_rej > 0
        residual = num_rej[nonzero] - lowess_res[nonzero]
        thresh = lowess_res.max() - np.sqrt(np.mean(residual**2))
        above = np.where(num_rej > thresh)[0]
        j = int(above[0]) if len(above) > 0 else 0

    return padj_table[:, j]
