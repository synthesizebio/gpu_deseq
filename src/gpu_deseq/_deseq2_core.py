"""Batched GPU kernels mirroring pydeseq2's numerics.

Every function here is a torch port of a specific pydeseq2 routine. Numerical
behavior is intended to match pydeseq2 within FP rounding. Where pydeseq2 uses
`scipy.optimize.minimize` per gene, we use the coarse+fine grid search that
pydeseq2 itself falls back to (`grid_fit_alpha` / `grid_fit_beta`) — it is
deterministic, batches on GPU, and matches pydeseq2's fallback path exactly.

References throughout cite paths under
`/home/max_synthesize_bio/text_to_rna/.venv/lib/python3.11/site-packages/pydeseq2/`.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import polygamma

MIN_DISP = 1e-8
MAX_DISP = 10.0
MIN_MU = 0.5
RIDGE = 1e-6
BETA_TOL = 1e-8
MIN_BETA = -30.0
MAX_BETA = 30.0
IRLS_MAXITER = 250
GRID_LENGTH = 100


# ---------------------------------------------------------------------------
# Size factors
# ---------------------------------------------------------------------------


def fit_size_factors(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Median-of-ratios size factors, with poscounts fallback.

    Args:
        counts: (n_genes, n_samples) non-negative raw counts.

    Returns:
        size_factors: (n_samples,) positive float64.
        normed_counts: (n_genes, n_samples) float64, counts / size_factors.

    Ports `pydeseq2.preprocessing.deseq2_norm_fit/transform` (default "ratio"
    mode, dds.py:692-703). Fallback to "poscounts" (dds.py:656-679) when every
    gene has at least one zero.
    """
    counts = counts.to(dtype=torch.float64)
    any_zero_per_gene = torch.any(counts == 0, dim=1)
    if torch.all(any_zero_per_gene):
        return _poscounts_size_factors(counts)

    # Ratio mode: logmeans per gene = mean(log(counts_g)); filter genes with
    # any zero (=> -inf logmean).
    filtered = ~any_zero_per_gene  # genes with all-positive counts
    logmeans = torch.log(counts[filtered]).mean(dim=1)  # (n_filtered,)
    log_ratios = torch.log(counts[filtered]).T - logmeans  # (n_samples, n_filtered)
    # NB: torch.median returns the lower-mid element for even-length inputs;
    # R/numpy/DESeq2 take the average of the two middle values. Use quantile.
    log_medians = torch.quantile(log_ratios, 0.5, dim=1)  # (n_samples,)
    size_factors = torch.exp(log_medians)
    normed_counts = counts / size_factors[None, :]
    return size_factors, normed_counts


def _poscounts_size_factors(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """pydeseq2 poscounts path (dds.py:656-679)."""
    n_samples = counts.shape[1]
    # log of positive counts; treat zeros as "missing" via mask.
    positive = counts > 0
    safe_log = torch.where(positive, torch.log(counts.clamp_min(1.0)), torch.zeros_like(counts))
    count_positive_per_gene = positive.sum(dim=1).clamp_min(1)
    logmeans = safe_log.sum(dim=1) / count_positive_per_gene
    # pydeseq2 uses np.log(self.X, out=log_counts, where=self.X != 0) so zeros
    # stay at 0 and mean is over ALL samples (including zeros), then filters
    # logmeans > 0 AND finite.
    logmeans_all = torch.where(positive, torch.log(counts.clamp_min(1.0)), torch.zeros_like(counts)).mean(dim=1)
    filtered_genes = torch.isfinite(logmeans_all) & (logmeans_all > 0)

    sf = torch.zeros(n_samples, dtype=torch.float64, device=counts.device)
    for j in range(n_samples):
        mask = filtered_genes & positive[:, j]
        if mask.any():
            log_ratio = torch.log(counts[mask, j]) - logmeans_all[mask]
            sf[j] = torch.exp(torch.quantile(log_ratio, 0.5))
        else:
            sf[j] = 1.0
    sf = sf / torch.exp(torch.log(sf).mean())  # geom mean 1
    normed = counts / sf[None, :]
    return sf, normed


# ---------------------------------------------------------------------------
# Initial dispersions (rough + method-of-moments, take min, clip)
# ---------------------------------------------------------------------------


def fit_rough_dispersions(normed_counts: torch.Tensor, design: torch.Tensor) -> torch.Tensor:
    """Linear-regression-based rough dispersion per gene.

    Port of `pydeseq2.utils.fit_rough_dispersions` (utils.py:814-853).

    Args:
        normed_counts: (n_genes, n_samples).
        design: (n_samples, n_vars).

    Returns:
        alpha_rde: (n_genes,) clamped at 0.
    """
    n_samples, n_vars = design.shape
    if n_samples == n_vars:
        raise ValueError(
            "n_samples == n_vars, no residual degrees of freedom for rough dispersions"
        )
    y = normed_counts.T  # (n_samples, n_genes)
    beta = torch.linalg.lstsq(design, y).solution  # (n_vars, n_genes)
    y_hat = (design @ beta).clamp_min(1.0)  # (n_samples, n_genes)
    term = ((y - y_hat) ** 2 - y_hat) / ((n_samples - n_vars) * y_hat**2)
    return term.sum(dim=0).clamp_min(0.0)  # (n_genes,)


def fit_mom_dispersions(normed_counts: torch.Tensor, size_factors: torch.Tensor) -> torch.Tensor:
    """Method-of-moments dispersion per gene.

    Port of `pydeseq2.utils.fit_moments_dispersions` (utils.py:856-885).
    """
    s_mean_inv = (1.0 / size_factors).mean()
    mu = normed_counts.mean(dim=1)
    # Unbiased variance (ddof=1), matching np.var(..., ddof=1).
    sigma = normed_counts.var(dim=1, correction=1)
    out = (sigma - s_mean_inv * mu) / (mu**2)
    return torch.nan_to_num(out, nan=0.0)


def fit_initial_dispersions(
    normed_counts: torch.Tensor,
    size_factors: torch.Tensor,
    design: torch.Tensor,
    min_disp: float = MIN_DISP,
    max_disp: float = MAX_DISP,
) -> torch.Tensor:
    """Initial alpha = clip(min(rough, MoM), min_disp, max_disp).

    Port of `DeseqDataSet._fit_MoM_dispersions` (dds.py:1142-1164). Note that
    `normed_counts` is assumed to already be restricted to non-zero genes.
    """
    rde = fit_rough_dispersions(normed_counts, design)
    mde = fit_mom_dispersions(normed_counts, size_factors)
    return torch.minimum(rde, mde).clamp(min_disp, max_disp)


# ---------------------------------------------------------------------------
# mu_hat initialization
# ---------------------------------------------------------------------------


def lin_reg_mu(
    counts: torch.Tensor,
    size_factors: torch.Tensor,
    design: torch.Tensor,
    min_mu: float = MIN_MU,
) -> torch.Tensor:
    """Saturated-design μ̂ via plain linear regression.

    Port of `pydeseq2.utils.fit_lin_mu` (utils.py:682-715).

    Args:
        counts: (n_genes, n_samples).
        size_factors: (n_samples,).
        design: (n_samples, n_vars).
    Returns:
        mu_hat: (n_genes, n_samples), floored at `min_mu`.
    """
    y = (counts / size_factors[None, :]).T  # (n_samples, n_genes)
    beta = torch.linalg.lstsq(design, y).solution  # (n_vars, n_genes)
    pred = design @ beta  # (n_samples, n_genes)
    mu_hat = (size_factors[:, None] * pred).T  # (n_genes, n_samples)
    return mu_hat.clamp_min(min_mu)


def is_saturated_design(design: torch.Tensor) -> bool:
    """True when the design has as many unique rows as columns.

    Per dds.py:747-749, pydeseq2 uses `lin_reg_mu` in this case; otherwise it
    initializes μ̂ via IRLS with MoM dispersion.
    """
    unique_rows = torch.unique(design, dim=0)
    return int(unique_rows.shape[0]) == int(design.shape[1])


# ---------------------------------------------------------------------------
# Negative log-likelihood (batched over genes AND a grid of log-α values)
# ---------------------------------------------------------------------------


def _nb_nll_batched(
    counts: torch.Tensor,   # (G, S)
    mu: torch.Tensor,       # (G, S)
    alpha: torch.Tensor,    # (K,)
) -> torch.Tensor:
    """Return NLL of shape (G, K).

    Matches `pydeseq2.utils.nb_nll` (utils.py:163-234) with `alpha` vectorized.
    """
    alpha_inv = 1.0 / alpha  # (K,)
    # Broadcast: (G, K, S)
    c = counts.unsqueeze(1)      # (G, 1, S)
    m = mu.unsqueeze(1)          # (G, 1, S)
    a = alpha.view(1, -1, 1)     # (1, K, 1)
    ai = alpha_inv.view(1, -1, 1)

    logbinom = torch.lgamma(c + ai) - torch.lgamma(c + 1.0) - torch.lgamma(ai)
    # per-sample term (the `sum(axis=0)` path in pydeseq2's vectorized branch)
    per_sample = (
        ai * torch.log(a)
        - logbinom
        + (c + ai) * torch.log(m + ai)
        - c * torch.log(m)
    )
    return per_sample.sum(dim=2)  # (G, K)


def _cr_term_batched(
    design: torch.Tensor,  # (S, P)
    mu: torch.Tensor,      # (G, S)
    alpha: torch.Tensor,   # (K,)
    ridge: float = 0.0,
) -> torch.Tensor:
    """Return 0.5 * log|det(X^T W X + ridge I)| of shape (G, K).

    W_g,k,s = mu_g,s / (1 + alpha_k * mu_g,s).
    """
    # (G, K, S)
    W = mu.unsqueeze(1) / (1.0 + alpha.view(1, -1, 1) * mu.unsqueeze(1))
    # X^T W X for each (g, k): einsum("sp,gks,sq->gkpq")
    xtwx = torch.einsum("sp,gks,sq->gkpq", design, W, design)
    if ridge > 0:
        p = design.shape[1]
        xtwx = xtwx + ridge * torch.eye(p, dtype=xtwx.dtype, device=xtwx.device)
    sign, logabsdet = torch.linalg.slogdet(xtwx)
    return 0.5 * logabsdet  # (G, K)


def _grid_fit_alpha(
    counts: torch.Tensor,   # (G, S)
    mu: torch.Tensor,       # (G, S)
    design: torch.Tensor,   # (S, P)
    *,
    alpha_hat: torch.Tensor | None = None,  # (G,) for prior term (MAP), else None
    prior_disp_var: float | None = None,
    min_disp: float = MIN_DISP,
    max_disp: float = MAX_DISP,
    grid_length: int = GRID_LENGTH,
) -> torch.Tensor:
    """Batched coarse+fine grid search over log(alpha) per gene.

    Ports `pydeseq2.grid_search.grid_fit_alpha` (grid_search.py:54-142) with
    CR regularization always on (matches fit_alpha_mle cr_reg=True default).
    When `prior_disp_var` is given, adds (log α − log α_hat)² / (2 σ²).

    Returns: α_hat_MLE of shape (G,).
    """
    device = counts.device
    dtype = counts.dtype
    log_min = float(np.log(min_disp))
    log_max = float(np.log(max_disp))
    coarse_log = torch.linspace(log_min, log_max, grid_length, dtype=dtype, device=device)
    coarse_alpha = torch.exp(coarse_log)

    def _loss(log_grid: torch.Tensor) -> torch.Tensor:
        """Return (G, K) loss at grid of log-alpha values."""
        a = torch.exp(log_grid)  # (K,)
        nll = _nb_nll_batched(counts, mu, a)   # (G, K)
        cr = _cr_term_batched(design, mu, a)   # (G, K)
        out = nll + cr
        if prior_disp_var is not None:
            assert alpha_hat is not None, "alpha_hat required when using prior_disp_var"
            log_ah = torch.log(alpha_hat).unsqueeze(1)  # (G, 1)
            prior = (log_grid.view(1, -1) - log_ah) ** 2 / (2.0 * prior_disp_var)
            out = out + prior
        return out

    coarse_losses = _loss(coarse_log)  # (G, K)
    coarse_idx = torch.argmin(coarse_losses, dim=1)  # (G,)
    coarse_best = coarse_log[coarse_idx]              # (G,)
    delta = coarse_log[1] - coarse_log[0]

    # Fine grid is per-gene: (G, K) of log-alpha values centered at coarse best.
    fine_offsets = torch.linspace(-1.0, 1.0, grid_length, dtype=dtype, device=device) * delta
    fine_log = coarse_best.unsqueeze(1) + fine_offsets.unsqueeze(0)  # (G, K)

    # We cannot reuse the CR einsum easily because log-alpha now varies per gene.
    # Expand per-gene: shape (G, K) for alpha.
    a_fine = torch.exp(fine_log)  # (G, K)

    # Batched NLL with per-gene alpha
    nll_fine = _nb_nll_batched_per_gene(counts, mu, a_fine)  # (G, K)
    cr_fine = _cr_term_batched_per_gene(design, mu, a_fine)  # (G, K)
    loss_fine = nll_fine + cr_fine
    if prior_disp_var is not None:
        log_ah = torch.log(alpha_hat).unsqueeze(1)
        loss_fine = loss_fine + (fine_log - log_ah) ** 2 / (2.0 * prior_disp_var)

    fine_idx = torch.argmin(loss_fine, dim=1)
    best_log = fine_log[torch.arange(fine_log.shape[0], device=device), fine_idx]
    return torch.exp(best_log).clamp(min_disp, max_disp)


def _nb_nll_batched_per_gene(
    counts: torch.Tensor,   # (G, S)
    mu: torch.Tensor,       # (G, S)
    alpha: torch.Tensor,    # (G, K)
) -> torch.Tensor:
    """NB NLL of shape (G, K) where alpha is per-gene."""
    ai = 1.0 / alpha  # (G, K)
    c = counts.unsqueeze(1)          # (G, 1, S)
    m = mu.unsqueeze(1)              # (G, 1, S)
    a_e = alpha.unsqueeze(2)         # (G, K, 1)
    ai_e = ai.unsqueeze(2)           # (G, K, 1)
    logbinom = torch.lgamma(c + ai_e) - torch.lgamma(c + 1.0) - torch.lgamma(ai_e)
    per_sample = (
        ai_e * torch.log(a_e)
        - logbinom
        + (c + ai_e) * torch.log(m + ai_e)
        - c * torch.log(m)
    )
    return per_sample.sum(dim=2)


def _cr_term_batched_per_gene(
    design: torch.Tensor,  # (S, P)
    mu: torch.Tensor,      # (G, S)
    alpha: torch.Tensor,   # (G, K)
) -> torch.Tensor:
    """0.5 * log|det(X^T W X)| of shape (G, K) with per-gene alpha."""
    # W[g, k, s] = mu[g, s] / (1 + alpha[g, k] * mu[g, s])
    W = mu.unsqueeze(1) / (1.0 + alpha.unsqueeze(2) * mu.unsqueeze(1))
    xtwx = torch.einsum("sp,gks,sq->gkpq", design, W, design)
    _, logabsdet = torch.linalg.slogdet(xtwx)
    return 0.5 * logabsdet


def _lp_and_dlp(
    counts: torch.Tensor,   # (G, S)
    mu: torch.Tensor,       # (G, S)
    design: torch.Tensor,   # (S, P)
    log_alpha: torch.Tensor,  # (G,)
    *,
    log_alpha_prior_mean: torch.Tensor | None = None,  # (G,) for usePrior=True
    log_alpha_prior_sigmasq: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cox-Reid log-posterior and its gradient w.r.t. log α — analytical
    port of DESeq2 src/DESeq2.cpp::log_posterior + dlog_posterior with
    useCR=TRUE, useWeights=FALSE. usePrior toggled by passing prior args.

    Returns (lp, dlp), each shape (G,).
    """
    alpha = torch.exp(log_alpha)                       # (G,)
    alpha_inv = 1.0 / alpha                             # (G,)
    a_col = alpha.unsqueeze(1)                          # (G, 1)
    ainv_col = alpha_inv.unsqueeze(1)                   # (G, 1)

    # Cox-Reid term: w_diag = 1 / (1/μ + α); dw = -w_diag^2
    inv_mu_plus_a = (1.0 / mu) + a_col                  # (G, S)
    w_diag = 1.0 / inv_mu_plus_a                        # (G, S)
    dw_diag = -w_diag * w_diag                          # (G, S)

    # b = X^T diag(w) X     — (G, P, P)
    b = torch.einsum("sp,gs,sq->gpq", design, w_diag, design)
    db = torch.einsum("sp,gs,sq->gpq", design, dw_diag, design)

    # logdet(b), and trace(b^{-1} db) for d/dα log det(b).
    sign_b, logdet_b = torch.linalg.slogdet(b)
    cr_term = -0.5 * logdet_b                           # (G,)
    # b_i_db[g] = b[g]^{-1} @ db[g]; trace per gene.
    b_i_db = torch.linalg.solve(b, db)
    tr_bi_db = torch.diagonal(b_i_db, dim1=-2, dim2=-1).sum(dim=-1)  # (G,)
    cr_grad_alpha = -0.5 * tr_bi_db                      # d/dα cr_term

    # Log-likelihood per sample (NB), summed.
    #   ll_s = lgamma(y + α^-1) - lgamma(α^-1) - y log(μ + α^-1) - α^-1 log(1 + μα)
    log_one_plus_mua = torch.log1p(mu * a_col)          # (G, S)
    ll_per_sample = (
        torch.lgamma(counts + ainv_col)
        - torch.lgamma(ainv_col)
        - counts * torch.log(mu + ainv_col)
        - ainv_col * log_one_plus_mua
    )
    ll_part = ll_per_sample.sum(dim=1)                   # (G,)

    lp = ll_part + cr_term                               # (G,)

    # d/dα ll_part = α^-2 * sum_s [digamma(α^-1) + log(1+μα) - μα/(1+μα)
    #                              - digamma(y + α^-1) + y/(μ + α^-1)]
    digamma_ainv = torch.digamma(ainv_col)               # (G, 1)
    mu_alpha = mu * a_col                                # (G, S)
    bracket = (
        digamma_ainv
        + log_one_plus_mua
        - mu_alpha / (1.0 + mu_alpha)
        - torch.digamma(counts + ainv_col)
        + counts / (mu + ainv_col)
    )                                                    # (G, S)
    ll_grad_alpha = alpha_inv * alpha_inv * bracket.sum(dim=1)  # α^-2 * sum (G,)

    # d/d(log α) lp = α * d/dα lp
    dlp = (ll_grad_alpha + cr_grad_alpha) * alpha        # (G,)

    if log_alpha_prior_mean is not None and log_alpha_prior_sigmasq is not None:
        diff = log_alpha - log_alpha_prior_mean
        lp = lp - 0.5 * diff * diff / log_alpha_prior_sigmasq
        dlp = dlp - diff / log_alpha_prior_sigmasq

    return lp, dlp


def _r_fit_alpha_mle(
    counts: torch.Tensor,   # (G, S)
    mu: torch.Tensor,       # (G, S)
    design: torch.Tensor,   # (S, P)
    alpha_init: torch.Tensor,  # (G,) starting α (rough/MoM, clipped)
    *,
    min_disp: float = MIN_DISP,
    max_disp: float = MAX_DISP,
    grid_length: int = GRID_LENGTH,
    maxit: int = 100,
    dispTol: float = 1e-6,
    kappa_0: float = 1.0,
    log_alpha_prior_mean: torch.Tensor | None = None,
    log_alpha_prior_sigmasq: float | None = None,
    apply_no_increase_revert: bool = True,
    apply_grid_fallback: bool = True,
) -> torch.Tensor:
    """Batched faithful port of DESeq2 src/DESeq2.cpp::fitDisp.

    Gradient ascent on log-posterior w.r.t. log α with Armijo backtracking.
    Per iteration t, with state (a, kappa, lp, dlp):

        a_propose = a + kappa * dlp
        # bound a_propose to [-30, 10] by adjusting kappa
        theta_kappa     = -lp(a + kappa * dlp)
        theta_hat_kappa = -lp - kappa * eps * dlp^2          # eps = 1e-4
        if theta_kappa <= theta_hat_kappa:
            iter_accept += 1
            a = a + kappa * dlp
            change = lp(a) - lp_prev
            if change < tol: break
            if a < min_log_alpha: break
            lp = lp(a); dlp = dlp(a)
            kappa = min(kappa * 1.1, kappa_0)
            if iter_accept % 5 == 0: kappa /= 2     # periodic halving
        else:
            kappa /= 2

    Followed by R's `noIncrease` revert to alpha_init and grid fallback for
    non-converged genes (matches estimateDispersionsGeneEst).

    Returns (G,) α estimates clipped to [min_disp, max_disp].
    """
    device, dtype = counts.device, counts.dtype
    eps = 1e-4
    log_lo, log_hi = -30.0, 10.0
    n_samples = counts.shape[1]
    max_disp_eff = float(max(max_disp, n_samples))      # R: maxDisp = max(10, ncol)
    min_log_alpha = float(np.log(min_disp / 10.0))      # R: log(minDisp/10)

    alpha_init_clip = alpha_init.clamp(min_disp, max_disp_eff)
    a = torch.log(alpha_init_clip).clamp(log_lo, log_hi).clone()

    prior_kwargs = dict(log_alpha_prior_mean=log_alpha_prior_mean,
                         log_alpha_prior_sigmasq=log_alpha_prior_sigmasq)
    lp, dlp = _lp_and_dlp(counts, mu, design, a, **prior_kwargs)
    initial_lp = lp.clone()

    G = counts.shape[0]
    kap = torch.full((G,), kappa_0, dtype=dtype, device=device)
    iter_accept = torch.zeros(G, dtype=torch.long, device=device)
    iter_count = torch.zeros(G, dtype=torch.long, device=device)
    done = torch.zeros(G, dtype=torch.bool, device=device)

    for _ in range(maxit):
        active = ~done
        if not active.any():
            break
        iter_count = torch.where(active, iter_count + 1, iter_count)

        # Bounds: shrink kappa to land exactly on boundary if proposal exits [-30, 10].
        a_prop_naive = a + kap * dlp
        # if a_prop_naive < -30: kap = (-30 - a)/dlp (assumes dlp<0 in this case)
        too_low = active & (a_prop_naive < log_lo) & (dlp != 0)
        kap = torch.where(too_low, (log_lo - a) / dlp, kap)
        too_high = active & (a_prop_naive > log_hi) & (dlp != 0)
        kap = torch.where(too_high, (log_hi - a) / dlp, kap)

        a_propose = a + kap * dlp
        lp_propose, _ = _lp_and_dlp(counts, mu, design, a_propose, **prior_kwargs)

        theta_kap = -lp_propose
        theta_hat = -lp - kap * eps * dlp * dlp
        accepted = active & (theta_kap <= theta_hat)
        rejected = active & ~accepted

        # Accepted branch: take step, recompute lp/dlp.
        a_new_acc = torch.where(accepted, a_propose, a)
        lp_new_acc = torch.where(accepted, lp_propose, lp)

        # change = lp_new - lp (pre-update)
        change = lp_new_acc - lp
        # Convergence on accepted: change < tol → done.
        conv_now = accepted & (change < dispTol)
        # Below-floor on accepted: a < min_log_alpha → done (without updating lp).
        below_floor = accepted & (a_new_acc < min_log_alpha)

        a = a_new_acc
        lp = lp_new_acc
        # Recompute dlp at new a (only matters for genes that take another iter).
        # We compute for all and let masks handle the rest.
        _lp_recompute, dlp_new = _lp_and_dlp(counts, mu, design, a, **prior_kwargs)
        # Use lp_recompute = lp_propose (already computed); dlp from analytical.
        dlp = torch.where(accepted, dlp_new, dlp)

        iter_accept = torch.where(accepted, iter_accept + 1, iter_accept)

        # κ updates
        # Accepted: κ ← min(κ*1.1, κ_0); if iter_accept % 5 == 0: κ /= 2
        kap_after_acc = torch.minimum(kap * 1.1,
                                       torch.full_like(kap, kappa_0))
        periodic_halve = accepted & (iter_accept > 0) & (iter_accept % 5 == 0)
        kap_after_acc = torch.where(periodic_halve, kap_after_acc / 2.0, kap_after_acc)
        # Rejected: κ /= 2
        kap_after_rej = kap / 2.0

        kap = torch.where(accepted, kap_after_acc, kap)
        kap = torch.where(rejected, kap_after_rej, kap)

        # Mark done (matching R's `break` semantics).
        done = done | conv_now | below_floor

    # noIncrease (R: revert to alpha_init if last_lp didn't substantially improve).
    if apply_no_increase_revert:
        no_increase = lp < initial_lp + initial_lp.abs() / 1e6
        a = torch.where(no_increase, torch.log(alpha_init_clip), a)

    # Grid fallback for genes flagged non-converged in R's sense:
    # dispGeneEstConv = iter < maxit AND iter != 1; refit when !conv AND α > min_disp*10.
    if apply_grid_fallback:
        not_conv_R = (iter_count >= maxit) | (iter_count == 1)
        refit_mask = not_conv_R & (torch.exp(a) > min_disp * 10)
        if refit_mask.any():
            idx = torch.nonzero(refit_mask, as_tuple=False).squeeze(-1)
            # The grid fallback in DESeq2 uses the same usePrior setting and
            # log_alpha_prior_mean/var if provided.
            prior_var_for_grid = (log_alpha_prior_sigmasq
                                  if log_alpha_prior_mean is not None else None)
            alpha_hat_for_grid = (torch.exp(log_alpha_prior_mean[idx])
                                  if log_alpha_prior_mean is not None else None)
            grid_alpha = _grid_fit_alpha(
                counts[idx], mu[idx], design,
                alpha_hat=alpha_hat_for_grid, prior_disp_var=prior_var_for_grid,
                min_disp=min_disp, max_disp=max_disp,
                grid_length=grid_length,
            )
            a[idx] = torch.log(grid_alpha)

    return torch.exp(a).clamp(min_disp, max_disp_eff)


def fit_alpha_mle(
    counts: torch.Tensor,
    mu: torch.Tensor,
    design: torch.Tensor,
    min_disp: float = MIN_DISP,
    max_disp: float = MAX_DISP,
    grid_length: int = GRID_LENGTH,
    *,
    alpha_init: torch.Tensor | None = None,
    use_nr: bool = True,
) -> torch.Tensor:
    """Batched Cox-Reid adjusted MLE dispersion per gene.

    By default uses Newton-Raphson with the same noIncrease+grid-fallback
    structure as R DESeq2's `estimateDispersionsGeneEst` so that we match R's
    behaviour bit-for-bit for genes where its NR fails to make progress.

    Pass `alpha_init` (the rough/MoM estimate, clipped) to mirror R's NR
    starting point exactly. If omitted, falls back to a quick MoM-style init
    from the supplied μ̂.

    When `use_nr=False`, uses the pure grid search path instead (finds the
    actual CR-NLL minimum but disagrees with R on ~30% of genes where R's
    NR reverts to alpha_init).
    """
    if use_nr:
        if alpha_init is None:
            rde = (((counts - mu) ** 2 - mu) /
                    ((counts.shape[1] - design.shape[1]) * mu**2)).sum(dim=1).clamp_min(0.0)
            alpha_init = rde.clamp(min_disp, max_disp)
        return _r_fit_alpha_mle(
            counts, mu, design, alpha_init,
            min_disp=min_disp, max_disp=max_disp, grid_length=grid_length,
        )
    return _grid_fit_alpha(
        counts, mu, design,
        alpha_hat=None, prior_disp_var=None,
        min_disp=min_disp, max_disp=max_disp,
        grid_length=grid_length,
    )


def fit_alpha_map(
    counts: torch.Tensor,
    mu: torch.Tensor,
    design: torch.Tensor,
    alpha_hat: torch.Tensor,  # per-gene trend dispersion (G,) — prior mean
    prior_disp_var: float,
    min_disp: float = MIN_DISP,
    max_disp: float = MAX_DISP,
    grid_length: int = GRID_LENGTH,
    *,
    alpha_init: torch.Tensor | None = None,
    use_nr: bool = True,
) -> torch.Tensor:
    """Batched MAP dispersion: CR-MLE + Gaussian prior on log α.

    By default, runs R DESeq2's `fitDisp` algorithm with `usePrior=TRUE`
    (port of src/DESeq2.cpp). When `use_nr=False`, uses the grid fallback.

    R DESeq2's `estimateDispersionsMAP` calls fitDisp initialized at
    log(dispGeneEst), with prior centered at log(dispFit), sigma² = priorVar.
    Pass `alpha_init` to mirror R exactly. Note that R does NOT apply the
    noIncrease revert / grid refit branches at the MAP step.
    """
    if use_nr:
        if alpha_init is None:
            alpha_init = alpha_hat
        return _r_fit_alpha_mle(
            counts, mu, design, alpha_init=alpha_init,
            min_disp=min_disp, max_disp=max_disp, grid_length=grid_length,
            log_alpha_prior_mean=torch.log(alpha_hat),
            log_alpha_prior_sigmasq=float(prior_disp_var),
            apply_no_increase_revert=False,
            apply_grid_fallback=False,
        )
    return _grid_fit_alpha(
        counts, mu, design,
        alpha_hat=alpha_hat, prior_disp_var=prior_disp_var,
        min_disp=min_disp, max_disp=max_disp,
        grid_length=grid_length,
    )


# ---------------------------------------------------------------------------
# Dispersion trend (CPU-side, one small fit per DE task)
# ---------------------------------------------------------------------------


def _mad(x: np.ndarray) -> float:
    """Scaled median absolute deviation (MAD).

    MAD = median(|x - median(x)|) / Φ⁻¹(0.75).

    Ports `pydeseq2.utils.mean_absolute_deviation` (utils.py:1210-1227).
    """
    from scipy.stats import norm
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)) / norm.ppf(0.75))


def fit_parametric_trend(
    genewise_disp: np.ndarray,  # (G_nonzero,)
    normed_means: np.ndarray,   # (G_nonzero,)
) -> tuple[np.ndarray, str]:
    """Fit the parametric trend α ≈ a0 + a1/μ̄ via gamma GLM with outlier loop.

    Port of `_fit_parametric_dispersion_trend` (dds.py:1201-1277). Falls back
    to mean trend if convergence fails. Returns (trend_per_gene, trend_type).
    """
    # Initial covariates/targets: filter infs/nans on 1/μ̄.
    cov = 1.0 / np.asarray(normed_means, dtype=np.float64)
    target = np.asarray(genewise_disp, dtype=np.float64)
    keep = np.isfinite(cov) & np.isfinite(target)
    cov_work = cov[keep]
    target_work = target[keep]

    old_coeffs = np.array([0.1, 0.1])
    coeffs = np.array([1.0, 1.0])

    success = False
    for _ in range(100):  # bound just in case
        if not ((coeffs > 1e-10).all()
                and (np.log(np.abs(coeffs / old_coeffs)) ** 2).sum() >= 1e-6):
            break
        old_coeffs = coeffs
        coeffs, predictions, converged = _dispersion_trend_gamma_glm(cov_work, target_work)
        if not converged or (coeffs <= 1e-10).any():
            success = False
            break
        # Filter genes outside (1e-4, 15) pred_ratio and refit
        pred_ratio = target_work / predictions
        mask = (pred_ratio >= 1e-4) & (pred_ratio < 15)
        if not mask.any():
            break
        cov_work = cov_work[mask]
        target_work = target_work[mask]
        success = True

    if not success:
        # Mean-based fallback (dds.py:1279-1301).
        from scipy.stats import trim_mean
        keep = genewise_disp > 10 * MIN_DISP
        if not keep.any():
            mean_disp = float(genewise_disp.mean())
        else:
            mean_disp = float(trim_mean(genewise_disp[keep], proportiontocut=0.001))
        trend = np.full_like(genewise_disp, mean_disp)
        return trend, "mean"

    trend = coeffs[0] + coeffs[1] / np.asarray(normed_means, dtype=np.float64)
    return trend, "parametric"


def fit_local_trend(
    genewise_disp: np.ndarray,
    normed_means: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Fit a LOESS dispersion trend on (log μ̄, log α), matching DESeq2 fitType="local".

    Uses statsmodels' lowess with the same span/iterations DESeq2 uses
    (frac=0.3, it=3). Falls back to the parametric path if lowess returns
    non-finite predictions (e.g. too few genes).
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess
    x = np.log(np.asarray(normed_means, dtype=np.float64))
    y = np.log(np.asarray(genewise_disp, dtype=np.float64))
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        # Too few genes for a reliable lowess; punt to parametric.
        return fit_parametric_trend(genewise_disp, normed_means)
    smoothed = lowess(y[mask], x[mask], frac=0.3, it=3, return_sorted=False)
    # Interpolate back to every gene (predictor is log μ̄).
    order = np.argsort(x[mask])
    xs = x[mask][order]; ys = smoothed[order]
    trend = np.interp(x, xs, ys, left=ys[0], right=ys[-1])
    return np.exp(trend), "local"


def fit_mean_trend(
    genewise_disp: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Constant trimmed-mean dispersion trend, matching DESeq2 fitType="mean".

    Shares the fallback used inside fit_parametric_trend but forced on.
    """
    from scipy.stats import trim_mean
    keep = genewise_disp > 10 * MIN_DISP
    if not keep.any():
        mean_disp = float(np.asarray(genewise_disp, dtype=np.float64).mean())
    else:
        mean_disp = float(trim_mean(genewise_disp[keep], proportiontocut=0.001))
    return np.full_like(genewise_disp, mean_disp, dtype=np.float64), "mean"


def _dispersion_trend_gamma_glm(covariates: np.ndarray, targets: np.ndarray):
    """Port of `default_inference.dispersion_trend_gamma_glm` (lines 200-230)."""
    cov_fit = np.column_stack([np.ones_like(covariates), covariates])
    tgt_fit = targets

    def loss(coeffs):
        mu = cov_fit @ coeffs
        return np.nanmean(tgt_fit / mu + np.log(mu), axis=0)

    def grad(coeffs):
        mu = cov_fit @ coeffs
        return -np.nanmean(
            ((tgt_fit / mu - 1)[:, None] * cov_fit) / mu[:, None], axis=0
        )

    try:
        res = minimize(
            loss, x0=np.array([1.0, 1.0]), jac=grad,
            method="L-BFGS-B",
            bounds=[(1e-12, np.inf), (1e-12, np.inf)],
        )
    except RuntimeWarning:
        return np.array([np.nan, np.nan]), np.array([np.nan] * len(targets)), False
    return res.x, cov_fit @ res.x, res.success


# ---------------------------------------------------------------------------
# Prior variance and outlier rule
# ---------------------------------------------------------------------------


def compute_prior_disp_var(
    genewise_disp: np.ndarray,
    fitted_disp: np.ndarray,
    n_samples: int,
    n_vars: int,
    min_disp: float = MIN_DISP,
) -> tuple[float, float]:
    """Return (prior_disp_var, squared_logres).

    Port of `fit_dispersion_prior` (dds.py:842-886).
    """
    residuals = np.log(genewise_disp) - np.log(fitted_disp)
    above = genewise_disp >= (100 * min_disp)
    if above.sum() == 0:
        squared_logres = 0.0
    else:
        squared_logres = _mad(residuals[above]) ** 2
    prior_var = max(squared_logres - polygamma(1, (n_samples - n_vars) / 2.0), 0.25)
    return float(prior_var), float(squared_logres)


def apply_outlier_keep_mle(
    map_disp: torch.Tensor,
    mle_disp: torch.Tensor,
    fitted_disp: torch.Tensor,
    squared_logres: float,
) -> torch.Tensor:
    """Genes with log(α_MLE) > log(α_trend) + 2√sqlogres keep MLE (dds.py:929-933)."""
    is_outlier = torch.log(mle_disp) > (torch.log(fitted_disp) + 2.0 * float(np.sqrt(squared_logres)))
    return torch.where(is_outlier, mle_disp, map_disp)


# ---------------------------------------------------------------------------
# IRLS (batched over genes) — matches irls_solver (utils.py:273-438)
# ---------------------------------------------------------------------------


def irls_batched(
    counts: torch.Tensor,         # (G, S)
    size_factors: torch.Tensor,   # (S,)
    design: torch.Tensor,         # (S, P)
    dispersions: torch.Tensor,    # (G,)
    *,
    min_mu: float = MIN_MU,
    beta_tol: float = BETA_TOL,
    ridge: float = RIDGE,
    max_iter: int = IRLS_MAXITER,
    min_beta: float = MIN_BETA,
    max_beta: float = MAX_BETA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched IRLS NB-GLM fit mirroring pydeseq2's per-gene `irls_solver`.

    Returns:
        beta: (G, P) fitted coefficients.
        mu:   (G, S) UN-thresholded μ = size_factors * exp(X @ β) (per line 437).
        H:    (G, S) diagonal of W^{1/2} X (X^T W X + ridge I)^{-1} X^T W^{1/2}.
        converged: (G,) bool.

    Genes that don't converge in `max_iter` or blow past |β|>max_beta are
    dispatched to a CPU fallback (per-gene scipy L-BFGS-B) so the final values
    match pydeseq2's behavior. This mirrors the structure of irls_solver.
    """
    G, S = counts.shape
    P = design.shape[1]
    device = counts.device
    dtype = counts.dtype
    log_sf = torch.log(size_factors).to(dtype=dtype)

    # Initialization: QR(X) for full-rank, else log(mean(counts/sf)) intercept.
    rank = int(torch.linalg.matrix_rank(design).item())
    if rank == P:
        Q, R = torch.linalg.qr(design)
        # y: (G, S) → solve per gene
        y = torch.log(counts / size_factors[None, :] + 0.1)  # (G, S)
        # beta = R^{-1} Q^T y^T  → shape (P, G)
        rhs = Q.T @ y.T  # (P, G)
        beta = torch.linalg.solve_triangular(R, rhs, upper=True).T  # (G, P)
    else:
        beta = torch.zeros((G, P), dtype=dtype, device=device)
        beta[:, 0] = torch.log((counts / size_factors[None, :]).mean(dim=1))

    beta_init = beta.clone()
    eye = ridge * torch.eye(P, dtype=dtype, device=device)

    # Pre-compute alpha_j = dispersions_g
    alpha_col = dispersions.unsqueeze(1)  # (G, 1)

    mu = (size_factors[None, :] * torch.exp(beta @ design.T)).clamp_min(min_mu)
    dev = torch.full((G,), 1e3, dtype=dtype, device=device)
    converged = torch.ones(G, dtype=torch.bool, device=device)
    active = torch.ones(G, dtype=torch.bool, device=device)  # still iterating

    for it in range(max_iter):
        W = mu / (1.0 + mu * alpha_col)                                   # (G, S)
        z = torch.log(mu / size_factors[None, :]) + (counts - mu) / mu    # (G, S)
        # X^T W X per gene: (G, P, P)
        xtwx = torch.einsum("sp,gs,sq->gpq", design, W, design) + eye
        xtwz = torch.einsum("sp,gs,gs->gp", design, W, z)
        # Solve per gene
        try:
            new_beta = torch.linalg.solve(xtwx, xtwz.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            new_beta = (torch.linalg.pinv(xtwx) @ xtwz.unsqueeze(-1)).squeeze(-1)

        # Deviance-based stopping per gene
        mu_new = (size_factors[None, :] * torch.exp(new_beta @ design.T)).clamp_min(min_mu)
        # -2 * NLL using per-gene alpha as scalar; batch per gene.
        dev_new = -2.0 * _nb_nll_per_gene(counts, mu_new, dispersions)  # (G,)
        dev_ratio = torch.abs(dev_new - dev) / (torch.abs(dev_new) + 0.1)

        diverging = (torch.abs(new_beta).max(dim=1).values > max_beta)
        # Genes with dev_ratio < tol are done; genes with diverging are flagged.
        done = (dev_ratio < beta_tol) & ~diverging
        # Only advance active genes
        to_advance = active & ~done
        # Commit only for active rows; non-active stay the same.
        update_mask = active.unsqueeze(1)
        beta = torch.where(update_mask, new_beta, beta)
        mu = torch.where(update_mask, mu_new, mu)
        dev = torch.where(active, dev_new, dev)

        # Genes that diverged are marked non-converged and inactive
        if diverging.any():
            bad = active & diverging
            converged = converged & ~bad
            active = active & ~bad

        # Genes that converged by dev-ratio are inactive
        active = active & ~done

        if not active.any():
            break

    # Any gene still active at max_iter: non-converged
    if active.any():
        converged = converged & ~active

    # CPU fallback for non-converged genes (should be rare)
    if (~converged).any():
        bad_idx = torch.nonzero(~converged, as_tuple=False).squeeze(-1).cpu().numpy()
        beta_np = beta.detach().cpu().numpy()
        counts_np = counts.detach().cpu().numpy()
        dispersions_np = dispersions.detach().cpu().numpy()
        design_np = design.detach().cpu().numpy()
        size_factors_np = size_factors.detach().cpu().numpy()
        ridge_mat = RIDGE * np.eye(P)
        from pydeseq2 import utils as _pud  # local import to keep core torch-only
        for g in bad_idx:
            b_init = beta_init[g].detach().cpu().numpy()
            def f(b, g=g):
                mu_ = np.maximum(size_factors_np * np.exp(design_np @ b), min_mu)
                return _pud.nb_nll(counts_np[g], mu_, dispersions_np[g]) + 0.5 * (ridge_mat @ b**2).sum()
            def df(b, g=g):
                mu_ = np.maximum(size_factors_np * np.exp(design_np @ b), min_mu)
                return (
                    -design_np.T @ counts_np[g]
                    + ((1.0 / dispersions_np[g] + counts_np[g]) * mu_ / (1.0 / dispersions_np[g] + mu_)) @ design_np
                    + ridge_mat @ b
                )
            res = minimize(f, b_init, jac=df, method="L-BFGS-B",
                           bounds=[(min_beta, max_beta)] * P)
            beta_np[g] = res.x
            if res.success:
                converged[g] = True
        beta = torch.from_numpy(beta_np).to(device=device, dtype=dtype)

    # Recompute H diagonal on final beta using UN-thresholded mu, per
    # irls_solver lines 427-438.
    mu_post = size_factors[None, :] * torch.exp(beta @ design.T)
    mu_post_clamped = mu_post.clamp_min(min_mu)
    W = mu_post_clamped / (1.0 + mu_post_clamped * alpha_col)
    xtwx = torch.einsum("sp,gs,sq->gpq", design, W, design) + eye
    xtwx_inv = torch.linalg.inv(xtwx)
    # diag(X (X^T W X + ridge)^-1 X^T) per gene, via einsum
    h_diag = torch.einsum("sp,gpq,sq->gs", design, xtwx_inv, design)
    H = torch.sqrt(W) * h_diag * torch.sqrt(W)

    return beta, mu_post, H, converged


def _nb_nll_per_gene(counts: torch.Tensor, mu: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Per-gene NLL (returns (G,)). Matches pydeseq2 nb_nll scalar-alpha branch."""
    ai = 1.0 / alpha                       # (G,)
    ai_e = ai.unsqueeze(1)                 # (G, 1)
    a_e = alpha.unsqueeze(1)               # (G, 1)
    logbinom = torch.lgamma(counts + ai_e) - torch.lgamma(counts + 1.0) - torch.lgamma(ai_e)
    n = counts.shape[1]
    term = (
        n * ai * torch.log(alpha)
        + (
            -logbinom
            + (counts + ai_e) * torch.log(ai_e + mu)
            - counts * torch.log(mu)
        ).sum(dim=1)
    )
    return term
