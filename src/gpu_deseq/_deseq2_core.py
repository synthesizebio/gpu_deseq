"""Batched GPU kernels reproducing R DESeq2 numerics.

Every function here is a torch implementation of a specific DESeq2 routine,
validated against the current R DESeq2 reference in
`tests/test_r_step_parity.py`. The dispersion fitter is an analytical port of
`DESeq2/src/DESeq2.cpp::fitDisp` (gradient ascent on the Cox-Reid log-posterior);
where a gene fails to converge we use a deterministic coarse+fine grid search
over log(alpha) that batches cleanly on GPU.

The package has no dependency on any other differential-expression library;
these kernels are the reference implementation.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import gammaln, polygamma

from . import _r_rng

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


def fit_size_factors(
    counts: torch.Tensor,
    method: str = "ratio",
) -> tuple[torch.Tensor, torch.Tensor]:
    """DESeq2-compatible median-of-ratios size factors.

    Args:
        counts: (n_genes, n_samples) non-negative raw counts.

    Returns:
        size_factors: (n_samples,) positive float64.
        normed_counts: (n_genes, n_samples) float64, counts / size_factors.

    ``method="ratio"`` matches DESeq2's default and raises when every gene
    contains a zero. ``method="poscounts"`` is the explicit alternative used
    by DESeq2 for such sparse matrices.
    """
    counts = counts.to(dtype=torch.float64)
    if method not in ("ratio", "poscounts"):
        raise ValueError("method must be 'ratio' or 'poscounts'")
    if method == "poscounts":
        return _poscounts_size_factors(counts)

    any_zero_per_gene = torch.any(counts == 0, dim=1)
    if torch.all(any_zero_per_gene):
        raise ValueError(
            "every gene contains at least one zero; use method='poscounts' "
            "to select DESeq2's positive-count estimator"
        )

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
    """DESeq2 "poscounts" size-factor path (geometric-mean over positive counts)."""
    n_samples = counts.shape[1]
    # log of positive counts; treat zeros as "missing" via mask.
    positive = counts > 0
    safe_log = torch.where(positive, torch.log(counts.clamp_min(1.0)), torch.zeros_like(counts))
    # DESeq2's ``geoMeanNZ`` takes the n-th root, where n is the total number
    # of samples, of the product of the non-zero counts. On the log scale this
    # is the sum of positive log-counts divided by all samples; all-zero genes
    # alone are excluded. Genes whose modified geometric mean is exactly one
    # remain eligible, matching ``estimateSizeFactorsForMatrix``.
    logmeans_all = safe_log.mean(dim=1)
    filtered_genes = positive.any(dim=1) & torch.isfinite(logmeans_all)

    sf = torch.zeros(n_samples, dtype=torch.float64, device=counts.device)
    for j in range(n_samples):
        mask = filtered_genes & positive[:, j]
        if mask.any():
            log_ratio = torch.log(counts[mask, j]) - logmeans_all[mask]
            sf[j] = torch.exp(torch.quantile(log_ratio, 0.5))
        else:
            raise ValueError(
                "cannot estimate a positive size factor for a sample with no "
                "positive counts"
            )
    sf = sf / torch.exp(torch.log(sf).mean())  # geom mean 1
    normed = counts / sf[None, :]
    return sf, normed


# ---------------------------------------------------------------------------
# Initial dispersions (rough + method-of-moments, take min, clip)
# ---------------------------------------------------------------------------


def fit_rough_dispersions(normed_counts: torch.Tensor, design: torch.Tensor) -> torch.Tensor:
    """Linear-regression-based rough dispersion per gene.

    Linear-regression rough dispersion, as in DESeq2's roughDispEstimate.

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

    Method-of-moments dispersion, as in DESeq2's momentsDispEstimate.
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

    DESeq2's initial dispersion estimate. Note that `normed_counts` is assumed
    to already be restricted to non-zero genes.
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

    Matches DESeq2's linear-regression mu initialization for saturated designs.

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

    DESeq2 uses `lin_reg_mu` in this case; otherwise it initializes μ̂ via
    IRLS with MoM dispersion.
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

    NB negative log-likelihood (DESeq2's nbinomLogLike), with `alpha` vectorized.
    """
    alpha_inv = 1.0 / alpha  # (K,)
    # Broadcast: (G, K, S)
    c = counts.unsqueeze(1)      # (G, 1, S)
    m = mu.unsqueeze(1)          # (G, 1, S)
    a = alpha.view(1, -1, 1)     # (1, K, 1)
    ai = alpha_inv.view(1, -1, 1)

    logbinom = torch.lgamma(c + ai) - torch.lgamma(c + 1.0) - torch.lgamma(ai)
    # per-sample term, summed over samples
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

    Deterministic coarse+fine grid search over log(alpha) with Cox-Reid
    regularization always on (the fallback DESeq2 uses when NR fails to
    converge). When `prior_disp_var` is given, adds (log α − log α_hat)² / (2 σ²).

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


def _logdet_and_tr_binv_db(
    b: torch.Tensor,   # (G, P, P), symmetric positive-definite
    db: torch.Tensor,  # (G, P, P)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logdet(b), trace(b^{-1} @ db)) via unrolled no-pivot LU.

    b = Xᵀ diag(w) X is SPD for full-rank X and w > 0, so no pivoting is needed
    and the leading pivots stay positive. Unlike torch.linalg.slogdet/solve
    (cuSOLVER, which host-syncs and cannot be captured in a CUDA graph), this is
    pure elementwise/index ops and is fully CUDA-graph-capturable. Matches the
    linalg path to ~1e-15 (validated across P = 2..6).
    """
    P = b.shape[-1]
    A = b.clone()  # becomes U (upper) with unit-lower multipliers in strict lower
    for k in range(P):
        piv = A[:, k, k].clone()
        for i in range(k + 1, P):
            m = A[:, i, k] / piv
            A[:, i, k] = m
            A[:, i, k + 1:] = A[:, i, k + 1:] - m.unsqueeze(-1) * A[:, k, k + 1:]
    diag_u = torch.diagonal(A, dim1=-2, dim2=-1)          # (G, P)
    logdet = torch.log(diag_u.abs()).sum(-1)              # (G,)

    # Solve b @ X = db for X = b^{-1} db (db has P columns).
    Y = db.clone()
    for i in range(P):                                    # forward: L Y = db
        for j in range(i):
            Y[:, i, :] = Y[:, i, :] - A[:, i, j].unsqueeze(-1) * Y[:, j, :]
    X = Y
    for i in range(P - 1, -1, -1):                        # back: U X = Y
        for j in range(i + 1, P):
            X[:, i, :] = X[:, i, :] - A[:, i, j].unsqueeze(-1) * X[:, j, :]
        X[:, i, :] = X[:, i, :] / A[:, i, i].unsqueeze(-1)
    tr = torch.diagonal(X, dim1=-2, dim2=-1).sum(-1)      # (G,)
    return logdet, tr


def _lp_and_dlp(
    counts: torch.Tensor,   # (G, S)
    mu: torch.Tensor,       # (G, S)
    design: torch.Tensor,   # (S, P)
    log_alpha: torch.Tensor,  # (G,)
    *,
    log_alpha_prior_mean: torch.Tensor | None = None,  # (G,) for usePrior=True
    # 0-dim tensor, not a float: `t / float` lowers to reciprocal-multiply and
    # would diverge by 1 ulp from the captured-graph path. See _r_fit_alpha_mle.
    log_alpha_prior_sigmasq: torch.Tensor | None = None,
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

    # logdet(b) and trace(b^{-1} db) for d/dα log det(b), via a capturable
    # no-pivot LU (b is SPD). Equivalent to slogdet/solve to ~1e-15 but does not
    # host-sync, so this runs inside a CUDA graph.
    logdet_b, tr_bi_db = _logdet_and_tr_binv_db(b, db)
    cr_term = -0.5 * logdet_b                            # (G,)
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


# One iteration of the fitDisp gradient-ascent + Armijo backtracking loop.
# Pure tensor ops (no host sync), so it is safe to trace into a CUDA graph. The
# eager driver checks `~done` for early exit around this; the captured driver
# runs a fixed iteration count and relies on the masks to freeze done genes.
def _nr_step(counts, mu, design, prior_mean, prior_sig, a, lp, dlp, kap,
             iter_accept, iter_count, done, *, eps, log_lo, log_hi, dispTol,
             min_log_alpha, kappa_0):
    active = ~done
    iter_count = torch.where(active, iter_count + 1, iter_count)

    a_prop_naive = a + kap * dlp
    too_low = active & (a_prop_naive < log_lo) & (dlp != 0)
    kap = torch.where(too_low, (log_lo - a) / dlp, kap)
    too_high = active & (a_prop_naive > log_hi) & (dlp != 0)
    kap = torch.where(too_high, (log_hi - a) / dlp, kap)

    a_propose = a + kap * dlp
    lp_propose, _ = _lp_and_dlp(counts, mu, design, a_propose,
                                log_alpha_prior_mean=prior_mean,
                                log_alpha_prior_sigmasq=prior_sig)
    theta_kap = -lp_propose
    theta_hat = -lp - kap * eps * dlp * dlp
    accepted = active & (theta_kap <= theta_hat)
    rejected = active & ~accepted

    a_new_acc = torch.where(accepted, a_propose, a)
    lp_new_acc = torch.where(accepted, lp_propose, lp)
    change = lp_new_acc - lp
    conv_now = accepted & (change < dispTol)
    below_floor = accepted & (a_new_acc < min_log_alpha)

    a = a_new_acc
    lp = lp_new_acc
    _lp_recompute, dlp_new = _lp_and_dlp(counts, mu, design, a,
                                         log_alpha_prior_mean=prior_mean,
                                         log_alpha_prior_sigmasq=prior_sig)
    dlp = torch.where(accepted, dlp_new, dlp)
    iter_accept = torch.where(accepted, iter_accept + 1, iter_accept)

    kap_after_acc = torch.minimum(kap * 1.1, torch.full_like(kap, kappa_0))
    periodic_halve = accepted & (iter_accept > 0) & (iter_accept % 5 == 0)
    kap_after_acc = torch.where(periodic_halve, kap_after_acc / 2.0, kap_after_acc)
    kap_after_rej = kap / 2.0
    kap = torch.where(accepted, kap_after_acc, kap)
    kap = torch.where(rejected, kap_after_rej, kap)

    done = done | conv_now | below_floor
    return a, lp, dlp, kap, iter_accept, iter_count, done


# Chunked CUDA-graph replay for the fitDisp NR loop.
#
# A fixed full-length capture is a poor trade: eager early-exits once every gene
# converges (typically ~15-40 iters), but a single graph must run all `maxit`
# iterations, so the extra iterations outweigh the launch-overhead savings.
#
# Instead we capture a graph of just GRAPH_CHUNK iterations that reads and writes
# a set of persistent state buffers in place, then replay it in a loop, checking
# convergence between chunks. That keeps the iteration count within one chunk of
# the eager count while collapsing ~6k per-call launches into a handful.
#
# GRAPH_CHUNK must divide `maxit` so the total iteration count is capped at
# exactly `maxit` — this makes the result bit-identical to the eager loop
# (done genes are frozen no-ops, so running to a chunk boundary changes nothing).
GRAPH_CHUNK = 10

# Cache key: (G, S, P, chunk, has_prior, dtype, device-index).
_GRAPH_CACHE: dict = {}


def _build_chunked_graph(key, counts, mu, design, prior_mean, prior_sig_t,
                         *, chunk, log_lo, log_hi, eps, dispTol, min_log_alpha,
                         kappa_0):
    """Allocate persistent input+state buffers and capture one `chunk`-iteration
    step. The captured region reads the state buffers, runs `chunk` NR iterations,
    and writes the final state back into the same buffers, so repeated replays
    advance the fit `chunk` iterations at a time."""
    G = counts.shape[0]
    dtype, device = counts.dtype, counts.device
    s_counts = counts.clone()
    s_mu = mu.clone()
    s_design = design.clone()
    s_pmean = prior_mean.clone() if prior_mean is not None else None
    s_psig = prior_sig_t.clone() if prior_sig_t is not None else None
    # Persistent state (re-initialized eagerly each fit before the replay loop).
    s_a = torch.zeros(G, dtype=dtype, device=device)
    s_lp = torch.zeros(G, dtype=dtype, device=device)
    s_dlp = torch.zeros(G, dtype=dtype, device=device)
    s_kap = torch.zeros(G, dtype=dtype, device=device)
    s_ia = torch.zeros(G, dtype=torch.long, device=device)
    s_ic = torch.zeros(G, dtype=torch.long, device=device)
    s_done = torch.zeros(G, dtype=torch.bool, device=device)

    def _chunk():
        a, lp, dlp, kap = s_a, s_lp, s_dlp, s_kap
        ia, ic, done = s_ia, s_ic, s_done
        for _ in range(chunk):
            a, lp, dlp, kap, ia, ic, done = _nr_step(
                s_counts, s_mu, s_design, s_pmean, s_psig, a, lp, dlp, kap,
                ia, ic, done, eps=eps, log_lo=log_lo, log_hi=log_hi,
                dispTol=dispTol, min_log_alpha=min_log_alpha, kappa_0=kappa_0)
        # Write final state back into the persistent buffers.
        s_a.copy_(a); s_lp.copy_(lp); s_dlp.copy_(dlp); s_kap.copy_(kap)
        s_ia.copy_(ia); s_ic.copy_(ic); s_done.copy_(done)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _chunk()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _chunk()

    entry = dict(graph=graph, counts=s_counts, mu=s_mu, design=s_design,
                 pmean=s_pmean, psig=s_psig, a=s_a, lp=s_lp, dlp=s_dlp,
                 kap=s_kap, ia=s_ia, ic=s_ic, done=s_done)
    _GRAPH_CACHE[key] = entry
    return entry


def _run_captured_nr(counts, mu, design, alpha_init_clip, prior_mean,
                     prior_sig_t, *, maxit, log_lo, log_hi, eps, dispTol,
                     min_log_alpha, kappa_0):
    """Run the fitDisp NR loop via chunked CUDA-graph replay.

    Returns (a, lp, iter_count, initial_lp) matching the eager path bit-for-bit.
    """
    chunk = GRAPH_CHUNK
    if maxit % chunk != 0:
        raise ValueError(f"GRAPH_CHUNK ({chunk}) must divide maxit ({maxit})")
    G, S = counts.shape
    P = design.shape[1]
    key = (G, S, P, chunk, prior_mean is not None,
           str(counts.dtype), counts.device.index)
    entry = _GRAPH_CACHE.get(key)
    if entry is None:
        entry = _build_chunked_graph(
            key, counts, mu, design, prior_mean, prior_sig_t, chunk=chunk,
            log_lo=log_lo, log_hi=log_hi, eps=eps, dispTol=dispTol,
            min_log_alpha=min_log_alpha, kappa_0=kappa_0)

    # Refresh input buffers.
    entry["counts"].copy_(counts)
    entry["mu"].copy_(mu)
    entry["design"].copy_(design)
    if prior_mean is not None:
        entry["pmean"].copy_(prior_mean)
        entry["psig"].copy_(prior_sig_t)

    # Initialize state eagerly (same as the eager loop's pre-loop state).
    a0 = torch.log(alpha_init_clip).clamp(log_lo, log_hi)
    lp0, dlp0 = _lp_and_dlp(entry["counts"], entry["mu"], entry["design"], a0,
                            log_alpha_prior_mean=entry["pmean"],
                            log_alpha_prior_sigmasq=entry["psig"])
    entry["a"].copy_(a0)
    entry["lp"].copy_(lp0)
    entry["dlp"].copy_(dlp0)
    entry["kap"].fill_(kappa_0)
    entry["ia"].zero_()
    entry["ic"].zero_()
    entry["done"].fill_(False)
    initial_lp = lp0.clone()

    # Replay chunk-by-chunk, stopping as soon as all genes have converged.
    for _ in range(maxit // chunk):
        entry["graph"].replay()
        if not (~entry["done"]).any():
            break

    return (entry["a"].clone(), entry["lp"].clone(),
            entry["ic"].clone(), initial_lp)


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
    use_cuda_graph: bool = False,
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

    step_scalars = dict(eps=eps, log_lo=log_lo, log_hi=log_hi, dispTol=dispTol,
                        min_log_alpha=min_log_alpha, kappa_0=kappa_0)

    # prior_sigmasq is a 0-dim tensor in BOTH paths: the captured graph needs it
    # as a buffer (so one graph serves datasets with different prior variance),
    # and the eager path must use the same operand type or the two paths stop
    # being bit-identical. `t / python_float` lowers to a reciprocal-multiply
    # while `t / 0-dim tensor` is a true divide, and they differ by 1 ulp on ~40%
    # of elements — amplified over the MAP loop's iterations to ~1e-6 relative.
    # True division is also what R's C++ does.
    prior_sig_t = (torch.as_tensor(log_alpha_prior_sigmasq, dtype=dtype, device=device)
                   if log_alpha_prior_sigmasq is not None else None)

    if use_cuda_graph and counts.is_cuda:
        a, lp, iter_count, initial_lp = _run_captured_nr(
            counts, mu, design, alpha_init_clip, log_alpha_prior_mean,
            prior_sig_t, maxit=maxit, **step_scalars)
    else:
        # Eager path with early exit — bit-for-bit the reference loop.
        prior_mean = log_alpha_prior_mean
        prior_sig = prior_sig_t
        a = torch.log(alpha_init_clip).clamp(log_lo, log_hi).clone()
        lp, dlp = _lp_and_dlp(counts, mu, design, a,
                              log_alpha_prior_mean=prior_mean,
                              log_alpha_prior_sigmasq=prior_sig)
        initial_lp = lp.clone()
        G = counts.shape[0]
        kap = torch.full((G,), kappa_0, dtype=dtype, device=device)
        iter_accept = torch.zeros(G, dtype=torch.long, device=device)
        iter_count = torch.zeros(G, dtype=torch.long, device=device)
        done = torch.zeros(G, dtype=torch.bool, device=device)
        for _ in range(maxit):
            if not (~done).any():
                break
            a, lp, dlp, kap, iter_accept, iter_count, done = _nr_step(
                counts, mu, design, prior_mean, prior_sig, a, lp, dlp, kap,
                iter_accept, iter_count, done, **step_scalars)

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
    use_cuda_graph: bool = False,
    use_triton: bool = False,
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
        if use_triton:
            from . import _triton_fit as _tri
            if counts.is_cuda and _tri.supports_p(design.shape[1]):
                return _tri.fit_alpha_mle_triton(
                    counts, mu, design, alpha_init,
                    min_disp=min_disp, max_disp=max_disp, grid_length=grid_length)
        return _r_fit_alpha_mle(
            counts, mu, design, alpha_init,
            min_disp=min_disp, max_disp=max_disp, grid_length=grid_length,
            use_cuda_graph=use_cuda_graph,
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
    use_cuda_graph: bool = False,
    use_triton: bool = False,
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
        if use_triton:
            from . import _triton_fit as _tri
            if counts.is_cuda and _tri.supports_p(design.shape[1]):
                return _tri.fit_alpha_map_triton(
                    counts, mu, design, alpha_hat, prior_disp_var, alpha_init,
                    min_disp=min_disp, max_disp=max_disp)
        return _r_fit_alpha_mle(
            counts, mu, design, alpha_init=alpha_init,
            min_disp=min_disp, max_disp=max_disp, grid_length=grid_length,
            log_alpha_prior_mean=torch.log(alpha_hat),
            log_alpha_prior_sigmasq=float(prior_disp_var),
            apply_no_increase_revert=False,
            apply_grid_fallback=False,
            use_cuda_graph=use_cuda_graph,
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

    Matches DESeq2's use of a normal-consistent MAD for the dispersion prior.
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
    """Fit the parametric trend α ≈ a0 + a1/μ̄, matching R DESeq2's
    `parametricDispersionFit` bit-for-bit. Falls back to a trimmed-mean trend if
    the parametric fit fails. Returns (trend_per_gene, trend_type).

    R's algorithm (DESeq2 core.R):
      * fit only on genes with dispGeneEst > 100*minDisp (`useForFit`);
      * start coefs = (0.1, 1); each iteration recompute residuals
        disp/(a0 + a1/μ̄) over the *full* fit set, keep those in (1e-4, 15),
        and refit `glm(disp ~ I(1/μ̄), family=Gamma(link="identity"))`
        warm-started at the current coefs;
      * converge when Σ log(coef/oldcoef)² < 1e-6 and the GLM converged, or
        after 10 iterations.
    The earlier port used L-BFGS-B, a progressively-shrunk fit set, and no
    `useForFit` filter, which agreed with R only at high residual d.o.f. (where
    the MAP barely uses the trend); at low d.o.f. the MAP leans on the trend and
    the discrepancy surfaced.
    """
    means = np.asarray(normed_means, dtype=np.float64)
    disps = np.asarray(genewise_disp, dtype=np.float64)
    # R: useForFit <- dispGeneEst > 100*minDisp
    use = np.isfinite(means) & np.isfinite(disps) & (means > 0) & (disps > 100 * MIN_DISP)

    coefs = np.array([0.1, 1.0])
    success = use.sum() >= 2
    if success:
        m_fit = means[use]
        d_fit = disps[use]
        for _ in range(10):  # R breaks after iter > 10
            resid = d_fit / (coefs[0] + coefs[1] / m_fit)
            good = (resid > 1e-4) & (resid < 15)
            if good.sum() < 2:
                success = False
                break
            oldcoefs = coefs
            coefs, converged = _gamma_identity_glm(1.0 / m_fit[good], d_fit[good], coefs)
            if not np.all(coefs > 0):
                success = False
                break
            if (np.sum(np.log(coefs / oldcoefs) ** 2) < 1e-6) and converged:
                break

    if not success:
        # Mean-based fallback (DESeq2 fitType fallback).
        from scipy.stats import trim_mean
        keep = genewise_disp > 10 * MIN_DISP
        if not keep.any():
            mean_disp = float(genewise_disp.mean())
        else:
            mean_disp = float(trim_mean(genewise_disp[keep], proportiontocut=0.001))
        trend = np.full_like(genewise_disp, mean_disp)
        return trend, "mean"

    trend = coefs[0] + coefs[1] / means
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


def _gamma_identity_glm(x: np.ndarray, y: np.ndarray, start: np.ndarray,
                        maxit: int = 25, eps: float = 1e-8):
    """Gamma GLM with identity link: E[y] = b0 + b1*x. Mirrors R's
    `glm(y ~ x, family=Gamma(link="identity"), start=...)` IRLS.

    Identity link + Gamma variance V(μ)=μ² give working weights w=1/μ² and
    working response z=y, i.e. each step is a weighted least squares of y on
    [1, x] with weights 1/μ². Deviance-based convergence, with step-halving to
    keep μ>0 (as R's glm.fit does). Returns (coefs, converged).
    """
    X = np.column_stack([np.ones_like(x), x])
    beta = np.asarray(start, dtype=np.float64).copy()

    def gamma_dev(mu):
        return -2.0 * np.sum(np.log(y / mu) - (y - mu) / mu)

    mu = X @ beta
    if np.any(mu <= 0) or not np.isfinite(gamma_dev(np.maximum(mu, 1e-300))):
        beta = np.array([max(float(np.mean(y)), 1e-6), 0.0])
        mu = X @ beta
    dev = gamma_dev(np.maximum(mu, 1e-300))
    converged = False
    for _ in range(maxit):
        w = 1.0 / (mu * mu)
        XtW = X.T * w
        try:
            beta_new = np.linalg.solve(XtW @ X, XtW @ y)
        except np.linalg.LinAlgError:
            break
        # step-halve until μ>0 and deviance is finite (R glm.fit behaviour)
        t = 1.0
        accepted = False
        for _h in range(30):
            b_try = beta + t * (beta_new - beta)
            mu_try = X @ b_try
            if np.all(mu_try > 0):
                d_try = gamma_dev(mu_try)
                if np.isfinite(d_try):
                    accepted = True
                    break
            t *= 0.5
        if not accepted:
            break
        beta, mu = b_try, mu_try
        if abs(d_try - dev) / (abs(d_try) + 0.1) < eps:
            dev = d_try
            converged = True
            break
        dev = d_try
    return beta, converged


# ---------------------------------------------------------------------------
# Prior variance and outlier rule
# ---------------------------------------------------------------------------


def _loess_kd_cuts(xs: np.ndarray, fc: int) -> list[float]:
    """Split points of R's loess kd-tree (`loessf.f` ehg124): recursive median
    split of the sorted predictor while a cell holds more than `fc` points. The
    cut is the order statistic at m = (ll+uu)//2, and the children are
    [ll, m] and [m+1, uu]."""
    cuts: list[float] = []

    def split(ll: int, uu: int) -> None:          # 1-indexed, inclusive
        if uu - ll + 1 <= fc:
            return
        m = (ll + uu) // 2
        cuts.append(float(xs[m - 1]))
        split(ll, m)
        split(m + 1, uu)

    split(1, xs.size)
    return cuts


def _loess_vertex_fit(
    x: np.ndarray, y: np.ndarray, x0: float, q: int
) -> tuple[float, float]:
    """Tricube-weighted local quadratic at `x0`, returning (value, slope)."""
    d = np.abs(x - x0)
    h = np.partition(d, q - 1)[q - 1]             # distance to the q-th nearest
    if h <= 0.0:
        return float(y[int(np.argmin(d))]), 0.0
    u = d / h
    w = np.where(u < 1.0, (1.0 - u**3) ** 3, 0.0)
    sel = w > 0.0
    dx = x[sel] - x0
    basis = np.stack([np.ones(dx.size), dx, dx * dx], axis=1)
    btw = basis.T * w[sel]
    coef = np.linalg.solve(btw @ basis, btw @ y[sel])
    return float(coef[0]), float(coef[1])


def _loess_quadratic(
    x: np.ndarray, y: np.ndarray, xout: np.ndarray,
    span: float = 0.2, cell: float = 0.2,
) -> np.ndarray:
    """R's `loess(y ~ x, span=span, degree=2)` evaluated at `xout`, including its
    default `surface="interpolate"`.

    R does not fit the local regression at every output point. It builds a
    kd-tree over the predictor, fits only at the tree's vertices -- taking the
    value *and* the slope there -- and cubic-Hermite interpolates in between.
    That approximation is part of what DESeq2 computes, so reproducing it is
    required, not optional: on the airway KL curve the exact ("direct") surface
    and this one pick argmins two grid points apart. Agrees with R to ~1e-14.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xout = np.asarray(xout, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    n = xs.size
    q = int(np.floor(span * n))
    fc = int(np.floor(n * span * cell))

    # Bounding box, expanded by 0.5% of the range at each end (ehg126).
    spread = xs[-1] - xs[0]
    box = [xs[0] - 0.005 * spread, xs[-1] + 0.005 * spread]
    verts = np.unique(np.array(box + _loess_kd_cuts(xs, fc), dtype=np.float64))

    vals = np.empty(verts.size, dtype=np.float64)
    slopes = np.empty(verts.size, dtype=np.float64)
    for i, v in enumerate(verts):
        vals[i], slopes[i] = _loess_vertex_fit(xs, ys, float(v), q)

    # Cubic Hermite inside the leaf cell bracketing each output point (ehg128).
    idx = np.clip(np.searchsorted(verts, xout, side="right") - 1, 0, verts.size - 2)
    step = verts[idx + 1] - verts[idx]
    h = (xout - verts[idx]) / step
    phi0 = (1.0 - h) ** 2 * (1.0 + 2.0 * h)
    phi1 = h**2 * (3.0 - 2.0 * h)
    psi0 = h * (1.0 - h) ** 2
    psi1 = h**2 * (h - 1.0)
    return (phi0 * vals[idx] + phi1 * vals[idx + 1]
            + (psi0 * slopes[idx] + psi1 * slopes[idx + 1]) * step)


def _prior_var_kl_grid(residuals_above: np.ndarray, m: int, p: int) -> float:
    """Small-residual-dof prior-variance estimator, matching R DESeq2's
    `estimateDispersionsPriorVar` branch for `(m - p) <= 3`.

    For a grid of candidate prior variances x, simulate the theoretical
    log-dispersion residual distribution log(chisq_{m-p}) + N(0, sqrt(x)) -
    log(m-p), histogram it, and pick the x minimizing the KL divergence from the
    observed residual histogram (loess-smoothed). Returns pmax(argmin, 0.25).

    Both stochastic ingredients are R's, not approximations of R's. The draws
    replay R's `set.seed(2)` Mersenne-Twister stream bit-for-bit (see
    `_r_rng`), and the KL curve is smoothed with R's loess including its default
    kd-tree `surface="interpolate"` (see `_loess_quadratic`). Substituting a
    different generator or a different smoother both move the reported argmin:
    on airway a PCG64 draw shifts it by 0.016 and a Savitzky-Golay smoother by
    0.080, against R's 0.5285285285285285, which this reproduces bit for bit.
    """
    brks = np.arange(-20, 21) / 2.0                       # R: -20:20/2
    lo, hi = brks[0], brks[-1]
    obs = residuals_above[(residuals_above > lo) & (residuals_above < hi)]
    obs_hist, _ = np.histogram(obs, bins=brks, density=True)
    var_grid = np.linspace(0.0, 8.0, 200)
    kl = np.empty_like(var_grid)
    dof = m - p
    draws = _r_rng.kl_grid_draws(dof, var_grid, n_samp=10000, seed=2)
    for i in range(var_grid.size):
        rand = draws[i]
        rand = rand[(rand > lo) & (rand < hi)]
        rand_hist, _ = np.histogram(rand, bins=brks, density=True)
        z = np.concatenate([obs_hist, rand_hist])
        small = z[z > 0].min()
        kl[i] = np.sum(obs_hist * (np.log(obs_hist + small) - np.log(rand_hist + small)))
    fine = np.linspace(0.0, 8.0, 1000)
    fitted = _loess_quadratic(var_grid, kl, fine, span=0.2)
    argmin_kl = float(fine[int(np.argmin(fitted))])
    return max(argmin_kl, 0.25)


def compute_prior_disp_var(
    genewise_disp: np.ndarray,
    fitted_disp: np.ndarray,
    n_samples: int,
    n_vars: int,
    min_disp: float = MIN_DISP,
) -> tuple[float, float]:
    """Return (prior_disp_var, squared_logres), matching R DESeq2's
    `estimateDispersionsPriorVar`.

    Two regimes on residual dof (m - p): for (m - p) <= 3 R uses a KL-divergence
    grid search (see `_prior_var_kl_grid`); otherwise the closed-form
    max(varLogDispEsts - trigamma((m-p)/2), 0.25). `squared_logres`
    (= varLogDispEsts) is returned for the downstream outlier rule either way.
    """
    m, p = n_samples, n_vars
    residuals = np.log(genewise_disp) - np.log(fitted_disp)
    above = genewise_disp >= (100 * min_disp)
    resid_above = residuals[above & np.isfinite(residuals)]
    squared_logres = _mad(resid_above) ** 2 if resid_above.size else 0.0

    if m <= p:
        prior_var = squared_logres
    elif (m - p) <= 3:
        prior_var = _prior_var_kl_grid(resid_above, m, p)
    else:
        prior_var = max(squared_logres - polygamma(1, (m - p) / 2.0), 0.25)
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
    """Batched IRLS NB-GLM fit, matching DESeq2's per-gene GLM solve.

    Returns:
        beta: (G, P) fitted coefficients.
        mu:   (G, S) UN-thresholded μ = size_factors * exp(X @ β) (per line 437).
        H:    (G, S) diagonal of W^{1/2} X (X^T W X + ridge I)^{-1} X^T W^{1/2}.
        converged: (G,) bool.

    Genes that don't converge in `max_iter` or blow past |β|>max_beta are
    dispatched to a CPU fallback (per-gene scipy L-BFGS-B) so the final values
    match DESeq2's behavior on hard genes.
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
        for g in bad_idx:
            b_init = beta_init[g].detach().cpu().numpy()
            def f(b, g=g):
                mu_ = np.maximum(size_factors_np * np.exp(design_np @ b), min_mu)
                return _nb_nll_np(counts_np[g], mu_, dispersions_np[g]) + 0.5 * (ridge_mat @ b**2).sum()
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
    """Per-gene NLL (returns (G,)) with scalar per-gene alpha (DESeq2 nbinomLogLike)."""
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


def _nb_nll_np(counts: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    """Scalar-alpha NB negative log-likelihood for a single gene (numpy).

    Used by the per-gene CPU IRLS fallback. Same closed form as the batched
    torch NLL kernels above, kept in numpy so the fallback stays dependency-free.
    """
    n = len(counts)
    alpha_inv = 1.0 / alpha
    logbinom = gammaln(counts + alpha_inv) - gammaln(counts + 1.0) - gammaln(alpha_inv)
    return float(
        n * alpha_inv * np.log(alpha)
        + (
            -logbinom
            + (counts + alpha_inv) * np.log(alpha_inv + mu)
            - counts * np.log(mu)
        ).sum()
    )
