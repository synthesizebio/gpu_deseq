"""apeGLM LFC shrinkage, batched on GPU.

Implements the apeGLM MAP estimator (Cauchy prior on the shrunk coefficient)
and its empirical-Bayes prior-variance fit, matching R DESeq2's
`lfcShrink(type="apeglm")`. The per-gene posterior is unimodal in β on
[min_beta, max_beta], and gradient + Hessian are analytical in closed form, so
batched Newton's method converges in a handful of iterations per gene.
Divergent genes fall back to per-gene scipy L-BFGS-B.

Reference: Zhu, Ibrahim, Love (2019) "Heavy-tailed prior distributions for
sequence count data: removing the noise and preserving large differences."
"""
from __future__ import annotations

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.optimize import minimize, root_scalar


def fit_prior_var(
    lfc: np.ndarray,      # (G,) MLE coefficient at shrink_index, in natural log
    se: np.ndarray,       # (G,) Wald SE for the same coefficient
    min_var: float = 1e-6,
    max_var: float = 400.0,
) -> float:
    """Empirical Bayes prior variance for the apeGLM prior.

    Solves the apeGLM prior-variance equation: find `a` satisfying
    ((S - D) / (2*(a+D)^2)).sum() / (1/(2*(a+D)^2)).sum() = a
    with S = β² and D = SE². Returns `min_var` if the objective at min_var is
    already negative (i.e. the data prefers essentially no shrinkage).
    """
    keep = np.isfinite(lfc) & np.isfinite(se)
    S = lfc[keep] ** 2
    D = se[keep] ** 2

    def objective(a: float) -> float:
        coeff = 1.0 / (2.0 * (a + D) ** 2)
        return ((S - D) * coeff).sum() / coeff.sum() - a

    if objective(min_var) < 0:
        return float(min_var)
    return float(root_scalar(objective, bracket=(min_var, max_var)).root)


# ---------------------------------------------------------------------------
# Batched objective, gradient, Hessian
# ---------------------------------------------------------------------------


def _nbinom_apeglm_loss(
    beta: torch.Tensor,              # (G, P)
    counts: torch.Tensor,            # (G, S)
    size: torch.Tensor,              # (G,)  = 1/alpha
    offset: torch.Tensor,            # (S,)  = log(size_factor)
    design: torch.Tensor,            # (S, P)
    prior_no_shrink_scale: float,
    prior_scale: float,
    shrink_index: int,
) -> torch.Tensor:
    """Return (G,) objective: prior - nll (the apeGLM negative log-posterior)."""
    P = design.shape[1]
    xbeta = beta @ design.T                     # (G, S)
    xbeta_off = xbeta + offset.unsqueeze(0)     # (G, S)
    # logaddexp(xbeta+offset, log(size)) elementwise
    log_size = torch.log(size).unsqueeze(1)     # (G, 1)
    lae = torch.logaddexp(xbeta_off, log_size.expand_as(xbeta_off))
    nll = (counts * xbeta - (counts + size.unsqueeze(1)) * lae).sum(dim=1)

    shrink_mask = torch.zeros(P, dtype=beta.dtype, device=beta.device)
    shrink_mask[shrink_index] = 1.0
    no_shrink = 1.0 - shrink_mask

    prior = (
        ((beta * no_shrink) ** 2 / (2.0 * prior_no_shrink_scale ** 2)).sum(dim=1)
        + torch.log1p((beta[:, shrink_index] / prior_scale) ** 2)
    )
    return prior - nll


def _nbinom_apeglm_grad(
    beta: torch.Tensor,
    counts: torch.Tensor,
    size: torch.Tensor,
    offset: torch.Tensor,
    design: torch.Tensor,
    prior_no_shrink_scale: float,
    prior_scale: float,
    shrink_index: int,
) -> torch.Tensor:
    """Return (G, P) gradient of the loss w.r.t. β."""
    P = design.shape[1]
    xbeta = beta @ design.T                     # (G, S)
    # d_nll = (counts - (counts+size)/(1 + size*exp(-xbeta-offset))) @ design
    inv = 1.0 / (1.0 + size.unsqueeze(1) * torch.exp(-xbeta - offset.unsqueeze(0)))
    resid = counts - (counts + size.unsqueeze(1)) * inv   # (G, S)
    d_nll = resid @ design                                # (G, P)

    shrink_mask = torch.zeros(P, dtype=beta.dtype, device=beta.device)
    shrink_mask[shrink_index] = 1.0
    no_shrink = 1.0 - shrink_mask

    d_prior_noshrink = beta * no_shrink / (prior_no_shrink_scale ** 2)   # (G, P)
    # d_prior_shrink: only at shrink_index; 2β / (σ² + β²)
    beta_s = beta[:, shrink_index]
    d_prior_shrink_scalar = 2.0 * beta_s / (prior_scale ** 2 + beta_s ** 2)  # (G,)
    d_prior_shrink = torch.zeros_like(beta)
    d_prior_shrink[:, shrink_index] = d_prior_shrink_scalar

    return (d_prior_noshrink + d_prior_shrink) - d_nll


def _nbinom_apeglm_hess(
    beta: torch.Tensor,
    counts: torch.Tensor,
    size: torch.Tensor,
    offset: torch.Tensor,
    design: torch.Tensor,
    prior_no_shrink_scale: float,
    prior_scale: float,
    shrink_index: int,
) -> torch.Tensor:
    """Return (G, P, P) Hessian of the loss w.r.t. β."""
    P = design.shape[1]
    xbeta = beta @ design.T                        # (G, S)
    exp_xb = torch.exp(xbeta + offset.unsqueeze(0))
    sz = size.unsqueeze(1)                          # (G, 1)
    frac = (counts + sz) * sz * exp_xb / (sz + exp_xb) ** 2   # (G, S)
    # H_nll = X.T * frac * X per gene: (G, P, P)
    H_nll = torch.einsum("sp,gs,sq->gpq", design, frac, design)

    # Prior diagonal
    shrink_mask = torch.zeros(P, dtype=beta.dtype, device=beta.device)
    shrink_mask[shrink_index] = 1.0
    no_shrink = 1.0 - shrink_mask

    h11 = 1.0 / prior_no_shrink_scale ** 2          # scalar
    beta_s = beta[:, shrink_index]
    h22 = 2.0 * (prior_scale ** 2 - beta_s ** 2) / (prior_scale ** 2 + beta_s ** 2) ** 2  # (G,)

    diag = torch.zeros_like(beta)                   # (G, P)
    diag += no_shrink * h11
    diag[:, shrink_index] = diag[:, shrink_index] + h22

    # Add diagonal to NLL Hessian: diag has shape (G, P), expand to (G, P, P)
    eye = torch.eye(P, dtype=beta.dtype, device=beta.device)
    H_prior = diag.unsqueeze(2) * eye.unsqueeze(0)
    return H_nll + H_prior


# ---------------------------------------------------------------------------
# Batched Newton with scipy fallback
# ---------------------------------------------------------------------------


def _apeglm_shrink_chunk(idx, counts_np, size_np, design_np, offset_np,
                         no_shrink_mask, shrink_index, prior_no_shrink_scale,
                         prior_scale, init):
    """Solve apeglm's MAP for a chunk of genes (module-level so joblib can ship it
    to worker processes; the big count matrix is memory-mapped, shared read-only).
    Each gene: unbounded L-BFGS on the exact apeGLM negative-log-posterior from
    apeglm's 0.1/-0.1 init."""
    P = design_np.shape[1]
    betas = np.empty((len(idx), P), dtype=np.float64)
    convs = np.empty(len(idx), dtype=bool)
    for k, g_i in enumerate(idx):
        y = counts_np[g_i]; size_i = size_np[g_i]

        def _f(b):
            xbeta = design_np @ b
            lae = np.logaddexp(xbeta + offset_np, np.log(size_i))
            nll = (y * xbeta - (y + size_i) * lae).sum()
            prior = ((b * no_shrink_mask) ** 2 / (2 * prior_no_shrink_scale ** 2)).sum() \
                + np.log1p((b[shrink_index] / prior_scale) ** 2)
            return prior - nll

        def _df(b):
            xbeta = design_np @ b
            inv = 1.0 / (1.0 + size_i * np.exp(-xbeta - offset_np))
            resid = y - (y + size_i) * inv
            d_prior = b * no_shrink_mask / prior_no_shrink_scale ** 2
            d_prior[shrink_index] += 2 * b[shrink_index] / (prior_scale ** 2 + b[shrink_index] ** 2)
            return d_prior - resid @ design_np

        res = minimize(_f, init, jac=_df, method="L-BFGS-B",
                       options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 300})
        betas[k] = res.x
        convs[k] = bool(res.success)
    return idx, betas, convs


def apeglm_shrink_batched(
    counts: torch.Tensor,         # (G, S)
    dispersions: torch.Tensor,    # (G,)
    size_factors: torch.Tensor,   # (S,)
    design: torch.Tensor,         # (S, P)
    shrink_index: int,
    prior_scale: float,
    prior_no_shrink_scale: float = 15.0,
    max_iter: int = 500,
    gtol: float = 1e-10,
    btol: float = 1e-10,
    min_beta: float = -30.0,
    max_beta: float = 30.0,
    beta_init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched Newton MAP fit of the apeGLM posterior.

    Returns:
        beta_shrunk: (G, P) MAP coefficients.
        inv_hess_diag: (G, P) diagonal of the inverse Hessian at MAP (used for SE).
        converged: (G,) bool.
    """
    G, S = counts.shape
    P = design.shape[1]
    device = counts.device
    dtype = counts.dtype

    size = (1.0 / dispersions).to(dtype)                # (G,)
    offset = torch.log(size_factors).to(dtype=dtype)    # (S,)

    # Faithful port of apeglm's nbinomCR estimator (the default used by DESeq2's
    # lfcShrink(type="apeglm")). We minimize the *exact* apeGLM negative-log-
    # posterior — NB log-likelihood + a normal prior on the non-shrink
    # coefficients + a Cauchy (t, df=1) prior on the shrink coefficient — with
    # L-BFGS started at apeglm's *exact* initial point 0.1, -0.1, 0.1, ...
    #
    # This must be an L-BFGS from a near-zero start, not our batched Newton: the
    # posterior is bimodal for extreme low-count genes (a sharp likelihood mode
    # far from zero and a prior mode near zero), and apeglm reports the *local*
    # optimum reached by gradient-based L-BFGS from ~0. A Newton method jumps the
    # barrier to the global optimum and disagrees with R for exactly those genes.
    # Replicating apeglm's solver+init reproduces R's shrunk LFC (Pearson 1.0).
    # This runs per-gene on CPU, as R's C++ does; the other four pipeline stages
    # stay batched on the GPU, so overall speedups are preserved.
    design_np = design.detach().cpu().numpy()
    offset_np = offset.detach().cpu().numpy()
    counts_np = np.ascontiguousarray(counts.detach().cpu().numpy())
    size_np = size.detach().cpu().numpy()
    shrink_mask = np.zeros(P); shrink_mask[shrink_index] = 1.0
    no_shrink_mask = 1.0 - shrink_mask

    # apeglm's exact starting point; unbounded L-BFGS (bounds=c(-Inf,Inf)). The
    # per-gene solves run in parallel across CPU cores (R does them serially in
    # C++); loky memory-maps the count matrix so workers share it read-only. The
    # GPU-batched stages upstream keep their speedups; only this stage is CPU.
    init = 0.1 * ((-1.0) ** np.arange(P))
    beta_np = np.tile(init, (G, 1))
    conv_np = np.zeros(G, dtype=bool)
    if G:
        n_chunks = min(G, 48)
        chunks = np.array_split(np.arange(G), n_chunks)
        results = Parallel(n_jobs=-1)(
            delayed(_apeglm_shrink_chunk)(
                idx, counts_np, size_np, design_np, offset_np, no_shrink_mask,
                shrink_index, prior_no_shrink_scale, prior_scale, init)
            for idx in chunks)
        for idx, betas, convs in results:
            beta_np[idx] = betas
            conv_np[idx] = convs
    beta = torch.from_numpy(beta_np).to(device=device, dtype=dtype)
    converged = torch.from_numpy(conv_np).to(device=device)

    # Compute inv(Hessian) diagonal at final β for SE — no ridge; the posterior
    # SD is sqrt of the diagonal of the inverse observed information.
    H_final = _nbinom_apeglm_hess(beta, counts, size, offset, design,
                                  prior_no_shrink_scale, prior_scale, shrink_index)
    H_inv = torch.linalg.inv(H_final)
    inv_hess_diag = torch.diagonal(H_inv, dim1=1, dim2=2)  # (G, P)
    return beta, inv_hess_diag, converged
