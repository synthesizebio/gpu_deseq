"""apeGLM LFC shrinkage, batched on GPU.

Ports pydeseq2's `nbinomGLM` / `nbinomFn` / `_fit_prior_var` from utils.py /
ds.py. The per-gene posterior is unimodal in β on [min_beta, max_beta], and
gradient + Hessian are analytical in closed form, so batched Newton's method
converges in a handful of iterations per gene. Divergent genes fall back to
per-gene scipy L-BFGS-B to match pydeseq2 exactly.

Reference: Zhu, Ibrahim, Love (2019) "Heavy-tailed prior distributions for
sequence count data: removing the noise and preserving large differences."
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize, root_scalar


def fit_prior_var(
    lfc: np.ndarray,      # (G,) MLE coefficient at shrink_index, in natural log
    se: np.ndarray,       # (G,) Wald SE for the same coefficient
    min_var: float = 1e-6,
    max_var: float = 400.0,
) -> float:
    """Empirical Bayes prior variance for the apeGLM prior.

    Solves pydeseq2._fit_prior_var (ds.py:556-593): find `a` satisfying
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
    """Return (G,) objective: prior - nll (matches pydeseq2.nbinomFn)."""
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
    min_beta: float = -30.0,
    max_beta: float = 30.0,
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

    # pydeseq2's initial guess: ±0.1 alternating across coefficients.
    beta = 0.1 * ((-1.0) ** torch.arange(P, dtype=dtype, device=device))
    beta = beta.unsqueeze(0).expand(G, P).contiguous()

    converged = torch.zeros(G, dtype=torch.bool, device=device)
    active = torch.ones(G, dtype=torch.bool, device=device)
    ridge_p = 1e-6 * torch.eye(P, dtype=dtype, device=device)

    for _ in range(max_iter):
        g = _nbinom_apeglm_grad(beta, counts, size, offset, design,
                                prior_no_shrink_scale, prior_scale, shrink_index)
        done_new = (g.abs().max(dim=1).values < gtol) & active
        converged = converged | done_new
        active = active & ~done_new
        if not active.any():
            break

        H = _nbinom_apeglm_hess(beta, counts, size, offset, design,
                                prior_no_shrink_scale, prior_scale, shrink_index)
        # Damped Newton step with tiny ridge for numerical safety.
        step = torch.linalg.solve(H + ridge_p, g.unsqueeze(-1)).squeeze(-1)
        # Line search: simple step-halving if loss increased.
        cur_loss = _nbinom_apeglm_loss(beta, counts, size, offset, design,
                                       prior_no_shrink_scale, prior_scale, shrink_index)
        t = torch.ones(G, dtype=dtype, device=device)
        for _ls in range(10):
            trial = beta - t.unsqueeze(1) * step
            trial = trial.clamp(min_beta, max_beta)
            new_loss = _nbinom_apeglm_loss(trial, counts, size, offset, design,
                                           prior_no_shrink_scale, prior_scale, shrink_index)
            bad = active & (new_loss > cur_loss + 1e-12)
            if not bad.any():
                break
            t = torch.where(bad, t * 0.5, t)
        # Apply step only to active rows.
        update_mask = active.unsqueeze(1)
        beta = torch.where(update_mask, trial, beta)

    # Any still-active gene: fall back to scipy L-BFGS-B per gene.
    if active.any():
        bad_idx = torch.nonzero(active, as_tuple=False).squeeze(-1).cpu().numpy()
        design_np = design.detach().cpu().numpy()
        offset_np = offset.detach().cpu().numpy()
        counts_np = counts.detach().cpu().numpy()
        size_np = size.detach().cpu().numpy()
        beta_np = beta.detach().cpu().numpy()
        shrink_mask = np.zeros(P); shrink_mask[shrink_index] = 1.0
        no_shrink_mask = 1.0 - shrink_mask

        def _f(b, g_i):
            xbeta = design_np @ b
            lae = np.logaddexp(xbeta + offset_np, np.log(size_np[g_i]))
            nll = (counts_np[g_i] * xbeta - (counts_np[g_i] + size_np[g_i]) * lae).sum()
            prior = ((b * no_shrink_mask) ** 2 / (2 * prior_no_shrink_scale ** 2)).sum() \
                + np.log1p((b[shrink_index] / prior_scale) ** 2)
            return prior - nll

        def _df(b, g_i):
            xbeta = design_np @ b
            inv = 1.0 / (1.0 + size_np[g_i] * np.exp(-xbeta - offset_np))
            resid = counts_np[g_i] - (counts_np[g_i] + size_np[g_i]) * inv
            d_nll = resid @ design_np
            d_prior = b * no_shrink_mask / prior_no_shrink_scale ** 2
            d_prior[shrink_index] += 2 * b[shrink_index] / (prior_scale ** 2 + b[shrink_index] ** 2)
            return d_prior - d_nll

        for g_i in bad_idx:
            x0 = beta_np[g_i]
            res = minimize(
                _f, x0, args=(g_i,), jac=_df, method="L-BFGS-B",
                bounds=[(min_beta, max_beta)] * P,
                options={"ftol": 1e-8, "gtol": 1e-8},
            )
            if res.success:
                beta_np[g_i] = res.x
                converged[int(g_i)] = True
        beta = torch.from_numpy(beta_np).to(device=device, dtype=dtype)

    # Compute inv(Hessian) diagonal at final β for SE — no ridge, matching
    # pydeseq2's `inv_hessian = np.linalg.inv(ddf(beta, 1))` (utils.py:1133).
    H_final = _nbinom_apeglm_hess(beta, counts, size, offset, design,
                                  prior_no_shrink_scale, prior_scale, shrink_index)
    H_inv = torch.linalg.inv(H_final)
    inv_hess_diag = torch.diagonal(H_inv, dim1=1, dim2=2)  # (G, P)
    return beta, inv_hess_diag, converged
