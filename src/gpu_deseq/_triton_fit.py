"""Fused Triton implementation of the fitDisp Newton-Raphson loop.

One program per gene runs the *entire* gradient-ascent loop in registers
(counts/mu/design loaded once, all iterations fused, per-gene early exit via a
`while`). This removes the per-kernel launch/latency overhead that the roofline
analysis identified as the dispersion stage's bottleneck.

Numerics match R DESeq2 to ~1e-14 (same tolerance the eager path meets): digamma
is hand-implemented to match ATen's `calc_digamma` (arguments are always
positive here), lgamma/log1p come from libdevice, and the Cox-Reid P×P logdet /
trace(b⁻¹ db) use an unrolled no-pivot LU (b is SPD). NOT bit-identical to the
eager path (reduction order differs), but R parity — the actual bar — holds.

Only P ∈ {2, 4} are implemented (single-factor / continuous, and additive
multi-factor with a 3-level batch). Callers must fall back to the eager/graph
path for other P; `supports_p()` reports what's covered.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - triton optional
    TRITON_AVAILABLE = False

from . import _deseq2_core as core

_SUPPORTED_P = (2, 4)


def supports_p(p: int) -> bool:
    return TRITON_AVAILABLE and p in _SUPPORTED_P


if TRITON_AVAILABLE:

    @triton.jit
    def _digamma_pos(x):
        # digamma for x > 0 (matches ATen calc_digamma; no reflection needed).
        result = x - x
        for _ in range(12):  # push x >= 10 via recurrence
            m = x < 10.0
            result = result - tl.where(m, 1.0 / x, 0.0)
            x = x + tl.where(m, 1.0, 0.0)
        z = 1.0 / (x * x)
        p = 8.33333333333333333333e-2
        p = p * z + -2.10927960927960927961e-2
        p = p * z + 7.57575757575757575758e-3
        p = p * z + -4.16666666666666666667e-3
        p = p * z + 3.96825396825396825397e-3
        p = p * z + -8.33333333333333333333e-3
        p = p * z + 8.33333333333333333333e-2
        return result + tl.log(x) - 0.5 / x - z * p

    @triton.jit
    def _lpdlp(counts, mu, x0, x1, x2, x3, mask, a, pmean, sigmasq,
               HAS_PRIOR: tl.constexpr, P: tl.constexpr):
        alpha = tl.exp(a)
        ainv = 1.0 / alpha
        w = tl.where(mask, 1.0 / (1.0 / mu + alpha), 0.0)
        dw = -w * w

        # Cox-Reid term: logdet(b) and trace(b^-1 db), b = Xᵀ W X (SPD).
        if P == 2:
            b00 = tl.sum(w * x0 * x0); b01 = tl.sum(w * x0 * x1); b11 = tl.sum(w * x1 * x1)
            e00 = tl.sum(dw * x0 * x0); e01 = tl.sum(dw * x0 * x1); e11 = tl.sum(dw * x1 * x1)
            det = b00 * b11 - b01 * b01
            logdet = tl.log(det)
            tr = (b11 * e00 - 2.0 * b01 * e01 + b00 * e11) / det
        else:  # P == 4
            b00 = tl.sum(w * x0 * x0); b01 = tl.sum(w * x0 * x1); b02 = tl.sum(w * x0 * x2); b03 = tl.sum(w * x0 * x3)
            b11 = tl.sum(w * x1 * x1); b12 = tl.sum(w * x1 * x2); b13 = tl.sum(w * x1 * x3)
            b22 = tl.sum(w * x2 * x2); b23 = tl.sum(w * x2 * x3)
            b33 = tl.sum(w * x3 * x3)
            e00 = tl.sum(dw * x0 * x0); e01 = tl.sum(dw * x0 * x1); e02 = tl.sum(dw * x0 * x2); e03 = tl.sum(dw * x0 * x3)
            e11 = tl.sum(dw * x1 * x1); e12 = tl.sum(dw * x1 * x2); e13 = tl.sum(dw * x1 * x3)
            e22 = tl.sum(dw * x2 * x2); e23 = tl.sum(dw * x2 * x3)
            e33 = tl.sum(dw * x3 * x3)
            # no-pivot LU of the symmetric 4x4 b (b_ji = b_ij)
            u00 = b00
            l10 = b01 / u00; l20 = b02 / u00; l30 = b03 / u00
            m11 = b11 - l10 * b01; m12 = b12 - l10 * b02; m13 = b13 - l10 * b03
            m22 = b22 - l20 * b02; m23 = b23 - l20 * b03
            m32 = b23 - l30 * b02; m33 = b33 - l30 * b03
            m21 = b12 - l20 * b01; m31 = b13 - l30 * b01
            u11 = m11
            l21 = m21 / u11; l31 = m31 / u11
            n22 = m22 - l21 * m12; n23 = m23 - l21 * m13
            n32 = m32 - l31 * m12; n33 = m33 - l31 * m13
            u22 = n22
            l32 = n32 / u22
            u33 = n33 - l32 * n23
            logdet = tl.log(u00) + tl.log(u11) + tl.log(u22) + tl.log(u33)

            # trace(b^-1 db): solve b x = db[:,c], take x_c, for c=0..3.
            # column 0 rhs = (e00,e01,e02,e03)
            y0 = e00; y1 = e01 - l10 * y0; y2 = e02 - l20 * y0 - l21 * y1; y3 = e03 - l30 * y0 - l31 * y1 - l32 * y2
            x3_ = y3 / u33; x2_ = (y2 - n23 * x3_) / u22; x1_ = (y1 - m12 * x2_ - m13 * x3_) / u11
            t0 = (y0 - b01 * x1_ - b02 * x2_ - b03 * x3_) / u00
            # column 1 rhs = (e01,e11,e12,e13)
            y0 = e01; y1 = e11 - l10 * y0; y2 = e12 - l20 * y0 - l21 * y1; y3 = e13 - l30 * y0 - l31 * y1 - l32 * y2
            x3_ = y3 / u33; x2_ = (y2 - n23 * x3_) / u22
            t1 = (y1 - m12 * x2_ - m13 * x3_) / u11
            # column 2 rhs = (e02,e12,e22,e23)
            y0 = e02; y1 = e12 - l10 * y0; y2 = e22 - l20 * y0 - l21 * y1; y3 = e23 - l30 * y0 - l31 * y1 - l32 * y2
            x3_ = y3 / u33
            t2 = (y2 - n23 * x3_) / u22
            # column 3 rhs = (e03,e13,e23,e33)
            y0 = e03; y1 = e13 - l10 * y0; y2 = e23 - l20 * y0 - l21 * y1; y3 = e33 - l30 * y0 - l31 * y1 - l32 * y2
            t3 = y3 / u33
            tr = t0 + t1 + t2 + t3

        lg_ainv = libdevice.lgamma(ainv)
        dg_ainv = _digamma_pos(ainv)
        lg_c = libdevice.lgamma(counts + ainv)
        dg_c = _digamma_pos(counts + ainv)
        log1p_mua = libdevice.log1p(mu * alpha)
        ll_s = tl.where(mask, lg_c - lg_ainv - counts * tl.log(mu + ainv) - ainv * log1p_mua, 0.0)
        lp = tl.sum(ll_s) - 0.5 * logdet
        mua = mu * alpha
        br = tl.where(mask, dg_ainv + log1p_mua - mua / (1.0 + mua) - dg_c + counts / (mu + ainv), 0.0)
        dlp = (ainv * ainv * tl.sum(br) - 0.5 * tr) * alpha
        if HAS_PRIOR:
            diff = a - pmean
            lp = lp - 0.5 * diff * diff / sigmasq
            dlp = dlp - diff / sigmasq
        return lp, dlp

    @triton.jit
    def _fit_kernel(counts_ptr, mu_ptr, x_ptr, ainit_ptr, pmean_ptr, out_a_ptr, out_ic_ptr,
                    sigmasq, S: tl.constexpr, BLOCK_S: tl.constexpr, maxit: tl.constexpr,
                    log_lo: tl.constexpr, log_hi: tl.constexpr, eps: tl.constexpr,
                    dispTol: tl.constexpr, min_log_alpha: tl.constexpr, kappa_0: tl.constexpr,
                    HAS_PRIOR: tl.constexpr, P: tl.constexpr):
        g = tl.program_id(0)
        s = tl.arange(0, BLOCK_S)
        mask = s < S
        counts = tl.load(counts_ptr + g * S + s, mask=mask, other=0.0)
        mu = tl.load(mu_ptr + g * S + s, mask=mask, other=1.0)
        x0 = tl.load(x_ptr + s * P + 0, mask=mask, other=0.0)
        x1 = tl.load(x_ptr + s * P + 1, mask=mask, other=0.0)
        x2 = tl.load(x_ptr + s * P + 2, mask=mask, other=0.0) if P > 2 else s * 0.0
        x3 = tl.load(x_ptr + s * P + 3, mask=mask, other=0.0) if P > 3 else s * 0.0
        pmean = tl.load(pmean_ptr + g) if HAS_PRIOR else 0.0

        a = tl.load(ainit_ptr + g)
        a = tl.minimum(tl.maximum(tl.log(a), log_lo), log_hi)
        lp, dlp = _lpdlp(counts, mu, x0, x1, x2, x3, mask, a, pmean, sigmasq, HAS_PRIOR, P)

        zero = a - a
        kap = zero + kappa_0
        ia = zero
        ic = zero
        done = zero > 1.0
        it = 0
        while (it < maxit) & (~done):
            ic = ic + 1.0
            a_prop_naive = a + kap * dlp
            too_low = (a_prop_naive < log_lo) & (dlp != 0.0)
            kap = tl.where(too_low, (log_lo - a) / dlp, kap)
            too_high = (a_prop_naive > log_hi) & (dlp != 0.0)
            kap = tl.where(too_high, (log_hi - a) / dlp, kap)
            a_propose = a + kap * dlp
            lp_prop, _ = _lpdlp(counts, mu, x0, x1, x2, x3, mask, a_propose, pmean, sigmasq, HAS_PRIOR, P)
            theta_kap = -lp_prop
            theta_hat = -lp - kap * eps * dlp * dlp
            accepted = theta_kap <= theta_hat
            rejected = ~accepted
            a_new = tl.where(accepted, a_propose, a)
            lp_new = tl.where(accepted, lp_prop, lp)
            change = lp_new - lp
            conv_now = accepted & (change < dispTol)
            below_floor = accepted & (a_new < min_log_alpha)
            a = a_new
            lp = lp_new
            _, dlp_new = _lpdlp(counts, mu, x0, x1, x2, x3, mask, a, pmean, sigmasq, HAS_PRIOR, P)
            dlp = tl.where(accepted, dlp_new, dlp)
            ia = tl.where(accepted, ia + 1.0, ia)
            kap_acc = tl.minimum(kap * 1.1, kappa_0)
            ia_mod = ia - 5.0 * libdevice.floor(ia / 5.0)
            ph = accepted & (ia > 0.0) & (ia_mod == 0.0)
            kap_acc = tl.where(ph, kap_acc / 2.0, kap_acc)
            kap = tl.where(accepted, kap_acc, kap)
            kap = tl.where(rejected, kap / 2.0, kap)
            done = done | conv_now | below_floor
            it = it + 1

        tl.store(out_a_ptr + g, a)
        tl.store(out_ic_ptr + g, ic)


def _launch(counts, mu, design, a_init_clip, prior_mean, sigmasq, maxit):
    G, S = counts.shape
    P = design.shape[1]
    dev = counts.device
    min_log_alpha = float(np.log(core.MIN_DISP / 10.0))
    out_a = torch.empty(G, dtype=torch.float64, device=dev)
    out_ic = torch.empty(G, dtype=torch.float64, device=dev)
    has_prior = prior_mean is not None
    pmean = prior_mean.contiguous() if has_prior else out_a  # dummy ptr if unused
    _fit_kernel[(G,)](counts.contiguous(), mu.contiguous(), design.contiguous(),
                      a_init_clip, pmean, out_a, out_ic,
                      float(sigmasq) if has_prior else 1.0,
                      S=S, BLOCK_S=triton.next_power_of_2(S), maxit=maxit,
                      log_lo=-30.0, log_hi=10.0, eps=1e-4, dispTol=1e-6,
                      min_log_alpha=min_log_alpha, kappa_0=1.0, HAS_PRIOR=has_prior, P=P)
    return out_a, out_ic.long()


def fit_alpha_mle_triton(counts, mu, design, alpha_init, *, min_disp=core.MIN_DISP,
                         max_disp=core.MAX_DISP, grid_length=core.GRID_LENGTH, maxit=100):
    """Gene-est CR-MLE via the fused Triton loop, then eager noIncrease + grid fallback."""
    S = counts.shape[1]
    max_disp_eff = float(max(max_disp, S))
    a_init_clip = alpha_init.clamp(min_disp, max_disp_eff).contiguous()
    a, iter_count = _launch(counts, mu, design, a_init_clip, None, None, maxit)
    a_init_log = torch.log(a_init_clip)
    lp_f, _ = core._lp_and_dlp(counts, mu, design, a)
    lp_i, _ = core._lp_and_dlp(counts, mu, design, a_init_log)
    a = torch.where(lp_f < lp_i + lp_i.abs() / 1e6, a_init_log, a)
    not_conv = (iter_count >= maxit) | (iter_count == 1)
    refit = not_conv & (torch.exp(a) > min_disp * 10)
    if refit.any():
        idx = torch.nonzero(refit, as_tuple=False).squeeze(-1)
        ga = core._grid_fit_alpha(counts[idx], mu[idx], design, alpha_hat=None,
                                  prior_disp_var=None, min_disp=min_disp,
                                  max_disp=max_disp, grid_length=grid_length)
        a = a.clone()
        a[idx] = torch.log(ga)
    return torch.exp(a).clamp(min_disp, max_disp_eff)


def fit_alpha_map_triton(counts, mu, design, alpha_hat, prior_disp_var, alpha_init, *,
                         min_disp=core.MIN_DISP, max_disp=core.MAX_DISP, maxit=100):
    """MAP dispersion via the fused Triton loop (no noIncrease / grid fallback, as in R)."""
    S = counts.shape[1]
    max_disp_eff = float(max(max_disp, S))
    a_init_clip = alpha_init.clamp(min_disp, max_disp_eff).contiguous()
    a, _ = _launch(counts, mu, design, a_init_clip, torch.log(alpha_hat).contiguous(),
                   float(prior_disp_var), maxit)
    return torch.exp(a).clamp(min_disp, max_disp_eff)
