"""Full NR loop fused into ONE Triton kernel (P=2, no prior): one program per
gene runs all maxit iterations in registers. counts/mu/design are loaded once
and reused across iterations. This is the maximal-fusion version — the true test
of whether Triton beats the CUDA-graph baseline. noIncrease + grid fallback run
eagerly afterward, matching _r_fit_alpha_mle."""
import sys, time
sys.path.insert(0, "src")
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import gpu_deseq._deseq2_core as core
from gpu_deseq import DESeqDataset, fit_size_factors


@triton.jit
def _digamma_pos(x):
    result = x - x
    for _ in range(12):
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
def _lpdlp(counts, mu, x0, x1, mask, a):
    alpha = tl.exp(a)
    ainv = 1.0 / alpha
    w = tl.where(mask, 1.0 / (1.0 / mu + alpha), 0.0)
    dw = -w * w
    b00 = tl.sum(w * x0 * x0); b01 = tl.sum(w * x0 * x1); b11 = tl.sum(w * x1 * x1)
    d00 = tl.sum(dw * x0 * x0); d01 = tl.sum(dw * x0 * x1); d11 = tl.sum(dw * x1 * x1)
    det = b00 * b11 - b01 * b01
    logdet = tl.log(det)
    tr = (b11 * d00 - 2.0 * b01 * d01 + b00 * d11) / det
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
    return lp, dlp


@triton.jit
def _fit_kernel(counts_ptr, mu_ptr, x_ptr, ainit_ptr, out_a_ptr, out_ic_ptr,
                S: tl.constexpr, BLOCK_S: tl.constexpr, maxit: tl.constexpr,
                log_lo: tl.constexpr, log_hi: tl.constexpr, eps: tl.constexpr,
                dispTol: tl.constexpr, min_log_alpha: tl.constexpr, kappa_0: tl.constexpr):
    g = tl.program_id(0)
    s = tl.arange(0, BLOCK_S)
    mask = s < S
    counts = tl.load(counts_ptr + g * S + s, mask=mask, other=0.0)
    mu = tl.load(mu_ptr + g * S + s, mask=mask, other=1.0)
    x0 = tl.load(x_ptr + s * 2 + 0, mask=mask, other=0.0)
    x1 = tl.load(x_ptr + s * 2 + 1, mask=mask, other=0.0)

    a = tl.load(ainit_ptr + g)
    a = tl.minimum(tl.maximum(tl.log(a), log_lo), log_hi)
    lp, dlp = _lpdlp(counts, mu, x0, x1, mask, a)

    zero = a - a
    kap = zero + kappa_0
    ia = zero
    ic = zero
    done = zero > 1.0  # all-False bool

    for it in range(maxit):
        active = ~done
        ic = tl.where(active, ic + 1.0, ic)
        a_prop_naive = a + kap * dlp
        too_low = active & (a_prop_naive < log_lo) & (dlp != 0.0)
        kap = tl.where(too_low, (log_lo - a) / dlp, kap)
        too_high = active & (a_prop_naive > log_hi) & (dlp != 0.0)
        kap = tl.where(too_high, (log_hi - a) / dlp, kap)
        a_propose = a + kap * dlp
        lp_prop, _ = _lpdlp(counts, mu, x0, x1, mask, a_propose)
        theta_kap = -lp_prop
        theta_hat = -lp - kap * eps * dlp * dlp
        accepted = active & (theta_kap <= theta_hat)
        rejected = active & (~accepted)
        a_new = tl.where(accepted, a_propose, a)
        lp_new = tl.where(accepted, lp_prop, lp)
        change = lp_new - lp
        conv_now = accepted & (change < dispTol)
        below_floor = accepted & (a_new < min_log_alpha)
        a = a_new
        lp = lp_new
        _, dlp_new = _lpdlp(counts, mu, x0, x1, mask, a)
        dlp = tl.where(accepted, dlp_new, dlp)
        ia = tl.where(accepted, ia + 1.0, ia)
        kap_acc = tl.minimum(kap * 1.1, kappa_0)
        # periodic halving every 5 accepts
        ia_mod = ia - 5.0 * libdevice.floor(ia / 5.0)
        ph = accepted & (ia > 0.0) & (ia_mod == 0.0)
        kap_acc = tl.where(ph, kap_acc / 2.0, kap_acc)
        kap = tl.where(accepted, kap_acc, kap)
        kap = tl.where(rejected, kap / 2.0, kap)
        done = done | conv_now | below_floor

    tl.store(out_a_ptr + g, a)
    tl.store(out_ic_ptr + g, ic)


def fit_alpha_mle_triton(counts, mu, design, alpha_init, maxit=100):
    """Triton full-loop gene-est MLE, then eager noIncrease + grid fallback."""
    G, S = counts.shape
    assert design.shape[1] == 2
    dev = counts.device
    min_disp, max_disp = core.MIN_DISP, core.MAX_DISP
    max_disp_eff = float(max(max_disp, S))
    min_log_alpha = float(np.log(min_disp / 10.0))
    a_init_clip = alpha_init.clamp(min_disp, max_disp_eff).contiguous()

    out_a = torch.empty(G, dtype=torch.float64, device=dev)
    out_ic = torch.empty(G, dtype=torch.float64, device=dev)
    BLOCK_S = triton.next_power_of_2(S)
    _fit_kernel[(G,)](counts.contiguous(), mu.contiguous(), design.contiguous(),
                      a_init_clip, out_a, out_ic, S=S, BLOCK_S=BLOCK_S, maxit=maxit,
                      log_lo=-30.0, log_hi=10.0, eps=1e-4, dispTol=1e-6,
                      min_log_alpha=min_log_alpha, kappa_0=1.0)
    a = out_a
    iter_count = out_ic.long()

    # noIncrease revert
    a_init_log = torch.log(a_init_clip)
    lp_final, _ = core._lp_and_dlp(counts, mu, design, a)
    lp_init, _ = core._lp_and_dlp(counts, mu, design, a_init_log)
    no_increase = lp_final < lp_init + lp_init.abs() / 1e6
    a = torch.where(no_increase, a_init_log, a)

    # grid fallback for non-converged
    not_conv = (iter_count >= maxit) | (iter_count == 1)
    refit = not_conv & (torch.exp(a) > min_disp * 10)
    if refit.any():
        idx = torch.nonzero(refit, as_tuple=False).squeeze(-1)
        grid_a = core._grid_fit_alpha(counts[idx], mu[idx], design,
                                      alpha_hat=None, prior_disp_var=None,
                                      min_disp=min_disp, max_disp=max_disp)
        a = a.clone()
        a[idx] = torch.log(grid_a)
    return torch.exp(a).clamp(min_disp, max_disp_eff)


def _bench(fn, n=15, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); s = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - s)
    return np.median(ts) * 1e3


if __name__ == "__main__":
    dev = "cuda"
    for (n_s, n_g) in [(60, 2000), (60, 20000), (6, 2000)]:
        rng = np.random.default_rng(n_g)
        half = n_s // 2
        base = np.clip(np.exp(rng.normal(5, 2, n_g)), 5, 5000)
        sf = np.exp(rng.normal(0, 0.2, n_s)); disp = 0.05 + 5.0 / base
        counts = np.zeros((n_g, n_s), dtype=np.int64)
        for gg in range(n_g):
            counts[gg] = rng.negative_binomial(1.0 / disp[gg], 1.0 / (1.0 + base[gg] * sf * disp[gg]))
        coldata = __import__("pandas").DataFrame({"condition": ["c"] * half + ["t"] * (n_s - half)})
        d = DESeqDataset(counts.astype(np.float64), coldata, design="~ condition", backend="torch").to(dev)
        fit_size_factors(d)
        nz = ~torch.all(d.counts == 0, dim=1); idx = torch.nonzero(nz).squeeze(-1)
        c, nm = d.counts[idx], d.normalized_counts[idx]
        a0 = core.fit_initial_dispersions(nm, d.size_factors, d.design_matrix)
        _, mu_hat, _, _ = core.irls_batched(c, d.size_factors, d.design_matrix, a0)
        mu_hat = mu_hat.clamp_min(core.MIN_MU)
        design = d.design_matrix

        ref = core.fit_alpha_mle(c, mu_hat, design, alpha_init=a0, use_cuda_graph=False)
        tri = fit_alpha_mle_triton(c, mu_hat, design, a0)
        rel = ((tri - ref).abs() / (ref.abs() + 1e-12)).max().item()

        t_eager = _bench(lambda: core.fit_alpha_mle(c, mu_hat, design, alpha_init=a0, use_cuda_graph=False))
        core._GRAPH_CACHE.clear()
        core.fit_alpha_mle(c, mu_hat, design, alpha_init=a0, use_cuda_graph=True)
        t_graph = _bench(lambda: core.fit_alpha_mle(c, mu_hat, design, alpha_init=a0, use_cuda_graph=True))
        t_tri = _bench(lambda: fit_alpha_mle_triton(c, mu_hat, design, a0))

        print(f"\n{n_s}x{n_g}: max|rel diff vs eager|={rel:.2e}")
        print(f"  eager={t_eager:7.2f}ms  graph={t_graph:7.2f}ms  triton={t_tri:7.2f}ms")
        print(f"  triton vs eager={t_eager/t_tri:.2f}x   triton vs graph={t_graph/t_tri:.2f}x")
