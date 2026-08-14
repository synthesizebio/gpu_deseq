"""Triton NR loop with optional Gaussian log-prior (MAP), P=2.
Extends the gene-est prototype: HAS_PRIOR adds -0.5*(a-m)^2/sig to lp and
-(a-m)/sig to dlp, matching _lp_and_dlp's prior branch. MAP wrapper skips the
noIncrease revert and grid fallback (as fit_alpha_map does)."""
import sys, time
sys.path.insert(0, "src")
import numpy as np, pandas as pd, torch
import triton, triton.language as tl
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
def _lpdlp(counts, mu, x0, x1, mask, a, prior_mean, sigmasq, HAS_PRIOR: tl.constexpr):
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
    if HAS_PRIOR:
        diff = a - prior_mean
        lp = lp - 0.5 * diff * diff / sigmasq
        dlp = dlp - diff / sigmasq
    return lp, dlp


@triton.jit
def _fit_kernel(counts_ptr, mu_ptr, x_ptr, ainit_ptr, pmean_ptr, out_a_ptr, out_ic_ptr,
                sigmasq, S: tl.constexpr, BLOCK_S: tl.constexpr, maxit: tl.constexpr,
                log_lo: tl.constexpr, log_hi: tl.constexpr, eps: tl.constexpr,
                dispTol: tl.constexpr, min_log_alpha: tl.constexpr, kappa_0: tl.constexpr,
                HAS_PRIOR: tl.constexpr):
    g = tl.program_id(0)
    s = tl.arange(0, BLOCK_S)
    mask = s < S
    counts = tl.load(counts_ptr + g * S + s, mask=mask, other=0.0)
    mu = tl.load(mu_ptr + g * S + s, mask=mask, other=1.0)
    x0 = tl.load(x_ptr + s * 2 + 0, mask=mask, other=0.0)
    x1 = tl.load(x_ptr + s * 2 + 1, mask=mask, other=0.0)
    pmean = tl.load(pmean_ptr + g) if HAS_PRIOR else 0.0

    a = tl.load(ainit_ptr + g)
    a = tl.minimum(tl.maximum(tl.log(a), log_lo), log_hi)
    lp, dlp = _lpdlp(counts, mu, x0, x1, mask, a, pmean, sigmasq, HAS_PRIOR)

    zero = a - a
    kap = zero + kappa_0
    ia = zero
    ic = zero
    done = zero > 1.0

    # Per-gene early exit: this program runs only until THIS gene converges.
    # Once `done`, the while exits — no fixed iteration count, no host round-trip.
    # Inside the loop ~done is always true, so the active-mask gating drops out.
    it = 0
    while (it < maxit) & (~done):
        ic = ic + 1.0
        a_prop_naive = a + kap * dlp
        too_low = (a_prop_naive < log_lo) & (dlp != 0.0)
        kap = tl.where(too_low, (log_lo - a) / dlp, kap)
        too_high = (a_prop_naive > log_hi) & (dlp != 0.0)
        kap = tl.where(too_high, (log_hi - a) / dlp, kap)
        a_propose = a + kap * dlp
        lp_prop, _ = _lpdlp(counts, mu, x0, x1, mask, a_propose, pmean, sigmasq, HAS_PRIOR)
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
        _, dlp_new = _lpdlp(counts, mu, x0, x1, mask, a, pmean, sigmasq, HAS_PRIOR)
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


def _launch(counts, mu, design, a_init_clip, prior_mean, sigmasq, maxit=100):
    G, S = counts.shape
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
                      min_log_alpha=min_log_alpha, kappa_0=1.0, HAS_PRIOR=has_prior)
    return out_a, out_ic.long()


def fit_alpha_mle_triton(counts, mu, design, alpha_init, maxit=100):
    S = counts.shape[1]
    max_disp_eff = float(max(core.MAX_DISP, S))
    a_init_clip = alpha_init.clamp(core.MIN_DISP, max_disp_eff).contiguous()
    a, iter_count = _launch(counts, mu, design, a_init_clip, None, None, maxit)
    a_init_log = torch.log(a_init_clip)
    lp_f, _ = core._lp_and_dlp(counts, mu, design, a)
    lp_i, _ = core._lp_and_dlp(counts, mu, design, a_init_log)
    a = torch.where(lp_f < lp_i + lp_i.abs() / 1e6, a_init_log, a)
    not_conv = (iter_count >= maxit) | (iter_count == 1)
    refit = not_conv & (torch.exp(a) > core.MIN_DISP * 10)
    if refit.any():
        idx = torch.nonzero(refit, as_tuple=False).squeeze(-1)
        ga = core._grid_fit_alpha(counts[idx], mu[idx], design, alpha_hat=None,
                                  prior_disp_var=None, min_disp=core.MIN_DISP, max_disp=core.MAX_DISP)
        a = a.clone(); a[idx] = torch.log(ga)
    return torch.exp(a).clamp(core.MIN_DISP, max_disp_eff)


def fit_alpha_map_triton(counts, mu, design, alpha_hat, prior_disp_var, alpha_init, maxit=100):
    S = counts.shape[1]
    max_disp_eff = float(max(core.MAX_DISP, S))
    a_init_clip = alpha_init.clamp(core.MIN_DISP, max_disp_eff).contiguous()
    a, _ = _launch(counts, mu, design, a_init_clip, torch.log(alpha_hat).contiguous(),
                   float(prior_disp_var), maxit)   # MAP: no noIncrease, no grid fallback
    return torch.exp(a).clamp(core.MIN_DISP, max_disp_eff)


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
    # 1) R parity for MAP against dispMAP fixture
    print("=== MAP parity vs R dispMAP ===")
    from pathlib import Path
    for label in ["medium_30x500", "large_60x2000"]:
        p = Path("fixtures/r_deseq2") / label
        counts = pd.read_csv(p / "counts.csv", index_col=0)
        coldata = pd.read_csv(p / "coldata.csv", index_col=0)
        dcols = pd.read_csv(p / "dispersions.csv").set_index("gene")
        r_map = dcols["dispMAP"].to_numpy() if "dispMAP" in dcols.columns else None
        dds = DESeqDataset(counts.to_numpy(np.float64), coldata, design="~ condition",
                           gene_ids=list(counts.index), sample_ids=list(counts.columns),
                           backend="torch").to(dev)
        fit_size_factors(dds); core_fit = __import__("gpu_deseq").fit_dispersions
        core_fit(dds)  # eager full, to get trend + prior var + mu_hat pieces
        # recompute the MAP inputs exactly
        nz = dds.non_zero_mask; idx = torch.nonzero(nz).squeeze(-1)
        c = dds.counts[idx]; design = dds.design_matrix
        a0 = core.fit_initial_dispersions(dds.normalized_counts[idx], dds.size_factors, design)
        mu_hat = core.lin_reg_mu(c, dds.size_factors, design) if core.is_saturated_design(design) \
            else core.irls_batched(c, dds.size_factors, design, a0)[1].clamp_min(core.MIN_MU)
        amle = dds.dispersions_gene_wise[idx]
        trend = dds.dispersion_trend[idx]
        pv = dds.prior_disp_var
        map_t = fit_alpha_map_triton(c, mu_hat, design, trend, pv, amle).cpu().numpy()
        map_e = core.fit_alpha_map(c, mu_hat, design, alpha_hat=trend, prior_disp_var=pv, alpha_init=amle).cpu().numpy()
        rel_te = (np.abs(map_t - map_e) / (np.abs(map_e) + 1e-12))
        line = f"  {label}: triton-vs-eager MAP p95={np.percentile(rel_te,95):.2e} max={rel_te.max():.2e}"
        if r_map is not None:
            rmnz = r_map[idx.cpu().numpy()]; m = np.isfinite(map_t) & np.isfinite(rmnz)
            rel_tr = np.abs(map_t[m]-rmnz[m])/(np.abs(rmnz[m])+1e-12)
            line += f"  triton-vs-R p95={np.percentile(rel_tr,95):.2e}"
        print(line)

    # 2) combined two-loop timing vs graph
    print("\n=== combined gene-est + MAP timing (Triton vs graph) ===")
    for (n_s, n_g) in [(6, 2000), (60, 2000), (60, 20000)]:
        rng = np.random.default_rng(n_g)
        half = n_s // 2; base = np.clip(np.exp(rng.normal(5, 2, n_g)), 5, 5000)
        sf = np.exp(rng.normal(0, 0.2, n_s)); disp = 0.05 + 5.0 / base
        cnt = np.zeros((n_g, n_s), dtype=np.int64)
        for gg in range(n_g):
            cnt[gg] = rng.negative_binomial(1.0/disp[gg], 1.0/(1.0+base[gg]*sf*disp[gg]))
        coldata = pd.DataFrame({"condition": ["c"]*half+["t"]*(n_s-half)})
        dds = DESeqDataset(cnt.astype(np.float64), coldata, design="~ condition", backend="torch").to(dev)
        fit_size_factors(dds)
        nz = ~torch.all(dds.counts == 0, dim=1); idx = torch.nonzero(nz).squeeze(-1)
        c = dds.counts[idx]; design = dds.design_matrix
        a0 = core.fit_initial_dispersions(dds.normalized_counts[idx], dds.size_factors, design)
        mu_hat = core.lin_reg_mu(c, dds.size_factors, design) if core.is_saturated_design(design) \
            else core.irls_batched(c, dds.size_factors, design, a0)[1].clamp_min(core.MIN_MU)
        amle = core.fit_alpha_mle(c, mu_hat, design, alpha_init=a0)
        trend = torch.as_tensor(core.fit_parametric_trend(amle.cpu().numpy(),
                    dds.normalized_counts[idx].mean(1).cpu().numpy())[0], dtype=torch.float64, device=dev)
        pv = 0.5

        def both_eager():
            m = core.fit_alpha_mle(c, mu_hat, design, alpha_init=a0, use_cuda_graph=False)
            core.fit_alpha_map(c, mu_hat, design, alpha_hat=trend, prior_disp_var=pv, alpha_init=m, use_cuda_graph=False)
        def both_graph():
            m = core.fit_alpha_mle(c, mu_hat, design, alpha_init=a0, use_cuda_graph=True)
            core.fit_alpha_map(c, mu_hat, design, alpha_hat=trend, prior_disp_var=pv, alpha_init=m, use_cuda_graph=True)
        def both_triton():
            m = fit_alpha_mle_triton(c, mu_hat, design, a0)
            fit_alpha_map_triton(c, mu_hat, design, trend, pv, m)

        core._GRAPH_CACHE.clear(); both_graph()
        te = _bench(both_eager); tg = _bench(both_graph); tt = _bench(both_triton)
        print(f"  {n_s}x{n_g}: eager={te:7.1f}ms graph={tg:7.1f}ms triton={tt:7.1f}ms  "
              f"triton vs graph={tg/tt:.2f}x  vs eager={te/tt:.2f}x")
