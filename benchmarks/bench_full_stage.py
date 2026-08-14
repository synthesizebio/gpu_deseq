"""Final full-stage fit_dispersions timing: eager vs graph vs triton (wired API)."""
import sys, time
sys.path.insert(0, "src")
import numpy as np, pandas as pd, torch
from gpu_deseq import DESeqDataset, fit_size_factors, fit_dispersions
import gpu_deseq._deseq2_core as core

dev = "cuda"


def sim(n_s, n_g, design, seed):
    rng = np.random.default_rng(seed)
    half = n_s // 2
    base = np.clip(np.exp(rng.normal(5, 2, n_g)), 5, 5000)
    sf = np.exp(rng.normal(0, 0.2, n_s)); disp = 0.05 + 5.0 / base
    lfc = np.zeros(n_g); lfc[:n_g // 5] = rng.choice([-2, -1, 1, 2], n_g // 5)
    counts = np.zeros((n_g, n_s), dtype=np.int64)
    for g in range(n_g):
        mu = base[g] * sf * np.exp((np.arange(n_s) >= half) * lfc[g] * np.log(2))
        counts[g] = rng.negative_binomial(1.0 / disp[g], 1.0 / (1.0 + mu * disp[g]))
    cols = {"condition": ["control"] * half + ["treated"] * (n_s - half)}
    if "batch" in design:
        cols["batch"] = [["b0", "b1", "b2"][i % 3] for i in range(n_s)]
    return counts.astype(np.float64), pd.DataFrame(cols)


def timed(fn, n=10, warmup=2):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); s = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - s)
    return np.median(ts) * 1e3


print(f"{'case':<26}{'eager':>9}{'graph':>9}{'triton':>9}{'tri/graph':>10}{'tri/eager':>10}")
for (n_s, n_g, design) in [(6, 2000, "~ condition"), (60, 2000, "~ condition"),
                           (60, 20000, "~ condition"), (60, 1500, "~ batch + condition")]:
    counts, coldata = sim(n_s, n_g, design, seed=n_g)
    d = DESeqDataset(counts, coldata, design=design, backend="torch").to(dev)
    fit_size_factors(d)
    te = timed(lambda: fit_dispersions(d, use_cuda_graph=False, use_triton=False))
    core._GRAPH_CACHE.clear(); fit_dispersions(d, use_cuda_graph=True)
    tg = timed(lambda: fit_dispersions(d, use_cuda_graph=True))
    fit_dispersions(d, use_triton=True)  # warm/compile
    tt = timed(lambda: fit_dispersions(d, use_triton=True))
    label = f"{n_s}x{n_g} {design.replace('~ ','~')}"
    print(f"{label:<26}{te:>8.1f}{tg:>9.1f}{tt:>9.1f}{tg/tt:>9.2f}x{te/tt:>9.2f}x")
