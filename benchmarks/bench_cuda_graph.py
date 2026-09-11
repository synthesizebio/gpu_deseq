"""Reproducible benchmark: eager vs CUDA-graph dispersion fitting.

Measures the effect of the ``use_cuda_graph`` path in ``fit_dispersions`` on the
dispersion Newton-Raphson loops — the pipeline's dominant GPU stage.

Two modes:
  (default)  full comparison across matrix sizes and designs (P=2 and P=4):
             eager vs graph for the isolated NR loop (fit_alpha_mle) and the full
             stage (fit_dispersions), with a bit-identity gate.
  --sweep    sweep GRAPH_CHUNK (the captured iterations per graph) to find the
             chunk size that best trades wasted iterations against replay/sync
             overhead.

Per case:
  1. Simulates a negative-binomial count matrix with a fixed seed.
  2. Verifies graph mode is BIT-IDENTICAL to eager (gene-wise, MAP, final α).
  3. Times, CUDA-synced, median-of-N: eager vs graph (WARM — cache primed). The
     one-time cold capture cost is reported separately.

Determinism: fixed seeds; no wall-clock in the numerics. Timings vary run to run
(~±10 %) but medians are stable. Run:

    PYTHONPATH=src .venv/bin/python benchmarks/bench_cuda_graph.py [--sweep] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
from gpu_deseq import DESeqDataset, fit_dispersions, fit_size_factors  # noqa: E402
import gpu_deseq._deseq2_core as core  # noqa: E402

# (n_samples, n_genes, design). "~ condition" is P=2; "~ batch + condition"
# (3-level batch) is P=4.
CASES = [
    (60, 2000, "~ condition"),
    (60, 20000, "~ condition"),
    (60, 2000, "~ batch + condition"),
    (60, 20000, "~ batch + condition"),
]
# Must all divide maxit (=100). chunk=100 captures the whole loop in one graph,
# i.e. it *is* the fixed-full-length capture, so the sweep doubles as the
# fixed-vs-chunked ablation rather than needing a separate code path.
SWEEP_CHUNKS = [2, 5, 10, 20, 25, 50, 100]
SWEEP_CASES = [(60, 2000, "~ condition"), (60, 20000, "~ condition")]
# Sample-count axis: fix genes, vary n_samples from a 2v2 pilot to biobank scale.
SAMPLE_SWEEP_GENES = 2000
SAMPLE_SWEEP_N = [4, 6, 30, 60, 200, 1000, 2000]
N_TIMED = 15
N_WARMUP = 3


def simulate(n_samples, n_genes, design, seed):
    """Fixed-seed NB counts, half control / half treated, 20% DE genes. When the
    design references `batch`, assigns a balanced 3-level batch factor."""
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    base = np.clip(np.exp(rng.normal(5, 2, n_genes)), 5, 5000)
    sf = np.exp(rng.normal(0, 0.2, n_samples))
    disp = 0.05 + 5.0 / base
    lfc = np.zeros(n_genes)
    lfc[: n_genes // 5] = rng.choice([-2, -1, 1, 2], n_genes // 5)
    counts = np.zeros((n_genes, n_samples), dtype=np.int64)
    for g in range(n_genes):
        mu = base[g] * sf * np.exp((np.arange(n_samples) >= half) * lfc[g] * np.log(2))
        counts[g] = rng.negative_binomial(1.0 / disp[g], 1.0 / (1.0 + mu * disp[g]))
    cols = {"condition": ["control"] * half + ["treated"] * (n_samples - half)}
    if "batch" in design:
        cols["batch"] = [["b0", "b1", "b2"][i % 3] for i in range(n_samples)]
    return counts.astype(np.float64), pd.DataFrame(cols)


def build(counts, coldata, design, device):
    d = DESeqDataset(counts, coldata, design=design, backend="torch").to(device)
    fit_size_factors(d)
    return d


def timed(fn, n=N_TIMED, warmup=N_WARMUP):
    """Median wall time in ms, CUDA-synchronized."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        s = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - s)
    return float(np.median(ts)) * 1e3


def _nr_inputs(d):
    """Extract the (counts, mu_hat, alpha_init) the gene-wise NR loop runs on."""
    nz = ~torch.all(d.counts == 0, dim=1)
    idx = torch.nonzero(nz).squeeze(-1)
    c, nm = d.counts[idx], d.normalized_counts[idx]
    a0 = core.fit_initial_dispersions(nm, d.size_factors, d.design_matrix)
    _, mu_hat, _, _ = core.irls_batched(c, d.size_factors, d.design_matrix, a0)
    return c, mu_hat.clamp_min(core.MIN_MU), a0


def _count_eager_iters(c, mu_hat, design, alpha_init):
    """Exact count of NR iterations the eager loop runs (uses the real init)."""
    a = torch.log(alpha_init.clamp(core.MIN_DISP, max(core.MAX_DISP, c.shape[1]))).clamp(-30, 10).clone()
    lp, dlp = core._lp_and_dlp(c, mu_hat, design, a)
    G = c.shape[0]
    kap = torch.full((G,), 1.0, dtype=c.dtype, device=c.device)
    ia = torch.zeros(G, dtype=torch.long, device=c.device)
    ic = torch.zeros(G, dtype=torch.long, device=c.device)
    done = torch.zeros(G, dtype=torch.bool, device=c.device)
    mla = float(np.log(core.MIN_DISP / 10))
    ran = 0
    for _ in range(100):
        if not (~done).any():
            break
        ran += 1
        a, lp, dlp, kap, ia, ic, done = core._nr_step(
            c, mu_hat, design, None, None, a, lp, dlp, kap, ia, ic, done,
            eps=1e-4, log_lo=-30.0, log_hi=10.0, dispTol=1e-6, min_log_alpha=mla, kappa_0=1.0)
    return ran


def bit_identity(counts, coldata, design, device):
    """max|Δ| of graph vs eager dispersions (gene-wise, MAP, final)."""
    d_e = build(counts, coldata, design, device)
    fit_dispersions(d_e, use_cuda_graph=False)
    d_g = build(counts, coldata, design, device)
    core._GRAPH_CACHE.clear()
    fit_dispersions(d_g, use_cuda_graph=True)
    fin = torch.isfinite(d_e.dispersions) & torch.isfinite(d_g.dispersions)
    return {
        "final": (d_e.dispersions[fin] - d_g.dispersions[fin]).abs().max().item(),
        "genewise": (d_e.dispersions_gene_wise[fin] - d_g.dispersions_gene_wise[fin]).abs().max().item(),
        "map": (d_e.dispersions_map[fin] - d_g.dispersions_map[fin]).abs().max().item(),
    }


def run_comparison(cases, device):
    out = []
    for (n_s, n_g, design) in cases:
        counts, coldata = simulate(n_s, n_g, design, seed=n_g)
        P = build(counts, coldata, design, device).design_matrix.shape[1]
        max_abs = bit_identity(counts, coldata, design, device)

        d = build(counts, coldata, design, device)
        t_disp_eager = timed(lambda: fit_dispersions(d, use_cuda_graph=False))
        core._GRAPH_CACHE.clear()
        fit_dispersions(d, use_cuda_graph=True)
        t_disp_graph = timed(lambda: fit_dispersions(d, use_cuda_graph=True))

        def cold():
            core._GRAPH_CACHE.clear()
            fit_dispersions(build(counts, coldata, design, device), use_cuda_graph=True)
        t_disp_cold = timed(cold, n=3, warmup=0)

        c, mu_hat, a0 = _nr_inputs(d)
        t_mle_eager = timed(lambda: core.fit_alpha_mle(c, mu_hat, d.design_matrix, alpha_init=a0, use_cuda_graph=False))
        core._GRAPH_CACHE.clear()
        core.fit_alpha_mle(c, mu_hat, d.design_matrix, alpha_init=a0, use_cuda_graph=True)
        t_mle_graph = timed(lambda: core.fit_alpha_mle(c, mu_hat, d.design_matrix, alpha_init=a0, use_cuda_graph=True))

        eager_iters = _count_eager_iters(c, mu_hat, d.design_matrix, a0)
        chunk = core.GRAPH_CHUNK
        out.append(dict(
            n_samples=n_s, n_genes=n_g, design=design, P=P, chunk=chunk,
            max_abs_diff=max_abs,
            fit_alpha_mle_eager_ms=round(t_mle_eager, 2),
            fit_alpha_mle_graph_ms=round(t_mle_graph, 2),
            fit_alpha_mle_speedup=round(t_mle_eager / t_mle_graph, 2),
            fit_dispersions_eager_ms=round(t_disp_eager, 2),
            fit_dispersions_graph_ms=round(t_disp_graph, 2),
            fit_dispersions_speedup=round(t_disp_eager / t_disp_graph, 2),
            fit_dispersions_graph_cold_ms=round(t_disp_cold, 2),
            eager_iters=eager_iters,
            graph_iters=-(-eager_iters // chunk) * chunk,
        ))
    return out


def run_sweep(cases, chunks, device):
    out = []
    saved = core.GRAPH_CHUNK
    try:
        for (n_s, n_g, design) in cases:
            counts, coldata = simulate(n_s, n_g, design, seed=n_g)
            d = build(counts, coldata, design, device)
            c, mu_hat, a0 = _nr_inputs(d)
            eager_iters = _count_eager_iters(c, mu_hat, d.design_matrix, a0)
            t_eager = timed(lambda: core.fit_alpha_mle(c, mu_hat, d.design_matrix, alpha_init=a0, use_cuda_graph=False))
            ref = core.fit_alpha_mle(c, mu_hat, d.design_matrix, alpha_init=a0, use_cuda_graph=False)
            rows = []
            for k in chunks:
                core.GRAPH_CHUNK = k
                core._GRAPH_CACHE.clear()
                got = core.fit_alpha_mle(c, mu_hat, d.design_matrix, alpha_init=a0, use_cuda_graph=True)
                max_abs = (got - ref).abs().max().item()
                t_g = timed(lambda: core.fit_alpha_mle(c, mu_hat, d.design_matrix, alpha_init=a0, use_cuda_graph=True))
                rows.append(dict(chunk=k, graph_ms=round(t_g, 2),
                                 speedup=round(t_eager / t_g, 2),
                                 graph_iters=-(-eager_iters // k) * k,
                                 max_abs_diff=max_abs))
            out.append(dict(n_samples=n_s, n_genes=n_g, design=design,
                            eager_ms=round(t_eager, 2), eager_iters=eager_iters,
                            sweep=rows))
    finally:
        core.GRAPH_CHUNK = saved
    return out


def run_sample_sweep(n_genes, samples_list, design, device, n_timed=N_TIMED):
    """Fix genes, vary n_samples. Measures how the graph payoff shifts as more
    samples make each NR iteration more compute-bound (less launch-bound)."""
    out = []
    for n_s in samples_list:
        counts, coldata = simulate(n_s, n_genes, design, seed=n_s * 7919)
        d = build(counts, coldata, design, device)
        max_abs = bit_identity(counts, coldata, design, device)

        t_disp_eager = timed(lambda: fit_dispersions(d, use_cuda_graph=False), n=n_timed)
        core._GRAPH_CACHE.clear()
        fit_dispersions(d, use_cuda_graph=True)
        t_disp_graph = timed(lambda: fit_dispersions(d, use_cuda_graph=True), n=n_timed)

        c, mu_hat, a0 = _nr_inputs(d)
        t_mle_eager = timed(lambda: core.fit_alpha_mle(c, mu_hat, d.design_matrix, alpha_init=a0, use_cuda_graph=False), n=n_timed)
        core._GRAPH_CACHE.clear()
        core.fit_alpha_mle(c, mu_hat, d.design_matrix, alpha_init=a0, use_cuda_graph=True)
        t_mle_graph = timed(lambda: core.fit_alpha_mle(c, mu_hat, d.design_matrix, alpha_init=a0, use_cuda_graph=True), n=n_timed)

        # Triton on the same axis. It is not bit-identical to eager, so its
        # agreement is recorded alongside its time; a size Triton cannot compile
        # (BLOCK_S grows with n_samples) records None rather than aborting the
        # sweep, and the paper must then say the column is short.
        t_disp_triton = t_mle_triton = tri_maxdiff = None
        try:
            ref = core.fit_alpha_mle(c, mu_hat, d.design_matrix, alpha_init=a0,
                                     use_cuda_graph=False)
            got = core.fit_alpha_mle(c, mu_hat, d.design_matrix, alpha_init=a0,
                                     use_triton=True)
            fin = torch.isfinite(ref) & torch.isfinite(got)
            tri_maxdiff = (ref[fin] - got[fin]).abs().max().item()
            t_mle_triton = timed(lambda: core.fit_alpha_mle(
                c, mu_hat, d.design_matrix, alpha_init=a0, use_triton=True), n=n_timed)
            fit_dispersions(d, use_triton=True)
            t_disp_triton = timed(lambda: fit_dispersions(d, use_triton=True), n=n_timed)
        except Exception as e:                                    # noqa: BLE001
            print(f"  [triton unavailable at n_samples={n_s}: {type(e).__name__}: {e}]")

        eager_iters = _count_eager_iters(c, mu_hat, d.design_matrix, a0)
        out.append(dict(
            n_samples=n_s, n_genes=n_genes, design=design,
            max_abs_diff=max_abs, eager_iters=eager_iters,
            triton_vs_eager_maxdiff=tri_maxdiff,
            fit_alpha_mle_eager_ms=round(t_mle_eager, 2),
            fit_alpha_mle_graph_ms=round(t_mle_graph, 2),
            fit_alpha_mle_speedup=round(t_mle_eager / t_mle_graph, 2),
            fit_alpha_mle_triton_ms=None if t_mle_triton is None else round(t_mle_triton, 2),
            fit_alpha_mle_triton_speedup=None if t_mle_triton is None
            else round(t_mle_eager / t_mle_triton, 2),
            fit_dispersions_eager_ms=round(t_disp_eager, 2),
            fit_dispersions_graph_ms=round(t_disp_graph, 2),
            fit_dispersions_speedup=round(t_disp_eager / t_disp_graph, 2),
            fit_dispersions_triton_ms=None if t_disp_triton is None else round(t_disp_triton, 2),
            fit_dispersions_triton_speedup=None if t_disp_triton is None
            else round(t_disp_eager / t_disp_triton, 2),
        ))
    return out


def provenance():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]).decode().strip())
    except Exception:
        commit, dirty = "unknown", False
    return dict(
        utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        cuda=torch.version.cuda, torch=torch.__version__,
        python=sys.version.split()[0], platform=platform.platform(),
        commit=commit, working_tree_dirty=dirty,
        n_timed=N_TIMED, default_chunk=core.GRAPH_CHUNK,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="sweep GRAPH_CHUNK")
    ap.add_argument("--sample-sweep", action="store_true", help="sweep n_samples")
    ap.add_argument(
        "--n-timed",
        type=int,
        default=N_TIMED,
        help=f"measured repetitions per case (default: {N_TIMED})",
    )
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA required for this benchmark.")

    prov = provenance()
    print("provenance:")
    for k, v in prov.items():
        print(f"  {k:20s}: {v}")

    payload = {"provenance": prov}

    if args.sweep:
        sweep = run_sweep(SWEEP_CASES, SWEEP_CHUNKS, "cuda")
        payload["sweep"] = sweep
        print("\nGRAPH_CHUNK sweep (fit_alpha_mle):")
        for s in sweep:
            print(f"\n  {s['n_samples']}x{s['n_genes']} {s['design']}  "
                  f"eager={s['eager_ms']}ms ({s['eager_iters']} iters)")
            print(f"    {'chunk':>6} {'graph_ms':>9} {'speedup':>8} {'iters':>6}  max|Δ|")
            for r in s["sweep"]:
                print(f"    {r['chunk']:>6} {r['graph_ms']:>9.2f} {r['speedup']:>7.2f}x "
                      f"{r['graph_iters']:>6}  {r['max_abs_diff']:.1e}")
    elif args.sample_sweep:
        ss = run_sample_sweep(
            SAMPLE_SWEEP_GENES,
            SAMPLE_SWEEP_N,
            "~ condition",
            "cuda",
            n_timed=args.n_timed,
        )
        payload["provenance"]["n_timed"] = args.n_timed
        payload["sample_sweep"] = ss
        print(f"\nSample-count sweep (n_genes={SAMPLE_SWEEP_GENES}, ~condition):")
        print(f"  {'samples':>7} {'iters':>5}  {'mle eager':>9} {'mle graph':>9} {'mle x':>6} "
              f"{'mle tri':>8} {'tri x':>6}  "
              f"{'disp eager':>10} {'disp graph':>10} {'disp x':>6} {'disp tri':>9} {'tri x':>6}  max|Δ|")

        def _f(v, w, p=1):
            return f"{'--':>{w}}" if v is None else f"{v:>{w}.{p}f}"

        for r in ss:
            print(f"  {r['n_samples']:>7} {r['eager_iters']:>5}  "
                  f"{r['fit_alpha_mle_eager_ms']:>9.1f} {r['fit_alpha_mle_graph_ms']:>9.1f} "
                  f"{r['fit_alpha_mle_speedup']:>5.2f}x "
                  f"{_f(r['fit_alpha_mle_triton_ms'], 8)} "
                  f"{_f(r['fit_alpha_mle_triton_speedup'], 5, 2)}x  "
                  f"{r['fit_dispersions_eager_ms']:>10.1f} {r['fit_dispersions_graph_ms']:>10.1f} "
                  f"{r['fit_dispersions_speedup']:>5.2f}x "
                  f"{_f(r['fit_dispersions_triton_ms'], 9)} "
                  f"{_f(r['fit_dispersions_triton_speedup'], 5, 2)}x  "
                  f"{r['max_abs_diff']['final']:.0e}")
    else:
        results = run_comparison(CASES, "cuda")
        payload["results"] = results
        print("\nresults:")
        for r in results:
            print(f"\n  {r['n_samples']} x {r['n_genes']}  {r['design']}  (P={r['P']})")
            print(f"    bit-identity max|Δ|: final={r['max_abs_diff']['final']:.1e} "
                  f"genewise={r['max_abs_diff']['genewise']:.1e} map={r['max_abs_diff']['map']:.1e}")
            print(f"    fit_alpha_mle   eager={r['fit_alpha_mle_eager_ms']:7.2f}ms  "
                  f"graph={r['fit_alpha_mle_graph_ms']:7.2f}ms  {r['fit_alpha_mle_speedup']}x")
            print(f"    fit_dispersions eager={r['fit_dispersions_eager_ms']:7.2f}ms  "
                  f"graph={r['fit_dispersions_graph_ms']:7.2f}ms  {r['fit_dispersions_speedup']}x  "
                  f"(cold {r['fit_dispersions_graph_cold_ms']:.0f}ms)")
            print(f"    iters: eager {r['eager_iters']}, chunked graph {r['graph_iters']} (chunk={r['chunk']})")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
