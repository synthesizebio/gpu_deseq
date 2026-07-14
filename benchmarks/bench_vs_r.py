"""Reproducible speedup-vs-R benchmark for gpu_deseq.

One self-contained driver:
  1. Simulate fixed-seed NB count matrices for each case.
  2. Time gpu_deseq on GPU — eager / CUDA-graph / Triton — for BOTH the
     dispersion stage (fit_dispersions) and the full pipeline (size factors ->
     dispersions -> Wald -> results).
  3. Shell out to Rscript to time R DESeq2 on the identical matrices
     (estimateDispersions, and full DESeq()+results()).
  4. Print two speedup tables (stage, full pipeline) and write JSON.

R DESeq2 is CPU single-threaded; gpu_deseq runs on the visible CUDA device — the
same "R vs GPU-accelerated" framing as the README (NOT a same-device compare).

Run:  PYTHONPATH=src .venv/bin/python benchmarks/bench_vs_r.py [--json out.json] [--reps N]
Needs: CUDA + a working `Rscript` with DESeq2 (set R_DESEQ2_LIB if not on the
default library path).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
from gpu_deseq import DESeqDataset, fit_size_factors, fit_dispersions, wald_test, results  # noqa: E402
import gpu_deseq._deseq2_core as core  # noqa: E402

# (n_samples, n_genes, design, contrast, tag)
CASES = [
    (6, 2000, "~ condition", "condition[T.treated]", "6x2000_cond"),
    (60, 2000, "~ condition", "condition[T.treated]", "60x2000_cond"),
    (60, 20000, "~ condition", "condition[T.treated]", "60x20000_cond"),
    (60, 1500, "~ batch + condition", "condition[T.treated]", "60x1500_multi"),
]


def simulate(n_s, n_g, design, seed):
    rng = np.random.default_rng(seed)
    half = n_s // 2
    base = np.clip(np.exp(rng.normal(5, 2, n_g)), 5, 5000)
    sf = np.exp(rng.normal(0, 0.2, n_s))
    disp = 0.05 + 5.0 / base
    lfc = np.zeros(n_g)
    lfc[: n_g // 5] = rng.choice([-2, -1, 1, 2], n_g // 5)
    counts = np.zeros((n_g, n_s), dtype=np.int64)
    for g in range(n_g):
        mu = base[g] * sf * np.exp((np.arange(n_s) >= half) * lfc[g] * np.log(2))
        counts[g] = rng.negative_binomial(1.0 / disp[g], 1.0 / (1.0 + mu * disp[g]))
    cols = {"condition": ["control"] * half + ["treated"] * (n_s - half)}
    if "batch" in design:
        cols["batch"] = [["b0", "b1", "b2"][i % 3] for i in range(n_s)]
    return counts.astype(np.int64), pd.DataFrame(cols)


def timed(fn, n, warmup=2):
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


def _stage(d, **kw):
    fit_dispersions(d, **kw)


def _full(counts, coldata, design, contrast, device, **kw):
    d = DESeqDataset(counts.astype(np.float64), coldata, design=design, backend="torch").to(device)
    fit_size_factors(d)
    fit_dispersions(d, **kw)
    results(wald_test(d, contrast=contrast))


def time_ours(casedir, device, reps):
    rows = {}
    for (n_s, n_g, design, contrast, tag) in CASES:
        counts, coldata = simulate(n_s, n_g, design, seed=n_g)
        # persist for R
        pd.DataFrame(counts, index=[f"g{i}" for i in range(n_g)],
                     columns=[f"s{j}" for j in range(n_s)]).to_csv(f"{casedir}/{tag}_counts.csv")
        cd = coldata.copy(); cd.index = [f"s{j}" for j in range(n_s)]
        cd.to_csv(f"{casedir}/{tag}_coldata.csv")

        d = DESeqDataset(counts.astype(np.float64), coldata, design=design, backend="torch").to(device)
        fit_size_factors(d)
        r = {}
        # dispersion stage
        r["disp_eager"] = timed(lambda: _stage(d, use_cuda_graph=False, use_triton=False), reps)
        core._GRAPH_CACHE.clear(); _stage(d, use_cuda_graph=True)
        r["disp_graph"] = timed(lambda: _stage(d, use_cuda_graph=True), reps)
        _stage(d, use_triton=True)
        r["disp_triton"] = timed(lambda: _stage(d, use_triton=True), reps)
        # full pipeline
        r["full_eager"] = timed(lambda: _full(counts, coldata, design, contrast, device), reps)
        core._GRAPH_CACHE.clear()
        r["full_graph"] = timed(lambda: _full(counts, coldata, design, contrast, device, use_cuda_graph=True), reps)
        r["full_triton"] = timed(lambda: _full(counts, coldata, design, contrast, device, use_triton=True), reps)
        rows[tag] = r
        print(f"  ours {tag}: disp(e/g/t)={r['disp_eager']:.0f}/{r['disp_graph']:.0f}/{r['disp_triton']:.0f}  "
              f"full={r['full_eager']:.0f}/{r['full_graph']:.0f}/{r['full_triton']:.0f} ms")

    manifest = pd.DataFrame([{"tag": t[4], "design": t[2]} for t in CASES])
    manifest.to_csv(f"{casedir}/manifest.csv", index=False)
    return rows


def time_r(casedir, reps):
    here = Path(__file__).resolve().parent
    print("  running Rscript (this is the slow part) ...")
    subprocess.run(["Rscript", str(here / "time_r_disp.R"), casedir, str(reps)], check=True)
    rt = pd.read_csv(f"{casedir}/r_timings.csv").set_index("tag")
    return {t: {"disp": float(rt.loc[t, "r_disp_ms"]), "full": float(rt.loc[t, "r_full_ms"])} for t in rt.index}


def provenance(reps):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]).decode().strip())
    except Exception:
        commit, dirty = "unknown", False
    return dict(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                torch=torch.__version__, python=sys.version.split()[0],
                platform=platform.platform(), commit=commit, working_tree_dirty=dirty, reps=reps)


def _table(title, ours, rmap, kind):
    print(f"\n{title} (speedup vs R DESeq2)")
    print(f"  {'case':<18}{'R (ms)':>9}{'eager':>10}{'graph':>10}{'triton':>10}")
    for (_, _, _, _, tag) in CASES:
        r = rmap[tag][kind]
        e = ours[tag][f"{kind}_eager"]; g = ours[tag][f"{kind}_graph"]; t = ours[tag][f"{kind}_triton"]
        print(f"  {tag:<18}{r:>9.0f}{r/e:>9.1f}x{r/g:>9.1f}x{r/t:>9.1f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--casedir", type=str, default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA required.")

    prov = provenance(args.reps)
    print("provenance:", {k: prov[k] for k in ("gpu", "torch", "cuda", "commit")})
    casedir = args.casedir or tempfile.mkdtemp(prefix="gpudeseq_vsr_")
    print("case dir:", casedir)

    ours = time_ours(casedir, "cuda", args.reps)
    rmap = time_r(casedir, args.reps)

    _table("Dispersion stage  (fit_dispersions vs estimateDispersions)", ours, rmap, "disp")
    _table("Full pipeline     (SF->disp->Wald->results vs DESeq()+results)", ours, rmap, "full")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"provenance": prov, "ours_ms": ours, "r_ms": rmap}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
