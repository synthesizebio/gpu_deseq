"""Per-step speedup table: 5 pipeline steps + total, columns R / eager / graph /
triton, one table per case.

Steps (gpu_deseq function // R DESeq2 call):
  normalization   fit_size_factors  // estimateSizeFactors
  dispersion      fit_dispersions   // estimateDispersions
  glm_fit         wald_test         // nbinomWaldTest
  significance    results           // results
  lfc_shrink      lfc_shrink        // lfcShrink(type="apeglm")

Only `dispersion` is affected by the eager/graph/triton accelerator; the other
four rows are identical across those three columns (the flags route only
fit_dispersions). R is CPU single-threaded; ours is one CUDA device.

Run:  PYTHONPATH=src .venv/bin/python benchmarks/bench_per_step.py [--json out.json] [--reps N]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
from gpu_deseq import (DESeqDataset, fit_size_factors, fit_dispersions,  # noqa: E402
                       wald_test, results, lfc_shrink)
import gpu_deseq._deseq2_core as core  # noqa: E402
from bench_vs_r import CASES, simulate, timed, provenance  # noqa: E402


def time_ours(casedir, device, reps):
    rows = {}
    for (n_s, n_g, design, contrast, tag) in CASES:
        counts, coldata = simulate(n_s, n_g, design, seed=n_g)
        pd.DataFrame(counts, index=[f"g{i}" for i in range(n_g)],
                     columns=[f"s{j}" for j in range(n_s)]).to_csv(f"{casedir}/{tag}_counts.csv")
        cd = coldata.copy(); cd.index = [f"s{j}" for j in range(n_s)]
        cd.to_csv(f"{casedir}/{tag}_coldata.csv")

        d = DESeqDataset(counts.astype(np.float64), coldata, design=design, backend="torch").to(device)
        # step 1
        t_norm = timed(lambda: fit_size_factors(d), reps)
        fit_size_factors(d)
        # step 2 (three modes)
        t_disp_e = timed(lambda: fit_dispersions(d, use_cuda_graph=False, use_triton=False), reps)
        core._GRAPH_CACHE.clear(); fit_dispersions(d, use_cuda_graph=True)
        t_disp_g = timed(lambda: fit_dispersions(d, use_cuda_graph=True), reps)
        fit_dispersions(d, use_triton=True)
        t_disp_t = timed(lambda: fit_dispersions(d, use_triton=True), reps)
        fit_dispersions(d)  # leave d with eager dispersions for downstream steps
        # step 3
        t_glm = timed(lambda: wald_test(d, contrast=contrast), reps)
        fit = wald_test(d, contrast=contrast)
        # step 4
        t_sig = timed(lambda: results(fit), reps)
        # step 5
        t_shrink = timed(lambda: lfc_shrink(fit, coeff=contrast), reps)

        rows[tag] = dict(norm=t_norm, disp_eager=t_disp_e, disp_graph=t_disp_g,
                         disp_triton=t_disp_t, glm=t_glm, sig=t_sig, shrink=t_shrink)
        print(f"  ours {tag}: norm={t_norm:.1f} disp(e/g/t)={t_disp_e:.0f}/{t_disp_g:.0f}/{t_disp_t:.0f} "
              f"glm={t_glm:.1f} sig={t_sig:.1f} shrink={t_shrink:.1f}")

    pd.DataFrame([{"tag": t[4], "design": t[2]} for t in CASES]).to_csv(f"{casedir}/manifest.csv", index=False)
    return rows


def time_r(casedir, reps, ncores):
    here = Path(__file__).resolve().parent
    print(f"  running Rscript time_r_steps.R (serial + {ncores}-core; slow) ...")
    subprocess.run(["Rscript", str(here / "time_r_steps.R"), casedir, str(reps), str(ncores)], check=True)
    rt = pd.read_csv(f"{casedir}/r_step_timings.csv").set_index("tag")
    return {t: dict(norm=float(rt.loc[t, "r_norm_ms"]), disp=float(rt.loc[t, "r_disp_ms"]),
                    glm=float(rt.loc[t, "r_glm_ms"]), sig=float(rt.loc[t, "r_sig_ms"]),
                    shrink=float(rt.loc[t, "r_shrink_ms"]),
                    deseqpar=float(rt.loc[t, "r_deseqpar_ms"]),
                    sig_par=float(rt.loc[t, "r_sig_par_ms"]),
                    shrink_par=float(rt.loc[t, "r_shrink_par_ms"]))
            for t in rt.index}


def print_table(tag, o, r, ncores):
    # Full 5-step x 5-column table. R 12-core: normalization isn't parallelized
    # (= serial); DESeq2 parallelizes dispersion+GLM as ONE bundle
    # (r["deseqpar"], incl. size factors) so those two rows share it; results and
    # lfc_shrink take their own parallel flag. ours: only dispersion changes
    # across eager/graph/triton.
    e_n, e_g, e_s, e_sh = o["norm"], o["glm"], o["sig"], o["shrink"]
    # R 12-core per row: dispersion carries (deseqpar - size factors); glm folded in.
    r12_disp_glm = max(r["deseqpar"] - r["norm"], 0.0)  # bundled dispersion+GLM, parallel
    rows = [
        # label, R1, R12, eager, graph, triton
        ("normalization", r["norm"], r["norm"],       e_n, e_n, e_n),
        ("dispersion",    r["disp"], r12_disp_glm,    o["disp_eager"], o["disp_graph"], o["disp_triton"]),
        ("glm_fit",       r["glm"],  None,            e_g, e_g, e_g),   # None => bundled above
        ("significance",  r["sig"],  r["sig_par"],    e_s, e_s, e_s),
        ("lfc_shrink",    r["shrink"], r["shrink_par"], e_sh, e_sh, e_sh),
    ]
    tot_r1 = r["norm"] + r["disp"] + r["glm"] + r["sig"] + r["shrink"]
    tot_r12 = r["deseqpar"] + r["sig_par"] + r["shrink_par"]  # measured multi-core end-to-end
    tot_e = e_n + o["disp_eager"] + e_g + e_s + e_sh
    tot_g = e_n + o["disp_graph"] + e_g + e_s + e_sh
    tot_t = e_n + o["disp_triton"] + e_g + e_s + e_sh

    def cell(v):
        return "  bundled↑" if v is None else f"{v:>9.1f}"

    print(f"\n=== {tag} — per-step time (ms) ===")
    print(f"  {'step':<15}{'R 1-thr':>10}{f'R {ncores}c':>10}{'eager':>9}{'graph':>9}{'triton':>9}")
    for lbl, r1, r12, e, g, t in rows:
        print(f"  {lbl:<15}{r1:>10.1f}{cell(r12):>10}{e:>9.1f}{g:>9.1f}{t:>9.1f}")
    r_best = min(tot_r1, tot_r12)
    print(f"  {'TOTAL':<15}{tot_r1:>10.1f}{tot_r12:>10.1f}{tot_e:>9.1f}{tot_g:>9.1f}{tot_t:>9.1f}")
    print(f"  {'x vs best R':<15}{r_best/tot_r1:>9.1f}x{r_best/tot_r12:>9.1f}x{r_best/tot_e:>8.1f}x{r_best/tot_g:>8.1f}x{r_best/tot_t:>8.1f}x")
    return dict(rows=rows, total=dict(R_1thread=tot_r1, R_multicore=tot_r12,
                                      eager=tot_e, graph=tot_g, triton=tot_t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ncores", type=int, default=os.cpu_count(),
                    help="cores for the charitable multi-core R column")
    ap.add_argument("--casedir", type=str, default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA required.")
    prov = provenance(args.reps)
    prov["r_ncores"] = args.ncores
    print("provenance:", {k: prov[k] for k in ("gpu", "torch", "cuda", "commit")}, "r_ncores:", args.ncores)
    casedir = args.casedir or tempfile.mkdtemp(prefix="gpudeseq_perstep_")
    print("case dir:", casedir)

    ours = time_ours(casedir, "cuda", args.reps)
    rmap = time_r(casedir, args.reps, args.ncores)
    tables = {tag: print_table(tag, ours[tag], rmap[tag], args.ncores) for (_, _, _, _, tag) in CASES}

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"provenance": prov, "ours_ms": ours, "r_ms": rmap, "tables": tables}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
