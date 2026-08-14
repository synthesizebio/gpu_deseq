"""Real-data validation, step 2 (cuDESeq2 side): run cuDESeq2 on the same real
Bioconductor datasets and quantify agreement with R DESeq2's reference output
(written by the benchmark reference run).

For each dataset it reports, joined per gene:
  - raw LFC (results):   Pearson/Spearman r, max|Δ|, p95 |Δ|
  - Wald stat / pvalue:  correlation
  - final dispersion:    p95 relative error
  - apeGLM-shrunk LFC:   Pearson r, max|Δ|
  - significance calls:  Jaccard of {padj < α} at α = 0.05 and 0.10
and writes validation/results/<name>.json + a parity scatter to
validation/figures/<name>.png.

Reference levels are read from meta.json and reproduced via a pandas Categorical
(base level first) so the contrast matches R exactly (no sign flip).

Usage: PYTHONPATH=src python validation/validate.py [--device cuda|cpu]
                                                      [--from-cache]
                                                      [--skip-timing]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "src")
import torch
from gpu_deseq import DESeqDataset, deseq, results, lfc_shrink
import gpu_deseq._deseq2_core as _core

DATA = Path("validation/data")
BENCH_CACHE = Path("bench/cache")
RESD = Path("validation/results"); RESD.mkdir(parents=True, exist_ok=True)
FIGD = Path("validation/figures"); FIGD.mkdir(parents=True, exist_ok=True)


def _pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else float("nan")


def _spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra = pd.Series(a[m]).rank().to_numpy()
    rb = pd.Series(b[m]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def _p95abs(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.percentile(np.abs(a[m] - b[m]), 95)) if m.any() else float("nan")


#: |shrunk LFC - R| above which a gene counts as landing in a different
#: posterior basin rather than merely differing numerically. Agreeing genes sit
#: at ~1e-4; basin flips are O(1), so anything in between separates them.
SHRUNK_DIVERGENCE_THRESHOLD = 0.1


def _shrunk_divergence(a, b, thresh=SHRUNK_DIVERGENCE_THRESHOLD):
    """Count genes whose apeGLM-shrunk LFC lands in a different basin than R's."""
    m = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[m] - b[m])
    return {
        "shrunk_lfc_divergent_threshold": float(thresh),
        "shrunk_lfc_n_divergent": int((d > thresh).sum()),
        "shrunk_lfc_n_compared": int(m.sum()),
        "shrunk_lfc_max_abs": float(d.max()) if d.size else float("nan"),
    }


def _jaccard(sa, sb):
    inter = (sa & sb).sum()
    union = (sa | sb).sum()
    return float(inter / union) if union else 1.0


def run_case(name, device, *, from_cache=False, skip_timing=False):
    d = DATA / name
    meta = json.loads((d / "meta.json").read_text())
    counts = pd.read_csv(d / "counts.csv", index_col=0)
    coldata = pd.read_csv(d / "coldata.csv", index_col=0)
    factor, ref, nonref = meta["factor"], meta["ref"], meta["nonref"]
    levels = list(pd.unique(coldata[factor]))
    # base level first => formulaic uses it as reference, matching R's relevel().
    coldata[factor] = pd.Categorical(coldata[factor], categories=[ref] + [l for l in levels if l != ref])
    contrast = f"{factor}[T.{nonref}]"

    reference = BENCH_CACHE / name
    if from_cache:
        cached = pd.read_csv(reference / "cu_reference_results.csv", index_col=0)
        if len(cached) != len(counts):
            raise ValueError(f"{name}: cached row count does not match counts")
        cached.index = counts.index
        cached.index.name = "gene"
        res = cached[["baseMean", "log2FoldChange", "stat", "padj"]].copy()
        from scipy.stats import norm
        res["pvalue"] = 2.0 * norm.sf(np.abs(res["stat"]))
        shr = cached[["shrunk_lfc"]].rename(columns={"shrunk_lfc": "log2FoldChange"})
        our_disp = cached["dispersion"]
    else:
        dds = DESeqDataset(counts.to_numpy(np.float64), coldata, design=meta["design"],
                           gene_ids=list(counts.index), sample_ids=list(counts.columns),
                           backend="torch").to(device)
        fit = deseq(dds, contrast=contrast)
        res = results(fit)
        shr = results(lfc_shrink(fit, coeff=contrast),
                      cooks_filter=False, independent_filter=False)
        our_disp = pd.Series(dds.dispersions.cpu().numpy(), index=list(counts.index))

    r_res = pd.read_csv(reference / "r_results.csv").set_index("gene")
    r_disp = pd.read_csv(d / "r_disp_details.csv").set_index("gene")["dispersion"]
    r_shr = pd.read_csv(reference / "r_shrink.csv").set_index("gene")

    j = res.join(r_res, rsuffix="_r")
    o_lfc, r_lfc = j["log2FoldChange"].to_numpy(), j["log2FoldChange_r"].to_numpy()
    o_p, r_p = j["pvalue"].to_numpy(), j["pvalue_r"].to_numpy()
    o_padj, r_padj = j["padj"].to_numpy(), j["padj_r"].to_numpy()
    js = shr["log2FoldChange"].reindex(r_shr.index)
    o_slfc, r_slfc = js.to_numpy(), r_shr["log2FoldChange"].to_numpy()
    dj = our_disp.reindex(r_disp.index)
    disp_rel = np.abs(dj.to_numpy() - r_disp.to_numpy()) / (np.abs(r_disp.to_numpy()) + 1e-12)

    def sig(padj, a):
        return (padj < a) & np.isfinite(padj)

    report = {
        "meta": meta,
        "n_genes_compared": int(np.isfinite(o_lfc) & np.isfinite(r_lfc)).sum() if False else int(len(j)),
        "lfc_pearson": _pearson(o_lfc, r_lfc),
        "lfc_spearman": _spearman(o_lfc, r_lfc),
        "lfc_p95_abs": _p95abs(o_lfc, r_lfc),
        "stat_pearson": _pearson(j["stat"].to_numpy(), j["stat_r"].to_numpy()),
        "pvalue_pearson": _pearson(o_p, r_p),
        "padj_p95_abs": _p95abs(o_padj, r_padj),
        "dispersion_p95_rel": float(np.percentile(disp_rel[np.isfinite(disp_rel)], 95)),
        "shrunk_lfc_pearson": _pearson(o_slfc, r_slfc),
        "shrunk_lfc_p95_abs": _p95abs(o_slfc, r_slfc),
        # apeGLM's posterior is bimodal for extreme-effect genes, so two L-BFGS
        # implementations can settle in different basins on a near-degenerate
        # gene and disagree by O(1) while every other gene agrees to ~1e-4.
        # Count them explicitly: correlation and p95 both hide a handful of
        # O(1) outliers, and the paper quotes this count.
        **_shrunk_divergence(o_slfc, r_slfc),
        "sig_jaccard_0.05": _jaccard(sig(o_padj, 0.05), sig(r_padj, 0.05)),
        "sig_jaccard_0.10": _jaccard(sig(o_padj, 0.10), sig(r_padj, 0.10)),
        "n_sig_ours_0.05": int(sig(o_padj, 0.05).sum()),
        "n_sig_r_0.05": int(sig(r_padj, 0.05).sum()),
    }
    if not skip_timing:
        report["timing_ms"] = _time_modes(
            counts, coldata, meta["design"], contrast, device, reps=5
        )
    timing_meta = json.loads((reference / "r_timings.json").read_text())
    meta["r_full_ms"] = timing_meta["total"]
    report["reference_versions"] = {
        key: timing_meta.get(key)
        for key in ("r_version", "deseq2_version", "apeglm_version")
    }
    _plot(name, o_lfc, r_lfc, o_slfc, r_slfc, o_padj, r_padj, report)
    (RESD / f"{name}.json").write_text(json.dumps(report, indent=2))
    return report


def _time_modes(counts, coldata, design, contrast, device, reps=5):
    """Median wall time (ms) of the full cuDESeq2 pipeline (size factors ->
    dispersions -> Wald -> results -> apeGLM shrink) in each execution mode."""
    def build():
        return DESeqDataset(counts.to_numpy(np.float64), coldata, design=design,
                            gene_ids=list(counts.index), sample_ids=list(counts.columns),
                            backend="torch").to(device)

    def run(kw):
        d = build()
        f = deseq(d, contrast=contrast, **kw)
        results(f)
        lfc_shrink(f, coeff=contrast)

    out = {}
    cuda = device == "cuda"
    for mode, kw in [("eager", {}), ("graph", {"use_cuda_graph": True}), ("triton", {"use_triton": True})]:
        _core._GRAPH_CACHE.clear()
        run(kw)                                   # warm / compile / capture
        if cuda: torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            if cuda: torch.cuda.synchronize()
            s = time.perf_counter(); run(kw)
            if cuda: torch.cuda.synchronize()
            ts.append(time.perf_counter() - s)
        out[mode] = float(np.median(ts)) * 1e3
    return out


def _plot(name, o_lfc, r_lfc, o_slfc, r_slfc, o_padj, r_padj, rep):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.9))
    def scatter(a, o, r, title, sub):
        m = np.isfinite(o) & np.isfinite(r)
        a.scatter(r[m], o[m], s=5, alpha=0.35, edgecolors="none", color="#3f3fd6")
        lo, hi = np.nanpercentile(np.concatenate([r[m], o[m]]), [0.5, 99.5])
        a.plot([lo, hi], [lo, hi], color="#b4530a", lw=1, ls="--")
        a.set_title(title, fontsize=11); a.set_xlabel("R DESeq2"); a.set_ylabel("cuDESeq2")
        a.text(0.04, 0.92, sub, transform=a.transAxes, fontsize=9, va="top", family="monospace")
    scatter(ax[0], o_lfc, r_lfc, "log2 fold change", f"r={rep['lfc_pearson']:.5f}")
    scatter(ax[1], o_slfc, r_slfc, "apeGLM-shrunk LFC", f"r={rep['shrunk_lfc_pearson']:.5f}")
    mp = np.isfinite(o_padj) & np.isfinite(r_padj)
    ax[2].scatter(-np.log10(r_padj[mp] + 1e-300), -np.log10(o_padj[mp] + 1e-300),
                  s=5, alpha=0.35, edgecolors="none", color="#067a54")
    lim = np.nanpercentile(-np.log10(r_padj[mp] + 1e-300), 99.5)
    ax[2].plot([0, lim], [0, lim], color="#b4530a", lw=1, ls="--")
    ax[2].set_title("-log10 padj"); ax[2].set_xlabel("R DESeq2"); ax[2].set_ylabel("cuDESeq2")
    ax[2].text(0.04, 0.92, f"Jaccard@.05={rep['sig_jaccard_0.05']:.3f}",
               transform=ax[2].transAxes, fontsize=9, va="top", family="monospace")
    fig.suptitle(f"cuDESeq2 vs R DESeq2 1.52.0 — {name}  ({rep['meta']['design']})", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGD / f"{name}.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if _cuda() else "cpu")
    ap.add_argument("--from-cache", action="store_true",
                    help="score the current benchmark captures without recomputing")
    ap.add_argument("--skip-timing", action="store_true",
                    help="do not run a timing loop")
    args = ap.parse_args()
    cases = sorted(p.name for p in DATA.iterdir() if (p / "meta.json").exists()) if DATA.exists() else []
    if not cases:
        sys.exit("No prepared datasets. Run: Rscript validation/fetch_and_reference.R")
    print(f"device={args.device}\n")
    reports = {}
    for name in cases:
        r = run_case(name, args.device, from_cache=args.from_cache,
                     skip_timing=args.skip_timing)
        reports[name] = r
        print(f"=== {name}  ({r['meta']['design']}, {r['meta']['n_samples']} samples) ===")
        print(f"  raw LFC        Pearson r = {r['lfc_pearson']:.6f}   Spearman = {r['lfc_spearman']:.6f}   p95|Δ| = {r['lfc_p95_abs']:.2e}")
        print(f"  dispersion     p95 rel   = {r['dispersion_p95_rel']:.2e}")
        print(f"  shrunk LFC     Pearson r = {r['shrunk_lfc_pearson']:.6f}")
        print(f"  significance   Jaccard@0.05 = {r['sig_jaccard_0.05']:.4f}"
              f"   (sig: ours={r['n_sig_ours_0.05']}, R={r['n_sig_r_0.05']})")
        print(f"  figure -> validation/figures/{name}.png\n")

    if args.skip_timing:
        return

    # Timing table: R (1-thread) vs cuDESeq2 eager/graph/triton, full pipeline.
    print("full-pipeline wall time (ms) + speedup vs R (1-thread):")
    print(f"  {'case':<14}{'P':>2}{'R 1-thr':>10}{'eager':>9}{'graph':>9}{'triton':>9}{'  triton×R':>10}")
    for name in cases:
        r = reports[name]; t = r["timing_ms"]; rf = r.get("r_full_ms")
        sx = f"{rf / t['triton']:>8.1f}×" if rf else "     n/a"
        rfs = f"{rf:>10.0f}" if rf else f"{'?':>10}"
        print(f"  {name:<14}{r['meta'].get('P','?'):>2}{rfs}{t['eager']:>9.0f}{t['graph']:>9.0f}{t['triton']:>9.0f}{sx:>10}")


def _cuda():
    try:
        import torch; return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
