"""Plot gpu_deseq vs R DESeq2 consistency on the generated fixtures.

Reads fixtures/r_deseq2/<label>/{counts,coldata,results}.csv, runs gpu_deseq
on the same counts, and produces a 2x3 grid per fixture:
    (1) log2FoldChange scatter              (2) lfcSE scatter
    (3) -log10(pvalue) scatter              (4) dispersion scatter (log-log)
    (5) LFC relative error histogram        (6) MA plot overlay

Each panel includes y=x reference, Pearson r, and a rel-err summary.
Plots go to fixtures/r_deseq2/plots/<label>.png plus a summary table CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gpu_deseq import (
    DESeqDataset,
    fit_dispersions,
    fit_size_factors,
    lfc_shrink,
    results,
    wald_test,
)


def _load(label: str):
    path = REPO_ROOT / "fixtures" / "r_deseq2" / label
    counts = pd.read_csv(path / "counts.csv", index_col=0)
    coldata = pd.read_csv(path / "coldata.csv", index_col=0)
    ref = pd.read_csv(path / "results.csv").set_index("gene")
    return counts, coldata, ref


def _run_gpu_deseq(counts: pd.DataFrame, coldata: pd.DataFrame):
    dds = DESeqDataset(
        counts.to_numpy(dtype=np.float64), coldata, design="~ condition",
        gene_ids=list(counts.index), backend="torch",
    )
    fit_size_factors(dds)
    fit_dispersions(dds)
    wald = wald_test(dds, contrast="condition[T.treated]")
    unshrunk = results(wald, cooks_filter=False, independent_filter=False)
    shrunk_fit = lfc_shrink(wald, coeff="condition[T.treated]")
    shrunk = results(shrunk_fit, cooks_filter=False, independent_filter=False)
    return unshrunk, shrunk, dds


def _scatter(ax, x, y, xlabel: str, ylabel: str, title: str, log: bool = False):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    ax.scatter(x, y, s=6, alpha=0.4, color="#1f77b4", edgecolor="none")
    lo = np.nanmin([x, y])
    hi = np.nanmax([x, y])
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--", label="y = x")
    if log:
        ax.set_xscale("log"); ax.set_yscale("log")
    r = np.corrcoef(x, y)[0, 1]
    rel = np.abs(x - y) / (np.abs(y) + 1e-12)
    txt = f"n = {len(x)}\nPearson r = {r:.6f}\np95 |rel err| = {np.percentile(rel, 95):.2e}"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, family="monospace",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)


def _ma_plot(ax, base_mean_r, lfc_r, lfc_ours, title: str):
    ax.scatter(base_mean_r, lfc_r,    s=6, alpha=0.35, color="black", label="R DESeq2")
    ax.scatter(base_mean_r, lfc_ours, s=6, alpha=0.35, color="#d62728", label="gpu_deseq", marker="x")
    ax.set_xscale("log")
    ax.set_xlabel("baseMean (R)")
    ax.set_ylabel("log2FoldChange")
    ax.set_title(title)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)


def _rel_err_hist(ax, x_ours, x_ref, label: str):
    mask = np.isfinite(x_ours) & np.isfinite(x_ref) & (np.abs(x_ref) > 1e-3)
    rel = np.abs(x_ours[mask] - x_ref[mask]) / np.abs(x_ref[mask])
    ax.hist(rel, bins=60, color="#2ca02c", edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlabel(f"relative error in {label}")
    ax.set_ylabel("gene count")
    ax.set_title(f"|gpu - R| / |R| ({label})")
    for q, c in [(50, "#ff7f0e"), (95, "#d62728")]:
        v = np.percentile(rel, q)
        ax.axvline(v, color=c, linestyle="--", linewidth=1, label=f"p{q} = {v:.2e}")
    ax.legend(loc="upper right", fontsize=8)


def _pvalue_scatter(ax, p_ours, p_ref):
    mask = np.isfinite(p_ours) & np.isfinite(p_ref) & (p_ours > 0) & (p_ref > 0)
    x = -np.log10(p_ref[mask]); y = -np.log10(p_ours[mask])
    ax.scatter(x, y, s=6, alpha=0.4, color="#9467bd", edgecolor="none")
    lim = max(x.max(), y.max())
    ax.plot([0, lim], [0, lim], color="black", linewidth=1, linestyle="--", label="y = x")
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.04, 0.96, f"n = {len(x)}\nPearson r = {r:.6f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8, family="monospace",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))
    ax.set_xlabel("-log10(pvalue)  R DESeq2")
    ax.set_ylabel("-log10(pvalue)  gpu_deseq")
    ax.set_title("p-values")
    ax.legend(loc="lower right", fontsize=8)


def _summary_row(label: str, ours: pd.DataFrame, ref: pd.DataFrame, our_disp: np.ndarray,
                 ref_disp: np.ndarray) -> dict:
    merged = ours.join(ref, rsuffix="_r")
    mask = np.isfinite(merged[["log2FoldChange", "log2FoldChange_r",
                                "lfcSE", "lfcSE_r"]]).all(axis=1)
    m = merged[mask]
    dmask = np.isfinite(our_disp) & np.isfinite(ref_disp)
    rel = lambda a, b: np.abs(a - b) / (np.abs(b) + 1e-12)
    return {
        "fixture": label,
        "n_genes": len(m),
        "dispersion_p95_rel": float(np.percentile(rel(our_disp[dmask], ref_disp[dmask]), 95)),
        "lfc_p95_rel":        float(np.percentile(rel(m["log2FoldChange"].to_numpy(), m["log2FoldChange_r"].to_numpy()), 95)),
        "lfc_corr":           float(np.corrcoef(m["log2FoldChange"], m["log2FoldChange_r"])[0, 1]),
        "lfcSE_p95_rel":      float(np.percentile(rel(m["lfcSE"].to_numpy(), m["lfcSE_r"].to_numpy()), 95)),
        "lfcSE_corr":         float(np.corrcoef(m["lfcSE"], m["lfcSE_r"])[0, 1]),
        "pvalue_corr":        float(np.corrcoef(m["pvalue"], m["pvalue_r"])[0, 1]),
    }


def main():
    fixtures_dir = REPO_ROOT / "fixtures" / "r_deseq2"
    plots_dir = fixtures_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(d.name for d in fixtures_dir.iterdir() if d.is_dir() and d.name != "plots")
    summaries = []

    for label in labels:
        print(f"\n[plot] {label}")
        counts, coldata, ref = _load(label)
        unshrunk, shrunk, dds = _run_gpu_deseq(counts, coldata)
        merged = unshrunk.join(ref, rsuffix="_r")
        our_disp = dds.dispersions.cpu().numpy()
        ref_disp = ref["dispersion"].to_numpy()
        has_apeglm = "log2FoldChange_apeglm" in ref.columns and \
            ref["log2FoldChange_apeglm"].notna().any()

        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        fig.suptitle(f"gpu_deseq vs R DESeq2 — fixture: {label}  (n={counts.shape[1]} samples, "
                     f"g={counts.shape[0]} genes)", fontsize=12)

        # Row 1: unshrunk Wald results
        _scatter(axes[0, 0],
                 merged["log2FoldChange_r"].to_numpy(), merged["log2FoldChange"].to_numpy(),
                 "log2FC  R DESeq2", "log2FC  gpu_deseq", "log2 fold change (MLE)")
        _scatter(axes[0, 1],
                 merged["lfcSE_r"].to_numpy(), merged["lfcSE"].to_numpy(),
                 "lfcSE  R DESeq2", "lfcSE  gpu_deseq", "standard error (MLE)")
        _pvalue_scatter(axes[0, 2],
                        merged["pvalue"].to_numpy(), merged["pvalue_r"].to_numpy())

        # Row 2: dispersions + error hist + MA plot
        _scatter(axes[1, 0], ref_disp, our_disp,
                 "dispersion  R DESeq2", "dispersion  gpu_deseq",
                 "dispersion (log-log)", log=True)
        _rel_err_hist(axes[1, 1],
                      merged["log2FoldChange"].to_numpy(), merged["log2FoldChange_r"].to_numpy(),
                      "log2FoldChange")
        _ma_plot(axes[1, 2], ref["baseMean"].to_numpy(),
                 merged["log2FoldChange_r"].to_numpy(), merged["log2FoldChange"].to_numpy(),
                 "MA plot overlay (MLE)")

        # Row 3: apeglm shrunk results
        if has_apeglm:
            shr_merged = shrunk.join(ref[["log2FoldChange_apeglm", "lfcSE_apeglm"]])
            _scatter(axes[2, 0],
                     shr_merged["log2FoldChange_apeglm"].to_numpy(),
                     shr_merged["log2FoldChange"].to_numpy(),
                     "log2FC (apeglm)  R DESeq2", "log2FC (apeglm)  gpu_deseq",
                     "log2 fold change (apeglm shrunk)")
            _scatter(axes[2, 1],
                     shr_merged["lfcSE_apeglm"].to_numpy(),
                     shr_merged["lfcSE"].to_numpy(),
                     "lfcSE (apeglm)  R DESeq2", "lfcSE (apeglm)  gpu_deseq",
                     "standard error (apeglm shrunk)")
            _ma_plot(axes[2, 2], ref["baseMean"].to_numpy(),
                     shr_merged["log2FoldChange_apeglm"].to_numpy(),
                     shr_merged["log2FoldChange"].to_numpy(),
                     "MA plot overlay (apeglm)")
        else:
            for j in range(3):
                axes[2, j].axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = plots_dir / f"{label}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  → {out}")
        summaries.append(_summary_row(label, unshrunk, ref, our_disp, ref_disp))

    df = pd.DataFrame(summaries)
    csv = plots_dir / "summary.csv"
    df.to_csv(csv, index=False)
    print(f"\nSummary written to {csv}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
