"""Standard differential-expression figures for the paper, drawn twice: once from
R DESeq2 1.52.0's reference output and once from cuDESeq2, so that parity is shown
in the plots a DE analyst actually reads rather than only as a table of residuals.

Two figures are produced in paper/figures/:

  de_diagnostics_parity.{pdf,png}
      2 rows (R DESeq2 / cuDESeq2) x 3 cols (dispersion-estimate plot, MA plot,
      volcano) for one dataset. The rows are drawn by identical code from the two
      pipelines' outputs; the point of the figure is that they are
      indistinguishable.

  de_parity_concordance.{pdf,png}
      Quantitative agreement across all prepared datasets: p-value histograms,
      concordance-at-top-N of the ranked gene lists, a Bland-Altman residual of
      the shrunk LFC against expression, and the Jaccard index of the called sets
      as a function of the significance threshold.

Inputs, per case in validation/data/<name>/:
  counts.csv, coldata.csv, meta.json      (written by fetch_and_reference.R)
  r_results.csv, r_shrink.csv             (      "                        )
  r_disp_details.csv                      (written by export_r_dispersion_details.R)
cuDESeq2's own output is computed here and cached as ours.csv next to them.

Usage:
  PYTHONPATH=src python validation/make_paper_figures.py [--device cuda|cpu]
                                                         [--case airway] [--refresh]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, "src")
from gpu_deseq import DESeqDataset, fit_size_factors, fit_dispersions, wald_test, results, lfc_shrink

DATA = Path("validation/data")
BENCH_CACHE = Path("bench/cache")
FIGD = Path("paper/figures"); FIGD.mkdir(parents=True, exist_ok=True)

#: Dataset shown in the six-panel diagnostics figure. airway is the canonical
#: Bioconductor example, and its ~cell+dex slice is the hardest of the six cases
#: (P=5 on n=8, i.e. 3 residual dof, the branch that enters R's simulation-based
#: prior-variance estimator).
DEFAULT_CASE = "airway"

#: Order cases largest-design-first for the multi-dataset panels, so the legend
#: reads in a stable order that does not depend on the filesystem.
CASE_ORDER = ["gtex_blood_muscle", "airway", "airway_cell", "airway_dex",
              "pasilla_2fac", "pasilla"]

# --- palette -----------------------------------------------------------------
# Categorical slots assigned in fixed order (never cycled); grey is context, not
# a slot. Validated for the light/print surface with the dataviz validator: the
# six-slot order clears the adjacent-pair CVD and normal-vision floors, and the
# low-contrast slots (aqua/yellow/magenta) are always additionally carried by a
# direct end label, never by colour alone.
SURFACE = "#ffffff"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9b9a95"
GRID = "#dedcd6"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BLUE, ORANGE = SLOTS[0], SLOTS[1]

ALPHA = 0.05          # significance threshold used in the MA / volcano panels
FS = 7.4              # base font size (figures are drawn at final print width)


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": FS, "axes.titlesize": FS + 0.8, "axes.labelsize": FS,
        "xtick.labelsize": FS - 0.8, "ytick.labelsize": FS - 0.8,
        "legend.fontsize": FS - 1.0, "legend.frameon": False,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.1, "pdf.fonttype": 42,
    })


# --- data --------------------------------------------------------------------
def _meta(name):
    return json.loads((DATA / name / "meta.json").read_text())


def r_side(name):
    """R DESeq2's reference output for one case, one row per gene."""
    d = DATA / name
    reference = BENCH_CACHE / name
    res = pd.read_csv(reference / "r_results.csv").set_index("gene")
    shr = pd.read_csv(reference / "r_shrink.csv").set_index("gene")
    det = pd.read_csv(d / "r_disp_details.csv").set_index("gene")
    out = pd.DataFrame({
        "baseMean": det["baseMean"],
        "dispGeneEst": det["dispGeneEst"],
        "dispFit": det["dispFit"],
        "dispersion": det["dispersion"],
        "dispOutlier": det["dispOutlier"].astype(bool),
        "log2FoldChange": res["log2FoldChange"],
        "stat": res["stat"],
        "pvalue": res["pvalue"],
        "padj": res["padj"],
        "shrunkLFC": shr["log2FoldChange"],
    })
    return out


def our_side(name, device, refresh=False):
    """cuDESeq2's output for one case, same columns as r_side(), cached on disk."""
    cache = DATA / name / "ours.csv"
    reference_cache = BENCH_CACHE / name / "cu_reference_results.csv"
    if reference_cache.exists() and not refresh:
        current = pd.read_csv(reference_cache, index_col=0)
        details = pd.read_csv(cache).set_index("gene") if cache.exists() else None
        if details is not None:
            for column in ("dispGeneEst", "dispFit", "dispOutlier"):
                current[column] = details[column]
        if "pvalue" not in current:
            from scipy.stats import norm
            current["pvalue"] = 2.0 * norm.sf(np.abs(current["stat"]))
        current = current.rename(columns={"shrunk_lfc": "shrunkLFC"})
        return current
    if cache.exists() and not refresh:
        return pd.read_csv(cache).set_index("gene")

    d, meta = DATA / name, _meta(name)
    counts = pd.read_csv(d / "counts.csv", index_col=0)
    coldata = pd.read_csv(d / "coldata.csv", index_col=0)
    factor, ref, nonref = meta["factor"], meta["ref"], meta["nonref"]
    levels = list(pd.unique(coldata[factor]))
    # Base level first so the contrast matches R's relevel() (no sign flip).
    coldata[factor] = pd.Categorical(
        coldata[factor], categories=[ref] + [l for l in levels if l != ref])
    contrast = f"{factor}[T.{nonref}]"

    dds = DESeqDataset(counts.to_numpy(np.float64), coldata, design=meta["design"],
                       gene_ids=list(counts.index), sample_ids=list(counts.columns),
                       backend="torch").to(device)
    fit_size_factors(dds)
    fit_dispersions(dds)
    fit = wald_test(dds, contrast=contrast)
    res = results(fit)
    shr = results(lfc_shrink(fit, coeff=contrast),
                  cooks_filter=False, independent_filter=False)

    def np_(t):
        return t.detach().cpu().numpy()

    genes = list(counts.index)
    # DESeq2's "mean of normalized counts", the x axis of plotDispEsts.
    base_mean = np_(dds.normalized_counts).mean(axis=1)
    gene_est, trend, final = (np_(dds.dispersions_gene_wise), np_(dds.dispersion_trend),
                              np_(dds.dispersions))
    # DESeq2's dispOutlier flag, by the same rule cuDESeq2 applies internally
    # (apply_outlier_keep_mle): a gene whose MLE sits more than 2 residual SDs
    # above the trend on the log scale keeps its MLE instead of the MAP estimate.
    with np.errstate(divide="ignore", invalid="ignore"):
        outlier = np.log(gene_est) > np.log(trend) + 2.0 * np.sqrt(dds.squared_logres)

    out = pd.DataFrame({
        "gene": genes,
        "baseMean": base_mean,
        "dispGeneEst": gene_est,
        "dispFit": trend,
        "dispersion": final,
        "dispOutlier": np.nan_to_num(outlier, nan=0.0).astype(bool),
        "log2FoldChange": res["log2FoldChange"].reindex(genes).to_numpy(),
        "stat": res["stat"].reindex(genes).to_numpy(),
        "pvalue": res["pvalue"].reindex(genes).to_numpy(),
        "padj": res["padj"].reindex(genes).to_numpy(),
        "shrunkLFC": shr["log2FoldChange"].reindex(genes).to_numpy(),
    }).set_index("gene")
    out.to_csv(cache)
    return out


def load(name, device, refresh=False):
    """R and cuDESeq2 output for one case, restricted to the shared gene set."""
    r, o = r_side(name), our_side(name, device, refresh)
    common = r.index.intersection(o.index)
    return r.loc[common], o.loc[common], _meta(name)


# --- panels ------------------------------------------------------------------
#: Dispersion floor for the log-scale dispersion panel. DESeq2's plotDispEsts
#: does the same (pmax(py, ymin)): the gene-wise MLE hits its own lower bound on
#: low-count genes, and those genes are drawn as a stripe along the axis.
DISP_FLOOR = 1e-8


def panel_dispersion(ax, df, lims=None, legend=False):
    """DESeq2's plotDispEsts: gene-wise MLE, fitted trend and final estimate
    against the mean of normalized counts."""
    m = (df["baseMean"] > 0) & np.isfinite(df["dispGeneEst"])
    x = df["baseMean"][m].to_numpy()
    ge, ft, fi = (df["dispGeneEst"][m].to_numpy(), df["dispFit"][m].to_numpy(),
                  df["dispersion"][m].to_numpy())
    out = df["dispOutlier"][m].to_numpy()
    ax.scatter(x, np.maximum(ge, DISP_FLOOR), s=0.7, alpha=0.30, linewidths=0,
               color=MUTED, rasterized=True)
    ax.scatter(x, np.maximum(fi, DISP_FLOOR), s=0.7, alpha=0.45, linewidths=0,
               color=BLUE, rasterized=True)
    o = np.argsort(x)
    ax.plot(x[o], ft[o], color=ORANGE, lw=1.2, solid_capstyle="round")
    if out.any():
        ax.scatter(x[out], np.maximum(fi[out], DISP_FLOOR), s=5.0, facecolors="none",
                   edgecolors=INK, linewidths=0.3, alpha=0.5, rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mean of normalized counts")
    ax.set_ylabel("dispersion")
    ax.text(0.96, 0.94, f"{out.sum():,} outliers", transform=ax.transAxes,
            fontsize=FS - 1.4, color=INK2, ha="right", va="top")
    if legend:
        ax.legend(handles=[
            Line2D([], [], ls="", marker="o", ms=2.6, mfc=MUTED, mec="none", label="gene-wise MLE"),
            Line2D([], [], color=ORANGE, lw=1.2, label="fitted trend"),
            Line2D([], [], ls="", marker="o", ms=2.6, mfc=BLUE, mec="none", label="final (MAP)"),
            Line2D([], [], ls="", marker="o", ms=3.0, mfc="none", mec=INK, mew=0.4, label="MLE kept (outlier)"),
        ], loc="lower left", handletextpad=0.3, borderpad=0.15, labelspacing=0.2)
    if lims:
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1])


def _edge_markers(ax, x, y, lo, hi, color, alpha):
    """Draw out-of-range points as triangles on the axis boundary, as plotMA does."""
    for sel, marker, edge in ((y > hi, "^", hi), (y < lo, "v", lo)):
        if sel.any():
            ax.scatter(x[sel], np.full(sel.sum(), edge), s=3.5, marker=marker,
                       linewidths=0, color=color, alpha=alpha, clip_on=False,
                       rasterized=True)


def panel_ma(ax, df, lims=None):
    """MA plot: apeGLM-shrunk LFC against expression, coloured by the call."""
    m = (df["baseMean"] > 0) & np.isfinite(df["shrunkLFC"])
    x, y = df["baseMean"][m].to_numpy(), df["shrunkLFC"][m].to_numpy()
    sig = df["padj"][m].to_numpy() < ALPHA
    ax.axhline(0.0, color=INK2, lw=0.5, zorder=1)
    for s, col, a in ((~sig, MUTED, 0.28), (sig, BLUE, 0.55)):
        ax.scatter(x[s], y[s], s=0.8, alpha=a, linewidths=0, color=col, rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("mean of normalized counts")
    ax.set_ylabel("shrunk log$_2$ fold change")
    ax.text(0.03, 0.04, f"{sig.sum():,} at padj < {ALPHA}", transform=ax.transAxes,
            fontsize=FS - 1.4, color=BLUE, va="bottom")
    if lims:
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1])
        lo, hi = lims[1]
        for s, col, a in ((~sig, MUTED, 0.28), (sig, BLUE, 0.55)):
            _edge_markers(ax, x[s], y[s], lo, hi, col, a)


def panel_volcano(ax, df, lims=None, legend=False):
    """Volcano: shrunk LFC against -log10 adjusted p-value."""
    m = np.isfinite(df["shrunkLFC"]) & np.isfinite(df["padj"])
    x = df["shrunkLFC"][m].to_numpy()
    y = -np.log10(np.clip(df["padj"][m].to_numpy(), 1e-300, None))
    sig = df["padj"][m].to_numpy() < ALPHA
    ax.axhline(-np.log10(ALPHA), color=INK2, lw=0.5, ls=(0, (3, 2)), zorder=1)
    for s, col, a in ((~sig, MUTED, 0.28), (sig, BLUE, 0.55)):
        ax.scatter(x[s], y[s], s=0.8, alpha=a, linewidths=0, color=col, rasterized=True)
    ax.set_xlabel("shrunk log$_2$ fold change")
    ax.set_ylabel(r"$-\log_{10}$ padj")
    ax.text(0.97, 0.94, f"$\\alpha$ = {ALPHA}", transform=ax.transAxes,
            fontsize=FS - 1.4, color=INK2, ha="right", va="top")
    if legend:
        # The call colouring is shared with the MA panel; label it once, here,
        # where there is empty space.
        ax.legend(handles=[
            Line2D([], [], ls="", marker="o", ms=2.6, mfc=BLUE, mec="none",
                   label=f"padj < {ALPHA}"),
            Line2D([], [], ls="", marker="o", ms=2.6, mfc=MUTED, mec="none",
                   label="not called"),
            Line2D([], [], ls="", marker="^", ms=2.6, mfc=INK2, mec="none",
                   label="beyond axis"),
        ], loc="upper left", handletextpad=0.3, borderpad=0.15, labelspacing=0.2)
    if lims:
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1])
        lo, hi = lims[1]
        for s, col, a in ((~sig, MUTED, 0.28), (sig, BLUE, 0.55)):
            _edge_markers(ax, x[s], y[s], lo, hi, col, a)


def _diagnostic_limits(r):
    """Axis limits for all six panels, taken from R's output and reused for
    cuDESeq2 so the two rows are compared on one scale rather than each rescaled
    to its own data."""
    bm = r["baseMean"][r["baseMean"] > 0].to_numpy()
    xlim = (10 ** (np.log10(bm.min()) - 0.15), 10 ** (np.log10(bm.max()) + 0.15))
    disp = np.concatenate([r["dispGeneEst"].dropna().to_numpy(),
                           r["dispersion"].dropna().to_numpy()])
    dlim = (DISP_FLOOR / 3, 10 ** (np.log10(max(disp.max(), 10)) + 0.3))
    # plotMA clamps the fold-change axis and puts the excess on the boundary.
    slfc = np.abs(r["shrunkLFC"].dropna().to_numpy())
    mlim = max(2.0, float(np.percentile(slfc, 99.9)))
    padj = -np.log10(np.clip(r["padj"].dropna().to_numpy(), 1e-300, None))
    vlim = float(np.percentile(padj, 99.98)) * 1.08
    vx = float(np.nanmax(np.abs(r["shrunkLFC"].to_numpy()))) * 1.06
    return {"disp": (xlim, dlim), "ma": (xlim, (-mlim, mlim)),
            "volcano": ((-vx, vx), (-vlim * 0.02, vlim))}


def fig_diagnostics(name, device, refresh):
    """Six panels: the three standard DESeq2 diagnostics, R over cuDESeq2."""
    r, o, meta = load(name, device, refresh)
    lims = _diagnostic_limits(r)
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.55))
    rows = [("R DESeq2 1.52.0", r), ("cuDESeq2", o)]
    for i, (label, df) in enumerate(rows):
        top = i == 0
        panel_dispersion(axes[i, 0], df, lims["disp"], legend=top)
        panel_ma(axes[i, 1], df, lims["ma"])
        panel_volcano(axes[i, 2], df, lims["volcano"], legend=top)
    # Column titles once, on the top row; the pipeline is named per row instead.
    for c, t in enumerate(["dispersion estimates", "MA plot", "volcano plot"]):
        axes[0, c].set_title(t, color=INK, pad=4)
    for ax in axes[0, :]:
        ax.set_xlabel("")
    fig.suptitle(f"{name}   ({meta['design']}, n={meta['n_samples']}, "
                 f"{meta['n_genes']:,} genes, P={meta['P']})",
                 fontsize=FS + 0.6, color=INK2, y=0.997)
    fig.tight_layout(rect=(0.045, 0, 1, 0.955), h_pad=1.2, w_pad=1.3)
    # Row labels, placed after layout so they track the final axes positions.
    for i, (label, _) in enumerate(rows):
        bb = axes[i, 0].get_position()
        fig.text(0.012, (bb.y0 + bb.y1) / 2, label, rotation=90, va="center",
                 ha="left", fontsize=FS + 0.8, color=INK)
    _save(fig, "de_diagnostics_parity")


# --- concordance figure ------------------------------------------------------
def _rank_order(stat):
    """Gene indices in the order a DE analyst would rank them: most significant
    first, genes with no test (filtered out) last.

    Ranking uses |Wald statistic| rather than the p-value, which is the same
    ordering without the numerical hazard: on large datasets R reports p = 0 for
    hundreds of genes whose true p-value has underflowed, so ranking by p-value
    would compare arbitrary tie-breaks among those genes instead of comparing the
    two pipelines.
    """
    key = np.where(np.isfinite(stat), -np.abs(stat), np.inf)
    return np.argsort(key, kind="stable")


def concordance_at_top(r_stat, o_stat, ns):
    """Fraction of the top-N ranked genes shared by the two rankings, per N."""
    ro, oo = _rank_order(r_stat), _rank_order(o_stat)
    return [len(set(ro[:n]) & set(oo[:n])) / n for n in ns]


def jaccard_vs_alpha(r_padj, o_padj, alphas):
    out = []
    for a in alphas:
        sr = np.isfinite(r_padj) & (r_padj < a)
        so = np.isfinite(o_padj) & (o_padj < a)
        u = (sr | so).sum()
        out.append(float((sr & so).sum() / u) if u else 1.0)
    return out


#: Marker per dataset, so the six near-coincident curves of panels (b) and (d)
#: are separable by shape as well as by hue.
MARKERS = ["o", "s", "^", "D", "v", "P"]


def _floor_ylim(worst, pad=0.04):
    """Lower y limit for an agreement panel: below the worst point, with room for
    the legend, so no curve is ever clipped out of the axes."""
    return min(0.99, np.floor((worst - pad) * 100) / 100)


def fig_concordance(cases, device, refresh, focus):
    loaded = {n: load(n, device, refresh) for n in cases}
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.3))

    # (a) p-value histograms, overlaid, on the focus dataset.
    ax = axes[0, 0]
    r, o, meta = loaded[focus]
    bins = np.linspace(0, 1, 26)
    ax.hist(r["pvalue"].dropna(), bins=bins, color=BLUE, alpha=0.5,
            label="R DESeq2 1.52.0", linewidth=0)
    ax.hist(o["pvalue"].dropna(), bins=bins, histtype="step", color=ORANGE,
            lw=1.0, label="cuDESeq2")
    ax.set_xlabel("p-value"); ax.set_ylabel("genes")
    ax.set_title(f"(a) p-value distribution — {focus}", color=INK)
    ax.legend(loc="upper center", handletextpad=0.5, borderpad=0.2, labelspacing=0.25)

    # (b) concordance at top N of the ranked gene lists, every dataset.
    ax = axes[0, 1]
    worst = 1.0
    for i, name in enumerate(cases):
        r, o, _ = loaded[name]
        top = max(20, min(2000, len(r) // 4))
        ns = np.unique(np.round(np.logspace(1, np.log10(top), 14)).astype(int))
        c = concordance_at_top(r["stat"].to_numpy(), o["stat"].to_numpy(), ns)
        worst = min(worst, min(c))
        ax.plot(ns, c, color=SLOTS[i], lw=1.0, marker=MARKERS[i], ms=2.4,
                mew=0, label=name)
    ax.set_xscale("log")
    ax.set_xlabel(r"top $N$ genes by $|$Wald statistic$|$")
    ax.set_ylabel("fraction of the top $N$ shared")
    ax.set_ylim(_floor_ylim(worst), 1.006)
    ax.set_title("(b) agreement of the ranked gene lists", color=INK)
    ax.legend(loc="lower right", ncol=2, handletextpad=0.4, columnspacing=0.9,
              borderpad=0.2, labelspacing=0.25, fontsize=FS - 2.0)

    # (c) Bland-Altman of the shrunk LFC against expression, on the focus set.
    ax = axes[1, 0]
    r, o, _ = loaded[focus]
    m = np.isfinite(r["shrunkLFC"]) & np.isfinite(o["shrunkLFC"])
    x = r["baseMean"][m].to_numpy()
    d = o["shrunkLFC"][m].to_numpy() - r["shrunkLFC"][m].to_numpy()
    p95 = float(np.percentile(np.abs(d), 95))
    ax.axhline(0.0, color=INK2, lw=0.5)
    for s in (+p95, -p95):
        ax.axhline(s, color=ORANGE, lw=0.8, ls=(0, (3, 2)))
    ax.scatter(x, d, s=0.7, alpha=0.30, linewidths=0, color=BLUE, rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("mean of normalized counts")
    ax.set_ylabel("shrunk LFC:  cuDESeq2 $-$ R")
    ax.set_title(f"(c) shrunk-LFC residual — {focus}", color=INK)
    ax.text(0.03, 0.95, f"p95 $|\\Delta|$ = {p95:.1e}   (dashed)\n"
            f"max $|\\Delta|$ = {np.abs(d).max():.1e}", transform=ax.transAxes,
            fontsize=FS - 1.6, color=INK2, va="top")

    # (d) agreement of the called sets as the threshold moves. The curves are
    # granular in proportion to 1/(set size), so each legend entry carries the
    # number of genes R calls at alpha = 0.05: the index moves by ~1/n per gene,
    # which is what makes the smallest set (airway_cell, ~200 genes) the noisiest.
    ax = axes[1, 1]
    alphas = np.logspace(-3, np.log10(0.2), 28)
    worst = 1.0
    for i, name in enumerate(cases):
        r, o, _ = loaded[name]
        rp = r["padj"].to_numpy()
        n_r = int((np.isfinite(rp) & (rp < ALPHA)).sum())
        j = jaccard_vs_alpha(rp, o["padj"].to_numpy(), alphas)
        worst = min(worst, min(j))
        ax.plot(alphas, j, color=SLOTS[i], lw=1.0, marker=MARKERS[i], ms=2.4,
                mew=0, label=f"{name}  ($n$={n_r:,})")
    ax.axvline(ALPHA, color=INK2, lw=0.5, ls=(0, (3, 2)))
    lo = _floor_ylim(worst)
    # Blended transform: pinned to the threshold in x, to clear space in y.
    ax.text(ALPHA, 0.28, rf" $\alpha$={ALPHA}", rotation=90, fontsize=FS - 2.0,
            color=INK2, va="bottom", ha="left",
            transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel(r"significance threshold $\alpha$ on padj")
    ax.set_ylabel("Jaccard index of the called sets")
    ax.set_ylim(lo, 1.006)
    ax.set_title(r"(d) agreement of the called sets vs $\alpha$", color=INK)
    ax.legend(loc="lower left", ncol=2, handletextpad=0.4, columnspacing=0.9,
              borderpad=0.2, labelspacing=0.25, fontsize=FS - 2.2)

    fig.tight_layout(h_pad=1.7, w_pad=1.6)
    _save(fig, "de_parity_concordance")


def _save(fig, stem):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        fig.savefig(FIGD / f"{stem}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  wrote {FIGD}/{stem}.pdf + .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if _cuda() else "cpu")
    ap.add_argument("--case", default=DEFAULT_CASE, help="dataset for the six-panel figure")
    ap.add_argument("--refresh", action="store_true", help="recompute cached cuDESeq2 output")
    args = ap.parse_args()

    have = {p.name for p in DATA.iterdir() if (p / "r_disp_details.csv").exists()}
    missing = {p.name for p in DATA.iterdir() if (p / "meta.json").exists()} - have
    if missing:
        print(f"  note: no r_disp_details.csv for {sorted(missing)} "
              f"(run validation/export_r_dispersion_details.R) — skipping")
    cases = [c for c in CASE_ORDER if c in have] + sorted(have - set(CASE_ORDER))
    if not cases:
        sys.exit("No cases with R dispersion details. "
                 "Run: Rscript validation/export_r_dispersion_details.R")
    focus = args.case if args.case in cases else cases[0]

    _style()
    print(f"device={args.device}  focus={focus}  cases={len(cases)}")
    fig_diagnostics(focus, args.device, args.refresh)
    fig_concordance(cases, args.device, args.refresh, focus)


def _cuda():
    try:
        import torch; return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
